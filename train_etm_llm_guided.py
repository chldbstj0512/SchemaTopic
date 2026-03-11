"""
LLM-guided 2nd-stage ETM training.

파이프라인 가정:
1) 1차 ETM 학습: train_etm.py → results/top_words.txt, metrics 등 생성
2) LLM refine: llm_refine.py --topic_words_file results/top_words.txt ...
   → llm_refine 결과 디렉토리(out_dir)에 words 파일(step3 기반)이 생성됨
3) 2차 ETM 학습 (이 스크립트):
   - LLM words 파일을 anchor로 사용
   - LLM 토픽 개수만큼 num_topics를 설정
   - anchor 단어 인덱스에 높은 확률이 가도록 loss에 regularization term 추가
"""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dataset import load_20news
from data import get_batch
from evaluation import run_evaluation, get_top_words_per_topic
from NTMs.etm import ETM
from llm_topic_prior_utils import (
    load_anchor_words_from_llm_words_file,
    load_anchor_words_from_step3_json,
    build_anchor_indices,
    summarize_anchor_coverage,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
if device.type == "cpu":
    print("Warning: CUDA not available, training on CPU (will be slow).")
else:
    print("Using device:", device)


def train_etm_with_anchors(
    train_bow: torch.Tensor,
    args,
    vocab,
    anchor_indices_per_topic,
    emsize: int,
    embeddings: np.ndarray,
):
    """
    2차 ETM 학습 루프.

    - num_topics = len(anchor_indices_per_topic)
    - anchor_indices_per_topic[k] = anchor 단어 인덱스 리스트
    - anchor loss: anchor 위치의 beta 확률이 커지도록 -log(beta)를 최소화
    """
    num_topics = len(anchor_indices_per_topic)
    vocab_size = len(vocab)

    print(f"LLM-guided ETM: num_topics = {num_topics}, vocab_size = {vocab_size}")

    emb_tensor = torch.from_numpy(embeddings).float().to(device)
    model = ETM(
        num_topics=num_topics,
        vocab_size=vocab_size,
        t_hidden_size=args.t_hidden_size,
        rho_size=emsize,
        emsize=emsize,
        theta_act=args.theta_act,
        embeddings=emb_tensor,
        train_embeddings=False,
        enc_drop=args.enc_drop,
    ).to(device)
    optimizer = model.get_optimizer(args)

    # 메인 학습 루프
    for epoch in range(1, args.epochs + 1):
        model.train()
        acc_loss = 0.0
        acc_kl_theta_loss = 0.0
        acc_anchor_loss = 0.0
        cnt = 0

        number_of_docs = torch.randperm(args.num_docs_train)
        batch_indices = torch.split(number_of_docs, args.batch_size)
        print("The number of the indices I am using for the training is ", len(batch_indices))

        for idx, indices in enumerate(batch_indices):
            optimizer.zero_grad()
            model.zero_grad()

            data_batch = get_batch(train_bow, indices, device)
            sums = data_batch.sum(1).unsqueeze(1)
            if args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch

            recon_loss, kld_theta = model.forward(
                data_batch, normalized_data_batch, theta=None, aggregate=True
            )

            # ------------------------------------------------------------------
            # Anchor loss: beta[k, anchor_idx] 확률을 키우기 위해 -log(beta)를 최소화
            # ------------------------------------------------------------------
            beta = model.get_beta()  # (K, V)
            beta = beta + 1e-10  # numerical stability

            anchor_loss = torch.tensor(0.0, device=device)
            anchor_topic_cnt = 0
            for k, idxs in enumerate(anchor_indices_per_topic):
                if not idxs:
                    continue
                anchor_topic_cnt += 1

                idx_tensor = torch.tensor(idxs, dtype=torch.long, device=device)
                beta_k = beta[k, idx_tensor]  # (num_anchors_for_topic,)

                # -log(prob)의 평균을 최소화 → 해당 anchor 위치 확률이 커지도록 유도
                anchor_loss_k = -torch.log(beta_k).mean()
                anchor_loss = anchor_loss + anchor_loss_k

            if anchor_topic_cnt > 0:
                anchor_loss = anchor_loss / anchor_topic_cnt

            total_loss = recon_loss + kld_theta
            if anchor_topic_cnt > 0 and args.lambda_anchor > 0.0:
                total_loss = total_loss + args.lambda_anchor * anchor_loss

            total_loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            acc_loss += float(recon_loss.detach().cpu())
            acc_kl_theta_loss += float(kld_theta.detach().cpu())
            acc_anchor_loss += float(anchor_loss.detach().cpu())
            cnt += 1

            if idx % args.log_interval == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2)
                cur_kl = round(acc_kl_theta_loss / cnt, 2)
                cur_anchor = round(acc_anchor_loss / max(cnt, 1), 4)
                cur_total = round(cur_loss + cur_kl + args.lambda_anchor * cur_anchor, 2)
                print(
                    "Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. "
                    "Anchor_loss: {} .. Total: {}".format(
                        epoch,
                        idx,
                        len(batch_indices),
                        optimizer.param_groups[0]["lr"],
                        cur_kl,
                        cur_loss,
                        cur_anchor,
                        cur_total,
                    )
                )

        cur_loss = round(acc_loss / cnt, 2)
        cur_kl = round(acc_kl_theta_loss / cnt, 2)
        cur_anchor = round(acc_anchor_loss / max(cnt, 1), 4)
        cur_total = round(cur_loss + cur_kl + args.lambda_anchor * cur_anchor, 2)
        print("*" * 100)
        print(
            "Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. "
            "Anchor_loss: {} .. Total: {}".format(
                epoch,
                optimizer.param_groups[0]["lr"],
                cur_kl,
                cur_loss,
                cur_anchor,
                cur_total,
            )
        )
        print("*" * 100)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets/20News")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--wdecay", type=float, default=1.2e-6)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--theta_act", type=str, default="relu")
    parser.add_argument("--t_hidden_size", type=int, default=500)
    parser.add_argument("--enc_drop", type=float, default=0.0)
    parser.add_argument("--bow_norm", type=int, default=1)
    parser.add_argument("--topk_words", type=int, default=15)
    parser.add_argument("--out_dir", type=str, default="results_llm_guided")
    parser.add_argument("--seed", type=int, default=42)

    # LLM anchor 관련 인자
    parser.add_argument(
        "--llm_words_file",
        type=str,
        default=None,
        help="llm_refine.py 가 저장한 anchor words 파일 (words_path). "
        "한 줄에 한 토픽의 단어들이 공백으로 구분되어 있어야 함.",
    )
    parser.add_argument(
        "--llm_step3_json",
        type=str,
        default=None,
        help="(선택) llm_refine.py step3 JSON 경로. 지정 시 JSON 기반으로 anchor 단어를 읽어옴.",
    )
    parser.add_argument(
        "--lambda_anchor",
        type=float,
        default=1.0,
        help="anchor loss 가중치 (anchor 단어 확률을 키우는 정규화 강도)",
    )

    args = parser.parse_args()

    args.bow_norm = bool(args.bow_norm)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1) 데이터 로드 (train_etm.py 와 동일한 방식)
    data_dir = ROOT / args.data_dir
    data = load_20news(data_dir)
    train_bow = data["train_bow"]
    test_bow = data["test_bow"]
    vocab = data["vocab"]
    embeddings = data["embeddings"]
    vocab_size = data["vocab_size"]
    train_labels = data["train_labels"]
    test_labels = data["test_labels"]

    N_train, N_test = train_bow.shape[0], test_bow.shape[0]
    emsize = embeddings.shape[1]
    args.num_docs_train = N_train
    args.num_docs_test = N_test
    args.num_docs_valid = 0
    args.num_words = args.topk_words

    V = test_bow.shape[1]
    half = V // 2
    test_1 = test_bow[:, :half]
    test_2 = test_bow[:, half:]
    args.num_docs_test_1 = N_test

    # train_bow를 GPU로 올려두기
    train_bow = train_bow.to(device)

    # 2) LLM anchor 단어 로드
    if args.llm_words_file is None and args.llm_step3_json is None:
        raise ValueError("Either --llm_words_file or --llm_step3_json must be provided.")

    if args.llm_words_file is not None:
        print(f"Loading LLM anchor words from words file: {args.llm_words_file}")
        anchor_words_per_topic = load_anchor_words_from_llm_words_file(args.llm_words_file)
    else:
        print(f"Loading LLM anchor words from JSON: {args.llm_step3_json}")
        anchor_words_per_topic = load_anchor_words_from_step3_json(args.llm_step3_json)

    # anchor 단어를 vocab 인덱스로 변환 (out-of-vocab 단어는 제외)
    anchor_indices_per_topic = build_anchor_indices(anchor_words_per_topic, vocab)
    coverage = summarize_anchor_coverage(anchor_indices_per_topic)
    print(
        "Anchor coverage:",
        coverage,
        "(topics with zero anchors will not receive anchor loss)",
    )

    # 3) LLM-guided 2차 ETM 학습
    model = train_etm_with_anchors(
        train_bow=train_bow,
        args=args,
        vocab=vocab,
        anchor_indices_per_topic=anchor_indices_per_topic,
        emsize=emsize,
        embeddings=embeddings,
    )

    # 4) 토픽 분포/평가/상위단어 추출 (train_etm.py 와 동일)
    model.eval()
    beta = model.get_beta()

    def get_theta_for_docs(model, bows, batch_size=256, bow_norm=True):
        model.eval()
        N, _ = bows.shape
        thetas = []
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = bows[start:end].to(device)
                if bow_norm:
                    s = batch.sum(1, keepdim=True).clamp(min=1e-6)
                    batch = batch / s
                theta, _ = model.get_theta(batch)
                thetas.append(theta.cpu().numpy())
        return np.vstack(thetas)

    theta_train = get_theta_for_docs(
        model, train_bow, batch_size=args.eval_batch_size, bow_norm=args.bow_norm
    )
    theta_test = get_theta_for_docs(
        model, test_bow, batch_size=args.eval_batch_size, bow_norm=args.bow_norm
    )

    train_bow_np = (
        train_bow.cpu().numpy()
        if train_bow.is_cuda
        else (train_bow.numpy() if torch.is_tensor(train_bow) else train_bow)
    )
    metrics = run_evaluation(
        beta,
        theta_train,
        theta_test,
        train_bow_np,
        train_labels,
        test_labels,
        vocab,
        beta.shape[0],
        topk_words=args.topk_words,
        n_docs_coherence=2000,
        root_dir=str(ROOT),
    )

    top_words = get_top_words_per_topic(
        beta.detach().cpu().numpy(), vocab, topk=args.topk_words
    )

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    import json

    with open(out_path / "metrics_llm_guided.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_path / "top_words_llm_guided.txt", "w", encoding="utf-8") as f:
        for k, words in enumerate(top_words):
            f.write("Topic {}: {}\n".format(k, " ".join(words)))

    print("\n--- LLM-guided ETM results saved to {} ---".format(args.out_dir))
    print("Metrics:", json.dumps(metrics, indent=2))
    print("\nTop words (first 3 topics):")
    for k in range(min(3, len(top_words))):
        print("  Topic {}: {}".format(k, " ".join(top_words[k][:10])))


if __name__ == "__main__":
    main()

