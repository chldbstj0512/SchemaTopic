"""
ETM 학습 메인 스크립트
- dataset 로드 → ETM 모델 생성 → optimizer → training loop → topic extraction → metric/상위단어 출력
- 20news, 250 epochs, 결과: metric (TC, TD, Purity, NMI) + 상위 단어 목록
"""
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np

# 프로젝트 루트를 path에 넣어 NTMs.etm, data, utils 등 import
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dataset import load_20news
from data import get_batch
from evaluation import run_evaluation, get_top_words_per_topic

# ETM은 data, utils를 같은 루트에서 import하므로 그대로 사용
from NTMs.etm import ETM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # 고정 입력 크기일 때 연산 빠르게
if device.type == "cpu":
    print("Warning: CUDA not available, training on CPU (will be slow).")
else:
    print("Using device:", device)


def get_theta_for_docs(model, bows, batch_size=256, bow_norm=True):
    """(N, V) BOW → (N, K) theta"""
    model.eval()
    N, V = bows.shape
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets/20News")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--num_topics", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2000, help="ETM_20News: 2000")
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.005, help="ETM_20News: 0.005")
    parser.add_argument("--wdecay", type=float, default=1.2e-6)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--clip", type=float, default=1.0, help="grad norm clip. 0=off. 중반 이후 NaN 방지용 1.0 권장")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--theta_act", type=str, default="relu")
    parser.add_argument("--t_hidden_size", type=int, default=500, help="LLM-ITL ETM과 동일")
    parser.add_argument("--enc_drop", type=float, default=0.0, help="LLM-ITL ETM과 동일")
    parser.add_argument("--bow_norm", type=int, default=1)
    parser.add_argument("--topk_words", type=int, default=15)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.bow_norm = bool(args.bow_norm)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1) dataset 로드
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
    # document completion용 test 반쪽 (vocab 절반씩)
    V = test_bow.shape[1]
    half = V // 2
    test_1 = test_bow[:, :half]
    test_2 = test_bow[:, half:]
    args.num_docs_test_1 = N_test

    # train_bow를 GPU로 한 번 올려서 매 배치마다 CPU→GPU 복사 제거 (속도 대폭 개선)
    train_bow = train_bow.to(device)

    # 2) ETM 모델 생성 (embeddings 사용, train_embeddings=False)
    emb_tensor = torch.from_numpy(embeddings).float().to(device)
    model = ETM(
        num_topics=args.num_topics,
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

    # 3) training loop (250 epochs)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train_for_epoch(epoch, args, train_bow)

    # 4) topic extraction & evaluation
    model.eval()
    beta = model.get_beta()
    theta_train = get_theta_for_docs(
        model, train_bow, batch_size=args.eval_batch_size, bow_norm=args.bow_norm
    )
    theta_test = get_theta_for_docs(
        model, test_bow, batch_size=args.eval_batch_size, bow_norm=args.bow_norm
    )

    train_bow_np = train_bow.cpu().numpy() if train_bow.is_cuda else (train_bow.numpy() if torch.is_tensor(train_bow) else train_bow)
    metrics = run_evaluation(
        beta,
        theta_train,
        theta_test,
        train_bow_np,
        train_labels,
        test_labels,
        vocab,
        args.num_topics,
        topk_words=args.topk_words,
        n_docs_coherence=2000,
        root_dir=str(ROOT),
    )

    # 5) 상위 단어 목록
    top_words = get_top_words_per_topic(
        beta.detach().cpu().numpy(), vocab, topk=args.topk_words
    )

    # 결과 저장
    out_path = Path(args.out_dir)
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_path / "top_words.txt", "w", encoding="utf-8") as f:
        for k, words in enumerate(top_words):
            f.write("Topic {}: {}\n".format(k, " ".join(words)))
    print("\n--- Results saved to {} ---".format(args.out_dir))
    print("Metrics:", json.dumps(metrics, indent=2))
    print("\nTop words (first 3 topics):")
    for k in range(min(3, len(top_words))):
        print("  Topic {}: {}".format(k, " ".join(top_words[k][:10])))
    return metrics, top_words


if __name__ == "__main__":
    main()
