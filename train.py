"""
Neural topic model training logic.

Argument parsing is intentionally handled in `main.py` only.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data import get_batch
from dataset import load_topic_dataset
from evaluation import get_top_words_per_topic, run_evaluation
from topic_models import create_topic_model
from utils import (
    build_anchor_indices,
    load_anchor_words_from_llm_words_file,
    load_anchor_words_from_step3_json,
    summarize_anchor_coverage,
)


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if DEVICE.type == "cpu":
    print("Warning: CUDA not available, training on CPU (will be slow).")
else:
    print("Using device:", DEVICE)


def load_training_data(root, data_dir):
    data = load_topic_dataset(root / data_dir)
    train_bow = data["train_bow"]
    test_bow = data["test_bow"]
    embeddings = data["embeddings"]
    vocab = data["vocab"]

    return {
        **data,
        "train_bow": train_bow,
        "test_bow": test_bow,
        "vocab": vocab,
        "embeddings": embeddings,
        "emsize": int(embeddings.shape[1]),
        "num_docs_train": int(train_bow.shape[0]),
        "num_docs_test": int(test_bow.shape[0]),
    }


def normalize_bows(bows, bow_norm):
    if not bow_norm:
        return bows
    sums = bows.sum(1, keepdim=True).clamp(min=1e-6)
    return bows / sums


def get_theta_for_docs(model, bows, device, batch_size=256, bow_norm=True):
    model.eval()
    num_docs = bows.shape[0]
    thetas = []
    model_expects_normalized_bows = getattr(model, "expects_normalized_bows", True)
    with torch.no_grad():
        for start in range(0, num_docs, batch_size):
            end = min(start + batch_size, num_docs)
            batch = bows[start:end].to(device)
            model_input = normalize_bows(batch, bow_norm=bow_norm) if model_expects_normalized_bows else batch
            theta, _ = model.get_document_topic_distribution(model_input)
            thetas.append(theta.cpu().numpy())
    return np.vstack(thetas)


def evaluate_topic_model(
    model,
    train_bow,
    test_bow,
    args,
    vocab,
    train_labels,
    test_labels,
    root_dir,
    device,
):
    model.eval()
    beta = model.get_topic_word_distribution()
    theta_train = get_theta_for_docs(
        model,
        train_bow,
        device=device,
        batch_size=args.eval_batch_size,
        bow_norm=args.bow_norm,
    )
    theta_test = get_theta_for_docs(
        model,
        test_bow,
        device=device,
        batch_size=args.eval_batch_size,
        bow_norm=args.bow_norm,
    )

    train_bow_np = (
        train_bow.cpu().numpy()
        if train_bow.is_cuda
        else (train_bow.numpy() if torch.is_tensor(train_bow) else train_bow)
    )
    metrics = run_evaluation(
        beta,
        theta_test,
        train_bow_np,
        test_labels,
        vocab,
        beta.shape[0],
        topk_words=args.topk_words,
        n_docs_coherence=2000,
        root_dir=root_dir,
    )
    top_words = get_top_words_per_topic(
        beta.detach().cpu().numpy(),
        vocab,
        topk=args.topk_words,
    )

    return {
        "beta": beta,
        "theta_train": theta_train,
        "theta_test": theta_test,
        "metrics": metrics,
        "top_words": top_words,
    }


def build_anchor_regularizer(anchor_indices_per_topic, anchor_weight, device):
    def _anchor_regularizer(model):
        beta = model.get_topic_word_distribution() + 1e-10
        anchor_loss = torch.tensor(0.0, device=device)
        anchor_topic_cnt = 0

        for topic_idx, anchor_indices in enumerate(anchor_indices_per_topic):
            if not anchor_indices:
                continue
            anchor_topic_cnt += 1
            index_tensor = torch.tensor(anchor_indices, dtype=torch.long, device=device)
            beta_topic = beta[topic_idx, index_tensor]
            anchor_loss = anchor_loss + (-torch.log(beta_topic).mean())

        if anchor_topic_cnt > 0:
            anchor_loss = anchor_loss / anchor_topic_cnt

        if anchor_topic_cnt == 0 or anchor_weight <= 0.0:
            return {"anchor_loss": torch.zeros((), device=device)}
        return {"anchor_loss": anchor_weight * anchor_loss}

    return _anchor_regularizer


def load_anchor_indices(anchor_words_file, anchor_topics_json, vocab):
    if anchor_words_file is None and anchor_topics_json is None:
        return None

    if anchor_words_file is not None:
        print("Loading anchor words from words file:", anchor_words_file)
        anchor_words_per_topic = load_anchor_words_from_llm_words_file(anchor_words_file)
    else:
        print("Loading anchor words from JSON:", anchor_topics_json)
        anchor_words_per_topic = load_anchor_words_from_step3_json(anchor_topics_json)

    if len(anchor_words_per_topic) == 0:
        raise ValueError(
            "Anchor source produced zero topics. Check whether '{}' is empty or whether schema refinement removed all topics.".format(
                anchor_words_file if anchor_words_file is not None else anchor_topics_json
            )
        )

    anchor_indices_per_topic = build_anchor_indices(anchor_words_per_topic, vocab)
    coverage = summarize_anchor_coverage(anchor_indices_per_topic)
    print(
        "Anchor coverage:",
        coverage,
        "(topics with zero anchors will not receive anchor loss)",
    )

    if coverage["num_topics"] == 0:
        raise ValueError("Anchor source produced zero topic groups.")
    if coverage["num_topics_with_anchors"] == 0:
        raise ValueError(
            "Anchor source did not match any vocabulary terms. Check whether the anchor words belong to the selected dataset vocabulary."
        )

    return anchor_indices_per_topic


def resolve_num_topics(requested_num_topics, anchor_indices_per_topic):
    if anchor_indices_per_topic is None:
        return requested_num_topics

    inferred_num_topics = len(anchor_indices_per_topic)
    if requested_num_topics != inferred_num_topics:
        print(
            "Overriding --num_topics={} with anchor-derived topic count={}.".format(
                requested_num_topics,
                inferred_num_topics,
            )
        )
    return inferred_num_topics


def build_output_paths(out_dir, output_suffix=None):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if output_suffix:
        metrics_name = "metrics_{}.json".format(output_suffix)
        top_words_name = "top_words_{}.txt".format(output_suffix)
    else:
        metrics_name = "metrics.json"
        top_words_name = "top_words.txt"

    return out_path, out_path / metrics_name, out_path / top_words_name


def _namespace_to_serializable_dict(args):
    serialized = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serialized[key] = value
        elif isinstance(value, Path):
            serialized[key] = str(value)
    return serialized


def save_model_artifacts(
    out_path,
    model,
    args,
    num_topics,
    vocab_size,
    emsize,
    metrics,
):
    checkpoint_path = out_path / "model.pt"
    metadata_path = out_path / "model_meta.json"

    model_kwargs = {
        "num_topics": num_topics,
        "vocab_size": vocab_size,
        "t_hidden_size": args.t_hidden_size,
        "rho_size": emsize,
        "emsize": emsize,
        "theta_act": args.theta_act,
        "train_embeddings": False,
        "enc_drop": args.enc_drop,
    }
    state_dict_cpu = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_name": args.model,
            "model_kwargs": model_kwargs,
            "state_dict": state_dict_cpu,
            "training_args": _namespace_to_serializable_dict(args),
            "metrics": metrics,
        },
        checkpoint_path,
    )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model,
                "model_kwargs": model_kwargs,
                "training_args": _namespace_to_serializable_dict(args),
                "metrics": metrics,
                "checkpoint_path": str(checkpoint_path),
            },
            f,
            indent=2,
        )

    return checkpoint_path, metadata_path


def train_topic_model(
    model,
    optimizer,
    train_bow,
    args,
    device,
    extra_regularizer_fn=None,
):
    for epoch in range(1, args.epochs + 1):
        _train_topic_model_epoch(
            model=model,
            optimizer=optimizer,
            train_bow=train_bow,
            args=args,
            device=device,
            epoch=epoch,
            extra_regularizer_fn=extra_regularizer_fn,
        )
    return model


def _train_topic_model_epoch(
    model,
    optimizer,
    train_bow,
    args,
    device,
    epoch,
    extra_regularizer_fn=None,
):
    model.train()
    metric_sums = defaultdict(float)
    cnt = 0

    number_of_docs = torch.randperm(args.num_docs_train)
    batch_indices = torch.split(number_of_docs, args.batch_size)
    print("The number of the indices I am using for the training is ", len(batch_indices))

    for idx, indices in enumerate(batch_indices):
        optimizer.zero_grad()
        model.zero_grad()

        data_batch = get_batch(train_bow, indices, device)
        normalized_data_batch = normalize_bows(data_batch, bow_norm=args.bow_norm)

        losses = model.compute_losses(
            data_batch,
            normalized_data_batch,
            theta=None,
            aggregate=True,
        )
        regularization_losses = dict(losses.regularization_losses)
        if extra_regularizer_fn is not None:
            extra_regularizers = extra_regularizer_fn(model)
            for name, loss_value in extra_regularizers.items():
                regularization_losses[name] = loss_value

        total_regularization = torch.zeros_like(losses.reconstruction_loss)
        for loss_value in regularization_losses.values():
            total_regularization = total_regularization + loss_value
        total_loss = losses.reconstruction_loss + total_regularization

        total_loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        metric_sums["reconstruction_loss"] += float(losses.reconstruction_loss.detach().cpu())
        for name, loss_value in regularization_losses.items():
            metric_sums[name] += float(loss_value.detach().cpu())
        metric_sums["total_loss"] += float(total_loss.detach().cpu())
        cnt += 1

        if idx % args.log_interval == 0 and idx > 0:
            _print_training_log(
                epoch=epoch,
                batch_idx=idx,
                num_batches=len(batch_indices),
                lr=optimizer.param_groups[0]["lr"],
                metric_sums=metric_sums,
                cnt=cnt,
                prefix="Epoch",
            )

    _print_training_log(
        epoch=epoch,
        batch_idx=len(batch_indices),
        num_batches=len(batch_indices),
        lr=optimizer.param_groups[0]["lr"],
        metric_sums=metric_sums,
        cnt=cnt,
        prefix="Epoch----->",
        surround=True,
    )
    return {
        name: total / max(cnt, 1)
        for name, total in metric_sums.items()
    }


def _print_training_log(
    epoch,
    batch_idx,
    num_batches,
    lr,
    metric_sums,
    cnt,
    prefix,
    surround=False,
):
    metrics = {
        name: round(total / max(cnt, 1), 4)
        for name, total in metric_sums.items()
    }

    ordered_keys = ["reconstruction_loss"]
    ordered_keys.extend(
        key
        for key in sorted(metrics.keys())
        if key not in {"reconstruction_loss", "total_loss"}
    )
    if "total_loss" in metrics:
        ordered_keys.append("total_loss")

    metric_text = " .. ".join(
        "{}: {}".format(name, metrics[name])
        for name in ordered_keys
    )
    line = "{}{} .. batch: {}/{} .. LR: {} .. {}".format(
        prefix,
        epoch,
        batch_idx,
        num_batches,
        lr,
        metric_text,
    )

    if surround:
        print("*" * 100)
        print(line)
        print("*" * 100)
    else:
        print(line)


def run_train(args):
    args.bow_norm = bool(args.bow_norm)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_training_data(ROOT, args.data_dir)
    train_bow = data["train_bow"]
    test_bow = data["test_bow"]
    vocab = data["vocab"]
    embeddings = data["embeddings"]
    vocab_size = data["vocab_size"]
    train_labels = data["train_labels"]
    test_labels = data["test_labels"]
    emsize = data["emsize"]

    args.num_docs_train = data["num_docs_train"]
    args.num_docs_test = data["num_docs_test"]
    args.num_docs_valid = 0
    args.num_words = args.topk_words

    train_bow = train_bow.to(DEVICE)

    anchor_indices_per_topic = load_anchor_indices(
        args.anchor_words_file,
        args.anchor_topics_json,
        vocab,
    )
    num_topics = resolve_num_topics(args.num_topics, anchor_indices_per_topic)
    run_mode = "anchor-guided" if anchor_indices_per_topic is not None else "standard"
    print(
        "Training {} model '{}' with {} topics".format(
            run_mode,
            args.model,
            num_topics,
        )
    )

    emb_tensor = torch.from_numpy(embeddings).float().to(DEVICE)
    model = create_topic_model(
        args.model,
        num_topics=num_topics,
        vocab_size=vocab_size,
        t_hidden_size=args.t_hidden_size,
        rho_size=emsize,
        emsize=emsize,
        theta_act=args.theta_act,
        embeddings=emb_tensor,
        train_embeddings=False,
        enc_drop=args.enc_drop,
    ).to(DEVICE)
    optimizer = model.build_optimizer(args)

    extra_regularizer_fn = None
    if anchor_indices_per_topic is not None:
        extra_regularizer_fn = build_anchor_regularizer(
            anchor_indices_per_topic=anchor_indices_per_topic,
            anchor_weight=args.lambda_anchor,
            device=DEVICE,
        )

    model = train_topic_model(
        model=model,
        optimizer=optimizer,
        train_bow=train_bow,
        args=args,
        device=DEVICE,
        extra_regularizer_fn=extra_regularizer_fn,
    )

    eval_outputs = evaluate_topic_model(
        model=model,
        train_bow=train_bow,
        test_bow=test_bow,
        args=args,
        vocab=vocab,
        train_labels=train_labels,
        test_labels=test_labels,
        root_dir=str(ROOT),
        device=DEVICE,
    )
    metrics = eval_outputs["metrics"]
    top_words = eval_outputs["top_words"]

    out_path, metrics_path, top_words_path = build_output_paths(
        args.out_dir,
        output_suffix=args.output_suffix,
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(top_words_path, "w", encoding="utf-8") as f:
        for topic_idx, words in enumerate(top_words):
            f.write("Topic {}: {}\n".format(topic_idx, " ".join(words)))

    checkpoint_path, metadata_path = save_model_artifacts(
        out_path=out_path,
        model=model,
        args=args,
        num_topics=num_topics,
        vocab_size=vocab_size,
        emsize=emsize,
        metrics=metrics,
    )

    print("\n--- Results saved to {} ---".format(out_path))
    print("Metrics:", json.dumps(metrics, indent=2))
    print("Checkpoint:", checkpoint_path)
    print("Metadata:", metadata_path)
    print("\nTop words (first 3 topics):")
    for topic_idx in range(min(3, len(top_words))):
        print("  Topic {}: {}".format(topic_idx, " ".join(top_words[topic_idx][:10])))

    return {
        "metrics": metrics,
        "top_words": top_words,
        "out_dir": str(out_path),
        "metrics_path": str(metrics_path),
        "top_words_path": str(top_words_path),
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(metadata_path),
    }


def run_eval_from_checkpoint(checkpoint_path, data_dir=None, root_dir=None):
    """
    Load a saved model checkpoint and run evaluation (TC, TD, Purity, NMI, PN).
    checkpoint_path: path to model.pt or directory containing model.pt
    data_dir: dataset path (required if not in checkpoint training_args)
    root_dir: project root for dataset resolution (default: ROOT)
    """
    root = Path(root_dir) if root_dir else ROOT
    ckpt = Path(checkpoint_path)
    if ckpt.is_dir():
        ckpt = ckpt / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(ckpt))

    print("Loading checkpoint:", ckpt)
    checkpoint = torch.load(ckpt, map_location="cpu")

    model_name = checkpoint.get("model_name", "etm")
    model_kwargs = checkpoint["model_kwargs"]
    state_dict = checkpoint["state_dict"]
    training_args = checkpoint.get("training_args", {})

    if data_dir is None:
        data_dir = training_args.get("data_dir")
    if data_dir is None:
        dataset = training_args.get("dataset", "20News")
        data_dir = str(Path("datasets") / dataset)
    data_dir = str(data_dir)

    print("Loading data from:", root / data_dir)
    data = load_training_data(root, data_dir)
    train_bow = data["train_bow"]
    test_bow = data["test_bow"]
    vocab = data["vocab"]
    train_labels = data["train_labels"]
    test_labels = data["test_labels"]
    embeddings = data["embeddings"]
    emsize = data["emsize"]

    emb_tensor = torch.from_numpy(embeddings).float().to(DEVICE)
    model_kwargs["embeddings"] = emb_tensor
    model = create_topic_model(model_name, **model_kwargs).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)

    args = type("Args", (), {})()
    args.eval_batch_size = training_args.get("eval_batch_size", 256)
    args.bow_norm = bool(training_args.get("bow_norm", 1))
    args.topk_words = training_args.get("topk_words", 15)

    print("\n--- Running evaluation ---")
    eval_outputs = evaluate_topic_model(
        model=model,
        train_bow=train_bow,
        test_bow=test_bow,
        args=args,
        vocab=vocab,
        train_labels=train_labels,
        test_labels=test_labels,
        root_dir=str(root),
        device=DEVICE,
    )
    metrics = eval_outputs["metrics"]
    top_words = eval_outputs["top_words"]

    print("\n--- Evaluation results ---")
    print(json.dumps(metrics, indent=2))
    return {"metrics": metrics, "top_words": top_words}
