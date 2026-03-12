"""
dataset: preprocessed topic-model dataset loader utilities.
"""
import pickle
import numpy as np
import torch
from pathlib import Path


def load_vocab(voc_path):
    """Load vocabulary terms from whitespace-separated `voc.txt`."""
    with open(voc_path, "r", encoding="utf-8") as f:
        return f.read().split()


def load_embeddings(emb_path):
    """word_embeddings.npy → (V, emsize) numpy"""
    return np.load(emb_path).astype(np.float32)


def tokens_counts_to_bow(tokens_list, counts_list, vocab_size):
    """문서별 tokens/counts 리스트 → (N, V) BOW 밀집 행렬"""
    n_docs = len(tokens_list)
    bow = np.zeros((n_docs, vocab_size), dtype=np.float32)
    for i in range(n_docs):
        tok = np.asarray(tokens_list[i]).flatten()
        cnt = np.asarray(counts_list[i]).flatten()
        if len(tok) == 0:
            continue
        # 인덱스가 vocab 범위를 벗어나지 않도록
        valid = tok < vocab_size
        np.add.at(bow[i], tok[valid], cnt[valid])
    return bow


def list_available_datasets(datasets_root=None):
    """Return dataset names that already have ETM-style preprocessed files."""
    root = Path(datasets_root) if datasets_root is not None else Path(__file__).resolve().parent / "datasets"
    if not root.exists():
        return []

    dataset_names = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        required_files = (
            child / "train.pkl",
            child / "test.pkl",
            child / "voc.txt",
            child / "word_embeddings.npy",
        )
        if all(path.exists() for path in required_files):
            dataset_names.append(child.name)
    return dataset_names


def infer_dataset_name(data_dir):
    return Path(data_dir).name


def load_topic_dataset(data_dir):
    """
    data_dir: preprocessed dataset directory such as `datasets/20News`
    Returns:
        train_bow: (N_train, V) tensor
        test_bow: (N_test, V) tensor
        train_labels: (N_train,) for clustering eval
        test_labels: (N_test,)
        vocab: list of str
        embeddings: (V, emsize) numpy
    """
    data_dir = Path(data_dir)
    required_files = {
        "train.pkl": data_dir / "train.pkl",
        "test.pkl": data_dir / "test.pkl",
        "voc.txt": data_dir / "voc.txt",
        "word_embeddings.npy": data_dir / "word_embeddings.npy",
    }
    missing_files = [name for name, path in required_files.items() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Dataset directory '{}' is missing required files: {}".format(
                data_dir,
                ", ".join(missing_files),
            )
        )

    with open(data_dir / "train.pkl", "rb") as f:
        train = pickle.load(f)
    with open(data_dir / "test.pkl", "rb") as f:
        test_data = pickle.load(f)

    vocab = load_vocab(data_dir / "voc.txt")
    vocab_size = len(vocab)
    embeddings = load_embeddings(data_dir / "word_embeddings.npy")

    train_bow = tokens_counts_to_bow(
        train["tokens"], train["counts"], vocab_size
    )
    test_in = test_data["test"]
    test_bow = tokens_counts_to_bow(
        test_in["tokens"], test_in["counts"], vocab_size
    )

    train_labels = np.asarray(train["labels"]).flatten()
    test_labels = np.asarray(test_data["labels"]).flatten()

    return {
        "train_bow": torch.from_numpy(train_bow),
        "test_bow": torch.from_numpy(test_bow),
        "train_labels": train_labels,
        "test_labels": test_labels,
        "vocab": vocab,
        "embeddings": embeddings,
        "vocab_size": vocab_size,
    }


def load_20news(data_dir):
    """Backward-compatible alias for the legacy loader name."""
    return load_topic_dataset(data_dir)
