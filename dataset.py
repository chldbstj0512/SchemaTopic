"""
dataset: train/test pkl, voc.txt, word_embeddings 로드 후 BOW 텐서 생성
"""
import pickle
import numpy as np
import torch
from pathlib import Path


def load_vocab(voc_path):
    """voc.txt 한 줄(공백 구분) 로드 → 단어 리스트"""
    with open(voc_path, "r") as f:
        line = f.read().strip()
    return line.split()


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


def load_20news(data_dir):
    """
    data_dir: datasets/20News 같은 경로
    Returns:
        train_bow: (N_train, V) tensor
        test_bow: (N_test, V) tensor
        train_labels: (N_train,) for clustering eval
        test_labels: (N_test,)
        vocab: list of str
        embeddings: (V, emsize) numpy
    """
    data_dir = Path(data_dir)
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
