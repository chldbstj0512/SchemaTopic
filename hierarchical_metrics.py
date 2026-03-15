"""
Hierarchy-specific metrics from TraCo (AAAI 2024).
Adapted for SchemaTopic schema_topics.json format.

Reference: https://github.com/bobxwu/TraCo
Paper: On the Affinity, Rationality, and Diversity of Hierarchical Topic Modeling (AAAI 2024)
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x


def _ensure_numpy(bow):
    """Convert BOW to numpy if needed."""
    if hasattr(bow, "cpu"):
        return bow.cpu().numpy().astype(np.float32)
    return np.asarray(bow, dtype=np.float32)


def compute_TD(texts):
    """Topic Diversity: fraction of unique words across topics. (TraCo eq.)"""
    if not texts:
        return 0.0
    K = len(texts)
    T = len(texts[0].split()) if texts[0] else 0
    if K == 0 or T == 0:
        return 0.0
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    counter = vectorizer.fit_transform(texts).toarray()
    TF = counter.sum(axis=0)
    return float((TF == 1).sum() / (K * T))


def _vocab_index(vocab, word):
    """Case-insensitive vocab index lookup."""
    w = word.strip().lower()
    for i, v in enumerate(vocab):
        if v.strip().lower() == w:
            return i
    return -1


def compute_CLNPMI(parent_diff_words, child_diff_words, all_bow, vocab):
    """Parent-Child NPMI for non-overlapping words. (TraCo)"""
    n_docs = len(all_bow)
    npmi_list = []
    for p_w in parent_diff_words:
        idx_p = _vocab_index(vocab, p_w)
        if idx_p < 0:
            continue
        flag_n = all_bow[:, idx_p] > 0
        p_n = np.sum(flag_n) / n_docs
        if p_n <= 0:
            continue
        for c_w in child_diff_words:
            idx_c = _vocab_index(vocab, c_w)
            if idx_c < 0:
                continue
            flag_l = all_bow[:, idx_c] > 0
            p_l = np.sum(flag_l) / n_docs
            p_nl = np.sum(flag_n * flag_l) / n_docs
            if p_nl >= 1.0 - 1e-9:
                npmi_score = 1.0
            elif p_nl <= 0 or p_l <= 0 or p_n <= 0:
                continue
            else:
                npmi_score = np.log(p_nl / (p_l * p_n)) / (-np.log(p_nl))
                npmi_score = float(np.clip(npmi_score, -1.0, 1.0))
            npmi_list.append(npmi_score)
    return npmi_list


def get_CLNPMI(PC_pair_groups, all_bow, vocab):
    """Parent-Child Coherence (PCC) via NPMI. (TraCo)"""
    CNPMI_list = []
    for group in tqdm(PC_pair_groups, desc="CLNPMI", leave=False):
        layer_CNPMI = []
        for parent_topic, child_topic in group:
            parent_words = set(parent_topic.split())
            child_words = set(child_topic.split())
            inter = parent_words.intersection(child_words)
            parent_diff_words = list(parent_words.difference(inter))
            child_diff_words = list(child_words.difference(inter))
            npmi_list = compute_CLNPMI(parent_diff_words, child_diff_words, all_bow, vocab)
            num_repetition = (len(parent_words) - len(parent_diff_words)) * (
                len(child_words) - len(child_diff_words)
            )
            npmi_list.extend([-1.0] * num_repetition)
            layer_CNPMI.extend(npmi_list)
        if layer_CNPMI:
            valid = [x for x in layer_CNPMI if x >= 0]
            CNPMI_list.append(np.mean(valid) if valid else 0.0)
        else:
            CNPMI_list.append(0.0)
    return CNPMI_list


def compute_diff_topic_pair(topic_str_a, topic_str_b):
    """Topic difference: fraction of words that appear in only one topic. (TraCo)"""
    word_counter = Counter()
    topic_words_a = topic_str_a.split()
    topic_words_b = topic_str_b.split()
    word_counter.update(topic_words_a)
    word_counter.update(topic_words_b)
    total = len(topic_words_a) + len(topic_words_b)
    if total == 0:
        return 0.0
    diff = sum(1 for v in word_counter.values() if v == 1) / total
    return float(diff)


def get_topics_difference(topic_pair_groups):
    """Average topic difference over pairs. (TraCo)"""
    diff_list = []
    for groups in topic_pair_groups:
        layer_diff = []
        for topic_a, topic_b in groups:
            diff = compute_diff_topic_pair(topic_a, topic_b)
            layer_diff.append(diff)
        diff_list.append(np.mean(layer_diff) if layer_diff else 0.0)
    return diff_list


def get_sibling_TD(sibling_groups):
    """Sibling Topic Diversity (SD). (TraCo)"""
    sibling_TD = []
    for group in sibling_groups:
        layer_sibling_TD = []
        for sibling_topics in group:
            td = compute_TD(sibling_topics)
            layer_sibling_TD.append(td)
        sibling_TD.append(np.mean(layer_sibling_TD) if layer_sibling_TD else 0.0)
    return sibling_TD


# ---------------------------------------------------------------------------
# SchemaTopic adapter: schema_topics.json -> TraCo-style groups
# ---------------------------------------------------------------------------


def load_schema_topics(schema_path):
    """Load schema_topics.json. Returns list of {label, topics: [{words}]}."""
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError("Schema not found: {}".format(schema_path))
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("schema", data) if isinstance(data, dict) else data


def schema_to_traco_groups(schema_data, num_top_words=15):
    """
    Convert SchemaTopic schema to TraCo-style PC pairs, PnonC pairs, sibling groups.

    SchemaTopic: 2-level hierarchy
      - Parent = schema label (use merged topic words as parent representation)
      - Child = each topic under that schema
    """
    PC_pair_groups = []   # [(parent_str, child_str), ...] per schema
    PnonC_pair_groups = []  # [(parent_str, non_child_str), ...]
    sibling_groups = []   # [[sibling_str, ...], ...] per schema

    all_parent_strs = []
    all_child_strs_by_schema = []

    for group in schema_data:
        label = group.get("label", "Misc")
        topics = group.get("topics", [])
        if not topics:
            continue

        child_strs = []
        for t in topics:
            words = t.get("words", [])
            if isinstance(words, list):
                w = " ".join(str(x).strip() for x in words[:num_top_words])
            else:
                w = str(words)
            if w.strip():
                child_strs.append(w)

        if not child_strs:
            continue

        # Parent = concatenate all child topic words (schema-level representation)
        all_words = []
        seen = set()
        for w in child_strs:
            for tok in w.split():
                t = tok.lower()
                if t not in seen:
                    seen.add(t)
                    all_words.append(tok)
        parent_str = " ".join(all_words[: num_top_words * 3])  # broader parent

        all_parent_strs.append(parent_str)
        all_child_strs_by_schema.append(child_strs)

        # PC pairs: (parent, child) for each child
        pc_pairs = [(parent_str, c) for c in child_strs]
        PC_pair_groups.append(pc_pairs)

        # Sibling group: all children under this schema
        sibling_groups.append([child_strs])

        # PnonC: (parent, non_child) - non_child = topic from a different schema
        my_idx = len(all_child_strs_by_schema) - 1
        pnonc_pairs = []
        for j, olist in enumerate(all_child_strs_by_schema):
            if j != my_idx:
                for c in olist:
                    pnonc_pairs.append((parent_str, c))
        PnonC_pair_groups.append(pnonc_pairs)

    return PC_pair_groups, PnonC_pair_groups, sibling_groups


def compute_hierarchical_metrics(
    schema_path,
    train_bow,
    test_bow,
    vocab,
    num_top_words=15,
):
    """
    Compute TraCo hierarchy metrics for SchemaTopic schema.

    Args:
        schema_path: path to schema_topics.json
        train_bow: (N_train, V) document-term matrix
        test_bow: (N_test, V)
        vocab: list of str
        num_top_words: top words per topic

    Returns:
        dict with keys: CLNPMI (PCC), PC_TD (PCD), Sibling_TD (SD), PnonC_TD (PnCD)
    """
    schema_data = load_schema_topics(schema_path)
    PC_pair_groups, PnonC_pair_groups, sibling_groups = schema_to_traco_groups(
        schema_data, num_top_words=num_top_words
    )

    if not PC_pair_groups:
        return {
            "CLNPMI": 0.0,
            "PC_TD": 0.0,
            "Sibling_TD": 0.0,
            "PnonC_TD": 0.0,
        }

    all_bow = np.vstack([
        _ensure_numpy(train_bow),
        _ensure_numpy(test_bow),
    ]).astype(np.float32)

    # Filter vocab to match BOW columns
    vocab = list(vocab)[: all_bow.shape[1]]

    CLNPMI = get_CLNPMI(PC_pair_groups, all_bow, vocab)
    PC_TD = get_topics_difference(PC_pair_groups)
    Sibling_TD = get_sibling_TD(sibling_groups)
    PnonC_TD = get_topics_difference(PnonC_pair_groups)

    return {
        "CLNPMI": float(np.mean(CLNPMI)),
        "PC_TD": float(np.mean(PC_TD)),
        "Sibling_TD": float(np.mean(Sibling_TD)),
        "PnonC_TD": float(np.mean(PnonC_TD)),
    }


def main():
    """CLI for standalone evaluation."""
    import argparse
    from dataset import load_topic_dataset

    parser = argparse.ArgumentParser(description="TraCo hierarchy metrics for SchemaTopic")
    parser.add_argument("--schema", required=True, help="path to schema_topics.json")
    parser.add_argument("--data_dir", required=True, help="e.g. datasets/20News")
    parser.add_argument("--num_top_words", type=int, default=15)
    args = parser.parse_args()

    data = load_topic_dataset(args.data_dir)
    train_bow = data["train_bow"]
    test_bow = data["test_bow"]
    vocab = data["vocab"]

    metrics = compute_hierarchical_metrics(
        args.schema,
        train_bow,
        test_bow,
        vocab,
        num_top_words=args.num_top_words,
    )
    print("Hierarchy metrics (TraCo-style):")
    for k, v in metrics.items():
        print("  {}: {:.5f}".format(k, v))


if __name__ == "__main__":
    main()
