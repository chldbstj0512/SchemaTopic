import json
from typing import List, Dict, Any


def load_anchor_words_from_llm_words_file(path: str) -> List[List[str]]:
    """
    Load topic-wise anchor words from the LLM words file.

    Expected format (as written by llm_refine.py: words_path):
        line 0: "word1 word2 word3 ..."
        line 1: "wordA wordB ..."
        ...
    Each line corresponds to one topic.
    """
    topics: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                topics.append([])
                continue
            topics.append(line.split())
    return topics


def load_anchor_words_from_step3_json(path: str) -> List[List[str]]:
    """
    Alternative loader when we want to use the JSON output of llm_refine.py directly.

    step3_json format (list of dicts):
        {
            "topic_id": int,
            "topic_name": str,
            "words": [str, ...],
            ...
        }
    We sort by topic_id and extract the "words" field.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("step3_json must be a list of topic dicts.")

    topics_dict: Dict[int, List[str]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        tid = item.get("topic_id", None)
        words = item.get("words", None)
        if tid is None or not isinstance(words, list):
            continue
        topics_dict[int(tid)] = [str(w).strip() for w in words if str(w).strip()]

    if not topics_dict:
        return []

    max_tid = max(topics_dict.keys())
    topics: List[List[str]] = []
    for tid in range(max_tid + 1):
        topics.append(topics_dict.get(tid, []))
    return topics


def build_anchor_indices(
    anchor_words_per_topic: List[List[str]],
    vocab: List[str],
) -> List[List[int]]:
    """
    Map LLM anchor words to vocabulary indices.

    - anchor_words_per_topic[k] = list of words for topic k
    - vocab[i] = word string at index i (same order as ETM/NTM uses)

    Returns:
        anchor_indices_per_topic[k] = list of integer indices into vocab.
    Out-of-vocabulary words are silently skipped.
    """
    vocab_to_idx = {w: i for i, w in enumerate(vocab)}
    anchor_indices_per_topic: List[List[int]] = []

    for words in anchor_words_per_topic:
        indices: List[int] = []
        for w in words:
            idx = vocab_to_idx.get(w)
            if idx is not None:
                indices.append(idx)
        anchor_indices_per_topic.append(indices)

    return anchor_indices_per_topic


def summarize_anchor_coverage(
    anchor_indices_per_topic: List[List[int]],
) -> Dict[str, Any]:
    """
    Small helper for logging / debugging:
    returns coverage stats about how many topics actually have at least one anchor.
    """
    num_topics = len(anchor_indices_per_topic)
    non_empty = sum(1 for idxs in anchor_indices_per_topic if len(idxs) > 0)
    return {
        "num_topics": num_topics,
        "num_topics_with_anchors": non_empty,
    }

