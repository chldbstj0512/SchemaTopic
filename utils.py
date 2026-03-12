"""
topic extraction 및 공통 helper 함수
- get_topics, get_topic_coherence (C_V), get_topic_diversity, nearest_neighbors
- Palmetto C_V (run_palmetto_cv) when jar + wikipedia_bd available
"""
import json
import os
import subprocess
import tempfile
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _progress(iterable, total=None, desc=None):
    if tqdm is None:
        if desc:
            print(desc)
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def get_topics(beta, vocab, topk=10):
    """
    beta: (K, V) numpy or tensor
    vocab: list of str
    returns: list of list of str, 각 토픽별 상위 topk 단어
    """
    if torch.is_tensor(beta):
        beta = beta.detach().cpu().numpy()
    beta = np.asarray(beta)
    K, V = beta.shape
    topics = []
    for k in range(K):
        top_idx = np.argsort(beta[k])[::-1][:topk]
        topics.append([vocab[i].strip() for i in top_idx])
    return topics


def _bow_to_texts(bow, vocab, max_docs=None):
    """BOW (N, V) → list of list of word strings (for gensim CoherenceModel)."""
    if torch.is_tensor(bow):
        bow = bow.cpu().numpy()
    bow = np.asarray(bow)
    N, V = bow.shape
    if max_docs is not None:
        N = min(N, max_docs)
    texts = []
    iterator = _progress(range(N), total=N, desc="Building texts for C_V")
    for i in iterator:
        row = bow[i]
        tokens = []
        for v in range(V):
            cnt = int(row[v])
            if cnt > 0:
                tokens.extend([vocab[v].strip()] * cnt)
        texts.append(tokens)
    return texts


def _read_tc(tc_content):
    """Parse Palmetto coherence output. Lines with index\\tvalue or one float per line."""
    result = tc_content.split("\n")[2:-1]
    tcs = []
    for line in result:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                tcs.append(float(parts[1]))
            except ValueError:
                continue
        else:
            try:
                tcs.append(float(line))
            except ValueError:
                continue
    if not tcs:
        for line in tc_content.strip().split("\n"):
            line = line.strip()
            if not line or (len(line) > 10 and line[0].isdigit() and "," in line):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    tcs.append(float(parts[1]))
                except ValueError:
                    pass
            else:
                try:
                    tcs.append(float(line))
                except ValueError:
                    pass
    return np.mean(tcs) if tcs else None


def run_palmetto_cv(root_dir, topics, temp_folder=None, timeout_per_topic=90, topk=10):
    """
    Palmetto C_V: 토픽당 한 줄씩 실행해 NPE 시에도 나머지 토픽 유지.
    root_dir: palmetto jar, wikipedia_bd 있는 디렉터리
    topics: list of list of str (각 토픽 상위 단어)
    Returns: mean C_V or None
    """
    jar = os.path.join(root_dir, "palmetto-0.1.5-exec.jar")
    wikipedia_bd = os.path.join(root_dir, "wikipedia_bd")
    if not os.path.isfile(jar) or not os.path.isdir(wikipedia_bd):
        return None
    out_dir = temp_folder or os.path.join(root_dir, "temp_tc_palmetto")
    os.makedirs(out_dir, exist_ok=True)
    tcs = []
    print("Computing topic coherence with Palmetto for {} topics...".format(len(topics)))
    topic_iter = _progress(
        enumerate(topics),
        total=len(topics),
        desc="Palmetto C_V",
    )
    for i, words in topic_iter:
        line = " ".join(words[:topk]) if isinstance(words, (list, tuple)) else words
        if isinstance(line, str) and not line.strip():
            continue
        single_path = os.path.join(out_dir, "topic_%d.txt" % i)
        with open(single_path, "w") as f:
            f.write(line.strip() + "\n")
        try:
            r = subprocess.run(
                ["java", "-jar", jar, wikipedia_bd, "C_V", os.path.abspath(single_path)],
                cwd=root_dir,
                capture_output=True,
                text=True,
                timeout=timeout_per_topic,
            )
            if r.returncode == 0 and r.stdout:
                val = _read_tc(r.stdout)
                if val is not None:
                    tcs.append(val)
        except (subprocess.TimeoutExpired, Exception):
            pass
    if not tcs:
        return None
    return float(np.mean(tcs))


def get_topic_coherence(beta, training_set, vocabulary, topk=10, n_docs_for_coherence=2000, root_dir=None):
    """
    C_V: root_dir에 Palmetto(jar + wikipedia_bd) 있으면 Palmetto 사용, 없으면 gensim fallback.
    """
    if torch.is_tensor(beta):
        beta = beta.detach().cpu().numpy()
    beta = np.asarray(beta)
    topics = get_topics(beta, vocabulary, topk=topk)
    print("Preparing topic coherence evaluation for {} topics...".format(len(topics)))

    if root_dir:
        tc = run_palmetto_cv(root_dir, topics, topk=topk)
        if tc is not None:
            print("Topic Coherence (C_V, Palmetto):", round(tc, 4))
            return float(tc)

    try:
        from gensim.models import CoherenceModel
    except ImportError:
        print("gensim not installed; skipping C_V coherence.")
        return 0.0

    print(
        "Palmetto unavailable; falling back to gensim C_V on up to {} documents.".format(
            n_docs_for_coherence,
        )
    )
    texts = _bow_to_texts(training_set, vocabulary, max_docs=n_docs_for_coherence)
    try:
        print("Running gensim CoherenceModel...")
        cm = CoherenceModel(topics=topics, texts=texts, coherence="c_v")
        score = cm.get_coherence()
    except Exception as e:
        print("CoherenceModel error:", e)
        return 0.0
    print("Topic Coherence (C_V, gensim):", round(score, 4))
    return float(score)


def get_topic_diversity(beta, topk=25):
    """
    Topic diversity: unique top-k words across topics / (K * topk).
    beta: (K, V) numpy or tensor
    """
    if torch.is_tensor(beta):
        beta = beta.detach().cpu().numpy()
    beta = np.asarray(beta)
    K, V = beta.shape
    all_words = set()
    for k in range(K):
        top_idx = np.argsort(beta[k])[::-1][:topk]
        for i in top_idx:
            all_words.add(i)
    diversity = len(all_words) / (K * topk)
    print("Topic Diversity (top-{}):".format(topk), round(diversity, 4))
    return float(diversity)


def nearest_neighbors(model_gensim, word, topn=10):
    """ETM visualize에서 사용. gensim 모델 없으면 빈 리스트 반환."""
    try:
        return [w for w, _ in model_gensim.wv.most_similar(word, topn=topn)]
    except Exception:
        return []


def load_anchor_words_from_llm_words_file(path):
    """
    Load topic-wise anchor words from a topic-word text file.

    Expected format:
        Topic 0: word1 word2 word3
        Topic 1: wordA wordB
    """
    topics = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                topics.append([])
                continue
            if line.lower().startswith("topic ") and ":" in line:
                line = line.split(":", 1)[1].strip()
            topics.append(line.split())
    return topics


def load_anchor_words_from_step3_json(path):
    """
    Load topic-wise anchor words from the refine JSON output.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("step3_json must be a list of topic dicts.")

    topics_dict = {}
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
    topics = []
    for tid in range(max_tid + 1):
        topics.append(topics_dict.get(tid, []))
    return topics


def build_anchor_indices(anchor_words_per_topic, vocab):
    """
    Map anchor words to vocabulary indices.
    """
    vocab_to_idx = {w: i for i, w in enumerate(vocab)}
    anchor_indices_per_topic = []

    for words in anchor_words_per_topic:
        indices = []
        for w in words:
            idx = vocab_to_idx.get(w)
            if idx is not None:
                indices.append(idx)
        anchor_indices_per_topic.append(indices)

    return anchor_indices_per_topic


def summarize_anchor_coverage(anchor_indices_per_topic):
    """
    Return simple coverage stats for anchor words.
    """
    num_topics = len(anchor_indices_per_topic)
    non_empty = sum(1 for idxs in anchor_indices_per_topic if len(idxs) > 0)
    return {
        "num_topics": num_topics,
        "num_topics_with_anchors": non_empty,
    }
