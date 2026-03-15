"""
topic extraction 및 공통 helper 함수
- get_topics, get_topic_coherence (C_V), get_topic_coherence_metrics (C_V, NPMI, UCI, UMass), get_topic_diversity
- Coherence 기본: Gensim. Palmetto 쓰려면 SCHEMATOPIC_USE_PALMETTO=1 + root_dir (jar, wikipedia_bd)
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


def run_palmetto_measure_batched(root_dir, topics, measure="C_V", temp_folder=None, timeout=600, topk=10):
    """
    Palmetto로 단일 coherence measure 계산. **전체 토픽을 한 파일에 넣고 measure당 1회만** Java 호출.
    (기존: 토픽당 1회 → N회. 배치: 1회) 훨씬 빠름.
    measure: C_V, NPMI, UCI, UMass, C_P, C_A
    timeout: 전체 실행 제한(초). 기본 600(10분).
    Returns: mean coherence or None
    """
    jar = os.path.join(root_dir, "palmetto-0.1.5-exec.jar")
    wikipedia_bd = os.path.join(root_dir, "wikipedia_bd")
    if not os.path.isfile(jar) or not os.path.isdir(wikipedia_bd):
        return None
    out_dir = temp_folder or os.path.join(root_dir, "temp_tc_palmetto")
    os.makedirs(out_dir, exist_ok=True)
    # 한 파일에 "한 줄에 한 토픽" (Palmetto 공식 포맷)
    all_path = os.path.join(out_dir, "all_topics_%s.txt" % measure.replace(" ", "_"))
    lines = []
    for words in topics:
        line = " ".join(words[:topk]) if isinstance(words, (list, tuple)) else (words if isinstance(words, str) else "")
        if line and line.strip():
            lines.append(line.strip())
    if not lines:
        return None
    with open(all_path, "w") as f:
        f.write("\n".join(lines))
    try:
        r = subprocess.run(
            ["java", "-jar", jar, wikipedia_bd, measure, os.path.abspath(all_path)],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode == 0 and r.stdout:
            return _read_tc(r.stdout)
    except (subprocess.TimeoutExpired, Exception):
        pass
    return None


def run_palmetto_measure(root_dir, topics, measure="C_V", temp_folder=None, timeout_per_topic=90, topk=10):
    """
    Palmetto 단일 measure. **배치 방식** 사용 (measure당 Java 1회). 빠름.
    timeout: 배치 1회당 상한(초). 기본 최대 10분.
    """
    batch_timeout = min(600, max(120, timeout_per_topic * max(1, len(topics))))
    return run_palmetto_measure_batched(
        root_dir, topics, measure=measure, temp_folder=temp_folder, timeout=batch_timeout, topk=topk
    )


def run_palmetto_cv(root_dir, topics, temp_folder=None, timeout_per_topic=90, topk=10):
    """Palmetto C_V only. Kept for backward compatibility."""
    return run_palmetto_measure(root_dir, topics, measure="C_V", temp_folder=temp_folder, timeout_per_topic=timeout_per_topic, topk=topk)


# Palmetto measure names for C_V, NPMI, UCI, UMass (jar 3rd argument)
PALMETTO_COHERENCE_MEASURES = ["C_V", "NPMI", "UCI", "UMass"]


def get_topic_coherence_metrics(beta, training_set, vocabulary, topk=10, n_docs_for_coherence=2000, root_dir=None, measures=None):
    """
    C_V, NPMI, UCI, UMass coherence 계산.
    - **기본(default)**: root_dir에 Palmetto(jar + wikipedia_bd)가 있으면 Palmetto 사용.
    - measures: None이면 4종 전부, ["C_V"]면 C_V만 (빠른 평가용).
    Returns: dict with keys topic_coherence_cv, topic_coherence_npmi, topic_coherence_uci, topic_coherence_umass
    """
    if torch.is_tensor(beta):
        beta = beta.detach().cpu().numpy()
    beta = np.asarray(beta)
    topics = get_topics(beta, vocabulary, topk=topk)
    print("Preparing topic coherence evaluation for {} topics...".format(len(topics)))

    out = {
        "topic_coherence_cv": None,
        "topic_coherence_npmi": None,
        "topic_coherence_uci": None,
        "topic_coherence_umass": None,
    }
    measures_to_run = measures if measures is not None else PALMETTO_COHERENCE_MEASURES

    jar = os.path.join(root_dir or "", "palmetto-0.1.5-exec.jar")
    wikipedia_bd = os.path.join(root_dir or "", "wikipedia_bd")
    palmetto_available = root_dir and os.path.isfile(jar) and os.path.isdir(wikipedia_bd)
    use_palmetto = palmetto_available and os.environ.get("SCHEMATOPIC_USE_PALMETTO", "1").strip().lower() not in ("0", "false", "no")
    if use_palmetto:
        print("Computing topic coherence with Palmetto (%s)..." % (", ".join(measures_to_run)))
        for measure in measures_to_run:
            val = run_palmetto_measure(root_dir, topics, measure=measure, topk=topk)
            key = "topic_coherence_cv" if measure == "C_V" else "topic_coherence_npmi" if measure == "NPMI" else "topic_coherence_uci" if measure == "UCI" else "topic_coherence_umass"
            out[key] = float(val) if val is not None else None
        for k, v in out.items():
            if v is not None:
                print("  %s: %s" % (k, round(v, 4)))
        return out

    try:
        from gensim.models import CoherenceModel
    except ImportError:
        print("gensim not installed; skipping coherence metrics.")
        return {k: 0.0 for k in out}

    gensim_map = [
        ("c_v", "topic_coherence_cv", "C_V"),
        ("c_npmi", "topic_coherence_npmi", "NPMI"),
        ("c_uci", "topic_coherence_uci", "UCI"),
        ("u_mass", "topic_coherence_umass", "UMass"),
    ]
    print("Computing topic coherence with Gensim (up to %d documents)..." % n_docs_for_coherence)
    texts = _bow_to_texts(training_set, vocabulary, max_docs=n_docs_for_coherence)
    for coherence_type, key, name in gensim_map:
        if name not in measures_to_run:
            continue
        try:
            cm = CoherenceModel(topics=topics, texts=texts, coherence=coherence_type)
            out[key] = float(cm.get_coherence())
            print("  %s (gensim): %s" % (key, round(out[key], 4)))
        except Exception as e:
            print("  %s: failed (%s)" % (key, e))
    return out


def get_topic_coherence(beta, training_set, vocabulary, topk=10, n_docs_for_coherence=2000, root_dir=None):
    """
    C_V만 반환 (기존 API 호환). 내부적으로 get_topic_coherence_metrics 사용.
    """
    metrics = get_topic_coherence_metrics(
        beta, training_set, vocabulary, topk=topk, n_docs_for_coherence=n_docs_for_coherence, root_dir=root_dir
    )
    cv = metrics.get("topic_coherence_cv")
    return float(cv) if cv is not None else 0.0


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
            if ":" in line:
                line = line.split(":", 1)[1].strip()
            topics.append(line.split())
    return topics


def load_anchor_words_from_step3_json(path):
    """
    Load topic-wise anchor words from the refine JSON output.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        schema_groups = data.get("schema", [])
        if not isinstance(schema_groups, list):
            raise ValueError("schema_topics_json must contain a 'schema' list.")

        topics_dict = {}
        for group in schema_groups:
            if not isinstance(group, dict):
                continue
            topics = group.get("topics", [])
            if not isinstance(topics, list):
                continue
            for item in topics:
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
        return [topics_dict.get(tid, []) for tid in range(max_tid + 1)]

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
