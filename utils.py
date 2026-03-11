"""
topic extraction 및 공통 helper 함수
- get_topics, get_topic_coherence (C_V), get_topic_diversity, nearest_neighbors
- Palmetto C_V (run_palmetto_cv) when jar + wikipedia_bd available
"""
import os
import subprocess
import tempfile
import numpy as np
import torch


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
    for i in range(N):
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
    for i, words in enumerate(topics):
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

    texts = _bow_to_texts(training_set, vocabulary, max_docs=n_docs_for_coherence)
    try:
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
