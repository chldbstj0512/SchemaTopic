"""
topic coherence (C_V), topic diversity, clustering 성능 (Purity, NMI)
"""
import numpy as np
from utils import get_topic_coherence, get_topic_diversity, get_topics


def compute_purity_nmi(theta, labels, num_topics):
    """
    theta: (N, K) document-topic distribution
    labels: (N,) ground truth class indices (0..C-1)
    num_topics: K
    Returns: purity, nmi (scalars)
    """
    N = theta.shape[0]
    pred_cluster = np.argmax(theta, axis=1)  # (N,)
    labels = np.asarray(labels).flatten().astype(int)
    if N == 0 or len(labels) == 0:
        return 0.0, 0.0
    num_classes = int(labels.max()) + 1
    if num_classes <= 0:
        return 0.0, 0.0

    # Purity: for each cluster, assign to majority class; purity = sum(max_j n_ij) / N
    contingency = np.zeros((num_topics, num_classes))
    for i in range(N):
        if labels[i] < num_classes:
            contingency[pred_cluster[i], labels[i]] += 1
    purity = np.sum(np.max(contingency, axis=1)) / N

    # NMI
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    pred_counts = np.bincount(pred_cluster, minlength=num_topics) / N
    true_counts = np.bincount(labels, minlength=num_classes) / N
    H_pred = entropy(pred_counts)
    H_true = entropy(true_counts)
    mutual = 0.0
    for k in range(num_topics):
        for c in range(num_classes):
            p_kc = contingency[k, c] / N
            if p_kc > 0:
                p_k = pred_counts[k]
                p_c = true_counts[c]
                if p_c > 0:
                    mutual += p_kc * np.log2(p_kc / (p_k * p_c))
    if H_pred <= 0 or H_true <= 0:
        nmi = 0.0
    else:
        nmi = 2 * mutual / (H_pred + H_true)
    return float(purity), float(nmi)


def run_evaluation(beta, theta_train, theta_test, train_bow, train_labels, test_labels,
                   vocab, num_topics, topk_words=10, n_docs_coherence=2000, root_dir=None):
    """
    TC (C_V), TD, (Purity + NMI)/2 계산.
    root_dir: Palmetto jar + wikipedia_bd 경로 (있으면 C_V에 Palmetto 사용)
    """
    if hasattr(beta, "detach"):
        beta_np = beta.detach().cpu().numpy()
    else:
        beta_np = np.asarray(beta)

    print("\n[Evaluation] Starting topic coherence / diversity / clustering metrics...")
    tc = get_topic_coherence(
        beta_np, train_bow, vocab,
        topk=topk_words, n_docs_for_coherence=n_docs_coherence,
        root_dir=root_dir,
    )
    print("[Evaluation] Topic coherence finished.")
    td = get_topic_diversity(beta_np, topk=25)
    print("[Evaluation] Topic diversity finished.")

    # Clustering on train (표준은 test로도 보고)
    purity_train, nmi_train = compute_purity_nmi(
        theta_train, train_labels, num_topics
    )
    purity_test, nmi_test = compute_purity_nmi(
        theta_test, test_labels, num_topics
    )
    purity = (purity_train + purity_test) / 2
    nmi = (nmi_train + nmi_test) / 2
    pn = (purity + nmi) / 2  # PN = (Purity + NMI) / 2
    print("Purity (train/test avg):", round(purity, 4))
    print("NMI (train/test avg):", round(nmi, 4))
    print("PN (Purity+NMI)/2:", round(pn, 4))
    print("[Evaluation] All metrics finished.\n")

    return {
        "topic_coherence_cv": tc,
        "topic_diversity": td,
        "purity": purity,
        "nmi": nmi,
        "PN": pn,
    }


def get_top_words_per_topic(beta, vocab, topk=15):
    """상위 단어 목록 (저장/출력용)."""
    return get_topics(beta, vocab, topk=topk)
