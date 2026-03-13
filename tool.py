# -*- coding: utf-8 -*-
"""
LLM 후처리 툴: stopwords 제거 등 topic word 정제 함수.
Step 3 최종 아웃풋에서 stopwords를 제거하고, schema_topics.json 등 모든 최종 출력에 반영.
"""
# ---------------------------------------------------------------------------
# Stopwords: NLTK + sklearn ENGLISH_STOP_WORDS (process_agnews와 동일 소스)
# NLTK/sklearn 없을 때 fallback용 기본 리스트
# ---------------------------------------------------------------------------
_FALLBACK_STOPWORDS = frozenset([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "need", "used", "know", "think",
    "make", "want", "way", "come", "going", "say", "time", "good", "need",
    "better", "actually", "really", "sure", "able", "example", "instance",
    "fact", "thing", "point", "let", "like", "get", "use", "see", "take",
])

_STOPWORDS = None  # type: set


def _load_stopwords():
    """stopwords 단어사전 로드. NLTK + sklearn 병합. 없으면 fallback 사용."""
    global _STOPWORDS
    if _STOPWORDS is not None:
        return _STOPWORDS
    words = set(_FALLBACK_STOPWORDS)
    try:
        from nltk.corpus import stopwords
        import nltk
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        words |= set(stopwords.words("english"))
    except Exception:
        pass
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        words |= set(ENGLISH_STOP_WORDS)
    except Exception:
        pass
    _STOPWORDS = words
    return _STOPWORDS


def filter_stopwords(words):
    """Topic words에서 stopwords 제거. Step 3 최종 아웃풋 후처리용."""
    if not words:
        return []
    stop = _load_stopwords()
    return [w for w in words if isinstance(w, str) and w.strip().lower() not in stop]
