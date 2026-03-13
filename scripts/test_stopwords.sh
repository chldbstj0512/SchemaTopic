#!/bin/bash
#
# Stopwords 후처리 실험: 학습 제외, refine(schema) 단계만 실행
# troubleshooting 5개 설정과 동일한 입력 사용
# 이전 결과(llm_test)와 topic_words 비교
#
# 사용법:
#   ./scripts/test_stopwords.sh [--dry-run]
#   CUDA_VISIBLE_DEVICES=1 ./scripts/test_stopwords.sh  # GPU 1 사용 (OOM 시)
#

cd "$(dirname "$0")/.."

PYTHON="${CONDA_PREFIX}/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="/home/ys0660/anaconda3/envs/ys0660/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
[[ -x "$PYTHON" ]] || PYTHON="python"

OUT_ROOT="troubleshooting/stopwords_test"
BASELINE_ROOT="troubleshooting/llm_test"
mkdir -p "$OUT_ROOT"

# run_troubleshoot.sh와 동일한 5개 조합 (troubleshooting/original 기준)
INPUTS=(
  "troubleshooting/original/20News_nstm_50_seed1/top_words.txt"
  "troubleshooting/original/20News_ecrtm_50_seed1/top_words.txt"
  "troubleshooting/original/R8_ecrtm_50_seed1/top_words.txt"
  "troubleshooting/original/20News_etm_50_seed1/top_words.txt"
  "troubleshooting/original/DBpedia_nstm_50_seed1/top_words.txt"
)

for i in "${!INPUTS[@]}"; do
  inp="${INPUTS[$i]}"
  name=$(basename "$(dirname "$inp")")
  out_dir="${OUT_ROOT}/${name}"
  if [[ ! -f "$inp" ]]; then
    echo "[SKIP] $name: $inp not found"
    continue
  fi
  echo ""
  echo "=========================================="
  echo "[$((i+1))/${#INPUTS[@]}] $name (refine only, stopwords filter 적용)"
  echo "  input: $inp"
  echo "  output: $out_dir"
  echo "=========================================="
  cmd="$PYTHON main.py schema --topic_words_file $inp --out_dir $out_dir --keep"
  if [[ "$1" == "--dry-run" ]]; then
    echo "  $cmd"
  else
    eval "$cmd"
  fi
done

echo ""
echo "=========================================="
echo "Stopwords 실험 완료"
echo "출력: ${OUT_ROOT}/"
echo "=========================================="

# 비교: stopwords_test vs llm_test (이전 결과)
if [[ "$1" != "--dry-run" ]] && [[ -d "$BASELINE_ROOT" ]]; then
  echo ""
  echo "=== topic_words 비교 (stopwords_test vs llm_test) ==="
  "$PYTHON" - "$OUT_ROOT" "$BASELINE_ROOT" << 'PYEOF'
import sys
from pathlib import Path

OUT_ROOT = Path(sys.argv[1])
BASELINE_ROOT = Path(sys.argv[2])

NAMES = [
    "20News_nstm_50_seed1",
    "20News_ecrtm_50_seed1",
    "R8_ecrtm_50_seed1",
    "20News_etm_50_seed1",
    "DBpedia_nstm_50_seed1",
]

# tool.py stopwords (비교용)
try:
    from tool import _load_stopwords
    STOPWORDS = _load_stopwords()
except Exception:
    STOPWORDS = set()

def load_topic_words(path):
    if not path.exists():
        return {}
    topics = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.lower().startswith("topic "):
                continue
            parts = line.split(":", 1)
            if len(parts) < 2:
                continue
            try:
                tid = int(parts[0].replace("Topic", "").strip())
                words = parts[1].strip().split()
                topics[tid] = words
            except (ValueError, IndexError):
                continue
    return topics

for name in NAMES:
    new_path = OUT_ROOT / name / "topic_words.txt"
    old_path = BASELINE_ROOT / name / "topic_words.txt"
    if not new_path.exists():
        print(f"\n[{name}] stopwords_test 결과 없음")
        continue
    if not old_path.exists():
        print(f"\n[{name}] llm_test 기준 없음 (비교 스킵)")
        continue

    new_topics = load_topic_words(new_path)
    old_topics = load_topic_words(old_path)

    removed_count = 0
    total_removed = 0
    examples = []
    for tid in sorted(set(new_topics) | set(old_topics)):
        old_w = set(w.lower() for w in old_topics.get(tid, []))
        new_w = set(w.lower() for w in new_topics.get(tid, []))
        removed = old_w - new_w
        stopwords_removed = removed & STOPWORDS
        if stopwords_removed:
            removed_count += 1
            total_removed += len(stopwords_removed)
            if len(examples) < 3:
                examples.append((tid, list(stopwords_removed)[:5], list(new_w)[:6]))

    print(f"\n[{name}]")
    print(f"  stopwords 제거된 토픽 수: {removed_count} / {len(new_topics)}")
    print(f"  제거된 stopword 총 개수: {total_removed}")
    if examples:
        for tid, rem, kept in examples:
            print(f"  예시 Topic {tid}: 제거됨 {rem} → 남은단어 {kept}...")
PYEOF
fi
