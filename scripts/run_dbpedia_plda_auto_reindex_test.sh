#!/usr/bin/env bash
# DBpedia PLDA auto 파이프라인 (step3 전 reindex 수정 적용) 실행 후 C_V·TD 비교
# 사용: LLM(예: Llama) 사용 가능한 환경에서 실행
# 비교 기준(수정 전): troubleshooting/1-llama-v2/dbpedia_CV_PLDA_auto → CV 0.440, TD 0.902

set -e
cd "$(dirname "$0")/.."
RESULTS_ROOT="${RESULTS_ROOT:-results}"
OUT_DIR="${RESULTS_ROOT}/dbpedia_plda_auto_reindex_test"

# 기존 vanilla top_words로 스키마+앵커만 실행 (vanilla 학습 생략, 동일 입력으로 스키마만 재실행)
VANILLA_TOP_WORDS="${1:-}"
if [ -z "$VANILLA_TOP_WORDS" ]; then
  # 기본: 전체 파이프라인 (vanilla 학습 포함)
  echo "=== Full pipeline: vanilla -> schema (with reindex fix) -> anchor ==="
  python3 main.py pipeline \
    --dataset DBpedia \
    --model plda \
    --num_topics 50 \
    --out_dir "$OUT_DIR" \
    --epochs 250 \
    --seed 1
else
  echo "=== Schema + anchor only (using existing vanilla: $VANILLA_TOP_WORDS) ==="
  python3 main.py pipeline \
    --dataset DBpedia \
    --model plda \
    --num_topics 50 \
    --out_dir "$OUT_DIR" \
    --topic_words_file "$VANILLA_TOP_WORDS" \
    --epochs 250 \
    --seed 1
fi

echo ""
echo "=== Results (reindex fix applied) ==="
if [ -f "${OUT_DIR}/anchor/metrics.json" ]; then
  cat "${OUT_DIR}/anchor/metrics.json"
  echo ""
  echo "Compare with (before fix): topic_coherence_cv ≈ 0.440, topic_diversity ≈ 0.902"
else
  echo "No anchor/metrics.json found."
fi
