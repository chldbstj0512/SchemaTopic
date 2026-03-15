#!/usr/bin/env bash
# 20NG NVDM auto 파이프라인 (step3 전 reindex 수정 적용) 테스트
# 비교: troubleshooting/1-llama-v2/20NG_nvdm_TD_auto → TD 0.852, CV 등
# reindex 적용 후 schema + anchor만 재실행하여 TD/CV 비교
#
# 사용: LLM(예: Llama) 사용 가능한 환경에서 실행.
#       C_V+TD만 계산: export SCHEMATOPIC_CV_TD_ONLY=1
#       다른 LLM: python3 main.py pipeline ... --model_name gpt-4o

set -e
cd "$(dirname "$0")/.."
RESULTS_ROOT="${RESULTS_ROOT:-results}"
OUT_DIR="${RESULTS_ROOT}/20ng_nvdm_auto_reindex_test"

# 기존 vanilla top_words 사용 시 스키마+앵커만 실행 (동일 입력, reindex 적용된 refine만 재실행)
VANILLA_TOP_WORDS="${1:-}"
if [ -z "$VANILLA_TOP_WORDS" ]; then
  VANILLA_TOP_WORDS="troubleshooting/1-llama-v2/20NG_nvdm_TD_auto/vanilla/top_words.txt"
  if [ ! -f "$VANILLA_TOP_WORDS" ]; then
    echo "Default vanilla not found: $VANILLA_TOP_WORDS"
    echo "Running full pipeline (vanilla -> schema -> anchor)..."
    VANILLA_TOP_WORDS=""
  fi
fi

if [ -n "$VANILLA_TOP_WORDS" ] && [ -f "$VANILLA_TOP_WORDS" ]; then
  echo "=== Schema + anchor only (reindex fix; vanilla: $VANILLA_TOP_WORDS) ==="
  python3 main.py pipeline \
    --dataset 20News \
    --model nvdm \
    --num_topics 50 \
    --out_dir "$OUT_DIR" \
    --topic_words_file "$VANILLA_TOP_WORDS" \
    --epochs 250 \
    --seed 1
else
  echo "=== Full pipeline: vanilla -> schema (reindex fix) -> anchor ==="
  python3 main.py pipeline \
    --dataset 20News \
    --model nvdm \
    --num_topics 50 \
    --out_dir "$OUT_DIR" \
    --epochs 250 \
    --seed 1
fi

echo ""
echo "=== Results (reindex fix applied) ==="
if [ -f "${OUT_DIR}/anchor/metrics.json" ]; then
  cat "${OUT_DIR}/anchor/metrics.json"
  echo ""
  echo "Compare with (before fix): topic_diversity ≈ 0.852, see ANALYSIS_10pct_drop.md §2"
else
  echo "No anchor/metrics.json found."
fi
