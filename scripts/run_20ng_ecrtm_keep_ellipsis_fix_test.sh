#!/usr/bin/env bash
# 20NG ECRTM keep: 이전에 Step 2에서 ... 로 깨졌던 케이스.
# 동일 vanilla로 스키마+앵커만 돌려서, 프롬프트/ellipsis 실패 처리 변경이 잘 들어갔는지 비교.
# 결과: results/reindex_validation/20ng_ecrtm_keep_ellipsis_fix/

set -e
cd "$(dirname "$0")/.."
export SCHEMATOPIC_CV_TD_ONLY=1

OUT_DIR="${RESULTS_ROOT:-results}/reindex_validation/20ng_ecrtm_keep_ellipsis_fix"
VANILLA="results/1_llama_origin_seed5/pipeline_20News_ecrtm_50_seed1/vanilla/top_words.txt"

if [ ! -f "$VANILLA" ]; then
  echo "Vanilla not found: $VANILLA"
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "=== 20NG ECRTM keep (ellipsis-fix test) ==="
echo "Vanilla: $VANILLA"
echo "Out: $OUT_DIR"
echo ""

python3 main.py pipeline --dataset 20News --model ecrtm --num_topics 50 \
  --out_dir "$OUT_DIR" --topic_words_file "$VANILLA" --epochs 250 --seed 1 --keep

echo ""
echo "=== Result ==="
if [ -f "$OUT_DIR/anchor/metrics.json" ]; then
  if grep -q '"truncated"' "$OUT_DIR/anchor/metrics.json"; then
    echo "FAILED (truncated):"
    cat "$OUT_DIR/anchor/metrics.json"
  else
    echo "SUCCESS (full metrics):"
    cat "$OUT_DIR/anchor/metrics.json"
    echo ""
    echo "Compare with (before fix): troubleshooting/1-llama-v2/20NG_ECRTM_TC_keep CV≈0.421 TD≈0.710"
  fi
else
  echo "No anchor/metrics.json"
fi
