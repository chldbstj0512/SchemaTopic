#!/usr/bin/env bash
# 4번 실험 (Step ablation): full, no_step1, no_step2, no_step3 네 가지 pipeline 실행.
# Vanilla 1회 후 동일 top_words 로 네 가지 schema 구성을 돌림.
# 사용: conda activate llm_itl 후 프로젝트 루트에서 bash scripts/run_experiment4_example.sh

set -e
cd "$(dirname "$0")/.."

MODEL="${1:-etm}"
DATASET="${2:-20News}"
NUM_TOPICS="${3:-50}"
BASE="results/vanilla_${DATASET}_${MODEL}_${NUM_TOPICS}"
TOPIC_WORDS="${BASE}/top_words.txt"
OUT_BASE="results/exp4_${MODEL}"

if [ -f .env ]; then set -a; source .env; set +a; fi

echo "=== Experiment 4: Step ablation ==="
echo "Model: $MODEL, Dataset: $DATASET, Num topics: $NUM_TOPICS"
echo ""

# 1) Vanilla (없으면 한 번만)
if [ ! -f "$TOPIC_WORDS" ]; then
  echo ">>> Running vanilla (once)"
  python main.py vanilla --model "$MODEL" --dataset "$DATASET" --num_topics "$NUM_TOPICS"
  echo ""
fi

# 2) Full
echo ">>> Pipeline: full (no skip)"
python main.py pipeline --model "$MODEL" --dataset "$DATASET" --num_topics "$NUM_TOPICS" \
  --topic_words_file "$TOPIC_WORDS" --out_dir "${OUT_BASE}_full"
echo ""

# 3) No step 1
echo ">>> Pipeline: without step 1"
python main.py pipeline --model "$MODEL" --dataset "$DATASET" --num_topics "$NUM_TOPICS" \
  --topic_words_file "$TOPIC_WORDS" --skip_step1 --out_dir "${OUT_BASE}_no_step1"
echo ""

# 4) No step 2
echo ">>> Pipeline: without step 2"
python main.py pipeline --model "$MODEL" --dataset "$DATASET" --num_topics "$NUM_TOPICS" \
  --topic_words_file "$TOPIC_WORDS" --skip_step2 --out_dir "${OUT_BASE}_no_step2"
echo ""

# 5) No step 3
echo ">>> Pipeline: without step 3"
python main.py pipeline --model "$MODEL" --dataset "$DATASET" --num_topics "$NUM_TOPICS" \
  --topic_words_file "$TOPIC_WORDS" --skip_step3 --out_dir "${OUT_BASE}_no_step3"
echo ""

echo "=== Done. Outputs: ==="
echo "  ${OUT_BASE}_full/"
echo "  ${OUT_BASE}_no_step1/"
echo "  ${OUT_BASE}_no_step2/"
echo "  ${OUT_BASE}_no_step3/"
