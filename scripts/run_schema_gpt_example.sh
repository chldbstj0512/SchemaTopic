#!/usr/bin/env bash
# Schema 단계만 GPT(gpt-4o)로 바로 실행하는 예시.
# 사용: conda activate llm_itl 후 ./scripts/run_schema_gpt_example.sh
# 또는: bash scripts/run_schema_gpt_example.sh (conda run 사용)

set -e
cd "$(dirname "$0")/.."

# .env 있으면 OPENAI_API_KEY 로드
if [ -f .env ]; then set -a; source .env; set +a; fi

TOPIC_WORDS="${1:-experiments/4/full/vanilla_20News_etm_50/top_words.txt}"
OUT_DIR="${2:-results/schema_gpt_example}"

if [ ! -f "$TOPIC_WORDS" ]; then
  echo "Not found: $TOPIC_WORDS"
  echo "Usage: $0 [topic_words_file] [out_dir]"
  echo "  Example: $0 results/vanilla_20News_etm_50/top_words.txt results/my_schema"
  exit 1
fi

echo "Topic words: $TOPIC_WORDS"
echo "Output dir:  $OUT_DIR"
echo ""

python main.py schema \
  --topic_words_file "$TOPIC_WORDS" \
  --model_name gpt-4o \
  --out_dir "$OUT_DIR"

echo ""
echo "Done. Output: $OUT_DIR"
