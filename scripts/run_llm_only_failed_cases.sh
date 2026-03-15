#!/usr/bin/env bash
# 실패했던 10건에 대해 vanilla output으로 LLM 추론(schema)만 실행. 앵커 학습 없음.
# 사용 (한 번에 10건 순서대로):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_llm_only_failed_cases.sh
#   또는
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_llm_only_failed_cases.sh all

set -e
cd "$(dirname "$0")/.."
BASE="results/1_llama_origin_seed5"
OUT_ROOT="${OUT_ROOT:-results/llm_only_run}"
mkdir -p "$OUT_ROOT"

# 10 cases: (vanilla_path, out_name, keep_flag)
# keep_flag: "" or "--keep"
CASES=(
  "$BASE/pipeline_20News_ecrtm_50_seed1/vanilla/top_words.txt|20ng_ecrtm_keep|--keep"
  "$BASE/pipeline_DBpedia_plda_50_seed1/vanilla/top_words.txt|dbpedia_plda_seed1|"
  "$BASE/pipeline_DBpedia_plda_50_seed2/vanilla/top_words.txt|dbpedia_plda_seed2|"
  "$BASE/pipeline_DBpedia_plda_50_seed3/vanilla/top_words.txt|dbpedia_plda_seed3|"
  "$BASE/pipeline_DBpedia_nvdm_50_seed5/vanilla/top_words.txt|dbpedia_nvdm_seed5|"
  "$BASE/pipeline_R8_nvdm_50_seed1/vanilla/top_words.txt|r8_nvdm_seed1|"
  "$BASE/pipeline_R8_nvdm_50_seed2/vanilla/top_words.txt|r8_nvdm_seed2|"
  "$BASE/pipeline_R8_nvdm_50_seed3/vanilla/top_words.txt|r8_nvdm_seed3|"
  "$BASE/pipeline_R8_nvdm_50_seed4/vanilla/top_words.txt|r8_nvdm_seed4|"
  "$BASE/pipeline_R8_nvdm_50_seed5/vanilla/top_words.txt|r8_nvdm_seed5|"
)

START=0
END=10
echo "=== LLM-only run (all 10 cases, one by one) ==="
for i in $(seq $START $((END-1))); do
  IFS='|' read -r vanilla_path out_name keep_flag <<< "${CASES[$i]}"
  if [ ! -f "$vanilla_path" ]; then
    echo "[$i] SKIP vanilla missing: $vanilla_path"
    continue
  fi
  out_dir="$OUT_ROOT/$out_name"
  echo "[$i] $out_name <- $vanilla_path"
  python3 main.py schema --topic_words_file "$vanilla_path" --out_dir "$out_dir" $keep_flag
  echo "[$i] Done: $out_dir"
done
echo "=== All 10 done ==="
