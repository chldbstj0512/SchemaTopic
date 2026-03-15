#!/usr/bin/env bash
# 10건: 스키마(LLM) 실행 → 전부 성공 시 각 케이스별 NTM(앵커) 재학습 → 메트릭 변화 요약
# 사용: CUDA_VISIBLE_DEVICES=0 bash scripts/run_ten_cases_schema_then_anchor.sh
#       또는 2>&1 | tee results/llm_only_run/log_schema_anchor.txt

set -e
cd "$(dirname "$0")/.."
BASE="results/1_llama_origin_seed5"
OUT_ROOT="${OUT_ROOT:-results/llm_only_run}"
mkdir -p "$OUT_ROOT"
NUM_TOPICS=50

# (vanilla_path, out_name, keep_flag, dataset, model)
CASES=(
  "$BASE/pipeline_20News_ecrtm_50_seed1/vanilla/top_words.txt|20ng_ecrtm_keep|--keep|20News|ecrtm"
  "$BASE/pipeline_DBpedia_plda_50_seed1/vanilla/top_words.txt|dbpedia_plda_seed1||DBpedia|plda"
  "$BASE/pipeline_DBpedia_plda_50_seed2/vanilla/top_words.txt|dbpedia_plda_seed2||DBpedia|plda"
  "$BASE/pipeline_DBpedia_plda_50_seed3/vanilla/top_words.txt|dbpedia_plda_seed3||DBpedia|plda"
  "$BASE/pipeline_DBpedia_nvdm_50_seed5/vanilla/top_words.txt|dbpedia_nvdm_seed5||DBpedia|nvdm"
  "$BASE/pipeline_R8_nvdm_50_seed1/vanilla/top_words.txt|r8_nvdm_seed1||R8|nvdm"
  "$BASE/pipeline_R8_nvdm_50_seed2/vanilla/top_words.txt|r8_nvdm_seed2||R8|nvdm"
  "$BASE/pipeline_R8_nvdm_50_seed3/vanilla/top_words.txt|r8_nvdm_seed3||R8|nvdm"
  "$BASE/pipeline_R8_nvdm_50_seed4/vanilla/top_words.txt|r8_nvdm_seed4||R8|nvdm"
  "$BASE/pipeline_R8_nvdm_50_seed5/vanilla/top_words.txt|r8_nvdm_seed5||R8|nvdm"
)

echo "========== Phase 1: Schema (LLM) for 10 cases =========="
FAILED=0
for i in $(seq 0 9); do
  IFS='|' read -r vanilla_path out_name keep_flag dataset model <<< "${CASES[$i]}"
  if [ ! -f "$vanilla_path" ]; then
    echo "[$i] SKIP vanilla missing: $vanilla_path"
    FAILED=1
    continue
  fi
  out_dir="$OUT_ROOT/$out_name"
  echo "[$i] Schema: $out_name"
  if ! python3 main.py schema --topic_words_file "$vanilla_path" --out_dir "$out_dir" $keep_flag; then
    echo "[$i] Schema FAILED: $out_name"
    FAILED=1
  else
    echo "[$i] Schema OK: $out_name"
  fi
done

if [ "$FAILED" -ne 0 ]; then
  echo "========== Some schema runs failed. Skipping anchor training. =========="
  exit 1
fi

echo ""
echo "========== Phase 2: Anchor (NTM) training for 10 cases =========="
for i in $(seq 0 9); do
  IFS='|' read -r vanilla_path out_name keep_flag dataset model <<< "${CASES[$i]}"
  schema_dir="$OUT_ROOT/$out_name"
  anchor_dir="$OUT_ROOT/$out_name/anchor"
  if [ ! -f "$schema_dir/schema_topics.json" ]; then
    echo "[$i] SKIP no schema_topics.json: $out_name"
    continue
  fi
  echo "[$i] Anchor: $out_name ($dataset $model)"
  python3 main.py anchor \
    --dataset "$dataset" \
    --model "$model" \
    --num_topics "$NUM_TOPICS" \
    --schema_dir "$schema_dir" \
    --out_dir "$anchor_dir"
  echo "[$i] Anchor done: $out_name"
done

echo ""
echo "========== Phase 3: Metric summary (vanilla vs schema+anchor) =========="
python3 - << 'PY'
import json
import os

OUT_ROOT = os.environ.get("OUT_ROOT", "results/llm_only_run")
BASE = "results/1_llama_origin_seed5"
cases = [
    ("20ng_ecrtm_keep", "pipeline_20News_ecrtm_50_seed1", "keep"),
    ("dbpedia_plda_seed1", "pipeline_DBpedia_plda_50_seed1", "auto"),
    ("dbpedia_plda_seed2", "pipeline_DBpedia_plda_50_seed2", "auto"),
    ("dbpedia_plda_seed3", "pipeline_DBpedia_plda_50_seed3", "auto"),
    ("dbpedia_nvdm_seed5", "pipeline_DBpedia_nvdm_50_seed5", "auto"),
    ("r8_nvdm_seed1", "pipeline_R8_nvdm_50_seed1", "auto"),
    ("r8_nvdm_seed2", "pipeline_R8_nvdm_50_seed2", "auto"),
    ("r8_nvdm_seed3", "pipeline_R8_nvdm_50_seed3", "auto"),
    ("r8_nvdm_seed4", "pipeline_R8_nvdm_50_seed4", "auto"),
    ("r8_nvdm_seed5", "pipeline_R8_nvdm_50_seed5", "auto"),
]
folder = "keep"  # pipeline has 'keep' or 'auto' subdir
print(f"{'case':<22} | {'C_V (vanilla)':>12} | {'C_V (anchor)':>12} | {'TD (vanilla)':>12} | {'TD (anchor)':>12}")
print("-" * 80)
for out_name, pipe_name, mode in cases:
    vanilla_metrics = os.path.join(BASE, pipe_name, "vanilla", "metrics.json")
    anchor_metrics = os.path.join(OUT_ROOT, out_name, "anchor", "metrics.json")
    cv_v = td_v = cv_a = td_a = "—"
    if os.path.exists(vanilla_metrics):
        with open(vanilla_metrics) as f:
            d = json.load(f)
        cv_v = d.get("topic_coherence_cv", "—")
        td_v = d.get("topic_diversity", "—")
        if isinstance(cv_v, float): cv_v = f"{cv_v:.4f}"
        if isinstance(td_v, float): td_v = f"{td_v:.4f}"
    if os.path.exists(anchor_metrics):
        with open(anchor_metrics) as f:
            d = json.load(f)
        cv_a = d.get("topic_coherence_cv", "—")
        td_a = d.get("topic_diversity", "—")
        if isinstance(cv_a, float): cv_a = f"{cv_a:.4f}"
        if isinstance(td_a, float): td_a = f"{td_a:.4f}"
    print(f"{out_name:<22} | {cv_v:>12} | {cv_a:>12} | {td_v:>12} | {td_a:>12}")
print("========== Done ==========")
PY
