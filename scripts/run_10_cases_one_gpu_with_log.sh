#!/usr/bin/env bash
# 10건 스키마 실행: GPU 1개만 사용, 순차 실행, 터미널에 진행 현황 출력, 결과를 파일에 기록
# 사용: bash scripts/run_10_cases_one_gpu_with_log.sh
#       (CUDA_VISIBLE_DEVICES=0 자동 설정)

set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0

BASE="results/1_llama_origin_seed5"
OUT_ROOT="${OUT_ROOT:-results/llm_only_run}"
mkdir -p "$OUT_ROOT"
RESULT_TSV="${RESULT_TSV:-troubleshooting/run_result_0315_2244.tsv}"
mkdir -p "$(dirname "$RESULT_TSV")"

# 헤더
echo -e "case\tstatus\tdetail" > "$RESULT_TSV"

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

echo "=============================================="
echo " 10건 스키마 실행 (GPU 0만 사용, 순차)"
echo " 결과 기록: $RESULT_TSV"
echo "=============================================="

for i in $(seq 0 9); do
  IFS='|' read -r vanilla_path out_name keep_flag <<< "${CASES[$i]}"
  out_dir="$OUT_ROOT/$out_name"
  n=$((i+1))
  echo ""
  echo "---------- [ $n / 10 ] $out_name ----------"
  if [ ! -f "$vanilla_path" ]; then
    echo "  SKIP: vanilla 없음 — $vanilla_path"
    echo -e "${out_name}\tSKIP\tvanilla 없음" >> "$RESULT_TSV"
    continue
  fi
  echo "  실행 중 ... (schema only)"
  if python3 main.py schema --topic_words_file "$vanilla_path" --out_dir "$out_dir" $keep_flag 2>&1; then
    if [ -f "$out_dir/schema_topics.json" ]; then
      n_topics=$(python3 -c '
import json,sys
p=sys.argv[1]
d=json.load(open(p))
groups=d.get("schema") or []
n=sum(len(g.get("topics") or []) for g in groups if isinstance(g,dict))
print(n)
' "$out_dir/schema_topics.json" 2>/dev/null) || n_topics="?"
      echo "  완료: OK (${n_topics} topics)"
      echo -e "${out_name}\tOK\t${n_topics} topics" >> "$RESULT_TSV"
    else
      echo "  완료: FAIL (schema_topics.json 없음)"
      echo -e "${out_name}\tFAIL\tschema_topics.json 없음" >> "$RESULT_TSV"
    fi
  else
    detail="실패"
    if [ -f "$out_dir/step3.txt" ]; then
      if grep -q '\.\.\.' "$out_dir/step3.txt" 2>/dev/null; then
        detail="Step3 truncation"
      elif grep -qi truncat "$out_dir/step3.txt" 2>/dev/null; then
        detail="Step3 truncation"
      fi
    fi
    if [ -f "$out_dir/step1.txt" ]; then
      lines=$(wc -l < "$out_dir/step1.txt" 2>/dev/null)
      if [ -n "${lines}" ] && [ "${lines}" -gt 500 ] 2>/dev/null; then
        detail="Step1 과다(${lines}줄) 또는 Step3 실패"
      fi
    fi
    echo "  완료: FAIL ($detail)"
    echo -e "${out_name}\tFAIL\t$detail" >> "$RESULT_TSV"
  fi
done

echo ""
echo "=============================================="
echo " 10건 모두 처리 완료. 결과: $RESULT_TSV"
echo "=============================================="
cat "$RESULT_TSV"
