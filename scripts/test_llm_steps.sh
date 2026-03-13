#!/bin/bash
#
# LLM 3단계(Step1 스키마, Step2 프루닝, Step3 refine)만 테스트
# 학습(anchor) 없이 여러 topic words 입력으로 실행
#
# 사용법:
#   ./scripts/test_llm_steps.sh [--dry-run]
#   CUDA_VISIBLE_DEVICES=0 ./scripts/test_llm_steps.sh
#

cd "$(dirname "$0")/.."

PYTHON="${CONDA_PREFIX}/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="/home/ys0660/anaconda3/envs/ys0660/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
[[ -x "$PYTHON" ]] || PYTHON="python"

OUT_ROOT="troubleshooting/llm_test"
mkdir -p "$OUT_ROOT"

# 입력: topic words 파일 (다양한 데이터셋/모델 조합)
INPUTS=(
  "troubleshooting/original/20News_nstm_50_seed1/top_words.txt"
  "troubleshooting/original/20News_ecrtm_50_seed1/top_words.txt"
  "troubleshooting/original/DBpedia_nstm_50_seed1/top_words.txt"
  "troubleshooting/original/R8_ecrtm_50_seed1/top_words.txt"
  "troubleshooting/original/20News_etm_50_seed1/top_words.txt"
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
  echo "[$((i+1))/${#INPUTS[@]}] $name"
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
echo "LLM 3단계 테스트 완료"
echo "출력: ${OUT_ROOT}/"
echo "=========================================="
