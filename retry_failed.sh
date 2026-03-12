#!/bin/bash
#
# 실패한 파이프라인만 재실행
# failed_gpu0.txt 또는 failed_gpu1.txt에서 읽어서 해당 실험만 다시 돌림
#
# 사용법:
#   CUDA_VISIBLE_DEVICES=0 ./retry_failed.sh gpu0   # GPU0 실패분 재실행
#   CUDA_VISIBLE_DEVICES=1 ./retry_failed.sh gpu1   # GPU1 실패분 재실행
#
# (기존 auto/keep도 지원: failed_auto.txt, failed_keep.txt)
#

cd "$(dirname "$0")"
PYTHON="${CONDA_PREFIX}/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="/home/ys0660/anaconda3/envs/ys0660/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
[[ -x "$PYTHON" ]] || PYTHON="python"
MODE=${1:-gpu0}
LOG_DIR="results/experiment_logs"
FAILED_FILE="${LOG_DIR}/failed_${MODE}.txt"
NUM_TOPICS=50

if [[ ! -f "$FAILED_FILE" ]]; then
  echo "실패 목록 없음: $FAILED_FILE"
  exit 0
fi

# "  - dataset model seedN [auto|keep] 시각" 형식 파싱
failed_lines=$(grep -E "^\s+-\s+" "$FAILED_FILE" | sed 's/^[[:space:]]*-[[:space:]]*//')
count=$(echo "$failed_lines" | grep -c . 2>/dev/null || echo 0)

if [[ "$count" -eq 0 ]]; then
  echo "재실행할 실패 실험이 없습니다."
  exit 0
fi

echo "재실행: ${count}개 (mode=${MODE})"
echo ""

i=0
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  read -r dataset model seed_rest mode_rest <<< "$line"
  seed_str=${seed_rest%% *}
  seed=${seed_str#seed}
  # gpu0/gpu1: line에 auto|keep 포함. auto|keep: filename이 mode
  if [[ "$mode_rest" == "auto" ]] || [[ "$mode_rest" == "keep" ]]; then
    run_mode="$mode_rest"
  else
    run_mode="$MODE"
  fi
  ((i++))

  out_auto="results/pipeline_${dataset}_${model}_${NUM_TOPICS}_seed${seed}"
  out_keep="results/pipeline_${dataset}_${model}_${NUM_TOPICS}_keep_seed${seed}"

  if [[ "$run_mode" == "keep" ]]; then
    echo "[${i}/${count}] $dataset $model $seed_str [keep] (vanilla 재사용)"
    mkdir -p "$out_keep"
    cmd_schema="$PYTHON main.py schema --topic_words_file $out_auto/vanilla/top_words.txt --keep --out_dir $out_keep/schema_50"
    cmd_anchor="$PYTHON main.py anchor --schema_dir $out_keep/schema_50 --out_dir $out_keep/anchor --model $model --dataset $dataset --num_topics $NUM_TOPICS --seed $seed"
    (eval "$cmd_schema" && eval "$cmd_anchor") 2>&1 | tee "${LOG_DIR}/pipeline_${dataset}_${model}_keep_seed${seed}.log"
  else
    echo "[${i}/${count}] $dataset $model $seed_str [auto]"
    cmd="$PYTHON main.py pipeline --model $model --dataset $dataset --num_topics $NUM_TOPICS --seed $seed --out_dir $out_auto"
    eval "$cmd" 2>&1 | tee "${LOG_DIR}/pipeline_${dataset}_${model}_auto_seed${seed}.log"
  fi

  if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    echo "  [OK]"
    sed -i "/  - ${dataset} ${model} ${seed_str} /d" "$FAILED_FILE"
  else
    echo "  [FAIL]"
  fi
done <<< "$failed_lines"

echo ""
echo "재실행 완료"
