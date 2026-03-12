#!/bin/bash
#
# 실패한 파이프라인만 재실행
# failed_auto.txt 또는 failed_keep.txt에서 읽어서 해당 실험만 다시 돌림
#
# 사용법:
#   CUDA_VISIBLE_DEVICES=0 ./retry_failed.sh auto   # auto 실패분 재실행
#   CUDA_VISIBLE_DEVICES=1 ./retry_failed.sh keep  # keep 실패분 재실행
#

cd "$(dirname "$0")"
PYTHON="${CONDA_PREFIX}/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="/home/ys0660/anaconda3/envs/ys0660/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
[[ -x "$PYTHON" ]] || PYTHON="python"
MODE=${1:-auto}
LOG_DIR="results/experiment_logs"
FAILED_FILE="${LOG_DIR}/failed_${MODE}.txt"
NUM_TOPICS=50

if [[ ! -f "$FAILED_FILE" ]]; then
  echo "실패 목록 없음: $FAILED_FILE"
  exit 0
fi

# "#" 또는 빈 줄 제외, "  - dataset model seedN 시각" 형식 파싱
failed_lines=$(grep -E "^\s+-\s+" "$FAILED_FILE" | sed 's/^[[:space:]]*-[[:space:]]*//' | cut -d' ' -f1-3 | grep -v '^$')
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
  read -r dataset model seed_rest <<< "$line"
  seed_str=${seed_rest%% *}
  seed=${seed_str#seed}
  ((i++))

  if [[ "$MODE" == "keep" ]]; then
    out_dir="results/pipeline_${dataset}_${model}_${NUM_TOPICS}_keep_seed${seed}"
    cmd="$PYTHON main.py pipeline --model $model --dataset $dataset --num_topics $NUM_TOPICS --seed $seed --out_dir $out_dir --keep"
  else
    out_dir="results/pipeline_${dataset}_${model}_${NUM_TOPICS}_seed${seed}"
    cmd="$PYTHON main.py pipeline --model $model --dataset $dataset --num_topics $NUM_TOPICS --seed $seed --out_dir $out_dir"
  fi

  log_file="${LOG_DIR}/pipeline_${dataset}_${model}_${MODE}_seed${seed}.log"
  echo "[${i}/${count}] $dataset $model $seed_str"
  eval "$cmd" 2>&1 | tee "$log_file"
  if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    echo "  [OK]"
    # 성공 시 실패 목록에서 제거
    sed -i "/  - ${dataset} ${model} ${seed_str} /d" "$FAILED_FILE"
  else
    echo "  [FAIL]"
  fi
done <<< "$failed_lines"

echo ""
echo "재실행 완료"
