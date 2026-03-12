#!/bin/bash
#
# SchemaTopic Pipeline - LLM 자동 (auto) 모드
# K 모드: auto (--keep 없음, LLM이 토픽 수 자동 조정)
#
# NTM 5종 x 데이터 4종 x 시드 5 = 100 실험
#
# 사용법:
#   CUDA_VISIBLE_DEVICES=0 ./run_pipeline_auto.sh           # GPU 0에서 실행
#   CUDA_VISIBLE_DEVICES=0 ./run_pipeline_auto.sh --dry-run # 명령만 출력
#

cd "$(dirname "$0")"
# conda env python 사용 (활성화 시 CONDA_PREFIX, 아니면 ys0660 경로)
PYTHON="${CONDA_PREFIX}/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="/home/ys0660/anaconda3/envs/ys0660/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
[[ -x "$PYTHON" ]] || PYTHON="python"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

NUM_TOPICS=50
LOG_DIR="results/experiment_logs"
STATUS_FILE="${LOG_DIR}/run_status_auto.txt"
FAILED_FILE="${LOG_DIR}/failed_auto.txt"
mkdir -p "$LOG_DIR"

NTM_MODELS=(ecrtm etm nstm nvdm plda)
DATASETS=(20News AGNews DBpedia R8)
SEEDS=(1 2 3 4 5)

total=$(( ${#NTM_MODELS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]} ))
current=0
success_count=0
fail_count=0
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

write_status() {
  local mode=$1
  local ts=$2
  cat > "$STATUS_FILE" << EOF
=== run_pipeline_auto.sh ===
시작: ${START_TIME}
진행: ${current}/${total}
성공: ${success_count}
실패: ${fail_count}
마지막: ${mode} (${ts})
EOF
  if [[ -f "$FAILED_FILE" ]] && [[ -s "$FAILED_FILE" ]]; then
    echo "" >> "$STATUS_FILE"
    echo "[실패 목록]" >> "$STATUS_FILE"
    cat "$FAILED_FILE" >> "$STATUS_FILE"
  fi
}

# 실패 목록 초기화 (새 실행 시)
echo "# 실패한 실험 (dataset model seed 시각)" > "$FAILED_FILE"

echo "=========================================="
echo "SchemaTopic Pipeline [auto] 실험 시작"
echo "총 ${total}개 실험"
echo "상태: $STATUS_FILE"
echo "실패기록: $FAILED_FILE"
echo "=========================================="

write_status "시작" "-"

for model in "${NTM_MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      current=$((current + 1))

      out_dir="results/pipeline_${dataset}_${model}_${NUM_TOPICS}_seed${seed}"
      log_file="${LOG_DIR}/pipeline_${dataset}_${model}_auto_seed${seed}.log"

      echo ""
      echo "[${current}/${total}] dataset=${dataset} model=${model} k_mode=auto seed=${seed}"
      echo "  out_dir=${out_dir}"
      echo "  log=${log_file}"

      cmd="$PYTHON main.py pipeline --model $model --dataset $dataset --num_topics $NUM_TOPICS --seed $seed --out_dir $out_dir"

      if $DRY_RUN; then
        echo "  $cmd"
      else
        eval "$cmd" 2>&1 | tee "$log_file"
        exit_code=${PIPESTATUS[0]}
        if [[ "$exit_code" -eq 0 ]]; then
          echo "  [OK] 완료"
          ((success_count++))
          write_status "성공" "${dataset} ${model} seed${seed}"
        else
          echo "  [FAIL] 실패 (exit code: $exit_code)"
          ((fail_count++))
          echo "  - ${dataset} ${model} seed${seed} $(date '+%Y-%m-%d %H:%M:%S')" >> "$FAILED_FILE"
          write_status "실패" "${dataset} ${model} seed${seed}"
        fi
      fi
    done
  done
done

write_status "종료" "-"
echo ""
echo "=========================================="
echo "Pipeline [auto] 완료 (${total}개)"
echo "성공: ${success_count} | 실패: ${fail_count}"
echo "실패 목록: $FAILED_FILE"
echo "=========================================="
