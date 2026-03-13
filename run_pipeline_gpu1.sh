#!/bin/bash
#
# SchemaTopic Pipeline - GPU 1 (데이터셋 2개: DBpedia, R8)
# vanilla 1회 공유 후 auto + keep 둘 다 실행
#
# 사용법:
#   CUDA_VISIBLE_DEVICES=1 ./run_pipeline_gpu1.sh
#   CUDA_VISIBLE_DEVICES=1 ./run_pipeline_gpu1.sh --dry-run
#

cd "$(dirname "$0")"
PYTHON="/home/hjhj97/miniconda3/envs/llm_itl/bin/python -u"
[[ -x "${PYTHON%% *}" ]] || PYTHON="${CONDA_PREFIX}/bin/python -u"
[[ -x "${PYTHON%% *}" ]] || PYTHON="$(command -v python3) -u"
[[ -x "${PYTHON%% *}" ]] || PYTHON="python -u"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

NUM_TOPICS=50
LOG_DIR="results/experiment_logs"
STATUS_FILE="${LOG_DIR}/run_status_gpu1.txt"
FAILED_FILE="${LOG_DIR}/failed_gpu1.txt"
mkdir -p "$LOG_DIR"

NTM_MODELS=(ecrtm etm nstm nvdm plda)
# GPU 1: DBpedia, R8
DATASETS=(DBpedia R8)
SEEDS=(1)

MODES=(auto keep)
total=$(( ${#NTM_MODELS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]} * ${#MODES[@]} ))
current=0
success_count=0
fail_count=0
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

write_status() {
  local mode=$1
  local ts=$2
  cat > "$STATUS_FILE" << EOF
=== run_pipeline_gpu1.sh ===
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

echo "# 실패한 실험 (dataset model seed mode 시각)" > "$FAILED_FILE"

echo "=========================================="
echo "SchemaTopic Pipeline [GPU1] DBpedia, R8"
echo "vanilla -> schema -> anchor (auto + keep)"
echo "총 ${total}개 실험"
echo "상태: $STATUS_FILE"
echo "=========================================="

write_status "시작" "-"

for model in "${NTM_MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      vanilla_out="results/pipeline_${dataset}_${model}_${NUM_TOPICS}_seed${seed}/vanilla"
      for mode in "${MODES[@]}"; do
        if [[ "$mode" == "keep" ]]; then
          out_dir="results/pipeline_${dataset}_${model}_${NUM_TOPICS}_keep_seed${seed}"
          extra_flag="--keep"
          if [[ -f "${vanilla_out}/top_words.txt" ]]; then
            extra_flag="--keep --vanilla_dir ${vanilla_out}"
          fi
        else
          out_dir="results/pipeline_${dataset}_${model}_${NUM_TOPICS}_seed${seed}"
          extra_flag=""
        fi

        current=$((current + 1))
        log="${LOG_DIR}/pipeline_${dataset}_${model}_${mode}_seed${seed}.log"
        echo ""
        echo "[${current}/${total}] ${dataset} ${model} seed${seed} [${mode}]"
        cmd="$PYTHON main.py pipeline --model $model --dataset $dataset --num_topics $NUM_TOPICS --seed $seed --out_dir $out_dir --batch_size 8000 $extra_flag"
        if $DRY_RUN; then
          echo "  $cmd"
        else
          eval "$cmd" 2>&1 | tee "$log"
          exit_code=${PIPESTATUS[0]}
          if [[ "$exit_code" -eq 0 ]]; then
            ((success_count++))
            write_status "성공" "${dataset} ${model} seed${seed} ${mode}"
          else
            ((fail_count++))
            echo "  - ${dataset} ${model} seed${seed} ${mode} $(date '+%Y-%m-%d %H:%M:%S')" >> "$FAILED_FILE"
            write_status "실패" "${dataset} ${model} seed${seed} ${mode}"
          fi
        fi

      done
    done
  done
done

write_status "종료" "-"
echo ""
echo "=========================================="
echo "Pipeline [GPU1] 완료"
echo "성공: ${success_count} | 실패: ${fail_count}"
echo "실패 목록: $FAILED_FILE"
echo "=========================================="
