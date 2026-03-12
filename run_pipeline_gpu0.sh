#!/bin/bash
#
# SchemaTopic Pipeline - GPU 0 (데이터셋 2개: 20News, AGNews)
# vanilla 1회 공유 후 auto + keep 둘 다 실행
#
# 사용법:
#   CUDA_VISIBLE_DEVICES=0 ./run_pipeline_gpu0.sh
#   CUDA_VISIBLE_DEVICES=0 ./run_pipeline_gpu0.sh --dry-run
#

cd "$(dirname "$0")"
PYTHON="${CONDA_PREFIX}/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="/home/ys0660/anaconda3/envs/ys0660/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
[[ -x "$PYTHON" ]] || PYTHON="python"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

NUM_TOPICS=50
LOG_DIR="results/experiment_logs"
STATUS_FILE="${LOG_DIR}/run_status_gpu0.txt"
FAILED_FILE="${LOG_DIR}/failed_gpu0.txt"
mkdir -p "$LOG_DIR"

NTM_MODELS=(ecrtm etm nstm nvdm plda)
# GPU 0: 20News, AGNews
DATASETS=(20News AGNews)
SEEDS=(1 2 3 4 5)

total=$(( ${#NTM_MODELS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]} * 2 ))  # x2 for auto+keep
current=0
success_count=0
fail_count=0
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

write_status() {
  local mode=$1
  local ts=$2
  cat > "$STATUS_FILE" << EOF
=== run_pipeline_gpu0.sh ===
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
echo "SchemaTopic Pipeline [GPU0] 20News, AGNews"
echo "vanilla 1회 공유 -> auto + keep"
echo "총 ${total}개 실험 (auto+keep)"
echo "상태: $STATUS_FILE"
echo "=========================================="

write_status "시작" "-"

for model in "${NTM_MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      out_auto="results/pipeline_${dataset}_${model}_${NUM_TOPICS}_seed${seed}"
      out_keep="results/pipeline_${dataset}_${model}_${NUM_TOPICS}_keep_seed${seed}"

      # --- 1) Pipeline (auto): vanilla + schema + anchor (vanilla 여기서 1회만)
      current=$((current + 1))
      log_auto="${LOG_DIR}/pipeline_${dataset}_${model}_auto_seed${seed}.log"
      echo ""
      echo "[${current}/${total}] ${dataset} ${model} seed${seed} [auto]"
      cmd="$PYTHON main.py pipeline --model $model --dataset $dataset --num_topics $NUM_TOPICS --seed $seed --out_dir $out_auto"
      if $DRY_RUN; then
        echo "  $cmd"
      else
        eval "$cmd" 2>&1 | tee "$log_auto"
        exit_code=${PIPESTATUS[0]}
        if [[ "$exit_code" -eq 0 ]]; then
          ((success_count++))
          write_status "성공" "${dataset} ${model} seed${seed} auto"
        else
          ((fail_count++))
          echo "  - ${dataset} ${model} seed${seed} auto $(date '+%Y-%m-%d %H:%M:%S')" >> "$FAILED_FILE"
          write_status "실패" "${dataset} ${model} seed${seed} auto"
          continue
        fi
      fi

      # --- 2) Keep: schema + anchor (vanilla 재사용)
      current=$((current + 1))
      log_keep="${LOG_DIR}/pipeline_${dataset}_${model}_keep_seed${seed}.log"
      echo ""
      echo "[${current}/${total}] ${dataset} ${model} seed${seed} [keep] (vanilla 재사용)"
      mkdir -p "$out_keep"
      # schema (keep)
      cmd_schema="$PYTHON main.py schema --topic_words_file $out_auto/vanilla/top_words.txt --keep --out_dir $out_keep/schema_50"
      # anchor (keep)
      cmd_anchor="$PYTHON main.py anchor --schema_dir $out_keep/schema_50 --out_dir $out_keep/anchor --model $model --dataset $dataset --num_topics $NUM_TOPICS --seed $seed"
      if $DRY_RUN; then
        echo "  $cmd_schema"
        echo "  $cmd_anchor"
      else
        (eval "$cmd_schema" && eval "$cmd_anchor") 2>&1 | tee "$log_keep"
        exit_code=${PIPESTATUS[0]}
        if [[ "$exit_code" -eq 0 ]]; then
          ((success_count++))
          write_status "성공" "${dataset} ${model} seed${seed} keep"
        else
          ((fail_count++))
          echo "  - ${dataset} ${model} seed${seed} keep $(date '+%Y-%m-%d %H:%M:%S')" >> "$FAILED_FILE"
          write_status "실패" "${dataset} ${model} seed${seed} keep"
        fi
      fi
    done
  done
done

write_status "종료" "-"
echo ""
echo "=========================================="
echo "Pipeline [GPU0] 완료"
echo "성공: ${success_count} | 실패: ${fail_count}"
echo "실패 목록: $FAILED_FILE"
echo "=========================================="
