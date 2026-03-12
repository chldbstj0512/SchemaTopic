#!/usr/bin/env bash

set -euo pipefail

# Run after activating the project environment, e.g.:
#   conda activate llm_itl

DATA_DIR="${DATA_DIR:-datasets/20News}"
NUM_TOPICS="${NUM_TOPICS:-50}"
EPOCHS="${EPOCHS:-250}"
LR="${LR:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-500}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
PATIENCE="${PATIENCE:-30}"
OUT_ROOT="${OUT_ROOT:-results/run_all_vanilla_k${NUM_TOPICS}}"

MODELS=(
  "etm"
  "ecrtm"
  "nvdm"
  "plda"
  "nstm"
  "scholar"
  "wete"
)

mkdir -p "${OUT_ROOT}"

echo "Running vanilla NTM sweep"
echo "  data_dir=${DATA_DIR}"
echo "  num_topics=${NUM_TOPICS}"
echo "  epochs=${EPOCHS}"
echo "  lr=${LR}"
echo "  batch_size=${BATCH_SIZE}"
echo "  eval_batch_size=${EVAL_BATCH_SIZE}"
echo "  patience=${PATIENCE}"
echo "  out_root=${OUT_ROOT}"
echo

for model in "${MODELS[@]}"; do
  out_dir="${OUT_ROOT}/${model}"
  echo "============================================================"
  echo "Model: ${model}"
  echo "Output: ${out_dir}"
  echo "============================================================"

  python main.py vanilla \
    --model "${model}" \
    --data_dir "${DATA_DIR}" \
    --num_topics "${NUM_TOPICS}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --patience "${PATIENCE}" \
    --out_dir "${out_dir}"

  echo
done

echo "All vanilla runs completed."
