#!/usr/bin/env bash
set -u

# =========================
# Config
# =========================
DATASET="20News"
NUM_TOPICS=50
EPOCHS=250
DEVICE="cuda"
LLM_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
JSON_RETRY_ATTEMPTS=2

# 실행할 모델들
MODELS=("nvdm" "plda" "nstm" "ecrtm" "etm")

# ablation case 이름과 skip 옵션
CASES=("full" "no_step1" "no_step2" "no_step3")

# 결과 루트
RESULTS_DIR="results"

# 로그 저장
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"

run_cmd() {
  local log_file="$1"
  shift
  echo
  echo ">>> Running: $*"
  echo ">>> Log: ${log_file}"
  "$@" 2>&1 | tee "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  if [ "${exit_code}" -ne 0 ]; then
    echo "!!! FAILED (exit code=${exit_code}): $*"
    return "${exit_code}"
  fi
  return 0
}

get_skip_flag() {
  local case_name="$1"
  case "${case_name}" in
    full) echo "" ;;
    no_step1) echo "--skip_step1" ;;
    no_step2) echo "--skip_step2" ;;
    no_step3) echo "--skip_step3" ;;
    *)
      echo "Unknown case: ${case_name}" >&2
      return 1
      ;;
  esac
}

echo "=============================="
echo " Schema Ablation Batch Runner "
echo "=============================="
echo "Dataset      : ${DATASET}"
echo "Num topics   : ${NUM_TOPICS}"
echo "Epochs       : ${EPOCHS}"
echo "Device       : ${DEVICE}"
echo "LLM model    : ${LLM_MODEL}"
echo "Models       : ${MODELS[*]}"
echo "Cases        : ${CASES[*]}"
echo

FAILED_RUNS=()

for MODEL in "${MODELS[@]}"; do
  echo
  echo "############################################################"
  echo "### MODEL: ${MODEL}"
  echo "############################################################"

  VANILLA_OUT="${RESULTS_DIR}/vanilla_${DATASET}_${MODEL}_${NUM_TOPICS}"
  TOP_WORDS_FILE="${VANILLA_OUT}/top_words.txt"

  # 1) Vanilla training
  if [ -f "${TOP_WORDS_FILE}" ]; then
    echo "[SKIP] Vanilla already exists: ${TOP_WORDS_FILE}"
  else
    VANILLA_LOG="${LOG_DIR}/${MODEL}_vanilla.log"
    run_cmd "${VANILLA_LOG}" \
      python main.py vanilla \
      --dataset "${DATASET}" \
      --model "${MODEL}" \
      --num_topics "${NUM_TOPICS}" \
      --epochs "${EPOCHS}" \
      --out_dir "${VANILLA_OUT}" \
    || FAILED_RUNS+=("${MODEL}:vanilla")
  fi

  if [ ! -f "${TOP_WORDS_FILE}" ]; then
    echo "!!! Missing top_words.txt after vanilla for model=${MODEL}. Skipping this model."
    FAILED_RUNS+=("${MODEL}:missing_top_words")
    continue
  fi

  # 2) Schema ablation
  for CASE_NAME in "${CASES[@]}"; do
    SKIP_FLAG="$(get_skip_flag "${CASE_NAME}")"

    SCHEMA_OUT="${RESULTS_DIR}/${MODEL}_ablation_${CASE_NAME}"
    SCHEMA_LOG="${LOG_DIR}/${MODEL}_schema_${CASE_NAME}.log"

    if [ -f "${SCHEMA_OUT}/schema_topics.json" ] && [ -f "${SCHEMA_OUT}/topic_words.txt" ]; then
      echo "[SKIP] Schema result already exists: ${SCHEMA_OUT}"
    else
      if [ -n "${SKIP_FLAG}" ]; then
        run_cmd "${SCHEMA_LOG}" \
          python main.py schema \
          --topic_words_file "${TOP_WORDS_FILE}" \
          --out_dir "${SCHEMA_OUT}" \
          --model_name "${LLM_MODEL}" \
          --max_new_tokens_step1 4096 \
          --max_new_tokens_step2 4096 \
          --max_new_tokens_step3 4096 \
          --json_retry_attempts "${JSON_RETRY_ATTEMPTS}" \
          --device "${DEVICE}" \
          "${SKIP_FLAG}" \
        || FAILED_RUNS+=("${MODEL}:schema:${CASE_NAME}")
      else
        run_cmd "${SCHEMA_LOG}" \
          python main.py schema \
          --topic_words_file "${TOP_WORDS_FILE}" \
          --out_dir "${SCHEMA_OUT}" \
          --model_name "${LLM_MODEL}" \
          --max_new_tokens_step1 4096 \
          --max_new_tokens_step2 4096 \
          --max_new_tokens_step3 4096 \
          --json_retry_attempts "${JSON_RETRY_ATTEMPTS}" \
          --device "${DEVICE}" \
        || FAILED_RUNS+=("${MODEL}:schema:${CASE_NAME}")
      fi
    fi

    # 3) Anchor training
    if [ -f "${SCHEMA_OUT}/schema_topics.json" ] && [ -f "${SCHEMA_OUT}/topic_words.txt" ]; then
      TRAIN_OUT="${RESULTS_DIR}/train_${MODEL}_${CASE_NAME}"
      TRAIN_LOG="${LOG_DIR}/${MODEL}_anchor_${CASE_NAME}.log"

      if [ -f "${TRAIN_OUT}/metrics.json" ]; then
        echo "[SKIP] Anchor training already exists: ${TRAIN_OUT}/metrics.json"
      else
        run_cmd "${TRAIN_LOG}" \
          python main.py anchor \
          --dataset "${DATASET}" \
          --model "${MODEL}" \
          --num_topics "${NUM_TOPICS}" \
          --schema_dir "${SCHEMA_OUT}" \
          --out_dir "${TRAIN_OUT}" \
        || FAILED_RUNS+=("${MODEL}:anchor:${CASE_NAME}")
      fi
    else
      echo "!!! Missing schema outputs for ${MODEL} / ${CASE_NAME}, skipping anchor."
      FAILED_RUNS+=("${MODEL}:anchor_skip_missing_schema:${CASE_NAME}")
    fi
  done
done

echo
echo "=============================="
echo " Batch run finished"
echo "=============================="

if [ "${#FAILED_RUNS[@]}" -eq 0 ]; then
  echo "All runs completed successfully."
else
  echo "Some runs failed:"
  for item in "${FAILED_RUNS[@]}"; do
    echo "  - ${item}"
  done
fi

echo
echo "Metrics files:"
find "${RESULTS_DIR}" -type f -path "*/metrics.json" | sort
