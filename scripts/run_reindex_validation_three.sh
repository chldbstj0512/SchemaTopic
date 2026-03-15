#!/usr/bin/env bash
# 1번 트러블슈팅(reindex) 검증: 문제됐던 3개 실험을 현재 코드(reindex 반영)로 재실행.
# Palmetto C_V + TD만 계산 (이전 metric과 비교용). 결과는 results/reindex_validation/ 아래에 저장.
#
# 사용: LLM 사용 가능한 환경에서 실행.
#   GPU 0에서 실험1, GPU 1에서 실험2 병렬 후, 실험3 실행 예:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_reindex_validation_three.sh dbpedia &
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_reindex_validation_three.sh 20ng_nvdm &
#   wait
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_reindex_validation_three.sh 20ng_ecrtm_keep
#
# 또는 한 실험만: bash scripts/run_reindex_validation_three.sh dbpedia

set -e
cd "$(dirname "$0")/.."

export SCHEMATOPIC_CV_TD_ONLY=1
RESULTS_ROOT="${RESULTS_ROOT:-results}"
BASE="${RESULTS_ROOT}/reindex_validation"
mkdir -p "$BASE"

# 이전(수정 전) 메트릭 경로 (비교용)
BEFORE_DBPEDIA="troubleshooting/1-llama-v2/dbpedia_CV_PLDA_auto/anchor/metrics.json"
BEFORE_20NG_NVDM="troubleshooting/1-llama-v2/20NG_nvdm_TD_auto/anchor/metrics.json"
BEFORE_20NG_ECRTM_KEEP="troubleshooting/1-llama-v2/20NG_ECRTM_TC_keep/anchor/metrics.json"

run_dbpedia() {
  local OUT_DIR="$BASE/dbpedia_plda_auto"
  local VANILLA="results/1_llama_origin_seed5/pipeline_DBpedia_plda_50_seed1/vanilla/top_words.txt"
  if [ ! -f "$VANILLA" ]; then
    echo "[dbpedia] Vanilla not found, running full pipeline."
    python3 main.py pipeline --dataset DBpedia --model plda --num_topics 50 \
      --out_dir "$OUT_DIR" --epochs 250 --seed 1
  else
    echo "[dbpedia] Schema + anchor only (reindex fix; same vanilla)."
    python3 main.py pipeline --dataset DBpedia --model plda --num_topics 50 \
      --out_dir "$OUT_DIR" --topic_words_file "$VANILLA" --epochs 250 --seed 1
  fi
  echo "[dbpedia] Done. Compare: before CV≈0.440 TD≈0.902 -> $OUT_DIR/anchor/metrics.json"
}

run_20ng_nvdm() {
  local OUT_DIR="$BASE/20ng_nvdm_auto"
  local VANILLA="troubleshooting/1-llama-v2/20NG_nvdm_TD_auto/vanilla/top_words.txt"
  if [ ! -f "$VANILLA" ]; then
    echo "[20ng_nvdm] Vanilla not found, running full pipeline."
    python3 main.py pipeline --dataset 20News --model nvdm --num_topics 50 \
      --out_dir "$OUT_DIR" --epochs 250 --seed 1
  else
    echo "[20ng_nvdm] Schema + anchor only (reindex fix; same vanilla)."
    python3 main.py pipeline --dataset 20News --model nvdm --num_topics 50 \
      --out_dir "$OUT_DIR" --topic_words_file "$VANILLA" --epochs 250 --seed 1
  fi
  echo "[20ng_nvdm] Done. Compare: before CV≈0.356 TD≈0.852 -> $OUT_DIR/anchor/metrics.json"
}

run_20ng_ecrtm_keep() {
  local OUT_DIR="$BASE/20ng_ecrtm_keep"
  local VANILLA="results/1_llama_origin_seed5/pipeline_20News_ecrtm_50_seed1/vanilla/top_words.txt"
  if [ ! -f "$VANILLA" ]; then
    echo "[20ng_ecrtm_keep] Vanilla not found, running full pipeline (keep mode)."
    python3 main.py pipeline --dataset 20News --model ecrtm --num_topics 50 \
      --out_dir "$OUT_DIR" --epochs 250 --seed 1 --keep
  else
    echo "[20ng_ecrtm_keep] Schema + anchor only (keep mode; same vanilla)."
    python3 main.py pipeline --dataset 20News --model ecrtm --num_topics 50 \
      --out_dir "$OUT_DIR" --topic_words_file "$VANILLA" --epochs 250 --seed 1 --keep
  fi
  echo "[20ng_ecrtm_keep] Done. Compare: before CV≈0.421 TD≈0.710 -> $OUT_DIR/anchor/metrics.json"
}

# 인자로 실험 선택: dbpedia | 20ng_nvdm | 20ng_ecrtm_keep
case "${1:-}" in
  dbpedia)           run_dbpedia ;;
  20ng_nvdm)         run_20ng_nvdm ;;
  20ng_ecrtm_keep)   run_20ng_ecrtm_keep ;;
  *)
    echo "Usage: $0 dbpedia | 20ng_nvdm | 20ng_ecrtm_keep"
    echo "  Or run in parallel:"
    echo "    CUDA_VISIBLE_DEVICES=0 $0 dbpedia &"
    echo "    CUDA_VISIBLE_DEVICES=1 $0 20ng_nvdm &"
    echo "    wait && CUDA_VISIBLE_DEVICES=0 $0 20ng_ecrtm_keep"
    exit 1
    ;;
esac
