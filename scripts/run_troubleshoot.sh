#!/bin/bash
#
# 트러블슈팅용 5개 실행 스크립트
#
# 목적: 버그 수정 후, 서로 다른 실패 경향성을 가진 조합에서 개선 여부를 검증
#
# 5개 조합:
#   1. NSTM + 20News seed1  [최악] generic verb 토픽, diversity 0.28
#   2. ECRTM + 20News seed1 [topic collapse] mattingly 10+회 반복
#   3. ECRTM + R8 seed1     [topic collapse] pay payout, forbes lucky 반복
#   4. ETM + 20News seed1   [generic] people make like know, purity 0.40
#   5. NSTM + DBpedia seed1 [양호] coherence 0.63, purity 0.86 - 잘 되는 케이스
#
# 사용법:
#   ./scripts/run_troubleshoot.sh <run_name>   # CUDA 미지정 시 GPU 0,1 자동 병렬
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_troubleshoot.sh myfix1  # GPU 0만 사용
#   ./scripts/run_troubleshoot.sh myfix1 --dry-run
#
# 출력: troubleshooting/<run_name>/
#   - {dataset}_{model}_50_seed{N}/  (5개 폴더)
#     - vanilla/ (results/pipeline_*_seed/vanilla에서 복사, 공유)
#     - auto/ (schema + anchor, pipeline auto 모드)
#     - keep/ (schema + anchor, pipeline keep 모드)
#   - analysis.csv (auto, keep 메트릭 모두 포함)
#

cd "$(dirname "$0")/.."

PYTHON="${CONDA_PREFIX}/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="/home/ys0660/anaconda3/envs/ys0660/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
[[ -x "$PYTHON" ]] || PYTHON="python"

# 인자 파싱: run_name [--dry-run] [--run-indices 0,1,2]
RUN_NAME=""
DRY_RUN=false
RUN_INDICES=""
args=("$@")
for i in "${!args[@]}"; do
  a="${args[$i]}"
  if [[ "$a" == "--dry-run" ]]; then
    DRY_RUN=true
  elif [[ "$a" == "--run-indices" ]] && [[ $((i+1)) -lt ${#args[@]} ]]; then
    RUN_INDICES="${args[$((i+1))]}"
  elif [[ -z "$RUN_NAME" ]] && [[ "$a" != --* ]]; then
    RUN_NAME="$a"
  fi
done

if [[ -z "$RUN_NAME" ]]; then
  echo "Usage: ./scripts/run_troubleshoot.sh <run_name> [--dry-run]"
  echo "  run_name: troubleshooting/<run_name>/ 아래에 결과 저장 (예: myfix1, fix_collapse)"
  exit 1
fi

# CUDA 미지정 시 GPU 0,1 병렬 실행 (--run-indices 없을 때만)
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ -z "$RUN_INDICES" ]] && ! $DRY_RUN; then
  SELF="$(realpath "$0")"
  echo "CUDA 미지정 → GPU 0, 1 병렬 실행"
  echo "  GPU 0: runs 1,2,3 (20News nstm, 20News ecrtm, R8 ecrtm)"
  echo "  GPU 1: runs 4,5 (20News etm, DBpedia nstm)"
  echo ""
  CUDA_VISIBLE_DEVICES=0 "$SELF" "$RUN_NAME" --run-indices "0,1,2" &
  pid0=$!
  CUDA_VISIBLE_DEVICES=1 "$SELF" "$RUN_NAME" --run-indices "3,4" &
  pid1=$!
  wait $pid0
  ret0=$?
  wait $pid1
  ret1=$?
  if [[ $ret0 -ne 0 ]] || [[ $ret1 -ne 0 ]]; then
    echo "일부 프로세스 실패 (gpu0=$ret0, gpu1=$ret1)"
    exit 1
  fi
  # analysis.csv 생성 (하위 프로세스에서 success>0이면 이미 생성됐을 수 있으나, 한쪽만 성공한 경우 대비)
  OUT_ROOT="troubleshooting/${RUN_NAME}"
  if [[ -d "$OUT_ROOT" ]]; then
    echo ""
    echo "analysis.csv 생성..."
    "$PYTHON" - "$OUT_ROOT" << 'PYEOF'
import json, sys
from pathlib import Path
OUT_ROOT = Path(sys.argv[1])
RUNS = [
    ("20News", "nstm", 1),
    ("20News", "ecrtm", 1),
    ("R8", "ecrtm", 1),
    ("20News", "etm", 1),
    ("DBpedia", "nstm", 1),
]
METRIC_KEYS = ["topic_coherence_cv", "topic_diversity", "purity", "nmi", "PN"]
def load_metrics(path):
    if not path.exists(): return {}
    try:
        with open(path) as f: return json.load(f)
    except Exception: return {}
rows = []
for dataset, model, seed in RUNS:
    folder = f"{dataset}_{model}_50_seed{seed}"
    base = OUT_ROOT / folder
    m_auto = load_metrics(base / "auto" / "anchor" / "metrics.json") or load_metrics(base / "anchor" / "metrics.json")
    m_keep = load_metrics(base / "keep" / "anchor" / "metrics.json")
    row = {"dataset": dataset, "model": model, "seed": seed}
    for k in METRIC_KEYS:
        row[f"{k}_auto"] = m_auto.get(k, "")
        row[f"{k}_keep"] = m_keep.get(k, "")
    rows.append(row)
cols = ["dataset", "model", "seed"] + [f"{k}_auto" for k in METRIC_KEYS] + [f"{k}_keep" for k in METRIC_KEYS]
with open(OUT_ROOT / "analysis.csv", "w") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
print("  Wrote", OUT_ROOT / "analysis.csv")
PYEOF
  fi
  echo ""
  echo "=========================================="
  echo "트러블슈팅 5개 실행 완료 (GPU 0+1 병렬)"
  echo "출력: ${OUT_ROOT}/"
  echo "=========================================="
  exit 0
fi

NUM_TOPICS=50
TROUBLESHOOT_ROOT="troubleshooting"
RESULTS_ROOT="results"
OUT_ROOT="${TROUBLESHOOT_ROOT}/${RUN_NAME}"
LOG_DIR="results/experiment_logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

# (dataset, model, seed, 설명)
RUNS=(
  "20News:nstm:1:최악_generic_verb_diversity_0.28"
  "20News:ecrtm:1:topic_collapse_mattingly_10회"
  "R8:ecrtm:1:topic_collapse_pay_forbes"
  "20News:etm:1:generic_verb_purity_0.40"
  "DBpedia:nstm:1:양호_coherence_0.63_purity_0.86"
)

total=${#RUNS[@]}
success=0
fail=0
FAILED_LIST=()

# --run-indices가 있으면 해당 인덱스만 실행
if [[ -n "$RUN_INDICES" ]]; then
  # "0,1,2" -> 공백으로 분리
  INDICES_ARR=()
  IFS=',' read -ra INDICES_ARR <<< "$RUN_INDICES"
  echo "=========================================="
  echo "SchemaTopic 트러블슈팅 (인덱스: ${RUN_INDICES})"
  echo "run_name: ${RUN_NAME} | GPU: ${CUDA_VISIBLE_DEVICES:-all}"
  echo "출력: ${OUT_ROOT}/"
  echo "=========================================="
else
  echo "=========================================="
  echo "SchemaTopic 트러블슈팅 5개 실행"
  echo "run_name: ${RUN_NAME}"
  echo "출력: ${OUT_ROOT}/"
  echo "=========================================="
fi
echo ""

for i in "${!RUNS[@]}"; do
  # --run-indices 지정 시 해당 인덱스만
  if [[ -n "$RUN_INDICES" ]]; then
    found=false
    for idx in "${INDICES_ARR[@]}"; do
      [[ "$i" == "$idx" ]] && { found=true; break; }
    done
    $found || continue
  fi
  IFS=':' read -r dataset model seed desc <<< "${RUNS[$i]}"
  idx=$((i + 1))
  out_dir="${OUT_ROOT}/${dataset}_${model}_${NUM_TOPICS}_seed${seed}"
  log_file="${LOG_DIR}/troubleshoot_${RUN_NAME}_${dataset}_${model}_seed${seed}.log"
  mkdir -p "$out_dir"

  pipeline_auto="${RESULTS_ROOT}/pipeline_${dataset}_${model}_${NUM_TOPICS}_seed${seed}"
  pipeline_keep="${RESULTS_ROOT}/pipeline_${dataset}_${model}_${NUM_TOPICS}_keep_seed${seed}"
  top_words="${pipeline_auto}/vanilla/top_words.txt"

  echo "[${idx}/${total}] ${dataset} + ${model} seed${seed}"
  echo "  → ${desc}"
  echo "  → out: ${out_dir}"
  echo "  → top_words: ${top_words}"

  if [[ ! -f "$top_words" ]]; then
    echo "  [SKIP] top_words not found. Run pipeline first: ${pipeline_auto}/"
    ((fail++))
    FAILED_LIST+=("${dataset} ${model} seed${seed} (no top_words)")
    echo ""
    continue
  fi

  # vanilla: pipeline_auto/vanilla에서 복사 (공유)
  vanilla_src="${pipeline_auto}/vanilla"
  vanilla_dst="${out_dir}/vanilla"
  if [[ -d "$vanilla_src" ]]; then
    if $DRY_RUN; then
      echo "  [dry-run] cp -r $vanilla_src $vanilla_dst"
    else
      cp -r "$vanilla_src" "$vanilla_dst"
      echo "  [vanilla] copied from ${vanilla_src}"
    fi
  fi

  run_ok=true

  # auto: schema + anchor (pipeline without --keep)
  echo "  [auto] schema + anchor..."
  cmd_auto="$PYTHON main.py pipeline --model $model --dataset $dataset --num_topics $NUM_TOPICS --seed $seed --out_dir ${out_dir}/auto --topic_words_file $top_words"
  if $DRY_RUN; then
    echo "  $cmd_auto"
  else
    eval "$cmd_auto" 2>&1 | tee -a "$log_file"
    ret_auto=${PIPESTATUS[0]}
    if [[ $ret_auto -ne 0 ]]; then
      run_ok=false
      echo "  [auto] FAIL"
    else
      echo "  [auto] OK"
    fi
  fi

  # keep: schema + anchor (pipeline with --keep)
  echo "  [keep] schema + anchor..."
  cmd_keep="$PYTHON main.py pipeline --model $model --dataset $dataset --num_topics $NUM_TOPICS --seed $seed --out_dir ${out_dir}/keep --topic_words_file $top_words --keep"
  if $DRY_RUN; then
    echo "  $cmd_keep"
  else
    eval "$cmd_keep" 2>&1 | tee -a "$log_file"
    ret_keep=${PIPESTATUS[0]}
    if [[ $ret_keep -ne 0 ]]; then
      run_ok=false
      echo "  [keep] FAIL"
    else
      echo "  [keep] OK"
    fi
  fi

  if ! $DRY_RUN; then
    if $run_ok; then
      ((success++))
      echo "  [OK] both auto and keep"
    else
      ((fail++))
      FAILED_LIST+=("${dataset} ${model} seed${seed}")
      echo "  [FAIL]"
    fi
  fi
  echo ""
done

# analysis.csv 생성 (auto, keep 메트릭 모두 포함)
if ! $DRY_RUN && [[ $success -gt 0 ]]; then
  echo "analysis.csv 생성..."
  "$PYTHON" - << PYEOF
import json
from pathlib import Path

OUT_ROOT = Path("$OUT_ROOT")
RUNS = [
    ("20News", "nstm", 1),
    ("20News", "ecrtm", 1),
    ("R8", "ecrtm", 1),
    ("20News", "etm", 1),
    ("DBpedia", "nstm", 1),
]

METRIC_KEYS = ["topic_coherence_cv", "topic_diversity", "purity", "nmi", "PN"]

def load_metrics(path):
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

rows = []
for dataset, model, seed in RUNS:
    folder = f"{dataset}_{model}_50_seed{seed}"
    base = OUT_ROOT / folder
    m_auto = load_metrics(base / "auto" / "anchor" / "metrics.json")
    m_keep = load_metrics(base / "keep" / "anchor" / "metrics.json")
    # 이전 구조 호환: auto/keep 없으면 anchor/ 사용
    if not m_auto:
        m_auto = load_metrics(base / "anchor" / "metrics.json")
    row = {"dataset": dataset, "model": model, "seed": seed}
    for k in METRIC_KEYS:
        row[f"{k}_auto"] = m_auto.get(k, "")
        row[f"{k}_keep"] = m_keep.get(k, "")
    rows.append(row)

cols = ["dataset", "model", "seed"]
for k in METRIC_KEYS:
    cols.append(f"{k}_auto")
    cols.append(f"{k}_keep")
out = OUT_ROOT / "analysis.csv"
with open(out, "w") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        vals = [str(r.get(c, "")) for c in cols]
        f.write(",".join(vals) + "\n")

print(f"  Wrote {out}")
PYEOF
fi

echo "=========================================="
echo "트러블슈팅 5개 실행 완료"
echo "성공: ${success} | 실패: ${fail}"
echo "출력: ${OUT_ROOT}/"
if [[ ${#FAILED_LIST[@]} -gt 0 ]]; then
  echo ""
  echo "실패 목록:"
  for f in "${FAILED_LIST[@]}"; do
    echo "  - $f"
  done
fi
echo "=========================================="
