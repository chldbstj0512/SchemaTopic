#!/bin/bash
#
# 현재 시점의 5가지 실행 결과를 troubleshooting/original/ 에 저장
# (기존 pipeline 결과에서 복사)
#
# 사용법: ./scripts/setup_troubleshoot_original.sh
#

set -e
cd "$(dirname "$0")/.."

RESULTS="results"
TROUBLESHOOT_ROOT="troubleshooting"
ORIGINAL="${TROUBLESHOOT_ROOT}/original"
mkdir -p "$ORIGINAL"

# (dataset, model, seed) -> pipeline source
RUNS=(
  "20News:nstm:1"
  "20News:ecrtm:1"
  "R8:ecrtm:1"
  "20News:etm:1"
  "DBpedia:nstm:1"
)

echo "=========================================="
echo "troubleshooting/original/ 설정"
echo "기존 pipeline 결과 복사"
echo "=========================================="

for run in "${RUNS[@]}"; do
  IFS=':' read -r dataset model seed <<< "$run"
  src="${RESULTS}/pipeline_${dataset}_${model}_50_seed${seed}/vanilla"
  dst="${ORIGINAL}/${dataset}_${model}_50_seed${seed}"
  mkdir -p "$dst"

  if [[ -d "$src" ]]; then
    cp -r "$src"/* "$dst/"
    echo "[OK] ${dataset} ${model} seed${seed}"
  else
    echo "[SKIP] ${dataset} ${model} seed${seed} (source not found: $src)"
  fi
done

# analysis.csv 생성
echo ""
echo "analysis.csv 생성..."
python3 - << 'PYEOF'
import json
from pathlib import Path

ORIGINAL = Path("troubleshooting/original")
RUNS = [
    ("20News", "nstm", 1),
    ("20News", "ecrtm", 1),
    ("R8", "ecrtm", 1),
    ("20News", "etm", 1),
    ("DBpedia", "nstm", 1),
]

METRIC_KEYS = ["topic_coherence_cv", "topic_diversity", "purity", "nmi", "PN"]
rows = []

for dataset, model, seed in RUNS:
    folder = f"{dataset}_{model}_50_seed{seed}"
    metrics_path = ORIGINAL / folder / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        row = {"dataset": dataset, "model": model, "seed": seed}
        for k in METRIC_KEYS:
            row[k] = m.get(k, "")
        rows.append(row)
    else:
        rows.append({"dataset": dataset, "model": model, "seed": seed, **{k: "" for k in METRIC_KEYS}})

out = ORIGINAL / "analysis.csv"
with open(out, "w") as f:
    cols = ["dataset", "model", "seed"] + METRIC_KEYS
    f.write(",".join(cols) + "\n")
    for r in rows:
        vals = [str(r.get(c, "")) for c in cols]
        f.write(",".join(vals) + "\n")

print(f"  Wrote {out}")
PYEOF

echo ""
echo "=========================================="
echo "완료: troubleshooting/original/"
echo "=========================================="
