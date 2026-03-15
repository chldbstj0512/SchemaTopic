#!/usr/bin/env python3
"""
1번(reindex) 검증 후 이전 메트릭과 비교.
사용: 세 실험을 run_reindex_validation_three.sh 로 돌린 뒤
  python3 scripts/compare_reindex_validation.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "reindex_validation"
TROUBLE = ROOT / "troubleshooting" / "1-llama-v2"

EXPERIMENTS = [
    ("dbpedia_plda_auto", "DBpedia PLDA auto", "dbpedia_CV_PLDA_auto"),
    ("20ng_nvdm_auto", "20NG nvdm auto", "20NG_nvdm_TD_auto"),
    ("20ng_ecrtm_keep", "20NG ECRTM keep", "20NG_ECRTM_TC_keep"),
]

def load_metrics(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    print("Reindex validation: before (troubleshooting) vs after (reindex fix)")
    print("=" * 70)
    for out_name, label, trouble_dir in EXPERIMENTS:
        before_path = TROUBLE / trouble_dir / "anchor" / "metrics.json"
        after_path = BASE / out_name / "anchor" / "metrics.json"
        before = load_metrics(before_path)
        after = load_metrics(after_path)
        cv_b = before.get("topic_coherence_cv") if before else None
        td_b = before.get("topic_diversity") if before else None
        cv_a = after.get("topic_coherence_cv") if after else None
        td_a = after.get("topic_diversity") if after else None
        print(f"\n{label}")
        print(f"  Before: CV={cv_b}, TD={td_b}  ({before_path})")
        print(f"  After:  CV={cv_a}, TD={td_a}  ({after_path})")
        if cv_b is not None and cv_a is not None:
            d_cv = (cv_a - cv_b) / cv_b * 100 if cv_b else 0
            print(f"  CV change: {d_cv:+.1f}%")
        if td_b is not None and td_a is not None:
            d_td = (td_a - td_b) / td_b * 100 if td_b else 0
            print(f"  TD change: {d_td:+.1f}%")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
