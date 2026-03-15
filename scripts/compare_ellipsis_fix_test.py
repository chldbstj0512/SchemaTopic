#!/usr/bin/env python3
"""
20NG ECRTM keep ellipsis-fix 테스트 결과 비교.
- 이전(깨진 run): reindex_validation/20ng_ecrtm_keep → Step 2 truncated
- 수정 후: reindex_validation/20ng_ecrtm_keep_ellipsis_fix
- 기준(원래 성공했던 keep): troubleshooting/1-llama-v2/20NG_ECRTM_TC_keep
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "reindex_validation"
TROUBLE = ROOT / "troubleshooting" / "1-llama-v2"

def load_metrics(path):
    if not path.exists():
        return None, "file missing"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("truncated"):
        return data, "truncated (step: {})".format(data.get("truncated_step", "?"))
    return data, "ok"

def main():
    prev_path = BASE / "20ng_ecrtm_keep" / "anchor" / "metrics.json"
    new_path = BASE / "20ng_ecrtm_keep_ellipsis_fix" / "anchor" / "metrics.json"
    ref_path = TROUBLE / "20NG_ECRTM_TC_keep" / "anchor" / "metrics.json"

    print("20NG ECRTM keep – ellipsis fix test comparison")
    print("=" * 60)

    ref_data, ref_status = load_metrics(ref_path)
    print("\n[Reference] troubleshooting/1-llama-v2/20NG_ECRTM_TC_keep (원래 성공한 keep)")
    print("  Status:", ref_status)
    if ref_data and ref_status == "ok":
        print("  CV:", ref_data.get("topic_coherence_cv"))
        print("  TD:", ref_data.get("topic_diversity"))

    prev_data, prev_status = load_metrics(prev_path)
    print("\n[Before fix] reindex_validation/20ng_ecrtm_keep (Step 2에서 깨졌던 run)")
    print("  Status:", prev_status)
    if prev_data and prev_status == "ok":
        print("  CV:", prev_data.get("topic_coherence_cv"))
        print("  TD:", prev_data.get("topic_diversity"))

    new_data, new_status = load_metrics(new_path)
    print("\n[After fix] reindex_validation/20ng_ecrtm_keep_ellipsis_fix")
    print("  Status:", new_status)
    if new_data and new_status == "ok":
        print("  CV:", new_data.get("topic_coherence_cv"))
        print("  TD:", new_data.get("topic_diversity"))
        if ref_data and ref_status == "ok":
            cv_ref = ref_data.get("topic_coherence_cv")
            td_ref = ref_data.get("topic_diversity")
            cv_new = new_data.get("topic_coherence_cv")
            td_new = new_data.get("topic_diversity")
            if cv_ref and cv_new:
                print("  CV vs reference: {:.4f} vs {:.4f}".format(cv_new, cv_ref))
            if td_ref and td_new:
                print("  TD vs reference: {:.4f} vs {:.4f}".format(td_new, td_ref))

    print("\n" + "=" * 60)
    if new_status == "ok" and prev_status != "ok":
        print("=> Fix effective: new run completed without truncation.")
    elif new_status != "ok":
        print("=> New run still failed (truncated or not run yet).")
    else:
        print("=> Both runs completed; compare metrics above.")

if __name__ == "__main__":
    main()
