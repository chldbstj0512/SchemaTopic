#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
results 폴더의 pipeline 결과를 분석하여 케이스별로 정리합니다.

폴더 구조:
  - {dataset}_{model}_50_seed{N}/
    - metrics_comparison.csv   : vanilla vs keep vs auto 메트릭 비교
    - topic_words_vanilla.txt  : 바닐라 결과
    - topic_words_keep.txt     : keep 결과
    - topic_words_auto.txt     : auto 결과

케이스 정의:
  - pipeline_{dataset}_{model}_50_seed{N} (vanilla + auto)
  - pipeline_{dataset}_{model}_50_keep_seed{N} (keep)
  두 실행을 하나의 케이스로 봄.
"""

import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
OUTPUT_ROOT = RESULTS_ROOT / "analysis"
OUTPUT_ROOT_V2 = RESULTS_ROOT / "analysis_v2"
OUTPUT_ROOT_V3 = RESULTS_ROOT / "analysis_v3"

METRIC_KEYS = ["topic_coherence_cv", "topic_diversity", "purity", "nmi", "PN", "truncated"]


def parse_pipeline_name(name: str) -> Optional[Dict]:
    """Parse pipeline folder name. Returns None if not matching."""
    # pipeline_20News_ecrtm_50_seed1
    # pipeline_20News_ecrtm_50_keep_seed1
    m = re.match(r"pipeline_(.+)_(\d+)_(?:keep_)?seed(\d+)$", name)
    if not m:
        return None
    rest, num_topics, seed = m.group(1), m.group(2), m.group(3)
    # rest = "20News_ecrtm" or "DBpedia_ecrtm" etc
    parts = rest.rsplit("_", 1)
    if len(parts) != 2:
        return None
    dataset, model = parts[0], parts[1]
    return {
        "dataset": dataset,
        "model": model,
        "num_topics": num_topics,
        "seed": seed,
        "keep": "keep" in name,
    }


def find_schema_dir(pipeline_dir: Path) -> Optional[Path]:
    """Find schema_N directory (N can vary). Prefer schema_50 for keep."""
    schema_dirs = list(pipeline_dir.glob("schema_*"))
    if not schema_dirs:
        return None
    # Prefer schema_50, else largest N
    schema_50 = next((d for d in schema_dirs if d.name == "schema_50"), None)
    if schema_50:
        return schema_50
    return max(schema_dirs, key=lambda d: int(d.name.replace("schema_", "") or "0"))


def load_metrics(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect_cases() -> List[Dict]:
    """Collect all (dataset, model, seed) cases that have both vanilla and keep pipelines."""
    cases = {}
    for d in RESULTS_ROOT.iterdir():
        if not d.is_dir() or not d.name.startswith("pipeline_"):
            continue
        info = parse_pipeline_name(d.name)
        if not info:
            continue
        key = (info["dataset"], info["model"], info["seed"])
        if key not in cases:
            cases[key] = {"vanilla_pipeline": None, "keep_pipeline": None}
        if info["keep"]:
            cases[key]["keep_pipeline"] = d
        else:
            cases[key]["vanilla_pipeline"] = d

    out = []
    for (dataset, model, seed), pipes in cases.items():
        if pipes["vanilla_pipeline"] and pipes["keep_pipeline"]:
            out.append({
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "vanilla_pipeline": pipes["vanilla_pipeline"],
                "keep_pipeline": pipes["keep_pipeline"],
            })
    return sorted(out, key=lambda c: (c["dataset"], c["model"], int(c["seed"])))


def process_case(case: Dict) -> Optional[Dict]:
    """Extract metrics and topic_words paths for one case."""
    vp = case["vanilla_pipeline"]
    kp = case["keep_pipeline"]

    # Vanilla: vanilla/metrics.json, vanilla/top_words.txt
    vanilla_metrics = load_metrics(vp / "vanilla" / "metrics.json")
    vanilla_words = vp / "vanilla" / "top_words.txt"

    # Keep: anchor/metrics.json, schema_50/topic_words.txt
    keep_metrics = load_metrics(kp / "anchor" / "metrics.json")
    keep_schema = find_schema_dir(kp)
    keep_words = (keep_schema / "topic_words.txt") if keep_schema else None

    # Auto: anchor/metrics.json from vanilla pipeline, schema_N/topic_words.txt
    auto_schema = find_schema_dir(vp)
    auto_metrics = load_metrics(vp / "anchor" / "metrics.json")
    auto_words = (auto_schema / "topic_words.txt") if auto_schema else None

    if not vanilla_metrics or not keep_metrics or not auto_metrics:
        return None

    return {
        "case": case,
        "vanilla_metrics": vanilla_metrics,
        "keep_metrics": keep_metrics,
        "auto_metrics": auto_metrics,
        "vanilla_words": vanilla_words if vanilla_words.exists() else None,
        "keep_words": keep_words if keep_words and keep_words.exists() else None,
        "auto_words": auto_words if auto_words and auto_words.exists() else None,
    }


def _is_truncated(metrics: Optional[Dict]) -> bool:
    """Check if metrics indicate truncated LLM output."""
    if not metrics:
        return False
    return metrics.get("truncated") is True


def _format_for_csv(val) -> str:
    """Format metric value for CSV."""
    if isinstance(val, (int, float)):
        return f"{val:.6f}"
    return str(val) if val is not None and val != "" else ""


def write_metrics_csv(out_dir: Path, data: dict):
    """Write metrics_comparison.csv with vanilla, keep, auto columns."""
    rows = []
    for key in METRIC_KEYS:
        if key == "truncated":
            v_str = "truncated" if _is_truncated(data["vanilla_metrics"]) else ""
            k_str = "truncated" if _is_truncated(data["keep_metrics"]) else ""
            a_str = "truncated" if _is_truncated(data["auto_metrics"]) else ""
        else:
            v = data["vanilla_metrics"].get(key, "")
            k = data["keep_metrics"].get(key, "")
            a = data["auto_metrics"].get(key, "")
            v_str = _format_for_csv(v)
            k_str = _format_for_csv(k)
            a_str = _format_for_csv(a)
        rows.append({"metric": key, "vanilla": v_str, "keep": k_str, "auto": a_str})

    out_path = out_dir / "metrics_comparison.csv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("metric,vanilla,keep,auto\n")
        for r in rows:
            f.write(f"{r['metric']},{r['vanilla']},{r['keep']},{r['auto']}\n")
    print(f"  Wrote {out_path}")


def copy_topic_words(out_dir: Path, data: dict):
    """Copy topic_words to topic_words_vanilla.txt, topic_words_keep.txt, topic_words_auto.txt."""
    for name, src in [
        ("topic_words_vanilla.txt", data["vanilla_words"]),
        ("topic_words_keep.txt", data["keep_words"]),
        ("topic_words_auto.txt", data["auto_words"]),
    ]:
        if src:
            dst = out_dir / name
            shutil.copy2(src, dst)
            print(f"  Copied {name}")
        else:
            print(f"  [skip] {name} (source not found)")


def get_existing_folders(root: Path) -> set:
    """Return set of folder names in given root."""
    if not root.exists():
        return set()
    return {d.name for d in root.iterdir() if d.is_dir()}


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT_V2.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT_V3.mkdir(parents=True, exist_ok=True)
    existing_v1 = get_existing_folders(OUTPUT_ROOT)
    existing_v2 = get_existing_folders(OUTPUT_ROOT_V2)
    cases = collect_cases()
    print(f"Found {len(cases)} cases (dataset+model+seed with both vanilla and keep pipelines)")
    print(f"analysis: {len(existing_v1)} | analysis_v2: {len(existing_v2)} | New -> analysis_v3\n")

    for case in cases:
        folder_name = f"{case['dataset']}_{case['model']}_50_seed{case['seed']}"
        if folder_name in existing_v1:
            out_root = OUTPUT_ROOT
            label = ""
        elif folder_name in existing_v2:
            out_root = OUTPUT_ROOT_V2
            label = "[v2]"
        else:
            out_root = OUTPUT_ROOT_V3
            label = "[v3]"
        out_dir = out_root / folder_name
        print(f"[{folder_name}] {label}")

        data = process_case(case)
        if not data:
            print("  [skip] Missing metrics")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        write_metrics_csv(out_dir, data)
        copy_topic_words(out_dir, data)
        print()

    print(f"Done. analysis: {OUTPUT_ROOT}")
    print(f"Done. analysis_v2: {OUTPUT_ROOT_V2}")
    print(f"Done. analysis_v3 (new): {OUTPUT_ROOT_V3}")


if __name__ == "__main__":
    main()
