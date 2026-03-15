#!/usr/bin/env python3
"""
results/ 아래 모든 step1.txt를 찾아 flat vs 계층(hierarchy) 집계.
llm_validation.validate_schema_step1_flat 사용.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from llm_validation import validate_schema_step1_flat


def main():
    results_dir = os.path.join(REPO_ROOT, "results")
    flat_list = []
    hierarchy_list = []

    for root, _dirs, files in os.walk(results_dir):
        for f in files:
            if f != "step1.txt":
                continue
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fp:
                    text = fp.read()
            except Exception as e:
                hierarchy_list.append((path, f"read_error:{e}"))
                continue
            is_flat, err = validate_schema_step1_flat(text)
            rel = os.path.relpath(path, REPO_ROOT)
            if is_flat:
                flat_list.append(rel)
            else:
                hierarchy_list.append((rel, err or "schema_hierarchy"))

    # 요약
    n_flat = len(flat_list)
    n_hier = len(hierarchy_list)
    print("=" * 60)
    print("Step1 flat vs 계층(hierarchy) 집계")
    print("=" * 60)
    print(f"총 step1.txt: {n_flat + n_hier}")
    print(f"  flat (1줄 1라벨, 계층 없음): {n_flat}")
    print(f"  계층 구조 감지:               {n_hier}")
    print()

    if hierarchy_list:
        print("--- 계층으로 나온 것 (path, 사유) ---")
        for rel, err in sorted(hierarchy_list, key=lambda x: x[0]):
            print(f"  {err}: {rel}")
        print()

    if flat_list:
        print("--- flat으로 나온 것 (경로만, 상위 50개) ---")
        for rel in sorted(flat_list)[:50]:
            print(f"  {rel}")
        if len(flat_list) > 50:
            print(f"  ... 외 {len(flat_list) - 50}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
