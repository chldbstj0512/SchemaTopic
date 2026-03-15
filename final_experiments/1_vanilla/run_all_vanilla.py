#!/usr/bin/env python3
"""
Run 4 datasets × 5 Neural Topic Models × 5 seeds vanilla experiments.
Output: final_experiments/1_vanilla/seed{1..5}/{dataset}_{model}/
Uses 2 GPUs: jobs run 2 at a time (CUDA_VISIBLE_DEVICES=0 and 1).
"""
import subprocess
import sys
from pathlib import Path

# Project root (parent of final_experiments)
ROOT = Path(__file__).resolve().parent.parent.parent
OUT_BASE = ROOT / "final_experiments" / "1_vanilla"

DATASETS = ["20News", "AGNews", "DBpedia", "R8"]
MODELS = ["etm", "nvdm", "nstm", "plda", "scholar"]  # 5 models
SEEDS = [1, 2, 3, 4, 5]
NUM_GPUS = 2


def build_jobs():
    jobs = []
    for seed in SEEDS:
        for dataset in DATASETS:
            for model in MODELS:
                out_dir = OUT_BASE / f"seed{seed}" / f"{dataset}_{model}"
                jobs.append({
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "out_dir": str(out_dir),
                })
    return jobs


def run_one(job, gpu_id):
    env = dict(__import__("os").environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        "vanilla",
        "--dataset", job["dataset"],
        "--model", job["model"],
        "--seed", str(job["seed"]),
        "--out_dir", job["out_dir"],
    ]
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir.parent / f"log_{job['dataset']}_{job['model']}.txt"
    with open(log_path, "w") as f:
        f.write(f"Command: {' '.join(cmd)}\n\n")
    with open(log_path, "a") as log:
        ret = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    return ret.returncode


def main():
    jobs = build_jobs()
    print(f"Total jobs: {len(jobs)} (4 datasets × 5 models × 5 seeds)")
    print(f"Output base: {OUT_BASE}")
    print(f"Using {NUM_GPUS} GPUs (2 jobs in parallel).\n")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    completed = 0
    failed = []
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as ex:
        futures = {}
        for i, job in enumerate(jobs):
            gpu_id = i % NUM_GPUS
            fut = ex.submit(run_one, job, gpu_id)
            futures[fut] = job
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                code = fut.result()
                completed += 1
                if code != 0:
                    failed.append((job, code))
                status = "FAIL" if code != 0 else "OK"
                print(f"[{completed}/{len(jobs)}] {status} seed{job['seed']} {job['dataset']} {job['model']}")
            except Exception as e:
                failed.append((job, str(e)))
                completed += 1
                print(f"[{completed}/{len(jobs)}] EXC seed{job['seed']} {job['dataset']} {job['model']}: {e}")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for job, err in failed:
            print(f"  seed{job['seed']} {job['dataset']} {job['model']}: {err}")
        sys.exit(1)
    print("\nAll jobs completed successfully.")


if __name__ == "__main__":
    main()
