#!/bin/bash
# results 삭제 후 파이프라인 실행
cd "$(dirname "$0")"

echo "=== Stopping any running pipelines ==="
tmux kill-session -t gpu0 2>/dev/null
tmux kill-session -t gpu1 2>/dev/null
pkill -f "main.py pipeline" 2>/dev/null

echo "=== Deleting results ==="
rm -rf results/*

echo "=== Starting pipelines (GPU 0 + GPU 1) ==="
ROOT="$(pwd)"
tmux new-session -d -s gpu0 "cd $ROOT && CUDA_VISIBLE_DEVICES=0 ./run_pipeline_gpu0.sh"
tmux new-session -d -s gpu1 "cd $ROOT && CUDA_VISIBLE_DEVICES=1 ./run_pipeline_gpu1.sh"

echo "Done. tmux attach -t gpu0  (or gpu1)"
