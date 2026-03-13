#!/bin/bash
# GPU0 + GPU1 동시 실행 (results 삭제 없음, 기존 vanilla 재사용)
# 사용법: ./run_pipeline_both.sh
cd "$(dirname "$0")"

echo "=== Stopping any running pipelines ==="
tmux kill-session -t gpu0 2>/dev/null
tmux kill-session -t gpu1 2>/dev/null
pkill -f "main.py pipeline" 2>/dev/null

echo "=== Starting pipelines (GPU 0 + GPU 1) ==="
ROOT="$(pwd)"
tmux new-session -d -s gpu0 "cd $ROOT && CUDA_VISIBLE_DEVICES=0 ./run_pipeline_gpu0.sh"
tmux new-session -d -s gpu1 "cd $ROOT && CUDA_VISIBLE_DEVICES=1 ./run_pipeline_gpu1.sh"

echo ""
echo "Done. GPU0(20News,AGNews) / GPU1(DBpedia,R8) 각각 tmux에서 실행 중"
echo ""
echo "  tmux attach -t gpu0   # GPU0 로그 보기"
echo "  tmux attach -t gpu1   # GPU1 로그 보기"
echo "  (detach: Ctrl+B, D)"
echo ""
