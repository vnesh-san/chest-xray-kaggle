#!/usr/bin/env bash
# start_training_tmux.sh — Launch full training in a monitored tmux session.
#
# Creates session "chest-training" with 3 windows:
#   0 training   — full pipeline (convert A+B → stage1→4 → ensemble), tee to log
#   1 gpu        — top: nvidia-smi dmon stream; bottom: filtered training log tail
#   2 tensorboard — TensorBoard server on port 6006
#
# Usage:
#   bash scripts/start_training_tmux.sh          # batch=16 (8 per A10G), workers=8
#   bash scripts/start_training_tmux.sh --batch 8   # if 16 OOMs

set -euo pipefail

REPO=/home/vignesh_santhalingam/playground/kaggle/chest-xray-kaggle
ENV_BIN=/home/vignesh_santhalingam/.airconda-environments/local--training--v0.1.0/bin
PYTHON=$ENV_BIN/python
TB=$ENV_BIN/tensorboard
SESSION=chest-training
LOG=$REPO/outputs/training_full.log

# Parse optional --batch override
BATCH=16
for arg in "$@"; do
  case $arg in
    --batch) shift; BATCH=$1 ;;
    --batch=*) BATCH="${arg#*=}" ;;
  esac
done

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

# ── Window 0: training ─────────────────────────────────────────────────────
tmux new-session -d -s "$SESSION" -n "training" -x 220 -y 50

tmux send-keys -t "$SESSION:training" \
  "cd $REPO && \
   echo '=== Phase 1: DICOM conversion (32 workers) ===' && \
   PYTHONPATH=. $PYTHON scripts/train_full.py --stage convert --window-set both --workers 32 \
     2>&1 | tee $LOG && \
   echo '=== Phase 2: Training (8 dataloader workers, 2 GPUs) ===' | tee -a $LOG && \
   PYTHONPATH=. $PYTHON scripts/train_full.py --stage all --batch $BATCH --workers 8 \
     2>&1 | tee -a $LOG" Enter

# ── Window 1: GPU + log monitor ────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "gpu"

# Top pane: streaming GPU power/util/memory/temp every 3s
tmux send-keys -t "$SESSION:gpu" \
  "nvidia-smi dmon -s pucmt -d 3" Enter

# Bottom pane: filtered training log (epochs, mAP, loss, errors)
tmux split-window -v -t "$SESSION:gpu"
tmux send-keys -t "$SESSION:gpu.1" \
  "echo 'Waiting for log...'; \
   while [ ! -f $LOG ]; do sleep 2; done; \
   tail -f $LOG | grep --line-buffered \
     -E '(Epoch|fold|mAP|map|loss|Loss|Error|ERROR|OOM|Killed|WARN|stage|Window set|Converting|Done|saved)'" Enter

# Resize: 60% top / 40% bottom
tmux resize-pane -t "$SESSION:gpu.0" -y 30

# ── Window 2: TensorBoard ──────────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "tensorboard"
tmux send-keys -t "$SESSION:tensorboard" \
  "cd $REPO && $TB --logdir outputs/runs --port 6006 --bind_all" Enter

# Start on training window
tmux select-window -t "$SESSION:training"

echo ""
echo "  Session started: $SESSION"
echo ""
echo "  tmux attach -t $SESSION          — view all panes"
echo "  Ctrl-B then 0/1/2                — switch windows"
echo "  Ctrl-B then d                    — detach (training keeps running)"
echo ""
echo "  TensorBoard → http://localhost:6006"
echo "  VSCode: Ctrl+Shift+P → 'Python: Launch TensorBoard' → select $REPO/outputs/runs"
echo ""
echo "  Training log: $LOG"
