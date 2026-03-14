#!/usr/bin/env bash
# run_24h.sh — 24-hour parallel training pipeline
#
# Runs 4 models on 2 GPUs in two parallel phases, then WBF ensemble.
#
#   Phase 1 (~10-11h):  YOLO26x-A on GPU 0  ||  YOLO26x-B on GPU 1
#   Phase 2 (~10-11h):  RT-DETR-X-A on GPU 0  ||  RT-DETR-X-B on GPU 1
#   Phase 3 (~1h):      WBF ensemble (both GPUs)
#
# Usage (from chest-xray-kaggle/):
#   bash scripts/run_24h.sh [--batch 8] [--workers 8]
#
# Monitor:
#   tail -f outputs/train_stage1.log outputs/train_stage2.log
#   nvidia-smi dmon -s u -d 10

set -euo pipefail
cd "$(dirname "$0")/.."   # always run from chest-xray-kaggle/

BATCH=${BATCH:-8}
WORKERS=${WORKERS:-8}

# Allow overrides via CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch)   BATCH="$2";   shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

LOG_DIR="outputs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  24h Parallel Training Pipeline"
echo "  batch=$BATCH  workers=$WORKERS"
echo "  $(date)"
echo "============================================================"

# ── Phase 1: YOLO26x — both window sets in parallel ──────────────────────────
echo ""
echo "=== Phase 1: YOLO26x-A (GPU 0) || YOLO26x-B (GPU 1) ==="
echo "  Started: $(date)"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_full.py \
    --stage 1 --no-folds --device 0 --batch "$BATCH" --workers "$WORKERS" \
    > "$LOG_DIR/train_stage1.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/train_full.py \
    --stage 2 --no-folds --device 0 --batch "$BATCH" --workers "$WORKERS" \
    > "$LOG_DIR/train_stage2.log" 2>&1 &
PID2=$!

echo "  Stage 1 PID: $PID1  (tail -f $LOG_DIR/train_stage1.log)"
echo "  Stage 2 PID: $PID2  (tail -f $LOG_DIR/train_stage2.log)"
wait $PID1 $PID2
echo "  Phase 1 complete: $(date)"

# ── Phase 2: RT-DETR-X — both window sets in parallel ────────────────────────
echo ""
echo "=== Phase 2: RT-DETR-X-A (GPU 0) || RT-DETR-X-B (GPU 1) ==="
echo "  Started: $(date)"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_full.py \
    --stage 3 --no-folds --device 0 --batch "$BATCH" --workers "$WORKERS" \
    > "$LOG_DIR/train_stage3.log" 2>&1 &
PID3=$!

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/train_full.py \
    --stage 4 --no-folds --device 0 --batch "$BATCH" --workers "$WORKERS" \
    > "$LOG_DIR/train_stage4.log" 2>&1 &
PID4=$!

echo "  Stage 3 PID: $PID3  (tail -f $LOG_DIR/train_stage3.log)"
echo "  Stage 4 PID: $PID4  (tail -f $LOG_DIR/train_stage4.log)"
wait $PID3 $PID4
echo "  Phase 2 complete: $(date)"

# ── Phase 3: WBF Ensemble ─────────────────────────────────────────────────────
echo ""
echo "=== Phase 3: WBF Ensemble ==="
echo "  Started: $(date)"

PYTHONPATH=. python scripts/train_full.py \
    --stage ensemble \
    > "$LOG_DIR/train_ensemble.log" 2>&1

echo "  Ensemble complete: $(date)"
echo ""
echo "============================================================"
echo "  ALL DONE: $(date)"
echo "  Submission: outputs/submission_ensemble.csv"
echo "  Logs:       $LOG_DIR/train_stage{1..4}.log  $LOG_DIR/train_ensemble.log"
echo "============================================================"
