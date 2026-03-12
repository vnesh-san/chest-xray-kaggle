#!/usr/bin/env bash
# Polls until unzip finishes, then runs the smoke test.
# Usage: bash scripts/wait_and_smoke_test.sh &

set -euo pipefail

UNZIP_PID=71359
DATA_DIR="data/raw/vinbigdata-chest-xray-abnormalities-detection"
CSV_PATH="$DATA_DIR/train.csv"
LOG="outputs/smoke_test_run.log"

mkdir -p outputs

echo "[wait_and_smoke_test] Waiting for unzip PID $UNZIP_PID to finish..."

while kill -0 "$UNZIP_PID" 2>/dev/null; do
    sleep 10
done

# Extra check: make sure train.csv exists (proves zip extracted correctly)
if [[ ! -f "$CSV_PATH" ]]; then
    echo "[ERROR] train.csv not found after unzip — extraction may have failed."
    exit 1
fi

echo "[wait_and_smoke_test] Unzip complete. Launching smoke test..."
echo "Log -> $LOG"

PYTHONPATH=. python scripts/smoke_test.py \
    --model yolo26n.pt \
    --n 500 \
    --epochs 1 \
    --batch 8 \
    2>&1 | tee "$LOG"

echo "[wait_and_smoke_test] Done. Check $LOG for results."
