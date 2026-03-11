# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style

Before starting any non-trivial change — new model, preprocessing pipeline, training config, evaluation run — ask clarifying questions to understand requirements, edge cases, and tradeoffs. Ask one focused question at a time if there are multiple unknowns.

Once requirements are clear, update `progress.md` with the details (see structure below), then begin implementation. Do not write any implementation code before progress.md is updated.

### progress.md structure

`progress.md` is a single long-lived file that accumulates the full history of every experiment, change, and decision. It always contains these sections, in this order:

---

**## TLDR**
A running bullet list of every request ever made and its status (`completed` / `in progress` / `abandoned`). Append a new bullet at the bottom when starting each new request. Update the status when done.

**## Current Work**
The full verbatim details of the work in progress: the original request, every question asked, every answer given, final agreed requirements, edge cases, and decisions. This section is replaced at the start of each new piece of work.

**## Previous Work**
An exact copy of what was in "Current Work" at the end of the last completed request. Gives Claude immediate context on what was most recently built without reading the full history. Replaced each time new work completes.

**## Full History**
An append-only log of every past experiment, oldest at top, newest at bottom. Each entry includes: date, branch name, original request, full Q&A, requirements agreed, and outcome. When "Current Work" is retired, paste it in full at the bottom of this section before overwriting it.

---

When starting new work: move "Current Work" into "Previous Work" and append it to "Full History", then populate a fresh "Current Work".

## Competition

[VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)

**Goal**: Detect and localize 14 types of thoracic abnormalities in chest X-ray DICOM images. Output bounding boxes with class labels and confidence scores.

**14 abnormality classes** (class_id 0–13) plus "No finding" (class_id 14):

| class_id | Name |
|----------|------|
| 0 | Aortic enlargement |
| 1 | Atelectasis |
| 2 | Calcification |
| 3 | Cardiomegaly |
| 4 | Consolidation |
| 5 | ILD |
| 6 | Infiltration |
| 7 | Lung Opacity |
| 8 | Nodule/Mass |
| 9 | Other lesion |
| 10 | Pleural effusion |
| 11 | Pleural thickening |
| 12 | Pneumothorax |
| 13 | Pulmonary fibrosis |
| 14 | No finding |

**Data**:
- ~18,000 DICOM training images; each read by at least 3 of 17 Vietnamese radiologists
- ~3,000 DICOM test images (no labels)
- `train.csv`: `image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max`
- "No finding" rows have `x_min=0, y_min=0, x_max=1, y_max=1` (normalized placeholder)

**Evaluation**: Mean Average Precision (mAP) averaged over IoU thresholds 0.4 to 0.75 (step 0.05).
- For "No finding" images: predict class 14 with a full-image box `0 0 width height`.
- A predicted "No finding" on a normal image counts as a true positive.

**Submission format** (`submission.csv`):
```
image_id,PredictionString
image_001,0.9 0 100 200 400 500 0.8 3 50 60 300 400
image_002,1.0 14 0 0 1024 1024
```
`PredictionString` = space-separated tuples of `confidence class_id xmin ymin xmax ymax`.

## Model Versioning

Every trained model is versioned. `outputs/model_versions.json` is the source of truth.

| Version | Dir | Method | Base | Data | Status |
|---------|-----|--------|------|------|--------|
| — | — | — | — | — | not started |

When adding a new version: update the table above, update `outputs/model_versions.json`, and use the version number in the output dir name.

## Post-Training Evaluation (MANDATORY)

**Every trained model must be evaluated immediately after training completes.** Do not move to the next version or declare a model "done" without running evaluation.

### What to evaluate

1. **Detection accuracy** — mAP at IoU 0.4:0.75:0.05 on the validation split:
   ```bash
   PYTHONPATH=. python scripts/eval.py \
     --checkpoint outputs/<version>/best.pt \
     --data-dir data/processed \
     --output-dir outputs/eval_<version>
   ```

2. **Per-class AP** — identify which classes are hardest (small lesions, rare classes).

3. **Failure analysis** — inspect false positives and false negatives for incorrect predictions:
   - **Missed detections** — IoU < threshold, low recall
   - **False positives** — high confidence but wrong location or wrong class
   - **Class confusion** — e.g., Infiltration vs Lung Opacity
   - **"No finding" errors** — predicted finding on a normal image or vice versa

4. **Analyze results**:
   ```bash
   PYTHONPATH=. python scripts/analyze_preds.py \
     --preds outputs/eval_<version>/predictions.csv \
     --gt data/processed/val_labels.csv \
     --output outputs/eval_<version>/analysis
   ```

### Where to record results

Update `outputs/model_versions.json` with:
```json
"results": {
  "mAP_40_75": 0.0,
  "mAP_50": 0.0,
  "per_class_AP": {},
  "failure_modes": {"missed_detections": 0, "false_positives": 0, "class_confusion": 0, "no_finding_errors": 0},
  "eval_notes": "qualitative summary"
}
```

## Commands

```bash
# From chest-xray-kaggle/

# Preprocess DICOM images → PNG (windowing + normalization)
PYTHONPATH=. python scripts/preprocess.py \
  --dicom-dir data/raw/train \
  --output-dir data/processed/images \
  --window-center 40 --window-width 400

# Convert train.csv annotations → YOLO / COCO format
PYTHONPATH=. python scripts/prepare_labels.py \
  --csv data/raw/train.csv \
  --images-dir data/processed/images \
  --output-dir data/processed \
  --format yolo  # or coco

# Train YOLOv8 detector
PYTHONPATH=. python scripts/train.py \
  --config configs/yolov8_default.yaml \
  --output-dir outputs/<version>

# Evaluate on validation split
PYTHONPATH=. python scripts/eval.py \
  --checkpoint outputs/<version>/best.pt \
  --data-dir data/processed \
  --output-dir outputs/eval_<version>

# Generate submission CSV
PYTHONPATH=. python scripts/predict.py \
  --checkpoint outputs/<version>/best.pt \
  --test-dir data/processed/test_images \
  --output submission.csv \
  --conf-thresh 0.25 --nms-iou 0.5

# Ensemble multiple model predictions
PYTHONPATH=. python scripts/ensemble.py \
  --preds outputs/eval_v1/predictions.csv outputs/eval_v2/predictions.csv \
  --method wbf \
  --output submission_ensemble.csv
```

## Architecture

This is an object detection pipeline for the [VinBigData Chest X-ray Abnormalities Detection Kaggle competition](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection).

**Approach**: Fine-tune a pretrained object detector (YOLOv8 or EfficientDet) on chest X-ray annotations, with careful DICOM windowing preprocessing and class-balanced sampling to handle the long-tailed class distribution.

### Planned Components

**`src/data/`** — Data loading and preprocessing:
- `dicom_utils.py` — DICOM → PNG conversion with windowing (lung window, mediastinum window, bone window)
- `dataset.py` — PyTorch Dataset for training/validation
- `augmentations.py` — Albumentations pipeline (flips, rotations, CLAHE, mosaic)
- `label_utils.py` — train.csv → YOLO/COCO format conversion, consensus label merging

**`src/models/`** — Model definitions:
- `detector.py` — Wrapper around YOLOv8 / EfficientDet / DETR

**`src/training/`** — Training loop:
- `train.py` — Main training entry point (supports multi-GPU via DDP)
- `loss.py` — Detection loss (box regression + classification)
- `scheduler.py` — LR scheduler utilities

**`src/evaluation/`** — Metrics:
- `metrics.py` — mAP at IoU 0.4:0.75:0.05, per-class AP
- `visualize.py` — Draw predictions and ground truth on images for inspection

**`src/inference/`** — Submission generation:
- `predict.py` — Run inference on test DICOM images, output submission CSV
- `ensemble.py` — Weighted Box Fusion (WBF) ensemble across multiple model predictions
- `postprocess.py` — NMS, confidence thresholding, "No finding" logic

**`scripts/`** — CLI entry points for preprocessing, training, evaluation, prediction.

**`configs/`** — YAML configs for each experiment (model arch, training hyperparams, augmentation).

### Data Flow

1. Raw DICOM images → `preprocess.py` → normalized PNG images in `data/processed/images/`
2. `train.csv` annotations → `prepare_labels.py` → YOLO/COCO label files in `data/processed/`
3. Train/val split (stratified by class presence) stored in `data/processed/splits/`
4. `train.py` trains detector on processed data → checkpoint at `outputs/<version>/best.pt`
5. `eval.py` runs validation → mAP scores + per-prediction CSV
6. `predict.py` runs test inference → `submission.csv` for Kaggle upload

### Key Design Decisions

- **DICOM windowing**: chest X-rays need windowing before feeding to ImageNet-pretrained models. Use lung window (WC=-600, WW=1500) or auto-windowing per image.
- **Consensus labels**: each image is annotated by multiple radiologists. Merge overlapping boxes (IoU > 0.4) by majority vote before training.
- **"No finding" handling**: images labeled as class 14 by all radiologists are treated as negatives for the detector. At inference, if no box exceeds the confidence threshold, output a single class-14 prediction with a full-image box.
- **Class imbalance**: oversample rare classes (Calcification, Pneumothorax, ILD) or use focal loss.

### Key Configs

| File | Purpose |
|---|---|
| `configs/yolov8_default.yaml` | YOLOv8x, 640px, 50 epochs, lr=1e-3 |
| `configs/data.yaml` | paths to processed images and labels |
| `configs/augmentation.yaml` | Albumentations pipeline settings |

### Useful Links

- [Competition](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)
- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [Albumentations](https://albumentations.ai/)
- [pydicom](https://pydicom.github.io/)
- [Weighted Box Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
- [timm (pretrained backbones)](https://timm.fast.ai/)
