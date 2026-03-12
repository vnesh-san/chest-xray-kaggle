"""
train_full.py — Full competition training pipeline.

Strategy:
  Stage 1: 15-fold CV with yolo26x   (NMS-free, MuSGD, 1024px)
  Stage 2: 15-fold CV with rtdetr-x  (transformer decoder, complementary errors)
  Stage ensemble: WBF across all 30 checkpoints (iou_thr=0.4)

Key improvements over naive baseline:
  - Consensus labels: WBF-merged radiologist annotations, min_votes=2 (biggest gain)
  - Multilabel-stratified CV: each fold gets representative samples of all 14 rare classes
  - 1024px input resolution for small lesion recall
  - hsv_s=0.2 (safe for multi-window RGB encoding Hounsfield ranges)
  - Full 15+15 fold WBF ensemble at inference

Usage:
    # Prepare consensus labels first (run once):
    PYTHONPATH=. python scripts/prepare_labels.py

    # Stage 1 — YOLO26x 15-fold CV
    PYTHONPATH=. python scripts/train_full.py --stage 1

    # Stage 2 — RT-DETR-X 15-fold CV
    PYTHONPATH=. python scripts/train_full.py --stage 2

    # Ensemble inference + submission
    PYTHONPATH=. python scripts/train_full.py --stage ensemble

    # Resume from a specific fold
    PYTHONPATH=. python scripts/train_full.py --stage 1 --start-fold 4
"""
import argparse
import gc
import os
import shutil
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.data.dicom_utils import dicom_to_multiwindow_png
from src.data.label_consensus import build_consensus_labels

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path("data/raw/vinbigdata-chest-xray-abnormalities-detection")
WORK_DIR   = Path("outputs")
YOLO_DIR   = WORK_DIR / "yolo_dataset"
RUNS_DIR   = WORK_DIR / "runs"
FOLD_DIR   = WORK_DIR / "folds"
IMAGES_DIR = YOLO_DIR / "images"
LABELS_DIR = YOLO_DIR / "labels"

N_FOLDS    = 15
IMG_SIZE   = 1024   # 640 → 1024: +3-6 mAP especially for small lesions
EPOCHS     = 50
SEED       = 42

CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification",
    "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
    "Lung Opacity", "Nodule/Mass", "Other lesion",
    "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
]
NC = len(CLASS_NAMES)

STAGE1_MODEL = "yolo26x.pt"
STAGE2_MODEL = "rtdetr-x.pt"


# ── Label helpers ─────────────────────────────────────────────────────────────
def write_yolo_label(path: Path, anns: list):
    with open(path, "w") as f:
        for cls, xc, yc, bw, bh in anns:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def load_consensus_annotations(train_data: pd.DataFrame) -> dict:
    """Build consensus-merged annotation map (WBF, min_votes=2)."""
    print("Building consensus labels (WBF merge, min_votes=2, iou_thr=0.4) ...")
    anns_map = build_consensus_labels(train_data, iou_thr=0.4, min_votes=2)
    n_pos = sum(1 for v in anns_map.values() if v)
    n_boxes = sum(len(v) for v in anns_map.values())
    print(f"  {n_pos:,} positive images, {n_boxes:,} consensus boxes")
    return anns_map


# ── DICOM → PNG conversion ────────────────────────────────────────────────────
def convert_all_dicoms(all_ids, workers=8):
    """Convert all training DICOMs to multi-window PNG (idempotent)."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    remaining = [i for i in all_ids if not (IMAGES_DIR / f"{i}.png").exists()]
    if not remaining:
        print(f"All {len(all_ids):,} PNGs already exist.")
        return
    print(f"Converting {len(remaining):,} DICOMs ...")

    def _convert(img_id):
        src = ROOT_DIR / "train" / f"{img_id}.dicom"
        dst = IMAGES_DIR / f"{img_id}.png"
        if src.exists():
            dicom_to_multiwindow_png(src, dst)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_convert, i): i for i in remaining}
        for fut in tqdm(as_completed(futs), total=len(remaining)):
            try:
                fut.result()
            except Exception as e:
                print(f"  WARN: {futs[fut]} — {e}")
    print(f"Conversion done: {len(list(IMAGES_DIR.glob('*.png'))):,} PNGs")


def write_all_labels(all_ids, anns_map):
    """Write YOLO .txt label files (idempotent)."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for img_id in tqdm(all_ids, desc="Labels"):
        lbl = LABELS_DIR / f"{img_id}.txt"
        if not lbl.exists():
            write_yolo_label(lbl, anns_map.get(img_id, []))


# ── Fold structure ────────────────────────────────────────────────────────────
def write_fold_yaml(fold_num, train_ids, val_ids) -> str:
    fold_path = FOLD_DIR / f"fold_{fold_num:02d}"
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        si = fold_path / split / "images"
        sl = fold_path / split / "labels"
        si.mkdir(parents=True, exist_ok=True)
        sl.mkdir(parents=True, exist_ok=True)
        for img_id in ids:
            src_i = IMAGES_DIR / f"{img_id}.png"
            src_l = LABELS_DIR / f"{img_id}.txt"
            dst_i = si / f"{img_id}.png"
            dst_l = sl / f"{img_id}.txt"
            if not dst_i.exists() and src_i.exists():
                os.symlink(src_i.resolve(), dst_i)
            if not dst_l.exists():
                if src_l.exists():
                    shutil.copy2(src_l, dst_l)
                else:
                    open(dst_l, "w").close()
    yaml_path = fold_path / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(
            {"path": str(fold_path.resolve()), "train": "train/images",
             "val": "val/images", "nc": NC, "names": CLASS_NAMES},
            f, default_flow_style=False
        )
    return str(yaml_path)


def make_fold_splits(img_ids, anns_map):
    """
    Multilabel-stratified 15-fold CV.
    Stratifies on per-class presence to ensure rare classes appear in every fold.
    Falls back to binary stratification if iterstrat is unavailable.
    """
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        # Build binary presence matrix: shape (N_images, 14)
        label_matrix = np.zeros((len(img_ids), NC), dtype=int)
        for i, img_id in enumerate(img_ids):
            for (cid, *_) in anns_map.get(img_id, []):
                if 0 <= cid < NC:
                    label_matrix[i, cid] = 1
        mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(mskf.split(img_ids, label_matrix))
        print(f"Using MultilabelStratifiedKFold (iterstrat) — all 14 classes balanced across folds.")
    except ImportError:
        from sklearn.model_selection import StratifiedKFold
        has_find = np.array([1 if anns_map.get(i) else 0 for i in img_ids])
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(skf.split(img_ids, has_find))
        print("WARNING: iterstrat not found, falling back to binary StratifiedKFold.")
    return splits


# ── Stage 1: YOLO26x 15-fold CV ───────────────────────────────────────────────
def stage1(batch: int, start_fold: int, workers: int):
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f" STAGE 1 — YOLO26x  |  {N_FOLDS}-fold CV  |  {EPOCHS} epochs/fold  |  {IMG_SIZE}px")
    print(f"{'='*60}\n")

    train_data = pd.read_csv(ROOT_DIR / "train.csv")
    all_ids    = train_data["image_id"].unique()
    anns_map   = load_consensus_annotations(train_data)

    convert_all_dicoms(all_ids, workers=workers)
    write_all_labels(all_ids, anns_map)

    img_ids = np.array(all_ids)
    splits  = make_fold_splits(img_ids, anns_map)

    results = []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        fold_num = fi + 1
        if fold_num < start_fold:
            print(f"Skipping fold {fold_num} (< --start-fold {start_fold})")
            continue

        yaml_path = write_fold_yaml(fold_num, img_ids[tr_idx], img_ids[va_idx])
        print(f"\n--- Fold {fold_num:02d}/{N_FOLDS} | train={len(tr_idx):,} val={len(va_idx):,} ---")

        model = YOLO(STAGE1_MODEL)
        res = model.train(
            data=yaml_path,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=batch,
            optimizer="MuSGD",
            end2end=True,
            amp=True,
            device=device,
            workers=workers,
            project=str(RUNS_DIR / "stage1"),
            name=f"fold_{fold_num:02d}",
            exist_ok=True,
            lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
            warmup_epochs=3.0,
            box=7.5, cls=1.0,        # cls=1.0 (up from 0.5) helps rare class discrimination
            # Augmentation — safe for multi-window Hounsfield RGB
            hsv_h=0.015,
            hsv_s=0.2,               # was 0.7 — reduced to protect HU channel encoding
            hsv_v=0.4,
            flipud=0.0,              # vertical flip breaks chest anatomy
            fliplr=0.5,
            mosaic=1.0,
            translate=0.1,
            scale=0.5,
            close_mosaic=10,
            plots=True,
            verbose=True,
            save=True,
            save_period=10,
            val=True,
            cache=False,
            seed=SEED + fi,
        )
        rd      = res.results_dict if hasattr(res, "results_dict") else {}
        map50   = rd.get("metrics/mAP50(B)", None)
        map5095 = rd.get("metrics/mAP50-95(B)", None)
        best_pt = str(RUNS_DIR / "stage1" / f"fold_{fold_num:02d}" / "weights" / "best.pt")
        results.append({"fold": fold_num, "map50": map50, "map50_95": map5095, "best": best_pt})
        print(f"Fold {fold_num:02d} → mAP50={map50}  mAP50-95={map5095}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    out = WORK_DIR / "stage1_results.csv"
    df.to_csv(out, index=False)
    print(f"\nStage 1 done. Mean mAP50={df['map50'].mean():.4f}  → {out}")


# ── Stage 2: RT-DETR-X 15-fold CV ─────────────────────────────────────────────
def stage2(batch: int, start_fold: int, workers: int):
    from ultralytics import RTDETR
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f" STAGE 2 — RT-DETR-X  |  {N_FOLDS}-fold CV  |  {EPOCHS} epochs/fold  |  {IMG_SIZE}px")
    print(f"{'='*60}\n")

    train_data = pd.read_csv(ROOT_DIR / "train.csv")
    all_ids    = train_data["image_id"].unique()
    anns_map   = load_consensus_annotations(train_data)

    convert_all_dicoms(all_ids, workers=workers)
    write_all_labels(all_ids, anns_map)

    img_ids = np.array(all_ids)
    splits  = make_fold_splits(img_ids, anns_map)

    results = []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        fold_num = fi + 1
        if fold_num < start_fold:
            continue

        yaml_path = write_fold_yaml(fold_num, img_ids[tr_idx], img_ids[va_idx])
        print(f"\n--- Fold {fold_num:02d}/{N_FOLDS} ---")

        model = RTDETR(STAGE2_MODEL)
        res = model.train(
            data=yaml_path,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=batch,
            optimizer="AdamW",
            amp=True,
            device=device,
            workers=workers,
            project=str(RUNS_DIR / "stage2"),
            name=f"fold_{fold_num:02d}",
            exist_ok=True,
            lr0=1e-4,
            weight_decay=0.0001,
            warmup_epochs=2.0,
            box=7.5, cls=1.0,
            hsv_h=0.015,
            hsv_s=0.2,               # safe for HU encoding
            hsv_v=0.4,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.5,
            translate=0.1,
            scale=0.5,
            close_mosaic=5,
            plots=True,
            verbose=True,
            save=True,
            save_period=10,
            val=True,
            cache=False,
            seed=SEED + fi,
        )
        rd      = res.results_dict if hasattr(res, "results_dict") else {}
        map50   = rd.get("metrics/mAP50(B)", None)
        map5095 = rd.get("metrics/mAP50-95(B)", None)
        best_pt = str(RUNS_DIR / "stage2" / f"fold_{fold_num:02d}" / "weights" / "best.pt")
        results.append({"fold": fold_num, "map50": map50, "map50_95": map5095, "best": best_pt})
        print(f"Fold {fold_num:02d} → mAP50={map50}  mAP50-95={map5095}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    out = WORK_DIR / "stage2_results.csv"
    df.to_csv(out, index=False)
    print(f"\nStage 2 done. Mean mAP50={df['map50'].mean():.4f}  → {out}")


# ── Stage 3: WBF Ensemble submission ──────────────────────────────────────────
def stage_ensemble(conf: float, iou_thr: float, workers: int):
    """
    Inference across all stage1 + stage2 fold models.
    WBF merge with iou_thr=0.4 (matches competition's lowest eval IoU).
    """
    from ensemble_boxes import weighted_boxes_fusion
    from ultralytics import YOLO, RTDETR

    device   = 0 if torch.cuda.is_available() else "cpu"
    test_dir = ROOT_DIR / "test"

    # Convert test DICOMs to PNG
    test_png_dir = YOLO_DIR / "test_images"
    test_png_dir.mkdir(parents=True, exist_ok=True)
    test_dcms = list(test_dir.glob("*.dicom"))
    remaining = [d for d in test_dcms if not (test_png_dir / f"{d.stem}.png").exists()]
    if remaining:
        print(f"Converting {len(remaining)} test DICOMs ...")
        def _conv(dcm):
            dicom_to_multiwindow_png(dcm, test_png_dir / f"{dcm.stem}.png")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in tqdm(as_completed({ex.submit(_conv, d): d for d in remaining}),
                            total=len(remaining)):
                fut.result()
    else:
        print(f"All {len(test_dcms)} test PNGs already converted.")

    # Gather all checkpoint paths
    stage1_ckpts = sorted((RUNS_DIR / "stage1").glob("fold_*/weights/best.pt"))
    stage2_ckpts = sorted((RUNS_DIR / "stage2").glob("fold_*/weights/best.pt"))
    all_ckpts = list(stage1_ckpts) + list(stage2_ckpts)
    if not all_ckpts:
        sys.exit("No checkpoints found. Run stage1 and/or stage2 first.")
    print(f"Ensemble: {len(all_ckpts)} models ({len(stage1_ckpts)} YOLO26x + {len(stage2_ckpts)} RT-DETR-X)")

    # Per-model validation mAP for weighted ensemble (load from results CSV if available)
    model_weights = _load_model_weights(stage1_ckpts, stage2_ckpts)
    print(f"Model weights: {[f'{w:.3f}' for w in model_weights]}")

    # Collect predictions from all models
    all_preds: dict[str, list] = {}
    for idx, ckpt in enumerate(all_ckpts):
        # Infer model type from path
        stage = "stage1" if "stage1" in str(ckpt) else "stage2"
        ModelClass = YOLO if stage == "stage1" else RTDETR
        model = ModelClass(str(ckpt))
        preds = model.predict(
            source=str(test_png_dir), imgsz=IMG_SIZE, conf=0.05,
            iou=0.5, device=device, verbose=False, save=False,
        )
        for res in preds:
            img_id = Path(res.path).stem
            if img_id not in all_preds:
                all_preds[img_id] = []
            if res.boxes and len(res.boxes):
                xyxyn  = res.boxes.xyxyn.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                labels = res.boxes.cls.cpu().numpy().astype(int)
                all_preds[img_id].append((
                    xyxyn.tolist(), scores.tolist(), labels.tolist(),
                    model_weights[idx]
                ))
            else:
                all_preds[img_id].append(([], [], [], model_weights[idx]))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # WBF per image
    rows = []
    for img_id, pred_list in tqdm(all_preds.items(), desc="WBF"):
        png = test_png_dir / f"{img_id}.png"
        w, h = Image.open(png).size if png.exists() else (1024, 1024)

        valid = [(b, s, l, wt) for b, s, l, wt in pred_list if b]
        if not valid:
            rows.append({"image_id": img_id, "PredictionString": "14 1.0 0 0 1 1"})
            continue

        boxes_list  = [p[0] for p in valid]
        scores_list = [p[1] for p in valid]
        labels_list = [p[2] for p in valid]
        weights     = [p[3] for p in valid]

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights,
            iou_thr=iou_thr,        # 0.4 — aligns with competition's lowest eval IoU
            skip_box_thr=0.05,
            conf_type="weighted_avg",
        )

        keep = fused_scores >= conf
        if not keep.any():
            rows.append({"image_id": img_id, "PredictionString": "14 1.0 0 0 1 1"})
        else:
            parts = []
            for (x1n, y1n, x2n, y2n), sc, cl in zip(
                fused_boxes[keep], fused_scores[keep], fused_labels[keep]
            ):
                parts.append(
                    f"{sc:.4f} {int(cl)} {int(x1n*w)} {int(y1n*h)} {int(x2n*w)} {int(y2n*h)}"
                )
            rows.append({"image_id": img_id, "PredictionString": " ".join(parts)})

    sub = pd.DataFrame(rows)
    out = WORK_DIR / "submission_ensemble.csv"
    sub.to_csv(out, index=False)
    print(f"\nSubmission saved: {len(sub)} rows → {out}")


def _load_model_weights(stage1_ckpts, stage2_ckpts) -> list[float]:
    """
    Load per-model weights from validation mAP results CSVs.
    Falls back to equal weights if CSVs not available.
    """
    def get_weights(ckpts, csv_path):
        if not csv_path.exists() or not ckpts:
            return [1.0] * len(ckpts)
        try:
            df = pd.read_csv(csv_path)
            # Match fold number from checkpoint path
            weights = []
            for ckpt in ckpts:
                fold_str = ckpt.parent.parent.name  # e.g. "fold_03"
                fold_num = int(fold_str.split("_")[1])
                row = df[df["fold"] == fold_num]
                if len(row) and pd.notna(row.iloc[0]["map50"]):
                    weights.append(float(row.iloc[0]["map50"]))
                else:
                    weights.append(1.0)
            return weights
        except Exception:
            return [1.0] * len(ckpts)

    w1 = get_weights(stage1_ckpts, WORK_DIR / "stage1_results.csv")
    w2 = get_weights(stage2_ckpts, WORK_DIR / "stage2_results.csv")
    return w1 + w2


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage",      type=str, default="1",
                    choices=["1", "2", "ensemble"])
    ap.add_argument("--batch",      type=int, default=4,
                    help="Batch size (default 4 for 1024px on single GPU)")
    ap.add_argument("--start-fold", type=int, default=1,
                    help="Resume from this fold number")
    ap.add_argument("--workers",    type=int, default=8)
    ap.add_argument("--conf",       type=float, default=0.25,
                    help="Final confidence threshold for ensemble predictions")
    ap.add_argument("--iou-thr",    type=float, default=0.4,
                    help="WBF IoU threshold (default 0.4, matches competition metric)")
    args = ap.parse_args()

    if args.stage == "1":
        stage1(args.batch, args.start_fold, args.workers)
    elif args.stage == "2":
        stage2(args.batch, args.start_fold, args.workers)
    elif args.stage == "ensemble":
        stage_ensemble(args.conf, args.iou_thr, args.workers)


if __name__ == "__main__":
    main()
