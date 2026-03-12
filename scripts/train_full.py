"""
train_full.py — Full competition training pipeline.

Strategy (run in order):
  Stage 1: 15-fold CV with yolo26x  (strong single-model baseline)
  Stage 2: RT-DETR-X ensemble       (transformer-based, complementary errors)
  Stage 3: WBF ensemble of both     (weighted box fusion at inference)

Usage:
    # Stage 1 only (recommended starting point after smoke test)
    PYTHONPATH=. python scripts/train_full.py --stage 1

    # Stage 2 (train RT-DETR after stage 1 is done)
    PYTHONPATH=. python scripts/train_full.py --stage 2

    # Inference + WBF ensemble
    PYTHONPATH=. python scripts/train_full.py --stage ensemble
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
import pydicom
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from ultralytics import YOLO

from src.data.dicom_utils import dicom_to_multiwindow_png

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR  = Path("data/raw/vinbigdata-chest-xray-abnormalities-detection")
WORK_DIR  = Path("outputs")
YOLO_DIR  = WORK_DIR / "yolo_dataset"
RUNS_DIR  = WORK_DIR / "runs"
FOLD_DIR  = WORK_DIR / "folds"
IMAGES_DIR = YOLO_DIR / "images"
LABELS_DIR = YOLO_DIR / "labels"

N_FOLDS  = 15
IMG_SIZE = 640
EPOCHS   = 50
SEED     = 42

CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification",
    "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
    "Lung Opacity", "Nodule/Mass", "Other lesion",
    "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
]
CLASS2ID = {n: i for i, n in enumerate(CLASS_NAMES)}
NC = len(CLASS_NAMES)

# Stage 1 model — yolo26x (best YOLO26 variant)
# Stage 2 model — rtdetr-x (transformer, strong complementary model)
STAGE1_MODEL = "yolo26x.pt"
STAGE2_MODEL = "rtdetr-x.pt"


# ── Shared helpers ────────────────────────────────────────────────────────────
def dicom_to_png(src: Path, dst: Path):
    """Multi-window 3-channel PNG: R=lung, G=mediastinum, B=soft-tissue."""
    dicom_to_multiwindow_png(src, dst)


def write_yolo_label(path: Path, anns: list):
    with open(path, "w") as f:
        for ann in anns:
            f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")


def load_annotations(train_data: pd.DataFrame) -> dict:
    """Build image_id → [(cls, xc, yc, bw, bh)] mapping with consensus merging."""
    findings = train_data[train_data["class_name"] != "No finding"]
    dims = (
        train_data.groupby("image_id")[["width", "height"]].first()
        if {"width", "height"} <= set(train_data.columns) else None
    )
    all_ids = train_data["image_id"].unique()
    anns_map = {i: [] for i in all_ids}

    for img_id, grp in findings.groupby("image_id"):
        if dims is not None and img_id in dims.index:
            iw, ih = float(dims.loc[img_id, "width"]), float(dims.loc[img_id, "height"])
        else:
            ds = pydicom.dcmread(str(ROOT_DIR / "train" / f"{img_id}.dicom"))
            ih, iw = ds.pixel_array.shape[:2]
        anns = []
        for _, r in grp.iterrows():
            cid = CLASS2ID.get(r["class_name"])
            if cid is None:
                continue
            x1, y1, x2, y2 = float(r.x_min), float(r.y_min), float(r.x_max), float(r.y_max)
            xc = (x1 + x2) / 2 / iw
            yc = (y1 + y2) / 2 / ih
            bw = (x2 - x1) / iw
            bh = (y2 - y1) / ih
            anns.append((cid, *[max(0., min(1., v)) for v in [xc, yc, bw, bh]]))
        anns_map[img_id] = anns
    return anns_map


def convert_all_dicoms(all_ids, workers=8):
    """Convert all training DICOMs to PNG (idempotent)."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    already = len(list(IMAGES_DIR.glob("*.png")))
    remaining = [i for i in all_ids if not (IMAGES_DIR / f"{i}.png").exists()]
    if not remaining:
        print(f"All {len(all_ids)} PNGs already exist.")
        return
    print(f"Converting {len(remaining)} DICOMs (already done: {already}) ...")
    def _convert(img_id):
        dicom_to_png(
            ROOT_DIR / "train" / f"{img_id}.dicom",
            IMAGES_DIR / f"{img_id}.png",
        )
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_convert, i): i for i in remaining}
        for fut in tqdm(as_completed(futs), total=len(remaining)):
            try:
                fut.result()
            except Exception as e:
                print(f"  WARN: {futs[fut]} — {e}")
    print(f"Conversion done: {len(list(IMAGES_DIR.glob('*.png')))} PNGs")


def write_all_labels(all_ids, anns_map):
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for img_id in tqdm(all_ids, desc="Labels"):
        write_yolo_label(LABELS_DIR / f"{img_id}.txt", anns_map[img_id])


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


# ── Stage 1: YOLO26x 15-fold CV ───────────────────────────────────────────────
def stage1(batch: int, start_fold: int, workers: int):
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f" STAGE 1 — YOLO26x  |  15-fold CV  |  {EPOCHS} epochs/fold")
    print(f"{'='*60}\n")

    train_data = pd.read_csv(ROOT_DIR / "train.csv")
    all_ids    = train_data["image_id"].unique()
    anns_map   = load_annotations(train_data)

    convert_all_dicoms(all_ids, workers=workers)
    write_all_labels(all_ids, anns_map)

    img_ids  = np.array(all_ids)
    has_find = np.array([1 if anns_map[i] else 0 for i in img_ids])
    skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits   = list(skf.split(img_ids, has_find))

    results = []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        fold_num = fi + 1
        if fold_num < start_fold:
            print(f"Skipping fold {fold_num} (< --start-fold {start_fold})")
            continue

        yaml_path = write_fold_yaml(fold_num, img_ids[tr_idx], img_ids[va_idx])
        print(f"\n--- Fold {fold_num:02d}/{N_FOLDS} | "
              f"train={len(tr_idx):,} val={len(va_idx):,} ---")

        model = YOLO(STAGE1_MODEL)
        res = model.train(
            data=yaml_path, epochs=EPOCHS, imgsz=IMG_SIZE, batch=batch,
            optimizer="MuSGD", end2end=True, amp=True, device=device,
            workers=workers, project=str(RUNS_DIR / "stage1"),
            name=f"fold_{fold_num:02d}", exist_ok=True,
            lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
            warmup_epochs=3.0, box=7.5, cls=0.5,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            flipud=0.0, fliplr=0.5, mosaic=1.0, translate=0.1, scale=0.5,
            close_mosaic=10, plots=True, verbose=True,
            save=True, save_period=10, val=True, cache=False,
            seed=SEED + fi,
        )
        rd      = res.results_dict if hasattr(res, "results_dict") else {}
        map50   = rd.get("metrics/mAP50(B)", None)
        map5095 = rd.get("metrics/mAP50-95(B)", None)
        results.append({"fold": fold_num, "map50": map50, "map50_95": map5095,
                         "best": str(RUNS_DIR / "stage1" / f"fold_{fold_num:02d}" / "weights" / "best.pt")})
        print(f"Fold {fold_num:02d} → mAP50={map50}  mAP50-95={map5095}")

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    df = pd.DataFrame(results)
    out = WORK_DIR / "stage1_results.csv"
    df.to_csv(out, index=False)
    print(f"\nStage 1 done. Mean mAP50={df['map50'].mean():.4f}  → {out}")


# ── Stage 2: RT-DETR-X 15-fold CV ─────────────────────────────────────────────
def stage2(batch: int, start_fold: int, workers: int):
    """
    RT-DETR-X: transformer-based detector from ultralytics.
    Complementary to YOLO26 — different backbone/architecture → different errors.
    WBF ensemble of both stages typically gives +2-4 mAP points.
    """
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f" STAGE 2 — RT-DETR-X  |  15-fold CV  |  {EPOCHS} epochs/fold")
    print(f"{'='*60}\n")

    train_data = pd.read_csv(ROOT_DIR / "train.csv")
    all_ids    = train_data["image_id"].unique()
    anns_map   = load_annotations(train_data)

    # Images/labels already exist from stage 1 (idempotent)
    convert_all_dicoms(all_ids, workers=workers)
    write_all_labels(all_ids, anns_map)

    img_ids  = np.array(all_ids)
    has_find = np.array([1 if anns_map[i] else 0 for i in img_ids])
    skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits   = list(skf.split(img_ids, has_find))

    results = []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        fold_num = fi + 1
        if fold_num < start_fold:
            continue

        yaml_path = write_fold_yaml(fold_num, img_ids[tr_idx], img_ids[va_idx])
        print(f"\n--- Fold {fold_num:02d}/{N_FOLDS} ---")

        model = YOLO(STAGE2_MODEL)
        res = model.train(
            data=yaml_path, epochs=EPOCHS, imgsz=IMG_SIZE, batch=batch,
            optimizer="AdamW", amp=True, device=device, workers=workers,
            project=str(RUNS_DIR / "stage2"), name=f"fold_{fold_num:02d}",
            exist_ok=True, lr0=1e-4, weight_decay=0.0001,
            warmup_epochs=2.0, box=7.5, cls=0.5,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            flipud=0.0, fliplr=0.5, mosaic=0.5, translate=0.1, scale=0.5,
            close_mosaic=5, plots=True, verbose=True,
            save=True, save_period=10, val=True, cache=False,
            seed=SEED + fi,
        )
        rd      = res.results_dict if hasattr(res, "results_dict") else {}
        map50   = rd.get("metrics/mAP50(B)", None)
        map5095 = rd.get("metrics/mAP50-95(B)", None)
        results.append({"fold": fold_num, "map50": map50, "map50_95": map5095,
                         "best": str(RUNS_DIR / "stage2" / f"fold_{fold_num:02d}" / "weights" / "best.pt")})
        print(f"Fold {fold_num:02d} → mAP50={map50}  mAP50-95={map5095}")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    df = pd.DataFrame(results)
    out = WORK_DIR / "stage2_results.csv"
    df.to_csv(out, index=False)
    print(f"\nStage 2 done. Mean mAP50={df['map50'].mean():.4f}  → {out}")


# ── Stage 3: WBF Ensemble submission ──────────────────────────────────────────
def stage_ensemble(conf: float, iou_thr: float, workers: int):
    """
    Run inference with all stage1 + stage2 fold models,
    merge predictions with Weighted Box Fusion (WBF),
    write submission.csv.
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        sys.exit("pip install ensemble-boxes")

    device   = 0 if torch.cuda.is_available() else "cpu"
    test_dir = ROOT_DIR / "test"

    # Convert test DICOMs
    test_png_dir = YOLO_DIR / "test_images"
    test_png_dir.mkdir(parents=True, exist_ok=True)
    test_dcms = list(test_dir.glob("*.dicom"))
    print(f"Converting {len(test_dcms)} test DICOMs ...")
    for dcm in tqdm(test_dcms):
        dst = test_png_dir / f"{dcm.stem}.png"
        if not dst.exists():
            dicom_to_png(dcm, dst)

    # Gather all checkpoint paths
    ckpts = []
    for stage in ["stage1", "stage2"]:
        ckpts += sorted((RUNS_DIR / stage).glob("fold_*/weights/best.pt"))
    if not ckpts:
        sys.exit("No checkpoints found. Run stage1 and/or stage2 first.")
    print(f"Ensemble: {len(ckpts)} models")

    # Run inference per model, collect boxes
    all_preds: dict[str, list] = {}   # image_id → list of (boxes, scores, labels)
    for ckpt in ckpts:
        model = YOLO(str(ckpt))
        preds = model.predict(
            source=str(test_png_dir), imgsz=IMG_SIZE, conf=conf,
            iou=0.5, device=device, verbose=False, save=False, end2end=False,
        )
        for res in preds:
            img_id = Path(res.path).stem
            if img_id not in all_preds:
                all_preds[img_id] = []
            if res.boxes and len(res.boxes):
                h, w = res.orig_shape
                boxes, scores, labels = [], [], []
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append([x1/w, y1/h, x2/w, y2/h])
                    scores.append(float(box.conf))
                    labels.append(int(box.cls))
                all_preds[img_id].append((boxes, scores, labels))
        del model; gc.collect(); torch.cuda.empty_cache()

    # WBF per image
    rows = []
    for img_id, pred_list in tqdm(all_preds.items(), desc="WBF"):
        # Get image size for denormalisation
        png = test_png_dir / f"{img_id}.png"
        w, h = Image.open(png).size if png.exists() else (1024, 1024)

        if not pred_list:
            rows.append({"image_id": img_id, "PredictionString": "14 1.0 0 0 1 1"})
            continue

        boxes_list  = [p[0] for p in pred_list]
        scores_list = [p[1] for p in pred_list]
        labels_list = [p[2] for p in pred_list]

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=iou_thr, skip_box_thr=conf,
        )

        if len(fused_boxes) == 0:
            rows.append({"image_id": img_id, "PredictionString": "14 1.0 0 0 1 1"})
        else:
            parts = []
            for (x1n, y1n, x2n, y2n), sc, cl in zip(fused_boxes, fused_scores, fused_labels):
                parts.append(f"{int(cl)} {sc:.4f} {int(x1n*w)} {int(y1n*h)} {int(x2n*w)} {int(y2n*h)}")
            rows.append({"image_id": img_id, "PredictionString": " ".join(parts)})

    sub = pd.DataFrame(rows)
    out = WORK_DIR / "submission_ensemble.csv"
    sub.to_csv(out, index=False)
    print(f"\nSubmission saved: {len(sub)} rows → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage",      type=str, default="1",
                    choices=["1", "2", "ensemble"],
                    help="Which stage to run")
    ap.add_argument("--batch",      type=int, default=16)
    ap.add_argument("--start-fold", type=int, default=1,
                    help="Resume from this fold number (skip earlier folds)")
    ap.add_argument("--workers",    type=int, default=8)
    ap.add_argument("--conf",       type=float, default=0.25,
                    help="Confidence threshold for ensemble inference")
    ap.add_argument("--iou-thr",    type=float, default=0.5,
                    help="IoU threshold for WBF")
    args = ap.parse_args()

    if args.stage == "1":
        stage1(args.batch, args.start_fold, args.workers)
    elif args.stage == "2":
        stage2(args.batch, args.start_fold, args.workers)
    elif args.stage == "ensemble":
        stage_ensemble(args.conf, args.iou_thr, args.workers)


if __name__ == "__main__":
    main()
