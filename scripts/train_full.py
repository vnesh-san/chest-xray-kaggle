"""
train_full.py — Full competition training pipeline.

6-channel dual-YOLO + RT-DETR ensemble strategy:

  Window Set A (images_A/):              Window Set B (images_B/):
    lung        (-600, 1500)               bone      ( 400, 1500)
    mediastinum (  40,  400)               pleural   (  50,  350)
    soft tissue (  80,  800)               vascular  ( 100,  700)
         ↓                                      ↓
  YOLO26x-A × 15 folds              YOLO26x-B × 15 folds
  (ILD, cardiomegaly,                (calcification,
   pneumothorax, infiltration,        pleural thickening,
   lung opacity, nodule)              aortic enlargement)
         ↓                                      ↓
                 RT-DETR-X-B × 15 folds
                 (transformer on bone/pleural/vascular view —
                  maximally different from YOLO26x-A)
                        ↓
              WBF ensemble (iou_thr=0.4)
              45 total checkpoints → submission

Key design decisions:
  - No architecture changes: all models use standard 3-ch RGB input
  - Full ImageNet pretrained weights preserved on both YOLO instances
  - Window Set B targets the 3 rare/hardest classes specifically
  - RT-DETR on Set B maximises ensemble diversity (different arch + different view)
  - Consensus labels: WBF-merge radiologist boxes, min_votes=2
  - Multilabel-stratified 15-fold CV (all 14 classes balanced across folds)
  - 1024px for small lesion recall
  - hsv_s=0.2 (safe for Hounsfield window encoding)

Usage:
    # Step 0: Build consensus labels + dims cache (run once, ~30s)
    PYTHONPATH=. python scripts/prepare_labels.py

    # Step 1: Convert all DICOMs to window-set PNGs (run once per set, ~20min each)
    PYTHONPATH=. python scripts/train_full.py --stage convert --window-set A
    PYTHONPATH=. python scripts/train_full.py --stage convert --window-set B

    # Stage 1: YOLO26x on window set A
    PYTHONPATH=. python scripts/train_full.py --stage 1 --window-set A

    # Stage 2: YOLO26x on window set B
    PYTHONPATH=. python scripts/train_full.py --stage 2 --window-set B

    # Stage 3: RT-DETR-X on window set B
    PYTHONPATH=. python scripts/train_full.py --stage 3 --window-set B

    # Stage ensemble: WBF across all 45 checkpoints
    PYTHONPATH=. python scripts/train_full.py --stage ensemble

    # Or run everything sequentially:
    PYTHONPATH=. python scripts/train_full.py --stage all --window-set both

    # Resume from a specific fold:
    PYTHONPATH=. python scripts/train_full.py --stage 1 --window-set A --start-fold 4
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

from src.data.dicom_utils import dicom_to_multiwindow_png, WINDOW_SETS
from src.data.label_consensus import build_consensus_labels, build_dims_cache

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path("data/raw/vinbigdata-chest-xray-abnormalities-detection")
WORK_DIR   = Path("outputs")
YOLO_DIR   = WORK_DIR / "yolo_dataset"
RUNS_DIR   = WORK_DIR / "runs"
FOLD_DIR   = WORK_DIR / "folds"
LABELS_DIR = YOLO_DIR / "labels"
DIMS_CACHE = str(WORK_DIR / "image_dims.json")

N_FOLDS    = 15
IMG_SIZE   = 1024
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

STAGE1_MODEL = "yolo26x.pt"    # window set A
STAGE2_MODEL = "yolo26x.pt"    # window set B
STAGE3_MODEL = "rtdetr-x.pt"   # window set B (transformer, max diversity)


# ── Helpers ───────────────────────────────────────────────────────────────────
def images_dir(window_set: str) -> Path:
    return YOLO_DIR / f"images_{window_set}"


def fold_dir(fold_num: int, window_set: str) -> Path:
    return FOLD_DIR / f"fold_{fold_num:02d}_{window_set}"


def write_yolo_label(path: Path, anns: list):
    with open(path, "w") as f:
        for cls, xc, yc, bw, bh in anns:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def load_consensus_annotations(train_data: pd.DataFrame, workers: int = 8) -> dict:
    print("Building consensus labels (WBF merge, min_votes=2, iou_thr=0.4) ...")
    all_ids = train_data["image_id"].unique().tolist()
    dims = build_dims_cache(all_ids, workers=workers * 4, cache_path=DIMS_CACHE)
    anns_map = build_consensus_labels(
        train_data, iou_thr=0.4, min_votes=2,
        workers=workers, dims_cache_path=DIMS_CACHE,
    )
    n_pos   = sum(1 for v in anns_map.values() if v)
    n_boxes = sum(len(v) for v in anns_map.values())
    print(f"  {n_pos:,} positive images, {n_boxes:,} consensus boxes")
    return anns_map


# ── DICOM → PNG conversion ────────────────────────────────────────────────────
def convert_all_dicoms(all_ids, window_set: str, workers: int = 8):
    """Convert all training DICOMs to multi-window PNG for given window set (idempotent)."""
    out_dir = images_dir(window_set)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = WINDOW_SETS[window_set]
    remaining = [i for i in all_ids if not (out_dir / f"{i}.png").exists()]
    if not remaining:
        print(f"Window set {window_set}: all {len(all_ids):,} PNGs already exist.")
        return
    print(f"Converting {len(remaining):,} DICOMs → window set {window_set} PNGs ...")

    def _convert(img_id):
        src = ROOT_DIR / "train" / f"{img_id}.dicom"
        dst = out_dir / f"{img_id}.png"
        if src.exists():
            dicom_to_multiwindow_png(src, dst, windows=windows)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_convert, i): i for i in remaining}
        for fut in tqdm(as_completed(futs), total=len(remaining)):
            try:
                fut.result()
            except Exception as e:
                print(f"  WARN: {futs[fut]} — {e}")
    print(f"  Done: {len(list(out_dir.glob('*.png'))):,} PNGs in {out_dir}")


def write_all_labels(all_ids, anns_map):
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    missing = [i for i in all_ids if not (LABELS_DIR / f"{i}.txt").exists()]
    if not missing:
        return
    for img_id in tqdm(missing, desc="Labels"):
        write_yolo_label(LABELS_DIR / f"{img_id}.txt", anns_map.get(img_id, []))


# ── Fold structure ────────────────────────────────────────────────────────────
def write_fold_yaml(fold_num: int, train_ids, val_ids, window_set: str) -> str:
    fp = fold_dir(fold_num, window_set)
    img_dir = images_dir(window_set)
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        si = fp / split / "images"
        sl = fp / split / "labels"
        si.mkdir(parents=True, exist_ok=True)
        sl.mkdir(parents=True, exist_ok=True)
        for img_id in ids:
            src_i = img_dir    / f"{img_id}.png"
            src_l = LABELS_DIR / f"{img_id}.txt"
            dst_i = si / f"{img_id}.png"
            dst_l = sl / f"{img_id}.txt"
            if not dst_i.exists() and src_i.exists():
                os.symlink(src_i.resolve(), dst_i)
            if not dst_l.exists():
                shutil.copy2(src_l, dst_l) if src_l.exists() else open(dst_l, "w").close()
    yaml_path = fp / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({"path": str(fp.resolve()), "train": "train/images",
                   "val": "val/images", "nc": NC, "names": CLASS_NAMES},
                  f, default_flow_style=False)
    return str(yaml_path)


def make_fold_splits(img_ids, anns_map):
    """Multilabel-stratified 15-fold CV — all 14 rare classes balanced per fold."""
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        label_matrix = np.zeros((len(img_ids), NC), dtype=int)
        for i, img_id in enumerate(img_ids):
            for (cid, *_) in anns_map.get(img_id, []):
                if 0 <= cid < NC:
                    label_matrix[i, cid] = 1
        mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(mskf.split(img_ids, label_matrix))
        print("MultilabelStratifiedKFold — all 14 classes balanced.")
    except ImportError:
        from sklearn.model_selection import StratifiedKFold
        has_find = np.array([1 if anns_map.get(i) else 0 for i in img_ids])
        splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(img_ids, has_find))
        print("WARNING: iterstrat not found, falling back to binary StratifiedKFold.")
    return splits


# ── Shared training hyperparams ───────────────────────────────────────────────
YOLO_TRAIN_KWARGS = dict(
    epochs=EPOCHS, imgsz=IMG_SIZE, optimizer="MuSGD", end2end=True,
    amp=True, workers=8,
    lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3.0, box=7.5, cls=1.0,
    hsv_h=0.015, hsv_s=0.2, hsv_v=0.4,   # hsv_s=0.2 safe for HU encoding
    flipud=0.0, fliplr=0.5,               # no vertical flip — breaks chest anatomy
    mosaic=1.0, translate=0.1, scale=0.5,
    close_mosaic=10, plots=True, verbose=True,
    save=True, save_period=10, val=True, cache=False,
)

RTDETR_TRAIN_KWARGS = dict(
    epochs=EPOCHS, imgsz=IMG_SIZE, optimizer="AdamW",
    amp=True, workers=8,
    lr0=1e-4, weight_decay=0.0001, warmup_epochs=2.0,
    box=7.5, cls=1.0,
    hsv_h=0.015, hsv_s=0.2, hsv_v=0.4,
    flipud=0.0, fliplr=0.5,
    mosaic=0.5, translate=0.1, scale=0.5,
    close_mosaic=5, plots=True, verbose=True,
    save=True, save_period=10, val=True, cache=False,
)


# ── Stage runners ─────────────────────────────────────────────────────────────
def _run_yolo_folds(model_name, window_set, stage_name, batch, start_fold, workers,
                    extra_kwargs=None):
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*65}")
    print(f" {stage_name}  |  window set {window_set}  |  {N_FOLDS}-fold  |  {EPOCHS} epochs  |  {IMG_SIZE}px")
    print(f"{'='*65}\n")

    train_data = pd.read_csv(ROOT_DIR / "train.csv")
    all_ids    = train_data["image_id"].unique()
    anns_map   = load_consensus_annotations(train_data, workers)

    convert_all_dicoms(all_ids, window_set, workers)
    write_all_labels(all_ids, anns_map)

    img_ids = np.array(all_ids)
    splits  = make_fold_splits(img_ids, anns_map)

    kwargs = {**YOLO_TRAIN_KWARGS, "batch": batch, "device": device,
              "workers": workers, **(extra_kwargs or {})}

    results = []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        fold_num = fi + 1
        if fold_num < start_fold:
            print(f"Skipping fold {fold_num}")
            continue
        yaml_path = write_fold_yaml(fold_num, img_ids[tr_idx], img_ids[va_idx], window_set)
        print(f"\n--- Fold {fold_num:02d}/{N_FOLDS} | train={len(tr_idx):,} val={len(va_idx):,} ---")

        model = YOLO(model_name)
        res = model.train(data=yaml_path,
                          project=str(RUNS_DIR / stage_name),
                          name=f"fold_{fold_num:02d}",
                          exist_ok=True, seed=SEED + fi, **kwargs)
        rd      = res.results_dict if hasattr(res, "results_dict") else {}
        map50   = rd.get("metrics/mAP50(B)", None)
        map5095 = rd.get("metrics/mAP50-95(B)", None)
        best_pt = str(RUNS_DIR / stage_name / f"fold_{fold_num:02d}" / "weights" / "best.pt")
        results.append({"fold": fold_num, "map50": map50, "map50_95": map5095,
                        "window_set": window_set, "best": best_pt})
        print(f"Fold {fold_num:02d} → mAP50={map50}  mAP50-95={map5095}")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    df = pd.DataFrame(results)
    out = WORK_DIR / f"{stage_name}_results.csv"
    df.to_csv(out, index=False)
    print(f"\n{stage_name} done. Mean mAP50={df['map50'].mean():.4f}  → {out}")


def _run_rtdetr_folds(window_set, stage_name, batch, start_fold, workers):
    from ultralytics import RTDETR
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*65}")
    print(f" {stage_name} (RT-DETR-X)  |  window set {window_set}  |  {N_FOLDS}-fold  |  {EPOCHS} epochs")
    print(f"{'='*65}\n")

    train_data = pd.read_csv(ROOT_DIR / "train.csv")
    all_ids    = train_data["image_id"].unique()
    anns_map   = load_consensus_annotations(train_data, workers)

    convert_all_dicoms(all_ids, window_set, workers)
    write_all_labels(all_ids, anns_map)

    img_ids = np.array(all_ids)
    splits  = make_fold_splits(img_ids, anns_map)

    kwargs = {**RTDETR_TRAIN_KWARGS, "batch": batch, "device": device, "workers": workers}

    results = []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        fold_num = fi + 1
        if fold_num < start_fold:
            continue
        yaml_path = write_fold_yaml(fold_num, img_ids[tr_idx], img_ids[va_idx], window_set)
        print(f"\n--- Fold {fold_num:02d}/{N_FOLDS} ---")

        model = RTDETR(STAGE3_MODEL)
        res = model.train(data=yaml_path,
                          project=str(RUNS_DIR / stage_name),
                          name=f"fold_{fold_num:02d}",
                          exist_ok=True, seed=SEED + fi, **kwargs)
        rd      = res.results_dict if hasattr(res, "results_dict") else {}
        map50   = rd.get("metrics/mAP50(B)", None)
        map5095 = rd.get("metrics/mAP50-95(B)", None)
        best_pt = str(RUNS_DIR / stage_name / f"fold_{fold_num:02d}" / "weights" / "best.pt")
        results.append({"fold": fold_num, "map50": map50, "map50_95": map5095,
                        "window_set": window_set, "best": best_pt})
        print(f"Fold {fold_num:02d} → mAP50={map50}  mAP50-95={map5095}")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    df = pd.DataFrame(results)
    out = WORK_DIR / f"{stage_name}_results.csv"
    df.to_csv(out, index=False)
    print(f"\n{stage_name} done. Mean mAP50={df['map50'].mean():.4f}  → {out}")


# ── Stage 1: YOLO26x Window Set A ─────────────────────────────────────────────
def stage1(batch, start_fold, workers):
    _run_yolo_folds(STAGE1_MODEL, "A", "stage1_yolo_A", batch, start_fold, workers)


# ── Stage 2: YOLO26x Window Set B ─────────────────────────────────────────────
def stage2(batch, start_fold, workers):
    _run_yolo_folds(STAGE2_MODEL, "B", "stage2_yolo_B", batch, start_fold, workers)


# ── Stage 3: RT-DETR-X Window Set B ───────────────────────────────────────────
def stage3(batch, start_fold, workers):
    _run_rtdetr_folds("B", "stage3_rtdetr_B", batch, start_fold, workers)


# ── Stage convert: DICOM → PNG only (no training) ─────────────────────────────
def stage_convert(window_set: str, workers: int):
    train_data = pd.read_csv(ROOT_DIR / "train.csv")
    all_ids    = train_data["image_id"].unique()
    sets = ["A", "B"] if window_set == "both" else [window_set]
    for ws in sets:
        convert_all_dicoms(all_ids, ws, workers)


# ── Stage ensemble: WBF across all 45 checkpoints ────────────────────────────
def stage_ensemble(conf: float, iou_thr: float, workers: int):
    from ensemble_boxes import weighted_boxes_fusion
    from ultralytics import YOLO, RTDETR

    device   = 0 if torch.cuda.is_available() else "cpu"
    test_dir = ROOT_DIR / "test"

    # Convert test DICOMs for both window sets
    for ws in ["A", "B"]:
        test_out = YOLO_DIR / f"test_images_{ws}"
        test_out.mkdir(parents=True, exist_ok=True)
        remaining = [d for d in test_dir.glob("*.dicom")
                     if not (test_out / f"{d.stem}.png").exists()]
        if remaining:
            print(f"Converting {len(remaining)} test DICOMs → window set {ws} ...")
            def _conv(dcm, _ws=ws):
                dicom_to_multiwindow_png(dcm, YOLO_DIR / f"test_images_{_ws}" / f"{dcm.stem}.png",
                                         windows=WINDOW_SETS[_ws])
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for fut in tqdm(as_completed({ex.submit(_conv, d): d for d in remaining}),
                                total=len(remaining)):
                    fut.result()
        else:
            print(f"Test PNGs (set {ws}) already converted.")

    # Gather all checkpoints + metadata
    stage_configs = [
        ("stage1_yolo_A", "A", YOLO),
        ("stage2_yolo_B", "B", YOLO),
        ("stage3_rtdetr_B", "B", RTDETR),
    ]

    all_ckpts = []   # list of (ckpt_path, window_set, ModelClass, weight)
    for stage_name, ws, ModelClass in stage_configs:
        csv_path = WORK_DIR / f"{stage_name}_results.csv"
        ckpts = sorted((RUNS_DIR / stage_name).glob("fold_*/weights/best.pt"))
        for ckpt in ckpts:
            fold_num = int(ckpt.parent.parent.name.split("_")[1])
            weight = 1.0
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    row = df[df["fold"] == fold_num]
                    if len(row) and pd.notna(row.iloc[0]["map50"]):
                        weight = float(row.iloc[0]["map50"])
                except Exception:
                    pass
            all_ckpts.append((ckpt, ws, ModelClass, weight))

    if not all_ckpts:
        sys.exit("No checkpoints found. Run stages 1, 2, 3 first.")
    print(f"Ensemble: {len(all_ckpts)} checkpoints  "
          f"({sum(1 for _,ws,_,_ in all_ckpts if ws=='A')} set-A, "
          f"{sum(1 for _,ws,_,_ in all_ckpts if ws=='B')} set-B)")

    # Run inference per model
    all_preds: dict[str, list] = {}
    for ckpt, ws, ModelClass, wt in tqdm(all_ckpts, desc="Inference"):
        test_src = str(YOLO_DIR / f"test_images_{ws}")
        model = ModelClass(str(ckpt))
        preds = model.predict(source=test_src, imgsz=IMG_SIZE, conf=0.05,
                              iou=0.5, device=device, verbose=False, save=False)
        for res in preds:
            img_id = Path(res.path).stem
            if img_id not in all_preds:
                all_preds[img_id] = []
            if res.boxes and len(res.boxes):
                all_preds[img_id].append((
                    res.boxes.xyxyn.cpu().numpy().tolist(),
                    res.boxes.conf.cpu().numpy().tolist(),
                    res.boxes.cls.cpu().numpy().astype(int).tolist(),
                    wt,
                ))
            else:
                all_preds[img_id].append(([], [], [], wt))
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # WBF per image
    rows = []
    for img_id, pred_list in tqdm(all_preds.items(), desc="WBF"):
        png = YOLO_DIR / f"test_images_A/{img_id}.png"
        w, h = Image.open(png).size if png.exists() else (1024, 1024)

        valid = [(b, s, l, wt) for b, s, l, wt in pred_list if b]
        if not valid:
            rows.append({"image_id": img_id, "PredictionString": "14 1.0 0 0 1 1"})
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            [p[0] for p in valid], [p[1] for p in valid], [p[2] for p in valid],
            weights=[p[3] for p in valid],
            iou_thr=iou_thr, skip_box_thr=0.05, conf_type="weighted_avg",
        )
        keep = fused_scores >= conf
        if not keep.any():
            rows.append({"image_id": img_id, "PredictionString": "14 1.0 0 0 1 1"})
        else:
            parts = [f"{sc:.4f} {int(cl)} {int(x1*w)} {int(y1*h)} {int(x2*w)} {int(y2*h)}"
                     for (x1, y1, x2, y2), sc, cl in zip(
                         fused_boxes[keep], fused_scores[keep], fused_labels[keep])]
            rows.append({"image_id": img_id, "PredictionString": " ".join(parts)})

    sub = pd.DataFrame(rows)
    out = WORK_DIR / "submission_ensemble.csv"
    sub.to_csv(out, index=False)
    print(f"\nSubmission saved: {len(sub)} rows → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, default="1",
                    choices=["convert", "1", "2", "3", "ensemble", "all"])
    ap.add_argument("--window-set", type=str, default="A",
                    choices=["A", "B", "both"],
                    help="Window set to use (A=lung/mediastinum/softtissue, "
                         "B=bone/pleural/vascular, both=run all)")
    ap.add_argument("--batch",      type=int,   default=4)
    ap.add_argument("--start-fold", type=int,   default=1)
    ap.add_argument("--workers",    type=int,   default=8)
    ap.add_argument("--conf",       type=float, default=0.25)
    ap.add_argument("--iou-thr",    type=float, default=0.4)
    args = ap.parse_args()

    if args.stage == "convert":
        stage_convert(args.window_set, args.workers)
    elif args.stage == "1":
        stage1(args.batch, args.start_fold, args.workers)
    elif args.stage == "2":
        stage2(args.batch, args.start_fold, args.workers)
    elif args.stage == "3":
        stage3(args.batch, args.start_fold, args.workers)
    elif args.stage == "ensemble":
        stage_ensemble(args.conf, args.iou_thr, args.workers)
    elif args.stage == "all":
        stage_convert(args.window_set if args.window_set != "both" else "both", args.workers)
        stage1(args.batch, args.start_fold, args.workers)
        stage2(args.batch, args.start_fold, args.workers)
        stage3(args.batch, args.start_fold, args.workers)
        stage_ensemble(args.conf, args.iou_thr, args.workers)


if __name__ == "__main__":
    main()
