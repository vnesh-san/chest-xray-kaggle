"""
smoke_test.py — Validate the full pipeline in ~5 min.

Usage:
    python scripts/smoke_test.py               # yolo26n, 500 images, 1 epoch
    python scripts/smoke_test.py --n 1000      # bigger subset
    python scripts/smoke_test.py --model yolo26x --epochs 3

After this passes, run full training:
    python scripts/train_full.py
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
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm
from ultralytics import YOLO

# ── Defaults ──────────────────────────────────────────────────────────────────
ROOT_DIR   = Path("data/raw/vinbigdata-chest-xray-abnormalities-detection")
WORK_DIR   = Path("outputs/smoke_test")
IMG_SIZE   = 640
SEED       = 42

CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification",
    "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
    "Lung Opacity", "Nodule/Mass", "Other lesion",
    "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
]
CLASS2ID = {n: i for i, n in enumerate(CLASS_NAMES)}
NC       = len(CLASS_NAMES)


# ── DICOM helpers ─────────────────────────────────────────────────────────────
def dicom_to_png(src: Path, dst: Path) -> tuple[int, int]:
    """Convert one DICOM to an 8-bit RGB PNG. Returns (w, h)."""
    ds = pydicom.dcmread(str(src))
    try:
        data = apply_voi_lut(ds.pixel_array.astype("float32"), ds)
    except Exception:
        data = ds.pixel_array.astype("float32")
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        data = data.max() - data
    lo, hi = data.min(), data.max()
    data = ((data - lo) / (hi - lo + 1e-8) * 255).astype("uint8")
    img = Image.fromarray(data).convert("RGB")
    img.save(str(dst))
    return img.size  # (w, h)


def write_yolo_label(path: Path, anns: list):
    with open(path, "w") as f:
        for cls, xc, yc, bw, bh in anns:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",      type=int,   default=500,       help="Number of images to use")
    ap.add_argument("--model",  type=str,   default="yolo26n.pt", help="YOLO model weights")
    ap.add_argument("--epochs", type=int,   default=1,         help="Training epochs")
    ap.add_argument("--batch",  type=int,   default=8,         help="Batch size")
    ap.add_argument("--workers",type=int,   default=4,         help="Dataloader workers")
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Model  : {args.model}")
    print(f"Images : {args.n}")
    print(f"Epochs : {args.epochs}")
    print()

    # ── Sanity check data dir ──────────────────────────────────────────────
    if not ROOT_DIR.exists():
        sys.exit(f"ERROR: data dir not found: {ROOT_DIR}\nMake sure unzip is complete.")

    train_csv = ROOT_DIR / "train.csv"
    if not train_csv.exists():
        sys.exit(f"ERROR: train.csv not found at {train_csv}")

    train_data = pd.read_csv(train_csv)
    print(f"Loaded train.csv: {len(train_data):,} rows, "
          f"{train_data['image_id'].nunique():,} images")

    # ── Sample N images (stratified: half with findings, half without) ─────
    all_ids   = train_data["image_id"].unique()
    finding_ids  = train_data[train_data["class_name"] != "No finding"]["image_id"].unique()
    nofind_ids   = np.setdiff1d(all_ids, finding_ids)

    n_find  = min(args.n // 2, len(finding_ids))
    n_nofind = min(args.n - n_find, len(nofind_ids))
    rng = np.random.default_rng(SEED)
    sample_ids = np.concatenate([
        rng.choice(finding_ids, n_find,   replace=False),
        rng.choice(nofind_ids,  n_nofind, replace=False),
    ])
    rng.shuffle(sample_ids)

    val_n    = max(50, len(sample_ids) // 5)
    train_ids = sample_ids[val_n:]
    val_ids   = sample_ids[:val_n]
    print(f"Subset  : {len(train_ids)} train | {len(val_ids)} val")

    # ── Build annotation lookup ────────────────────────────────────────────
    findings = train_data[train_data["class_name"] != "No finding"]
    dims = (
        train_data.groupby("image_id")[["width", "height"]].first()
        if {"width", "height"} <= set(train_data.columns) else None
    )

    def get_anns(img_id):
        rows = findings[findings["image_id"] == img_id]
        if len(rows) == 0:
            return []
        if dims is not None and img_id in dims.index:
            iw, ih = float(dims.loc[img_id, "width"]), float(dims.loc[img_id, "height"])
        else:
            ds = pydicom.dcmread(str(ROOT_DIR / "train" / f"{img_id}.dicom"))
            ih, iw = ds.pixel_array.shape[:2]
        anns = []
        for _, r in rows.iterrows():
            cid = CLASS2ID.get(r["class_name"])
            if cid is None:
                continue
            x1, y1, x2, y2 = float(r.x_min), float(r.y_min), float(r.x_max), float(r.y_max)
            xc = (x1 + x2) / 2 / iw
            yc = (y1 + y2) / 2 / ih
            bw = (x2 - x1) / iw
            bh = (y2 - y1) / ih
            anns.append((cid, *[max(0., min(1., v)) for v in [xc, yc, bw, bh]]))
        return anns

    # ── DICOM → PNG + labels ───────────────────────────────────────────────
    imgs_dir   = WORK_DIR / "images"
    labels_dir = WORK_DIR / "labels"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    def convert_one(img_id):
        png  = imgs_dir   / f"{img_id}.png"
        lbl  = labels_dir / f"{img_id}.txt"
        if not png.exists():
            src = ROOT_DIR / "train" / f"{img_id}.dicom"
            if src.exists():
                dicom_to_png(src, png)
            else:
                return  # skip missing files
        write_yolo_label(lbl, get_anns(img_id))

    print("\nConverting DICOMs → PNG ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(convert_one, i): i for i in sample_ids}
        for fut in tqdm(as_completed(futs), total=len(sample_ids)):
            fut.result()

    print(f"PNGs    : {len(list(imgs_dir.glob('*.png'))):,}")

    # ── Build fold symlinks ────────────────────────────────────────────────
    def make_split(ids, split):
        si = WORK_DIR / split / "images"
        sl = WORK_DIR / split / "labels"
        si.mkdir(parents=True, exist_ok=True)
        sl.mkdir(parents=True, exist_ok=True)
        for img_id in ids:
            dst_i = si / f"{img_id}.png"
            dst_l = sl / f"{img_id}.txt"
            if not dst_i.exists():
                src = imgs_dir / f"{img_id}.png"
                if src.exists():
                    os.symlink(src.resolve(), dst_i)
            if not dst_l.exists():
                src = labels_dir / f"{img_id}.txt"
                if src.exists():
                    shutil.copy2(src, dst_l)
                else:
                    open(dst_l, "w").close()

    make_split(train_ids, "train")
    make_split(val_ids,   "val")

    dataset_yaml = WORK_DIR / "dataset.yaml"
    with open(dataset_yaml, "w") as f:
        yaml.dump({
            "path"  : str(WORK_DIR.resolve()),
            "train" : "train/images",
            "val"   : "val/images",
            "nc"    : NC,
            "names" : CLASS_NAMES,
        }, f, default_flow_style=False)

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\nTraining {args.model} for {args.epochs} epoch(s) ...")
    model = YOLO(args.model)
    results = model.train(
        data         = str(dataset_yaml),
        epochs       = args.epochs,
        imgsz        = IMG_SIZE,
        batch        = args.batch,
        optimizer    = "MuSGD",
        end2end      = True,
        amp          = True,
        device       = device,
        workers      = args.workers,
        project      = str(WORK_DIR / "runs"),
        name         = "smoke",
        exist_ok     = True,
        plots        = False,
        verbose      = True,
        seed         = SEED,
        close_mosaic = 0,   # skip warm-up mosaic-close for 1-epoch test
        cache        = False,
    )

    # ── Report ─────────────────────────────────────────────────────────────
    rd = results.results_dict if hasattr(results, "results_dict") else {}
    map50   = rd.get("metrics/mAP50(B)",    "n/a")
    map5095 = rd.get("metrics/mAP50-95(B)", "n/a")
    print("\n" + "=" * 50)
    print(" SMOKE TEST COMPLETE")
    print("=" * 50)
    print(f"  mAP@50    : {map50}")
    print(f"  mAP@50-95 : {map5095}")
    print(f"  Weights   : {WORK_DIR}/runs/smoke/weights/best.pt")
    print("=" * 50)
    if map50 != "n/a" and float(map50) >= 0:
        print("\n✓ Pipeline validated. Ready for full training.")
        print("  Next: python scripts/train_full.py")
    else:
        print("\n✗ Something looks off — check the logs above.")


if __name__ == "__main__":
    main()
