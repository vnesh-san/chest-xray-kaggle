"""
prepare_labels.py — Generate consensus YOLO labels from train.csv.

Run this ONCE before training. Reads train.csv, applies multi-rater WBF consensus
(min 2 radiologist votes, IoU >= 0.4), and writes YOLO .txt label files.

Usage:
    PYTHONPATH=. python scripts/prepare_labels.py
    PYTHONPATH=. python scripts/prepare_labels.py --min-votes 3 --iou-thr 0.4
    PYTHONPATH=. python scripts/prepare_labels.py --dry-run   # stats only
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data.label_consensus import build_consensus_labels, consensus_stats

ROOT_DIR   = Path("data/raw/vinbigdata-chest-xray-abnormalities-detection")
LABELS_DIR = Path("outputs/yolo_dataset/labels")


def write_yolo_label(path: Path, anns: list):
    with open(path, "w") as f:
        for cls, xc, yc, bw, bh in anns:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-votes", type=int, default=2,
                    help="Min radiologist votes to keep a box (default 2)")
    ap.add_argument("--iou-thr", type=float, default=0.4,
                    help="IoU threshold for matching radiologist boxes (default 0.4)")
    ap.add_argument("--no-finding-strategy", type=str, default="any",
                    choices=["any", "majority", "all"],
                    help="How to determine 'No finding' images (default: any)")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel worker processes for WBF (0 = all CPU cores)")
    ap.add_argument("--output-dir", type=Path, default=LABELS_DIR,
                    help="Where to write YOLO .txt label files")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print stats only, do not write files")
    args = ap.parse_args()

    train_csv = ROOT_DIR / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")

    print(f"Loading {train_csv} ...")
    df = pd.read_csv(train_csv)
    print(f"  {len(df):,} rows, {df['image_id'].nunique():,} unique images")

    print(f"\nBuilding consensus labels (min_votes={args.min_votes}, iou_thr={args.iou_thr}) ...")
    import multiprocessing as mp
    n_workers = args.workers if args.workers > 0 else mp.cpu_count()
    print(f"  Using {n_workers} parallel workers")
    anns_map = build_consensus_labels(
        df,
        iou_thr=args.iou_thr,
        min_votes=args.min_votes,
        no_finding_strategy=args.no_finding_strategy,
        workers=n_workers,
    )

    stats = consensus_stats(anns_map)
    print(f"\n--- Consensus label statistics ---")
    print(f"  Total images   : {stats['total_images']:,}")
    print(f"  Positive images: {stats['positive_images']:,}")
    print(f"  Negative images: {stats['negative_images']:,}")
    print(f"  Total boxes    : {stats['total_boxes']:,}")
    print(f"  Boxes/positive : {stats['boxes_per_positive']:.2f}")
    print(f"\n  Per-class box counts:")
    for cls_name, cnt in stats["per_class"].items():
        print(f"    {cls_name:<30s}: {cnt:,}")

    if args.dry_run:
        print("\n[Dry run — no files written]")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting YOLO labels to {args.output_dir} ...")
    for img_id, anns in tqdm(anns_map.items()):
        write_yolo_label(args.output_dir / f"{img_id}.txt", anns)

    # Save stats as JSON for reference
    stats_path = args.output_dir / "consensus_stats.json"
    with open(stats_path, "w") as f:
        json.dump({"min_votes": args.min_votes, "iou_thr": args.iou_thr, **stats}, f, indent=2)
    print(f"  Stats saved: {stats_path}")
    print(f"\nDone. {len(anns_map):,} label files written to {args.output_dir}")


if __name__ == "__main__":
    main()
