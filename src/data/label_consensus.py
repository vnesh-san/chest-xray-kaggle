"""
label_consensus.py — Merge multi-rater radiologist annotations into consensus labels.

The VinBigData train.csv has 3-17 independent radiologist reads per image.
Training on raw union labels introduces significant noise from single-rater boxes.
This module merges boxes using WBF, keeping only boxes that >= min_votes radiologists drew.

Usage:
    from src.data.label_consensus import build_consensus_labels

    anns_map = build_consensus_labels(
        df=pd.read_csv("data/raw/.../train.csv"),
        iou_thr=0.4,    # matches competition's lowest eval IoU
        min_votes=2,    # keep boxes drawn by >= 2 radiologists
    )
    # anns_map: dict image_id -> list of (class_id, xc, yc, bw, bh) normalized YOLO format
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm


CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification",
    "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
    "Lung Opacity", "Nodule/Mass", "Other lesion",
    "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
]
CLASS2ID = {n: i for i, n in enumerate(CLASS_NAMES)}


def _wbf_single_class(
    boxes_xyxy_norm: list[list[float]],
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run WBF on a list of normalized [x1,y1,x2,y2] boxes (all same class, one image).
    Returns (merged_boxes [N,4], vote_counts [N]).

    We encode radiologist votes as uniform scores so that after WBF the fused
    score == average_score == 1/k * k_votes.  Specifically, each box gets
    score = 1/n_rads so the merged score = sum_of_matching_scores / n_fused.
    But we just want vote counts, so we pass score=1 for each box and divide
    back out by the number of inputs.
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        raise ImportError("pip install ensemble-boxes")

    n = len(boxes_xyxy_norm)
    if n == 0:
        return np.zeros((0, 4)), np.zeros(0)

    # WBF expects: list of per-model box lists. We treat each radiologist as a "model"
    # with a single box.  Pass all as one "model" — WBF will cluster by IoU.
    scores = [1.0] * n
    labels = [0] * n
    merged_boxes, merged_scores, _ = weighted_boxes_fusion(
        [boxes_xyxy_norm], [scores], [labels],
        iou_thr=iou_thr,
        skip_box_thr=0.0,
        conf_type="avg",
    )
    # merged_scores = avg of fused scores.  Since each input score = 1,
    # avg = n_votes / n_fused_boxes... but WBF normalizes differently.
    # More reliable: use the count of input boxes that got merged into each cluster.
    # Re-implement: cluster by IoU manually to get exact vote counts.
    vote_counts = _count_votes(boxes_xyxy_norm, merged_boxes.tolist(), iou_thr)
    return merged_boxes, np.array(vote_counts)


def _count_votes(
    input_boxes: list[list[float]],
    merged_boxes: list[list[float]],
    iou_thr: float,
) -> list[int]:
    """Count how many input boxes overlap each merged box at >= iou_thr."""
    if not merged_boxes or not input_boxes:
        return []
    inp = np.array(input_boxes)   # (N, 4)
    mer = np.array(merged_boxes)  # (M, 4)
    counts = []
    for mb in mer:
        # IoU of mb vs every input box
        xi1 = np.maximum(inp[:, 0], mb[0])
        yi1 = np.maximum(inp[:, 1], mb[1])
        xi2 = np.minimum(inp[:, 2], mb[2])
        yi2 = np.minimum(inp[:, 3], mb[3])
        inter = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
        area_inp = (inp[:, 2] - inp[:, 0]) * (inp[:, 3] - inp[:, 1])
        area_mb  = (mb[2] - mb[0]) * (mb[3] - mb[1])
        union = area_inp + area_mb - inter
        iou = np.where(union > 0, inter / union, 0.0)
        counts.append(int((iou >= iou_thr).sum()))
    return counts


def _process_image_chunk(args) -> list[tuple[str, list]]:
    """
    Worker function: process a CHUNK of images in one process.
    Much faster than per-image dispatch because it amortises process startup
    and pickle overhead across many images.

    args: (chunk, iou_thr, min_votes)
    chunk: list of (img_id, img_findings_rows, iw, ih)
    Returns: list of (img_id, anns_list)
    """
    chunk, iou_thr, min_votes = args
    results = []
    for img_id, img_findings_rows, iw, ih in chunk:
        merged_anns = []
        cls_groups: dict[str, list] = {}
        for row in img_findings_rows:
            cls_name, x_min, y_min, x_max, y_max = row
            cls_groups.setdefault(cls_name, []).append((x_min, y_min, x_max, y_max))

        for cls_name, coords in cls_groups.items():
            cid = CLASS2ID.get(cls_name)
            if cid is None:
                continue
            boxes_norm = []
            for x_min, y_min, x_max, y_max in coords:
                x1 = max(0.0, min(1.0, float(x_min) / iw))
                y1 = max(0.0, min(1.0, float(y_min) / ih))
                x2 = max(0.0, min(1.0, float(x_max) / iw))
                y2 = max(0.0, min(1.0, float(y_max) / ih))
                if x2 > x1 and y2 > y1:
                    boxes_norm.append([x1, y1, x2, y2])
            if not boxes_norm:
                continue
            merged_boxes, vote_counts = _wbf_single_class(boxes_norm, iou_thr)
            for box, votes in zip(merged_boxes, vote_counts):
                if votes < min_votes:
                    continue
                x1, y1, x2, y2 = box
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                bw = x2 - x1
                bh = y2 - y1
                merged_anns.append((cid, xc, yc, bw, bh))
        results.append((img_id, merged_anns))
    return results


def build_dims_cache(
    image_ids: list[str],
    dicom_dir: str | None = None,
    cache_path: str | None = "outputs/image_dims.json",
    workers: int = 0,
) -> dict[str, tuple[float, float]]:
    """
    Build a {image_id: (width, height)} cache by reading all DICOMs in parallel.
    Saves to cache_path as JSON so subsequent runs skip DICOM reads entirely.

    Parameters
    ----------
    image_ids  : list of image IDs to resolve
    dicom_dir  : directory containing .dicom files
                 (default: data/raw/vinbigdata.../train)
    cache_path : where to save/load the JSON cache (None = no cache file)
    workers    : threads for parallel DICOM reading (0 = cpu_count * 2 for I/O)
    """
    import json
    import pydicom
    import multiprocessing as mp
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    if dicom_dir is None:
        dicom_dir = "data/raw/vinbigdata-chest-xray-abnormalities-detection/train"
    dicom_dir = Path(dicom_dir)

    # Load existing cache
    dims: dict[str, tuple[float, float]] = {}
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            raw = json.load(f)
        dims = {k: tuple(v) for k, v in raw.items()}

    missing = [i for i in image_ids if i not in dims]
    if not missing:
        return dims

    n_threads = workers if workers > 0 else min(64, (mp.cpu_count() or 1) * 4)
    print(f"  Reading dims for {len(missing):,} DICOMs ({n_threads} threads) ...")

    def _read_dim(img_id):
        p = dicom_dir / f"{img_id}.dicom"
        if not p.exists():
            return img_id, (1.0, 1.0)
        ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        rows = int(getattr(ds, "Rows", 0)) or 1
        cols = int(getattr(ds, "Columns", 0)) or 1
        return img_id, (float(cols), float(rows))  # (width, height)

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futs = {ex.submit(_read_dim, i): i for i in missing}
        for fut in tqdm(as_completed(futs), total=len(missing), desc="DICOM dims"):
            img_id, wh = fut.result()
            dims[img_id] = wh

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(dims, f)
        print(f"  Dims cached → {cache_path}")

    return dims


def build_consensus_labels(
    df: pd.DataFrame,
    iou_thr: float = 0.4,
    min_votes: int = 2,
    no_finding_strategy: str = "any",
    workers: int = 0,
    dims_cache_path: str | None = "outputs/image_dims.json",
) -> dict[str, list[tuple]]:
    """
    Build YOLO-format consensus annotations for all images.

    Parameters
    ----------
    df            : raw train.csv DataFrame
    iou_thr       : IoU threshold for matching radiologist boxes (default 0.4)
    min_votes     : minimum radiologist agreement to keep a box (default 2)
    no_finding_strategy:
        "any"  — if any radiologist marks no finding only, keep image as negative
        "all"  — only mark as negative if ALL radiologists say no finding
        "majority" — majority of radiologists say no finding
    workers       : number of parallel worker processes (0 = use all CPU cores)

    Returns
    -------
    anns_map : dict image_id -> list of (class_id, xc, yc, bw, bh)
               Empty list means "No finding" (negative image).
               Coordinates are normalized [0, 1].
    """
    import multiprocessing as mp
    if workers == 0:
        workers = mp.cpu_count()

    anns_map: dict[str, list] = {}
    all_ids = df["image_id"].unique()

    findings = df[df["class_name"] != "No finding"].copy()
    no_find  = df[df["class_name"] == "No finding"].copy()

    # Get image dimensions: from train.csv cols if available, else parallel DICOM read
    has_dims = {"width", "height"} <= set(df.columns)
    if has_dims:
        _dims_df = df.groupby("image_id")[["width", "height"]].first()
        dims = {img_id: (float(_dims_df.loc[img_id, "width"]), float(_dims_df.loc[img_id, "height"]))
                for img_id in all_ids if img_id in _dims_df.index}
    else:
        dims = build_dims_cache(list(all_ids), workers=workers * 2,
                                cache_path=dims_cache_path)

    def get_dims(img_id: str) -> tuple[float, float]:
        return dims.get(img_id, (1.0, 1.0))

    # Build per-image radiologist vote counts for "No finding"
    nofind_votes: dict[str, int] = {}
    total_rads:   dict[str, int] = {}
    for img_id, grp in df.groupby("image_id"):
        total_rads[img_id] = grp["rad_id"].nunique() if "rad_id" in grp.columns else len(grp)
        nofind_votes[img_id] = no_find[no_find["image_id"] == img_id]["rad_id"].nunique() \
            if "rad_id" in df.columns else int((no_find["image_id"] == img_id).sum())

    # Separate negative images (no WBF needed) from positive images
    positive_ids = []
    worker_args  = []

    for img_id in all_ids:
        img_findings = findings[findings["image_id"] == img_id]
        n_rads   = total_rads.get(img_id, 1)
        n_nofind = nofind_votes.get(img_id, 0)

        # Decide if this image is a "No finding" negative
        if no_finding_strategy == "all":
            is_negative = n_nofind == n_rads
        elif no_finding_strategy == "majority":
            is_negative = n_nofind > n_rads / 2
        else:  # "any"
            is_negative = len(img_findings) == 0

        if len(img_findings) == 0 or is_negative:
            anns_map[img_id] = []
            continue

        iw, ih = get_dims(img_id)
        rows = [(r["class_name"], r["x_min"], r["y_min"], r["x_max"], r["y_max"])
                for _, r in img_findings.iterrows()]
        worker_args.append((img_id, rows, iw, ih))
        positive_ids.append(img_id)

    # Run WBF in parallel — chunk images across workers to minimise IPC overhead
    if worker_args:
        n_proc = min(workers, len(worker_args))
        chunk_size = max(1, len(worker_args) // n_proc)
        chunks = [
            (worker_args[i : i + chunk_size], iou_thr, min_votes)
            for i in range(0, len(worker_args), chunk_size)
        ]
        with mp.Pool(processes=n_proc) as pool:
            for chunk_results in pool.imap_unordered(_process_image_chunk, chunks):
                for img_id, anns in chunk_results:
                    anns_map[img_id] = anns

    # Remaining images that were never assigned
    for img_id in all_ids:
        if img_id not in anns_map:
            anns_map[img_id] = []

    return anns_map




def consensus_stats(anns_map: dict[str, list]) -> dict:
    """Print summary statistics for the consensus labels."""
    total = len(anns_map)
    negatives = sum(1 for v in anns_map.values() if len(v) == 0)
    total_boxes = sum(len(v) for v in anns_map.values())

    class_counts = [0] * len(CLASS_NAMES)
    for anns in anns_map.values():
        for (cid, *_) in anns:
            class_counts[cid] += 1

    return {
        "total_images": total,
        "negative_images": negatives,
        "positive_images": total - negatives,
        "total_boxes": total_boxes,
        "boxes_per_positive": total_boxes / max(1, total - negatives),
        "per_class": {CLASS_NAMES[i]: class_counts[i] for i in range(len(CLASS_NAMES))},
    }
