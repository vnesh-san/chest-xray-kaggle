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


def build_consensus_labels(
    df: pd.DataFrame,
    iou_thr: float = 0.4,
    min_votes: int = 2,
    no_finding_strategy: str = "any",
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
                 (conservative; default)
        "all"  — only mark as negative if ALL radiologists say no finding
        "majority" — majority of radiologists say no finding

    Returns
    -------
    anns_map : dict image_id -> list of (class_id, xc, yc, bw, bh)
               Empty list means "No finding" (negative image).
               Coordinates are normalized [0, 1].
    """
    anns_map: dict[str, list] = {}
    all_ids = df["image_id"].unique()

    findings = df[df["class_name"] != "No finding"].copy()
    no_find  = df[df["class_name"] == "No finding"].copy()

    # Pre-extract image dimensions (from train.csv if available, else fallback)
    has_dims = {"width", "height"} <= set(df.columns)
    if has_dims:
        dims = df.groupby("image_id")[["width", "height"]].first()
    else:
        dims = None

    def get_dims(img_id: str) -> tuple[float, float]:
        if dims is not None and img_id in dims.index:
            return float(dims.loc[img_id, "width"]), float(dims.loc[img_id, "height"])
        # Fallback: read from DICOM (slow, avoid if possible)
        import pydicom
        from pathlib import Path
        dcm_path = Path("data/raw/vinbigdata-chest-xray-abnormalities-detection/train") / f"{img_id}.dicom"
        if dcm_path.exists():
            ds = pydicom.dcmread(str(dcm_path))
            ih, iw = ds.pixel_array.shape[:2]
            return float(iw), float(ih)
        return 1.0, 1.0  # shouldn't happen

    # Build per-image radiologist vote counts for "No finding"
    nofind_votes: dict[str, int] = {}
    total_rads:   dict[str, int] = {}
    for img_id, grp in df.groupby("image_id"):
        total_rads[img_id] = grp["rad_id"].nunique() if "rad_id" in grp.columns else len(grp)
        nofind_votes[img_id] = no_find[no_find["image_id"] == img_id]["rad_id"].nunique() \
            if "rad_id" in df.columns else int((no_find["image_id"] == img_id).sum())

    for img_id in all_ids:
        img_findings = findings[findings["image_id"] == img_id]
        n_rads = total_rads.get(img_id, 1)
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
        merged_anns = []

        for cls_name, cls_grp in img_findings.groupby("class_name"):
            cid = CLASS2ID.get(cls_name)
            if cid is None:
                continue

            # Build normalized xyxy boxes for this class
            boxes_norm = []
            for _, r in cls_grp.iterrows():
                x1 = max(0.0, min(1.0, float(r["x_min"]) / iw))
                y1 = max(0.0, min(1.0, float(r["y_min"]) / ih))
                x2 = max(0.0, min(1.0, float(r["x_max"]) / iw))
                y2 = max(0.0, min(1.0, float(r["y_max"]) / ih))
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

        anns_map[img_id] = merged_anns

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
