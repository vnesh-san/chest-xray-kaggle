"""
dicom_utils.py — DICOM → multi-window 3-channel PNG conversion.

Two complementary 3-channel window sets for dual-YOLO ensemble strategy:

  Window Set A (general anatomy):
    R = lung window        (WC=-600, WW=1500)  — pneumothorax, ILD, infiltration, atelectasis
    G = mediastinum window (WC= 40,  WW= 400)  — cardiomegaly, effusion, aorta
    B = soft tissue window (WC= 80,  WW= 800)  — nodules, consolidation, lung opacity

  Window Set B (specific pathologies):
    R = bone window        (WC= 400, WW=1500)  — calcification (rare, needs bone contrast)
    G = pleural window     (WC=  50, WW= 350)  — pleural thickening vs effusion distinction
    B = vascular window    (WC= 100, WW= 700)  — aortic enlargement, pulmonary fibrosis

Training two YOLO26x models on A and B (no architecture changes — both 3-ch RGB)
then WBF-ensembling gives complementary detection coverage across all 14 classes.
"""
from __future__ import annotations

import numpy as np
import pydicom
from pathlib import Path
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut


# ── Window presets (center, width) ────────────────────────────────────────────
WINDOW_PRESETS = {
    "lung":        (-600, 1500),   # parenchyma
    "mediastinum": (  40,  400),   # heart / vessels
    "soft_tissue": (  80,  800),   # nodules / masses
    "bone":        ( 400, 1500),   # calcification / ribs
    "pleural":     (  50,  350),   # pleural effusion / thickening
    "vascular":    ( 100,  700),   # aorta / pulmonary vessels
}

# Window Set A — general anatomy (current default)
WINDOW_SET_A = [
    WINDOW_PRESETS["lung"],
    WINDOW_PRESETS["mediastinum"],
    WINDOW_PRESETS["soft_tissue"],
]

# Window Set B — specific pathologies (new)
WINDOW_SET_B = [
    WINDOW_PRESETS["bone"],
    WINDOW_PRESETS["pleural"],
    WINDOW_PRESETS["vascular"],
]

# Backward-compat alias
DEFAULT_WINDOWS = WINDOW_SET_A

WINDOW_SETS = {"A": WINDOW_SET_A, "B": WINDOW_SET_B}


def _apply_window(pixels: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Hard VOI window clamp → rescale to uint8 [0, 255].
    pixels: raw float32 HU (or raw pixel values after VOI-LUT).
    """
    lo = center - width / 2.0
    hi = center + width / 2.0
    windowed = np.clip(pixels, lo, hi)
    windowed = ((windowed - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return windowed


def dicom_to_multiwindow_png(
    src: Path | str,
    dst: Path | str,
    windows: list[tuple[float, float]] | None = None,
    invert_monochrome1: bool = True,
) -> tuple[int, int]:
    """
    Convert a DICOM file to a 3-channel PNG using multiple VOI windows.

    Parameters
    ----------
    src : path to .dicom file
    dst : path to output .png file
    windows : list of (center, width) tuples — default: lung/mediastinum/soft-tissue
    invert_monochrome1 : flip MONOCHROME1 images (some scanners invert polarity)

    Returns
    -------
    (width, height) of saved image
    """
    if windows is None:
        windows = DEFAULT_WINDOWS
    if isinstance(windows, str):
        windows = WINDOW_SETS[windows]   # accept "A" or "B" directly
    assert len(windows) == 3, "Exactly 3 windows required for RGB PNG output."

    ds = pydicom.dcmread(str(src))

    # 1. Extract raw pixel values (float32)
    try:
        pixels = apply_voi_lut(ds.pixel_array.astype(np.float32), ds)
    except Exception:
        pixels = ds.pixel_array.astype(np.float32)

    # 2. Handle MONOCHROME1 (intensity inverted relative to standard)
    if invert_monochrome1 and getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() - pixels

    # 3. Rescale Intercept / Slope → Hounsfield Units (if available)
    slope     = float(getattr(ds, "RescaleSlope",     1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        pixels = pixels * slope + intercept

    # 4. Apply each window → one 8-bit channel
    channels = [_apply_window(pixels, c, w) for c, w in windows]

    # 5. Stack as RGB and save
    rgb = np.stack(channels, axis=-1)   # (H, W, 3) uint8
    img = Image.fromarray(rgb, mode="RGB")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    img.save(str(dst))
    return img.size  # (width, height)


def dicom_to_single_window_png(
    src: Path | str,
    dst: Path | str,
    center: float = -600,
    width: float  = 1500,
) -> tuple[int, int]:
    """
    Single-window grayscale → 3-channel PNG (all channels identical).
    Provided for ablation / backward-compat.
    """
    ds = pydicom.dcmread(str(src))
    try:
        pixels = apply_voi_lut(ds.pixel_array.astype(np.float32), ds)
    except Exception:
        pixels = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() - pixels
    slope     = float(getattr(ds, "RescaleSlope",     1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    pixels    = pixels * slope + intercept
    ch = _apply_window(pixels, center, width)
    rgb = np.stack([ch, ch, ch], axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    img.save(str(dst))
    return img.size
