"""
dicom_utils.py — DICOM → multi-window 3-channel PNG conversion.

Gold standard for chest X-ray competitions:
  R = lung window        (WC=-600, WW=1500)  — pneumothorax, ILD, infiltration
  G = mediastinum window (WC=40,   WW=400)   — cardiomegaly, effusion, aorta
  B = soft tissue window (WC=80,   WW=800)   — nodules, calcification, mass

Each channel is an independent 8-bit windowed view of the full 12/16-bit DICOM.
No dynamic range is lost — different tissue contrasts are captured per channel.
Works natively with ultralytics YOLO / RT-DETR (standard uint8 RGB PNG).
"""
from __future__ import annotations

import numpy as np
import pydicom
from pathlib import Path
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut


# ── Window presets (center, width) ────────────────────────────────────────────
WINDOW_PRESETS = {
    "lung":        (-600, 1500),
    "mediastinum": (  40,  400),
    "soft_tissue": (  80,  800),
    "bone":        ( 400, 1800),
}

# Default 3-channel stack used for training
DEFAULT_WINDOWS = [
    WINDOW_PRESETS["lung"],
    WINDOW_PRESETS["mediastinum"],
    WINDOW_PRESETS["soft_tissue"],
]


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
    assert len(windows) == 3, "Exactly 3 windows required for RGB output."

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
