"""
yolo_dicom_dataset.py — Plug native DICOM reading directly into YOLO26 training.

Subclasses ultralytics.data.dataset.YOLODataset and overrides the image
loading path so YOLO26 sees full-quality float32 multi-window DICOM tensors
instead of pre-baked 8-bit PNGs. All of YOLO26's losses, augmentations
(mosaic, mixup, copy-paste) and training loop are preserved.

Usage:
    from src.data.yolo_dicom_dataset import DicomYOLODataset, DicomYOLOTrainer

    trainer = DicomYOLOTrainer(
        dicom_root = "data/raw/vinbigdata-chest-xray-abnormalities-detection/train",
        overrides  = { "model": "yolo26n.pt", "data": "outputs/folds/fold_01/dataset.yaml",
                       "epochs": 1, "imgsz": 1024, "batch": 4, ... }
    )
    trainer.train()
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG

from src.data.dataset import load_dicom_hu, hu_to_3channel_float, WINDOWS


class DicomYOLODataset(YOLODataset):
    """
    YOLODataset that loads images from raw DICOM files instead of PNGs.

    The dataset YAML still lists PNG filenames (YOLO uses these as keys to
    find label .txt files). We intercept the image-load step and replace it
    with our DICOM pipeline: HU extraction → multi-window → CLAHE → float32.

    All other YOLODataset behaviour (mosaic, label parsing, augmentation
    transforms, caching) is unchanged.
    """

    def __init__(
        self,
        *args,
        dicom_root: str | Path,
        windows: list[tuple[float, float]] | None = None,
        clahe_clip: float = 2.0,
        clahe_tile: tuple[int, int] = (8, 8),
        **kwargs,
    ):
        self.dicom_root = Path(dicom_root)
        self.windows    = windows or list(WINDOWS.values())
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        super().__init__(*args, **kwargs)

    # ── Override image loading ────────────────────────────────────────────────
    def load_image(self, i: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """
        Load the i-th image.
        Returns: (img_uint8_rgb, (orig_h, orig_w), (resized_h, resized_w))

        ultralytics expects a uint8 [H, W, 3] BGR or RGB array here.
        We load the DICOM, build multi-window float32, scale to uint8.
        All windowing/CLAHE detail is preserved at the uint8 level.
        """
        # Derive DICOM path from the PNG path stored by YOLO
        png_path  = Path(self.im_files[i])
        img_id    = png_path.stem
        dicom_path = self.dicom_root / f"{img_id}.dicom"

        if not dicom_path.exists():
            # Fallback: try reading the PNG if DICOM isn't there yet
            img = cv2.imread(str(png_path))
            if img is None:
                raise FileNotFoundError(f"Neither DICOM nor PNG found for {img_id}")
            h0, w0 = img.shape[:2]
        else:
            # Full DICOM pipeline
            hu     = load_dicom_hu(dicom_path)
            h0, w0 = hu.shape

            # Multi-window + CLAHE → float32 [H, W, 3] in [0, 1]
            img_f  = hu_to_3channel_float(
                hu, self.windows, self.clahe_clip, self.clahe_tile
            )
            # Convert to uint8 [0, 255] in RGB order
            img = (img_f * 255).astype(np.uint8)

        # Resize to self.imgsz (YOLO's target resolution)
        r  = self.imgsz / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            img = cv2.resize(img, (round(w0 * r), round(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]

        return img, (h0, w0), (h, w)


class DicomYOLOTrainer(DetectionTrainer):
    """
    DetectionTrainer that uses DicomYOLODataset for image loading.

    Pass dicom_root to point to the folder containing .dicom files.
    Everything else (model, data yaml, epochs, etc.) is standard YOLO.
    """

    def __init__(
        self,
        *args,
        dicom_root: str | Path,
        windows: list[tuple[float, float]] | None = None,
        clahe_clip: float = 2.0,
        clahe_tile: tuple[int, int] = (8, 8),
        **kwargs,
    ):
        self._dicom_root  = Path(dicom_root)
        self._windows     = windows
        self._clahe_clip  = clahe_clip
        self._clahe_tile  = clahe_tile
        super().__init__(*args, **kwargs)

    def build_dataset(
        self,
        img_path: str,
        mode: str = "train",
        batch: int | None = None,
    ) -> DicomYOLODataset:
        """
        Override DetectionTrainer.build_dataset to inject our DICOM dataset.
        Called internally by YOLO's get_dataloader().
        """
        gs = max(int(self.model.stride.max() if self.model else 0), 32)
        return DicomYOLODataset(
            img_path  = img_path,
            imgsz     = self.args.imgsz,
            batch_size= batch,
            augment   = mode == "train",
            hyp       = self.args,
            rect      = self.args.rect or mode != "train",
            cache     = self.args.cache or None,
            single_cls= self.args.single_cls or False,
            stride    = int(gs),
            pad       = 0.0 if mode == "train" else 0.5,
            prefix    = ("train: " if mode == "train" else "val: "),
            task      = self.args.task,
            classes   = self.args.classes,
            data      = self.data,
            fraction  = self.args.fraction if mode == "train" else 1.0,
            # ── DICOM-specific args ──
            dicom_root = self._dicom_root,
            windows    = self._windows,
            clahe_clip = self._clahe_clip,
            clahe_tile = self._clahe_tile,
        )
