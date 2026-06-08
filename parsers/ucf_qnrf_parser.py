"""
ucf_qnrf_parser.py
───────────────────
Parser for the UCF-QNRF (ECCV 2018) crowd counting dataset.

Annotation format
─────────────────
Each image  img_XXXX.jpg  has a paired annotation file  img_XXXX_ann.mat.

The .mat stores a variable  annPoints  of shape (N, 2) where each row is
(x, y) in pixel coordinates.  Files can be either MATLAB v5/v6 (scipy) or
v7.3 HDF5 (h5py); both are handled automatically.

References
──────────
  Idrees et al., "Composition Loss for Counting, Density Map Estimation
  and Localisation in Dense Crowds", ECCV 2018.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)


class UCFQNRFParser(BaseParser):
    """
    Parse UCF-QNRF Train and Test splits.

    Parameters
    ----------
    root : str | Path
        Path that contains Train/ and Test/ subdirectories.
    """

    def __init__(self, root: str | Path) -> None:
        super().__init__(root, dataset_name="ucf_qnrf")

    # ── Public API ──────────────────────────────────────────────────────────

    def parse(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for split in ("Train", "Test"):
            df = self._parse_split(split)
            if df is not None:
                frames.append(df)

        if not frames:
            raise RuntimeError(
                f"[{self.dataset_name}] No annotation files found under {self.root}"
            )

        combined = pd.concat(frames, ignore_index=True)
        combined = self.validate_schema(combined)
        self.sanity_check(combined)
        return combined

    # ── Private helpers ─────────────────────────────────────────────────────

    def _parse_split(self, split_dir: str) -> pd.DataFrame | None:
        split_path = self.root / split_dir
        if not split_path.exists():
            logger.warning(
                "[%s] Split directory not found: %s",
                self.dataset_name, split_path,
            )
            return None

        ann_files = sorted(split_path.glob("*_ann.mat"))
        if not ann_files:
            logger.warning(
                "[%s] No *_ann.mat files in %s", self.dataset_name, split_path
            )
            return None

        split_label = split_dir.lower()  # "train" | "test"
        logger.info(
            "[%s] %s — found %d annotation files",
            self.dataset_name, split_dir, len(ann_files),
        )

        rows: list[dict] = []

        for ann_path in ann_files:
            # img_0001_ann.mat → img_0001
            image_id = ann_path.stem.replace("_ann", "")
            img_path = split_path / f"{image_id}.jpg"

            try:
                points = self._load_ann_points(ann_path)
            except Exception as exc:
                logger.error(
                    "[%s] Failed to load %s: %s",
                    self.dataset_name, ann_path.name, exc,
                )
                continue

            gt_count = len(points)

            if gt_count == 0:
                rows.append(
                    self._make_row(split_label, image_id, gt_count,
                                   np.nan, np.nan)
                )
            else:
                for pt in points:
                    rows.append(
                        self._make_row(split_label, image_id, gt_count,
                                       float(pt[0]), float(pt[1]))
                    )

        return pd.DataFrame(rows) if rows else None

    # ── .mat loading ────────────────────────────────────────────────────────

    def _load_ann_points(self, ann_path: Path) -> np.ndarray:
        """
        Load annPoints from a UCF-QNRF annotation .mat file.
        Returns an (N, 2) float array.
        """
        if HAS_SCIPY:
            try:
                mat  = sio.loadmat(str(ann_path))
                pts  = mat["annPoints"]
                return np.array(pts, dtype=float)
            except Exception:
                pass

        if HAS_H5PY:
            with h5py.File(str(ann_path), "r") as f:
                for key in ("annPoints", "ann_points", "points"):
                    if key in f:
                        pts = np.array(f[key])
                        # Shape can be (2,N) or (N,2)
                        return pts.T if pts.shape[0] == 2 else pts

        raise IOError(
            f"Cannot read annotation from {ann_path.name}. "
            "Make sure scipy and/or h5py are installed."
        )

    # ── Utilities ────────────────────────────────────────────────────────────

    def _make_row(
        self,
        split: str,
        image_id: str,
        gt_count: int,
        cx: float,
        cy: float,
    ) -> dict:
        return {
            "dataset":           self.dataset_name,
            "split":             split,
            "image_id":          image_id,
            "frame_id":          -1,
            "gt_count":          gt_count,
            "trajectory_id":     "",
            "cx":                cx,
            "cy":                cy,
            "bbox_x":            np.nan,
            "bbox_y":            np.nan,
            "bbox_w":            np.nan,
            "bbox_h":            np.nan,
            "density_available": False,
        }
