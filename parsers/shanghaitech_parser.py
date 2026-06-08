"""
shanghaitech_parser.py
───────────────────────
Parser for ShanghaiTech Part A & Part B crowd counting datasets.

Annotation format
─────────────────
Each image has a matching .mat file under ground-truth/:
    GT_IMG_<N>.mat

The .mat can be either:
  • scipy-readable  (MATLAB v5/v6)  → image_info[0,0][0,0][0] is the (N,2) point array
  • h5py-readable   (MATLAB v7.3)   → dataset key 'image_info/image_info[0,0]/location'

Both formats are handled automatically.

Output schema: see base_parser.SCHEMA_COLUMNS
"""

import logging
import re
from pathlib import Path
from typing import Literal

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

Part = Literal["A", "B", "both"]


class ShanghaiTechParser(BaseParser):
    """
    Parse ShanghaiTech Part A and/or Part B.

    Parameters
    ----------
    root : str | Path
        Path to the ShanghaiTech directory that contains part_A/ and part_B/.
    part : "A" | "B" | "both"
        Which part(s) to parse.
    """

    def __init__(self, root: str | Path, part: Part = "both") -> None:
        super().__init__(root, dataset_name=f"shanghaitech_{part}")
        self.part = part

    # ── Public API ──────────────────────────────────────────────────────────

    def parse(self) -> pd.DataFrame:
        """Return normalised DataFrame for all requested parts/splits."""
        parts_to_run: list[str] = (
            ["A", "B"] if self.part == "both" else [self.part]
        )
        frames: list[pd.DataFrame] = []
        for p in parts_to_run:
            for split in ("train", "test"):
                df = self._parse_split(p, split)
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

    def _parse_split(
        self, part: str, split: str
    ) -> pd.DataFrame | None:
        split_dir = self.root / f"part_{part}" / f"{split}_data"
        gt_dir    = split_dir / "ground-truth"
        img_dir   = split_dir / "images"

        if not gt_dir.exists():
            logger.warning(
                "[%s] Ground-truth directory not found: %s",
                self.dataset_name, gt_dir,
            )
            return None

        mat_files = sorted(gt_dir.glob("GT_IMG_*.mat"))
        if not mat_files:
            logger.warning(
                "[%s] No .mat files found in %s", self.dataset_name, gt_dir
            )
            return None

        logger.info(
            "[%s] Part %s / %s — found %d annotation files",
            self.dataset_name, part, split, len(mat_files),
        )

        rows: list[dict] = []
        dataset_tag = f"shanghaitech_{part}"

        for mat_path in mat_files:
            image_id = self._image_id_from_mat(mat_path)
            img_path = img_dir / f"{image_id}.jpg"
            density_available = (img_dir.parent / f"{image_id}.h5").exists()

            try:
                points = self._load_points(mat_path)
            except Exception as exc:
                logger.error(
                    "[%s] Failed to load %s: %s",
                    self.dataset_name, mat_path.name, exc,
                )
                continue

            gt_count = len(points)

            if gt_count == 0:
                # Still emit one row so the image appears in the table
                rows.append(
                    self._build_row(
                        dataset_tag, split, image_id, -1, gt_count,
                        "", np.nan, np.nan,
                        np.nan, np.nan, np.nan, np.nan,
                        density_available,
                    )
                )
            else:
                for pt in points:
                    cx, cy = float(pt[0]), float(pt[1])
                    rows.append(
                        self._build_row(
                            dataset_tag, split, image_id, -1, gt_count,
                            "", cx, cy,
                            np.nan, np.nan, np.nan, np.nan,
                            density_available,
                        )
                    )

        return pd.DataFrame(rows) if rows else None

    # ── .mat loading ────────────────────────────────────────────────────────

    def _load_points(self, mat_path: Path) -> np.ndarray:
        """
        Load dot-annotation points from a ShanghaiTech .mat file.
        Returns an (N, 2) array of (x, y) coordinates.
        """
        # Try scipy first (MATLAB < v7.3)
        if HAS_SCIPY:
            try:
                mat = sio.loadmat(str(mat_path))
                # Structure: image_info[0,0][0,0][0]  → (N,2)
                pts = mat["image_info"][0, 0][0, 0][0]
                return np.array(pts, dtype=float)
            except Exception:
                pass  # fall through to h5py

        # Try h5py (MATLAB v7.3 HDF5 format)
        if HAS_H5PY:
            with h5py.File(str(mat_path), "r") as f:
                # Navigate the HDF5 group structure
                pts = self._h5_find_points(f)
                if pts is not None:
                    return pts

        raise IOError(
            f"Cannot read {mat_path.name}. "
            "Ensure scipy and h5py are installed and the file is valid."
        )

    def _h5_find_points(self, f: "h5py.File") -> np.ndarray | None:
        """
        Recursively search an h5py File for the (N,2) or (2,N) location array.
        ShanghaiTech v7.3 stores it under 'image_info' with object references.
        """
        try:
            ref = f["image_info"][0, 0]
            sub = f[ref]
            loc_ref = sub["location"][0, 0]
            pts = np.array(f[loc_ref]).T  # shape (N, 2)
            return pts
        except Exception:
            pass

        # Fallback: search all datasets for an (N,2) float array
        for key in f:
            item = f[key]
            if hasattr(item, "shape") and len(item.shape) == 2:
                arr = np.array(item)
                if arr.shape[1] == 2 or arr.shape[0] == 2:
                    return arr.T if arr.shape[0] == 2 else arr
        return None

    # ── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _image_id_from_mat(mat_path: Path) -> str:
        """GT_IMG_42.mat → IMG_42"""
        name = mat_path.stem  # e.g. GT_IMG_42
        m = re.match(r"GT_(IMG_\d+)", name)
        return m.group(1) if m else name

    @staticmethod
    def _build_row(
        dataset: str, split: str, image_id: str,
        frame_id: int, gt_count: int, trajectory_id: str,
        cx: float, cy: float,
        bbox_x: float, bbox_y: float, bbox_w: float, bbox_h: float,
        density_available: bool,
    ) -> dict:
        return {
            "dataset":           dataset,
            "split":             split,
            "image_id":          image_id,
            "frame_id":          frame_id,
            "gt_count":          gt_count,
            "trajectory_id":     trajectory_id,
            "cx":                cx,
            "cy":                cy,
            "bbox_x":            bbox_x,
            "bbox_y":            bbox_y,
            "bbox_w":            bbox_w,
            "bbox_h":            bbox_h,
            "density_available": density_available,
        }
