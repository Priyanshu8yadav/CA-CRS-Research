"""
base_parser.py
──────────────
Abstract base class for all dataset parsers.

Every concrete parser must implement `parse()` and return a DataFrame
with the normalised flat schema defined here.

Flat CSV schema
───────────────
dataset          : str   — dataset name (e.g. "shanghaitech_A")
split            : str   — "train" | "test"
image_id         : str   — filename stem, e.g. "IMG_1"
frame_id         : int   — frame index for video datasets; -1 for still images
gt_count         : int   — ground-truth head / person count
trajectory_id    : str   — person track ID (PETS2009); "" for still-image sets
cx               : float — dot annotation centre-x  (NaN if not available)
cy               : float — dot annotation centre-y  (NaN if not available)
bbox_x           : float — bounding box left edge   (NaN if not available)
bbox_y           : float — bounding box top edge    (NaN if not available)
bbox_w           : float — bounding box width       (NaN if not available)
bbox_h           : float — bounding box height      (NaN if not available)
density_available: bool  — True if a density map (.h5/.npy) exists
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical column order for all parsers
SCHEMA_COLUMNS: list[str] = [
    "dataset",
    "split",
    "image_id",
    "frame_id",
    "gt_count",
    "trajectory_id",
    "cx",
    "cy",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "density_available",
]


class BaseParser(ABC):
    """Abstract base for crowd-dataset parsers."""

    def __init__(self, root: str | Path, dataset_name: str) -> None:
        self.root = Path(root)
        self.dataset_name = dataset_name
        if not self.root.exists():
            raise FileNotFoundError(
                f"[{self.dataset_name}] Dataset root not found: {self.root}"
            )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    # ── Abstract interface ──────────────────────────────────────────────────

    @abstractmethod
    def parse(self) -> pd.DataFrame:
        """
        Read raw annotations and return a normalised DataFrame.
        The returned frame MUST contain exactly SCHEMA_COLUMNS.
        """
        ...

    # ── Shared helpers ──────────────────────────────────────────────────────

    def validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce the canonical column set and dtypes.
        Missing columns are added with appropriate null values.
        """
        for col in SCHEMA_COLUMNS:
            if col not in df.columns:
                logger.warning(
                    "[%s] Column '%s' missing — filling with null",
                    self.dataset_name, col,
                )
                df[col] = None
        df = df[SCHEMA_COLUMNS].copy()

        # Coerce dtypes
        df["gt_count"]   = df["gt_count"].astype(int)
        df["frame_id"]   = df["frame_id"].fillna(-1).astype(int)
        df["density_available"] = df["density_available"].astype(bool)
        df["trajectory_id"]     = df["trajectory_id"].fillna("").astype(str)
        for col in ("cx", "cy", "bbox_x", "bbox_y", "bbox_w", "bbox_h"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def sanity_check(self, df: pd.DataFrame) -> None:
        """Print and log basic dataset statistics."""
        total_images  = df["image_id"].nunique()
        total_annots  = len(df)
        count_summary = (
            df.groupby(["dataset", "split"])["gt_count"]
            .agg(["mean", "min", "max", "sum"])
            .round(2)
        )
        logger.info(
            "[%s] images=%d  annotations=%d\n%s",
            self.dataset_name, total_images, total_annots, count_summary,
        )
        print(f"\n── {self.dataset_name} sanity check ──")
        print(f"  Unique images : {total_images}")
        print(f"  Total rows    : {total_annots}")
        print(count_summary.to_string())
        print()

        # Guard: no image should have gt_count < 0
        neg = (df["gt_count"] < 0).sum()
        if neg:
            logger.warning("[%s] %d rows have gt_count < 0", self.dataset_name, neg)

    def save_csv(
        self,
        df: pd.DataFrame,
        out_dir: str | Path,
        filename: Optional[str] = None,
    ) -> Path:
        """Save the normalised DataFrame to CSV."""
        out_dir  = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname    = filename or f"{self.dataset_name}.csv"
        out_path = out_dir / fname
        df.to_csv(out_path, index=False)
        logger.info("[%s] CSV saved → %s", self.dataset_name, out_path)
        return out_path

    def save_json_summary(
        self,
        df: pd.DataFrame,
        out_dir: str | Path,
        filename: Optional[str] = None,
    ) -> Path:
        """Save a compact JSON summary of dataset statistics."""
        out_dir  = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname    = filename or f"{self.dataset_name}_summary.json"
        out_path = out_dir / fname

        summary: dict = {"dataset": self.dataset_name, "splits": {}}
        for split, grp in df.groupby("split"):
            per_image = grp.groupby("image_id")["gt_count"].first()
            summary["splits"][split] = {
                "n_images":       int(per_image.shape[0]),
                "total_persons":  int(per_image.sum()),
                "mean_count":     round(float(per_image.mean()), 2),
                "std_count":      round(float(per_image.std()), 2),
                "min_count":      int(per_image.min()),
                "max_count":      int(per_image.max()),
            }

        out_path.write_text(json.dumps(summary, indent=2))
        logger.info("[%s] JSON summary saved → %s", self.dataset_name, out_path)
        return out_path
