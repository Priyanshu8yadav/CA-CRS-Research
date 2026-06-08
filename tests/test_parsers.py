"""
tests/test_parsers.py
──────────────────────
Unit tests for all three dataset parsers.

These tests run against the real data on disk (paths from config.yaml).
They verify the output schema, non-empty frames, and basic count sanity.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parsers.base_parser import SCHEMA_COLUMNS
from parsers.shanghaitech_parser import ShanghaiTechParser
from parsers.ucf_qnrf_parser import UCFQNRFParser
from parsers.pets2009_parser import PETS2009Parser


@pytest.fixture(scope="session")
def cfg() -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with cfg_path.open() as f:
        return yaml.safe_load(f)


# ── Schema helper ─────────────────────────────────────────────────────────────

def assert_schema(df: pd.DataFrame, dataset_name: str) -> None:
    """Assert all required columns are present and basic dtypes hold."""
    for col in SCHEMA_COLUMNS:
        assert col in df.columns, f"[{dataset_name}] Missing column: {col}"
    assert (df["gt_count"] >= 0).all(), f"[{dataset_name}] Negative gt_count found"
    assert df["image_id"].notna().all(), f"[{dataset_name}] Null image_id found"
    assert df["split"].isin(["train", "test"]).all(), \
        f"[{dataset_name}] Unknown split values"


# ── ShanghaiTech ─────────────────────────────────────────────────────────────

class TestShanghaiTechParser:

    def test_parse_returns_nonempty_dataframe(self, cfg):
        parser = ShanghaiTechParser(root=cfg["datasets"]["shanghaitech"], part="both")
        df = parser.parse()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "ShanghaiTech parse returned empty DataFrame"

    def test_schema_columns(self, cfg):
        parser = ShanghaiTechParser(root=cfg["datasets"]["shanghaitech"], part="both")
        df = parser.parse()
        assert_schema(df, "shanghaitech")

    def test_both_parts_present(self, cfg):
        parser = ShanghaiTechParser(root=cfg["datasets"]["shanghaitech"], part="both")
        df = parser.parse()
        datasets = df["dataset"].unique()
        assert "shanghaitech_A" in datasets, "Part A missing"
        assert "shanghaitech_B" in datasets, "Part B missing"

    def test_both_splits_present(self, cfg):
        parser = ShanghaiTechParser(root=cfg["datasets"]["shanghaitech"], part="both")
        df = parser.parse()
        splits = df["split"].unique()
        assert "train" in splits
        assert "test" in splits

    def test_gt_counts_reasonable(self, cfg):
        parser = ShanghaiTechParser(root=cfg["datasets"]["shanghaitech"], part="A")
        df = parser.parse()
        per_image = df.groupby("image_id")["gt_count"].first()
        # Part A has dense crowds; typical counts 33–3139
        assert per_image.max() > 50, "Max count unexpectedly low for Part A"
        assert (per_image >= 0).all()

    def test_dot_annotations_present(self, cfg):
        parser = ShanghaiTechParser(root=cfg["datasets"]["shanghaitech"], part="A")
        df = parser.parse()
        # Most rows should have cx/cy
        has_cx = df["cx"].notna().mean()
        assert has_cx > 0.5, f"Too few rows have cx annotation: {has_cx:.2%}"


# ── UCF-QNRF ──────────────────────────────────────────────────────────────────

class TestUCFQNRFParser:

    def test_parse_returns_nonempty_dataframe(self, cfg):
        parser = UCFQNRFParser(root=cfg["datasets"]["ucf_qnrf"])
        df = parser.parse()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_schema_columns(self, cfg):
        parser = UCFQNRFParser(root=cfg["datasets"]["ucf_qnrf"])
        df = parser.parse()
        assert_schema(df, "ucf_qnrf")

    def test_both_splits(self, cfg):
        parser = UCFQNRFParser(root=cfg["datasets"]["ucf_qnrf"])
        df = parser.parse()
        assert "train" in df["split"].values
        assert "test"  in df["split"].values

    def test_counts_reasonable(self, cfg):
        parser = UCFQNRFParser(root=cfg["datasets"]["ucf_qnrf"])
        df = parser.parse()
        per_image = df.groupby("image_id")["gt_count"].first()
        # UCF-QNRF: 49–12865 heads
        assert per_image.max() > 100
        assert (per_image >= 0).all()


# ── PETS2009 ──────────────────────────────────────────────────────────────────

class TestPETS2009Parser:

    def test_parse_returns_nonempty_dataframe(self, cfg):
        parser = PETS2009Parser(root=cfg["datasets"]["pets2009"])
        df = parser.parse()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_schema_columns(self, cfg):
        parser = PETS2009Parser(root=cfg["datasets"]["pets2009"])
        df = parser.parse()
        assert_schema(df, "pets2009")

    def test_split_is_test(self, cfg):
        parser = PETS2009Parser(root=cfg["datasets"]["pets2009"])
        df = parser.parse()
        assert (df["split"] == "test").all(), "PETS2009 should all be 'test' split"

    def test_frame_id_nonnegative(self, cfg):
        parser = PETS2009Parser(root=cfg["datasets"]["pets2009"])
        df = parser.parse()
        assert (df["frame_id"] >= 0).all()

    def test_trajectory_ids_present(self, cfg):
        parser = PETS2009Parser(root=cfg["datasets"]["pets2009"])
        df = parser.parse()
        assert df["trajectory_id"].notna().all()
        assert (df["trajectory_id"] != "").any(), "Trajectory IDs should not all be empty"

    def test_bbox_columns_populated(self, cfg):
        parser = PETS2009Parser(root=cfg["datasets"]["pets2009"])
        df = parser.parse()
        assert df["bbox_x"].notna().mean() > 0.8, "PETS2009 should have bbox annotations"
        assert df["bbox_w"].notna().mean() > 0.8
        assert (df["bbox_w"].dropna() > 0).all(), "bbox_w must be positive"
        assert (df["bbox_h"].dropna() > 0).all(), "bbox_h must be positive"
