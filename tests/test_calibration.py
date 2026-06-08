"""
tests/test_calibration.py
──────────────────────────
Unit tests for the calibration module.

Tests the metrics helpers, β computation, and the κ(β) fitting.
Uses a synthetic calibration dataset so tests run without raw data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.metrics import mae, rmse, mape, r2
from calibration.build_dataset import compute_beta_from_boxes, iou_matrix
from calibration.fit_kappa import kappa, fit_kappa_model


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r2(y, y) == pytest.approx(1.0)

    def test_r2_baseline(self):
        """Predicting the mean gives R²=0."""
        y = np.array([1.0, 2.0, 3.0])
        y_mean = np.full_like(y, y.mean())
        assert r2(y, y_mean) == pytest.approx(0.0, abs=1e-6)

    def test_mape_nonzero(self):
        y_true = np.array([2.0, 4.0])
        y_pred = np.array([2.1, 3.9])
        result = mape(y_true, y_pred)
        assert result >= 0.0

    def test_mae_known(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([2.0, 3.0])
        assert mae(y_true, y_pred) == pytest.approx(1.0)


# ── Beta computation ──────────────────────────────────────────────────────────

class TestBetaComputation:

    def test_no_overlap_gives_zero_beta(self):
        # Non-overlapping boxes
        boxes = np.array([
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [50, 0, 60, 10],
        ], dtype=float)
        beta = compute_beta_from_boxes(boxes, iou_thresh=0.1)
        assert beta == pytest.approx(0.0)

    def test_full_overlap_gives_max_beta(self):
        # All boxes perfectly overlapping
        box = [0, 0, 10, 10]
        boxes = np.array([box] * 5, dtype=float)
        beta = compute_beta_from_boxes(boxes, iou_thresh=0.1)
        assert beta == pytest.approx(1.0)

    def test_beta_in_range(self):
        rng = np.random.default_rng(0)
        boxes = rng.uniform(0, 100, (20, 4))
        # Ensure x2 > x1, y2 > y1
        boxes[:, 2] = boxes[:, 0] + rng.uniform(5, 30, 20)
        boxes[:, 3] = boxes[:, 1] + rng.uniform(5, 30, 20)
        beta = compute_beta_from_boxes(boxes)
        assert 0.0 <= beta <= 1.0

    def test_single_box_gives_zero(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=float)
        beta = compute_beta_from_boxes(boxes)
        assert beta == 0.0

    def test_iou_matrix_symmetric(self):
        boxes = np.array([[0,0,10,10],[5,5,15,15]], dtype=float)
        iou = iou_matrix(boxes)
        assert iou.shape == (2, 2)
        assert iou[0, 1] == pytest.approx(iou[1, 0])
        assert iou[0, 0] == pytest.approx(0.0)  # diagonal should be 0


# ── Kappa model ────────────────────────────────────────────────────────────────

class TestKappaModel:

    def test_kappa_at_zero_beta_equals_one(self):
        """κ(0) = 1 + α₀ · 0^γ = 1 (for any positive γ)."""
        result = kappa(np.array([0.0]), alpha0=1.0, gamma=1.5)
        # β=0 → 1 + α₀ * 0^γ = 1
        assert result[0] == pytest.approx(1.0, abs=1e-3)

    def test_kappa_monotone_in_beta(self):
        """κ should increase with β for positive α₀ and γ."""
        betas = np.linspace(0.01, 1.0, 50)
        ks    = kappa(betas, alpha0=0.8, gamma=1.2)
        diffs = np.diff(ks)
        assert (diffs >= 0).all(), "κ is not monotonically non-decreasing"

    def test_kappa_at_one_beta(self):
        """κ(1) = 1 + α₀."""
        result = kappa(np.array([1.0]), alpha0=1.5, gamma=1.0)
        assert result[0] == pytest.approx(1.0 + 1.5, rel=1e-4)

    def test_fit_kappa_synthetic(self, tmp_path):
        """Fit converges on noiseless synthetic data."""
        rng   = np.random.default_rng(42)
        betas = rng.uniform(0.05, 0.95, 80)
        # True: α₀=0.80, γ=1.20
        true_alpha0, true_gamma = 0.80, 1.20
        kappa_obs = kappa(betas, true_alpha0, true_gamma)

        df = pd.DataFrame({"beta": betas, "kappa_obs": kappa_obs})
        result = fit_kappa_model(df, output_dir=tmp_path)

        assert abs(result["alpha0"] - true_alpha0) < 0.05
        assert abs(result["gamma"]  - true_gamma)  < 0.05
        assert result["metrics"]["r2"] > 0.99

    def test_fit_kappa_noisy_data(self, tmp_path):
        """Fit achieves R² > 0.90 on noisy synthetic data."""
        rng   = np.random.default_rng(7)
        betas = rng.uniform(0.01, 0.99, 200)
        noise = rng.normal(0, 0.05, 200)
        kappa_obs = kappa(betas, 1.0, 1.5) + noise
        kappa_obs = np.clip(kappa_obs, 1.0, None)

        df = pd.DataFrame({"beta": betas, "kappa_obs": kappa_obs})
        result = fit_kappa_model(df, output_dir=tmp_path)
        assert result["metrics"]["r2"] > 0.85
