"""
metrics.py
──────────
Regression metrics for evaluating κ(β) fit quality.

Implements: MAE, RMSE, MAPE, R²
All functions accept plain numpy arrays or lists.
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Mean Absolute Percentage Error (in %)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 1.0  # perfect fit when all targets are equal
    return float(1.0 - ss_res / ss_tot)


def report_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "Fit",
) -> dict[str, float]:
    """
    Compute and print all four metrics.

    Returns
    -------
    dict with keys: mae, rmse, mape, r2
    """
    results = {
        "mae":  mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2":   r2(y_true, y_pred),
    }
    print(f"\n── {label} Metrics ──────────────────────────────")
    print(f"  MAE  : {results['mae']:.6f}")
    print(f"  RMSE : {results['rmse']:.6f}")
    print(f"  MAPE : {results['mape']:.4f} %")
    print(f"  R²   : {results['r2']:.6f}")
    print()
    return results
