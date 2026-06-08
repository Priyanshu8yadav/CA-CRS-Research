"""
plots.py  (calibration)
───────────────────────
Matplotlib plotting helpers for the κ(β) calibration module.

All functions save directly to disk and return the output Path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Shared style ────────────────────────────────────────────────────────────

_STYLE = {
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":      "DejaVu Sans",
}


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)


# ── Plot functions ──────────────────────────────────────────────────────────

def plot_kappa_curve(
    beta: np.ndarray,
    kappa_obs: np.ndarray,
    alpha0: float,
    gamma: float,
    out_path: str | Path,
) -> Path:
    """
    Scatter of observed (β, κ) points with the fitted κ(β) curve overlay.
    """
    _apply_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    beta_line  = np.linspace(0, 1, 300)
    kappa_line = 1.0 + alpha0 * np.power(np.clip(beta_line, 1e-9, 1.0), gamma)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        beta, kappa_obs,
        s=20, alpha=0.55, color="#4C72B0", label="Observed κ",
        zorder=3,
    )
    ax.plot(
        beta_line, kappa_line,
        color="#DD4949", linewidth=2.2,
        label=f"Fitted κ(β) = 1 + {alpha0:.4f}·β^{gamma:.4f}",
        zorder=4,
    )

    ax.set_xlabel("Occlusion Ratio β", fontsize=12)
    ax.set_ylabel("Correction Factor κ", fontsize=12)
    ax.set_title("Occlusion Correction Factor — κ(β) Fit", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    title: str = "Actual vs Predicted",
) -> Path:
    """Scatter of actual vs predicted values with identity line."""
    _apply_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lo = min(y_true.min(), y_pred.min()) * 0.95
    hi = max(y_true.max(), y_pred.max()) * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=20, alpha=0.55, color="#2ca02c", zorder=3)
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1.2, label="Perfect fit")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    xlabel: str = "Actual",
    title: str = "Residuals",
) -> Path:
    """Residual plot: (actual - predicted) vs actual."""
    _apply_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_true, residuals, s=20, alpha=0.55, color="#9467bd", zorder=3)
    ax.axhline(0, color="grey", linewidth=1.2, linestyle="--")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Residual (actual − predicted)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
