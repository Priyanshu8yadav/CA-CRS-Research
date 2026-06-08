"""
fit_kappa.py
─────────────
Fit the occlusion correction factor κ(β) = 1 + α₀ · β^γ
using scipy.optimize.curve_fit (non-linear least squares).

The calibration dataset is expected to have columns:
  beta       : float in [0, 1]  — estimated occlusion ratio
  kappa_obs  : float ≥ 1        — observed correction factor (gt / detected)

Outputs
───────
  outputs/calibration/kappa_params.json   — fitted α₀ and γ with metrics
  outputs/calibration/kappa_fit_curve.png — κ(β) curve overlay
  outputs/calibration/actual_vs_predicted.png
  outputs/calibration/residuals.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning

from calibration.metrics import report_metrics
from calibration import plots as cal_plots

logger = logging.getLogger(__name__)


# ── Model definition ────────────────────────────────────────────────────────

def kappa(beta: np.ndarray, alpha0: float, gamma: float) -> np.ndarray:
    """
    κ(β) = 1 + α₀ · β^γ

    Parameters
    ----------
    beta   : occlusion ratio, values in [0, 1]
    alpha0 : scale factor  (>0)
    gamma  : exponent      (>0)
    """
    beta = np.asarray(beta, dtype=float)
    return 1.0 + alpha0 * np.power(np.clip(beta, 1e-9, 1.0), gamma)


# ── Fitting ─────────────────────────────────────────────────────────────────

def fit_kappa_model(
    df: pd.DataFrame,
    alpha0_init: float = 0.80,
    gamma_init:  float = 1.20,
    alpha0_bounds: tuple[float, float] = (0.1, 5.0),
    gamma_bounds:  tuple[float, float] = (0.3, 3.0),
    output_dir: str | Path = "outputs/calibration",
) -> dict:
    """
    Fit κ(β) using scipy.optimize.curve_fit.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'beta' and 'kappa_obs'.
    alpha0_init, gamma_init : float
        Initial parameter guesses.
    alpha0_bounds, gamma_bounds : (lower, upper)
        Allowed parameter ranges.
    output_dir : str | Path
        Where to save parameters JSON and plots.

    Returns
    -------
    dict with keys: alpha0, gamma, metrics (dict), covariance (array)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data prep ──────────────────────────────────────────────────────────
    clean = df[["beta", "kappa_obs"]].dropna()
    clean = clean[(clean["beta"] >= 0) & (clean["kappa_obs"] >= 1.0)]

    if len(clean) < 5:
        raise ValueError(
            f"Not enough valid calibration points ({len(clean)}). "
            "Need at least 5 rows with beta ≥ 0 and kappa_obs ≥ 1."
        )

    beta_vals   = clean["beta"].values
    kappa_vals  = clean["kappa_obs"].values

    logger.info(
        "[fit_kappa] Fitting on %d samples — β∈[%.3f, %.3f], κ∈[%.3f, %.3f]",
        len(clean), beta_vals.min(), beta_vals.max(),
        kappa_vals.min(), kappa_vals.max(),
    )

    # ── Curve fit ──────────────────────────────────────────────────────────
    p0     = [alpha0_init, gamma_init]
    bounds = (
        [alpha0_bounds[0], gamma_bounds[0]],
        [alpha0_bounds[1], gamma_bounds[1]],
    )

    try:
        popt, pcov = curve_fit(
            kappa,
            beta_vals,
            kappa_vals,
            p0=p0,
            bounds=bounds,
            maxfev=10_000,
            method="trf",          # Trust-Region Reflective — respects bounds
        )
    except (RuntimeError, OptimizeWarning) as exc:
        logger.error("[fit_kappa] curve_fit failed: %s", exc)
        raise

    alpha0_fit, gamma_fit = float(popt[0]), float(popt[1])
    kappa_pred = kappa(beta_vals, alpha0_fit, gamma_fit)

    logger.info(
        "[fit_kappa] Fitted α₀=%.6f  γ=%.6f", alpha0_fit, gamma_fit
    )

    # ── Metrics ────────────────────────────────────────────────────────────
    metrics = report_metrics(kappa_vals, kappa_pred, label="κ(β) Fit")

    # ── Persist parameters ─────────────────────────────────────────────────
    params_out = {
        "alpha0":     alpha0_fit,
        "gamma":      gamma_fit,
        "n_samples":  len(clean),
        "metrics":    metrics,
    }
    params_path = output_dir / "kappa_params.json"
    params_path.write_text(json.dumps(params_out, indent=2))
    logger.info("[fit_kappa] Parameters saved → %s", params_path)

    # ── Plots ──────────────────────────────────────────────────────────────
    cal_plots.plot_kappa_curve(
        beta_vals, kappa_vals, alpha0_fit, gamma_fit,
        output_dir / "kappa_fit_curve.png",
    )
    cal_plots.plot_actual_vs_predicted(
        kappa_vals, kappa_pred,
        output_dir / "actual_vs_predicted.png",
        xlabel="κ observed",
        ylabel="κ predicted",
        title="Actual vs Predicted κ(β)",
    )
    cal_plots.plot_residuals(
        kappa_vals, kappa_pred,
        output_dir / "residuals.png",
        xlabel="κ observed",
        title="Residuals — κ(β) Fit",
    )

    return {
        "alpha0":     alpha0_fit,
        "gamma":      gamma_fit,
        "metrics":    metrics,
        "covariance": pcov.tolist(),
    }
