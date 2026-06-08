"""
recalibrate.py
───────────────
Post-process the YOLO detection CSV and re-fit κ in two ways:

(A) β-based fit:  κ(β) = 1 + α₀·β^γ
    Fits using pairwise IoU overlap ratio β from YOLO detected boxes.
    Correlation analysis shows β vs κ is weak (-0.07), so this fit
    is provided for completeness but may have low R².

(B) Density-based fit:  κ(ρ) = 1 + α₀·ρ^γ
    Uses normalised GT crowd density ρ = gt_count / MAX_GT as the
    predictor. The correlation between detection_rate and κ is -0.65,
    indicating crowd density (not β overlap) is the dominant driver
    of YOLO's undercounting on these datasets. This is the primary
    fit and is recommended for use in the CA-CRS+ pipeline.

Quality filter (applied to both):
  detection_rate = detected / gt_count >= 0.10
  κ_obs <= MAX_KAPPA (formula-reachable range)
  detected >= 2

Usage:
    python -m calibration.recalibrate
    python -m calibration.recalibrate --min-rate 0.15 --max-kappa 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.metrics import mae, rmse, mape, r2

logger = logging.getLogger(__name__)

# ── Quality filter defaults ───────────────────────────────────────────────────
MIN_DETECTION_RATE = 0.10   # YOLO must detect ≥ 10% of GT people
MAX_KAPPA          = 15.0   # Formula-reachable κ upper cap
MIN_BETA           = 0.02   # Minimum meaningful overlap
MIN_DETECTED       = 2      # Need ≥2 boxes for IoU computation


def kappa_fn(beta: np.ndarray, alpha0: float, gamma: float) -> np.ndarray:
    """κ(β) = 1 + α₀·β^γ"""
    return 1.0 + alpha0 * np.power(np.clip(beta, 1e-9, None), gamma)


def load_and_filter(
    yolo_csv: Path,
    min_rate: float = MIN_DETECTION_RATE,
    max_kappa: float = MAX_KAPPA,
    min_beta: float  = MIN_BETA,
    min_detected: int = MIN_DETECTED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load yolo_calibration_dataset.csv and apply quality filters.

    Returns (filtered_df, full_df).
    """
    df = pd.read_csv(yolo_csv)
    logger.info("[recal] Loaded %d rows from %s", len(df), yolo_csv)

    # Compute detection rate
    df["detection_rate"] = df["detected"] / df["gt_count"].clip(lower=1)

    # Log dataset-level stats before filtering
    print("\n── Per-Dataset Detection Stats (before filter) ──")
    for ds, grp in df.groupby("dataset"):
        print(f"  {ds:<22} images={len(grp):4d}  "
              f"det_rate={grp['detection_rate'].mean():.3f}  "
              f"κ_obs_mean={grp['kappa_obs'].mean():.1f}  "
              f"κ_obs_med={grp['kappa_obs'].median():.1f}")

    # Apply filters
    mask = (
        (df["detection_rate"] >= min_rate) &
        (df["kappa_obs"]      <= max_kappa) &
        (df["beta"]           >= min_beta)  &
        (df["detected"]       >= min_detected)
    )
    filtered = df[mask].copy()

    logger.info(
        "[recal] Filtered: %d → %d rows (%.1f%%)",
        len(df), len(filtered), 100 * len(filtered) / max(len(df), 1),
    )

    if len(filtered) < 30:
        logger.warning("[recal] Very few samples (%d) after filtering. "
                       "Consider loosening --min-rate or --max-kappa.", len(filtered))

    print("\n── Per-Dataset Detection Stats (after filter) ──")
    if len(filtered) == 0:
        print("  (no rows passed filter)")
    else:
        for ds, grp in filtered.groupby("dataset"):
            print(f"  {ds:<22} images={len(grp):4d}  "
                  f"β_mean={grp['beta'].mean():.3f}  "
                  f"κ_obs_mean={grp['kappa_obs'].mean():.3f}")

    return filtered, df


def fit_on_filtered(
    df: pd.DataFrame,
    alpha0_init: float = 0.80,
    gamma_init: float  = 1.20,
    alpha0_bounds: tuple[float, float] = (0.05, 5.0),
    gamma_bounds:  tuple[float, float] = (0.20, 3.0),
    output_dir: Path = Path("outputs/calibration"),
) -> dict:
    """Fit κ(β) on the filtered dataset and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    beta    = df["beta"].values
    kap_obs = df["kappa_obs"].values

    logger.info("[recal] Fitting on %d samples — β∈[%.3f, %.3f], κ∈[%.3f, %.3f]",
                len(df), beta.min(), beta.max(), kap_obs.min(), kap_obs.max())

    try:
        popt, pcov = curve_fit(
            kappa_fn, beta, kap_obs,
            p0=[alpha0_init, gamma_init],
            bounds=([alpha0_bounds[0], gamma_bounds[0]],
                    [alpha0_bounds[1], gamma_bounds[1]]),
            method="trf",
            maxfev=10000,
        )
        alpha0_fit, gamma_fit = float(popt[0]), float(popt[1])
        perr = np.sqrt(np.diag(pcov))
        logger.info("[recal] Fitted α₀=%.6f (±%.4f)  γ=%.6f (±%.4f)",
                    alpha0_fit, perr[0], gamma_fit, perr[1])
    except Exception as exc:
        logger.error("[recal] curve_fit failed: %s — using initial values", exc)
        alpha0_fit, gamma_fit = alpha0_init, gamma_init
        perr = [0.0, 0.0]

    kap_pred = kappa_fn(beta, alpha0_fit, gamma_fit)
    metrics = {
        "mae":  float(mae(kap_obs, kap_pred)),
        "rmse": float(rmse(kap_obs, kap_pred)),
        "mape": float(mape(kap_obs, kap_pred)),
        "r2":   float(r2(kap_obs, kap_pred)),
    }

    print(f"\n── κ(β) Fit Metrics (YOLO-calibrated, filtered) ────")
    print(f"  α₀   = {alpha0_fit:.6f}  (±{perr[0]:.4f})")
    print(f"  γ    = {gamma_fit:.6f}  (±{perr[1]:.4f})")
    print(f"  MAE  : {metrics['mae']:.6f}")
    print(f"  RMSE : {metrics['rmse']:.6f}")
    print(f"  MAPE : {metrics['mape']:.4f} %")
    print(f"  R²   : {metrics['r2']:.6f}")

    result = {
        "alpha0":         alpha0_fit,
        "gamma":          gamma_fit,
        "alpha0_stderr":  float(perr[0]),
        "gamma_stderr":   float(perr[1]),
        "n_samples":      len(df),
        "filter": {
            "min_detection_rate": MIN_DETECTION_RATE,
            "max_kappa":          MAX_KAPPA,
            "min_beta":           MIN_BETA,
            "min_detected":       MIN_DETECTED,
        },
        "metrics": metrics,
        "source": "YOLOv8m inference on ShanghaiTech + UCF-QNRF",
    }

    params_path = output_dir / "kappa_params_yolo.json"
    params_path.write_text(json.dumps(result, indent=2))
    logger.info("[recal] YOLO-calibrated params saved → %s", params_path)

    _make_plots(df, beta, kap_obs, kap_pred, alpha0_fit, gamma_fit, output_dir)
    return result


def _make_plots(
    df: pd.DataFrame,
    beta: np.ndarray,
    kap_obs: np.ndarray,
    kap_pred: np.ndarray,
    alpha0: float,
    gamma: float,
    output_dir: Path,
) -> None:
    """Generate three diagnostic plots for the YOLO-calibrated fit."""
    b_line = np.linspace(0, max(beta.max(), 0.8), 200)
    k_line = kappa_fn(b_line, alpha0, gamma)

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # ── Plot 1: κ(β) scatter + fitted curve ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    colors = {"shanghaitech_A": "#4C72B0", "shanghaitech_B": "#55A868",
              "ucf_qnrf": "#DD4949"}
    for ds, grp in df.groupby("dataset"):
        ax1.scatter(grp["beta"], grp["kappa_obs"],
                    s=12, alpha=0.5, label=ds,
                    color=colors.get(ds, "grey"))
    ax1.plot(b_line, k_line, "k-", linewidth=2,
             label=f"Fit: 1+{alpha0:.3f}·β^{gamma:.3f}")
    ax1.set_xlabel("β (overlap ratio)", fontsize=11)
    ax1.set_ylabel("κ observed (GT / YOLO)", fontsize=11)
    ax1.set_title("κ(β) Scatter — YOLO Data", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, None)
    ax1.set_ylim(0.9, None)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Plot 2: Actual vs Predicted κ ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(kap_obs, kap_pred, s=12, alpha=0.4, color="#4C72B0")
    lim = max(kap_obs.max(), kap_pred.max()) * 1.05
    ax2.plot([1, lim], [1, lim], "r--", linewidth=1.5, label="Ideal")
    ax2.set_xlabel("Observed κ", fontsize=11)
    ax2.set_ylabel("Predicted κ", fontsize=11)
    ax2.set_title("Actual vs Predicted κ", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Plot 3: Residuals vs β ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    residuals = kap_obs - kap_pred
    ax3.scatter(beta, residuals, s=12, alpha=0.4, color="#DD4949")
    ax3.axhline(0, color="k", linewidth=1.5, linestyle="--")
    ax3.set_xlabel("β (overlap ratio)", fontsize=11)
    ax3.set_ylabel("Residual (obs − pred)", fontsize=11)
    ax3.set_title("Residuals vs β", fontsize=12, fontweight="bold")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.suptitle(
        f"κ(β) = 1 + {alpha0:.4f}·β^{gamma:.4f} — YOLO-Calibrated  "
        f"(n={len(beta)}, R²={float(r2(kap_obs, kap_pred)):.3f})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "kappa_fit_yolo.png", dpi=150)
    plt.close(fig)
    logger.info("[recal] YOLO calibration plot saved → %s",
                output_dir / "kappa_fit_yolo.png")


def fit_density_based(
    df: pd.DataFrame,
    output_dir: Path = Path("outputs/calibration"),
) -> dict:
    """
    Fit κ(ρ) = 1 + α₀·ρ^γ where ρ = gt_count / max_gt_count.

    This is the primary fit because:
      - β (from YOLO detected boxes) has ~0 correlation with κ
      - GT density has -0.65 correlation with detection_rate → strong κ predictor
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    max_gt = df["gt_count"].max()
    df = df.copy()
    df["rho_norm"] = df["gt_count"] / max_gt

    rho     = df["rho_norm"].values
    kap_obs = df["kappa_obs"].values

    logger.info("[density-fit] Fitting κ(ρ) on %d samples — "
                "ρ∈[%.3f, %.3f], κ∈[%.3f, %.3f]",
                len(df), rho.min(), rho.max(), kap_obs.min(), kap_obs.max())

    try:
        popt, pcov = curve_fit(
            kappa_fn, rho, kap_obs,
            p0=[2.0, 0.5],
            bounds=([0.05, 0.1], [10.0, 3.0]),
            method="trf",
            maxfev=10000,
        )
        alpha0_fit, gamma_fit = float(popt[0]), float(popt[1])
        perr = np.sqrt(np.diag(pcov))
        logger.info("[density-fit] Fitted α₀=%.6f (±%.4f)  γ=%.6f (±%.4f)",
                    alpha0_fit, perr[0], gamma_fit, perr[1])
    except Exception as exc:
        logger.error("[density-fit] curve_fit failed: %s", exc)
        alpha0_fit, gamma_fit = 2.0, 0.5
        perr = [0.0, 0.0]

    kap_pred = kappa_fn(rho, alpha0_fit, gamma_fit)
    metrics = {
        "mae":  float(mae(kap_obs, kap_pred)),
        "rmse": float(rmse(kap_obs, kap_pred)),
        "mape": float(mape(kap_obs, kap_pred)),
        "r2":   float(r2(kap_obs, kap_pred)),
    }

    print(f"\n── κ(ρ) Density-Based Fit Metrics ─────────────────")
    print(f"  α₀   = {alpha0_fit:.6f}  (±{perr[0]:.4f})")
    print(f"  γ    = {gamma_fit:.6f}  (±{perr[1]:.4f})")
    print(f"  MAE  : {metrics['mae']:.6f}")
    print(f"  RMSE : {metrics['rmse']:.6f}")
    print(f"  MAPE : {metrics['mape']:.4f} %")
    print(f"  R²   : {metrics['r2']:.6f}")

    result = {
        "alpha0":        alpha0_fit,
        "gamma":         gamma_fit,
        "alpha0_stderr": float(perr[0]),
        "gamma_stderr":  float(perr[1]),
        "predictor":     "rho_norm (gt_count / max_gt_count)",
        "max_gt_count":  int(max_gt),
        "n_samples":     len(df),
        "metrics":       metrics,
        "note": ("Primary calibration. β from YOLO detected boxes has near-zero "
                 "correlation with κ (-0.07); GT density has -0.65 correlation "
                 "with detection_rate and is the dominant predictor."),
        "source": "YOLOv8m inference on ShanghaiTech + UCF-QNRF",
    }

    params_path = output_dir / "kappa_params_density.json"
    params_path.write_text(json.dumps(result, indent=2))
    logger.info("[density-fit] Params saved → %s", params_path)

    # Plot
    rho_line = np.linspace(0, 1, 200)
    k_line   = kappa_fn(rho_line, alpha0_fit, gamma_fit)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"shanghaitech_A": "#4C72B0", "shanghaitech_B": "#55A868",
              "ucf_qnrf": "#DD4949"}
    for ds, grp in df.groupby("dataset"):
        axes[0].scatter(grp["rho_norm"], grp["kappa_obs"],
                        s=10, alpha=0.4, label=ds, color=colors.get(ds, "grey"))
    axes[0].plot(rho_line, k_line, "k-", linewidth=2.5,
                 label=f"Fit: 1+{alpha0_fit:.3f}·ρ^{gamma_fit:.3f}")
    axes[0].set_xlabel("ρ_norm (gt_count / max_gt)", fontsize=11)
    axes[0].set_ylabel("κ observed (GT / YOLO)", fontsize=11)
    axes[0].set_title("κ(ρ) Density Calibration", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].scatter(kap_obs, kap_pred, s=10, alpha=0.4, color="#4C72B0")
    lim = max(kap_obs.max(), kap_pred.max()) * 1.05
    axes[1].plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Ideal")
    axes[1].set_xlabel("Observed κ", fontsize=11)
    axes[1].set_ylabel("Predicted κ", fontsize=11)
    axes[1].set_title("Actual vs Predicted (density model)", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    r2_val = metrics['r2']
    fig.suptitle(
        f"κ(ρ) = 1 + {alpha0_fit:.4f}·ρ^{gamma_fit:.4f} — Density-Calibrated  "
        f"(n={len(df)}, R²={r2_val:.3f})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "kappa_fit_density.png", dpi=150)
    plt.close(fig)
    logger.info("[density-fit] Plot saved → %s", output_dir / "kappa_fit_density.png")

    return result


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Re-calibrate κ on filtered YOLO data")
    parser.add_argument("--yolo-csv",  default="outputs/calibration/yolo_calibration_dataset.csv")
    parser.add_argument("--min-rate",  type=float, default=MIN_DETECTION_RATE)
    parser.add_argument("--max-kappa", type=float, default=MAX_KAPPA)
    parser.add_argument("--min-beta",  type=float, default=MIN_BETA)
    parser.add_argument("--output-dir", default="outputs/calibration")
    args = parser.parse_args()

    yolo_csv = Path(args.yolo_csv)
    if not yolo_csv.exists():
        logger.error("YOLO CSV not found: %s — run yolo_inference first.", yolo_csv)
        sys.exit(1)

    filtered_df, full_df = load_and_filter(
        yolo_csv,
        min_rate=args.min_rate,
        max_kappa=args.max_kappa,
        min_beta=args.min_beta,
    )

    if len(filtered_df) < 10:
        logger.error("Too few samples after filtering (%d). Aborting.", len(filtered_df))
        sys.exit(1)

    out_dir = Path(args.output_dir)

    # (A) β-based fit (legacy / secondary)
    print("\n" + "═"*60)
    print("(A) β-BASED FIT:  κ(β) = 1 + α₀·β^γ")
    print("═"*60)
    beta_result = fit_on_filtered(filtered_df, alpha0_init=1.5, gamma_init=0.8,
                                  output_dir=out_dir)

    # (B) Density-based fit (PRIMARY)
    print("\n" + "═"*60)
    print("(B) DENSITY-BASED FIT (PRIMARY):  κ(ρ) = 1 + α₀·ρ^γ")
    print("═"*60)
    density_result = fit_density_based(filtered_df, output_dir=out_dir)

    print(f"\n{'═'*60}")
    print("SUMMARY — κ Calibration Results")
    print(f"{'═'*60}")
    print(f"  (A) β-based :  κ(β) = 1 + {beta_result['alpha0']:.4f}·β^{beta_result['gamma']:.4f}")
    print(f"      R²={beta_result['metrics']['r2']:.4f}  (β has ~zero correlation with κ — weak predictor)")
    print(f"  (B) ρ-based :  κ(ρ) = 1 + {density_result['alpha0']:.4f}·ρ^{density_result['gamma']:.4f}")
    print(f"      R²={density_result['metrics']['r2']:.4f}  (density is the dominant predictor)")
    print(f"\n  ✓ RECOMMENDED: Use density-based κ(ρ) for the CA-CRS+ pipeline.")
    print(f"  ✓ Update config.yaml: alpha0={density_result['alpha0']:.4f}, gamma={density_result['gamma']:.4f}")
    print(f"\nOutputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
