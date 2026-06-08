"""
analyzer.py
────────────
Evaluate the CA-CRS+ risk formula on Saltelli samples, run SALib Sobol'
analysis, and produce a ranked sensitivity report.

CA-CRS+ Formula (from risk_scoring.py — read-only reference)
─────────────────────────────────────────────────────────────
  CRS = min(w1·D + Φ(D,S) + w3·C, 1.0)
  Φ(D,S) = w2·S·(1-D) + γ_exp · exp(λ·(D-S))

The weights are re-normalised so that w1+w2+w3 = 1 even when sampled
independently by SALib. This avoids artefacts from infeasible weight sums.

Fixed (non-sampled) inputs use representative mid-range values:
  D_norm  = 0.55   (moderately dense crowd)
  S_norm  = 0.40   (moderate speed)
  C_norm  = 0.35   (moderate conflict)

These values are chosen to keep the system in the WARNING zone where
parameter sensitivity is highest and most practically meaningful.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Representative "mid-range" crowd state for sensitivity evaluation
_D_NORM = 0.55
_S_NORM = 0.40
_C_NORM = 0.35

_FIX_THRESHOLD_ST = 0.05   # total-order index below which param → fix


def _cacrs_score(
    w1: float, w2: float, w3: float,
    gamma_exp: float, lam: float,
    d: float = _D_NORM,
    s: float = _S_NORM,
    c: float = _C_NORM,
) -> float:
    """
    Evaluate CA-CRS+ score with weight renormalisation.
    Clamps to [0, 1].
    """
    # Re-normalise weights so they always sum to 1.0
    total = w1 + w2 + w3
    if total < 1e-9:
        return 0.0
    w1 /= total
    w2 /= total
    w3 /= total

    # Φ term — exponential gridlock crush penalty
    phi = w2 * s * (1.0 - d) + gamma_exp * math.exp(lam * (d - s))

    crs = w1 * d + phi + w3 * c
    return float(min(max(crs, 0.0), 1.0))


def evaluate_model(samples: np.ndarray) -> np.ndarray:
    """
    Evaluate the CA-CRS+ risk formula on all Saltelli samples.

    Parameters
    ----------
    samples : np.ndarray, shape (M, 5)
        Columns order: w1, w2, w3, gamma_exp, lambda

    Returns
    -------
    np.ndarray, shape (M,) — CRS values
    """
    Y = np.empty(len(samples))
    for i, row in enumerate(samples):
        w1, w2, w3, gamma_exp, lam = row
        Y[i] = _cacrs_score(w1, w2, w3, gamma_exp, lam)
    return Y


def run_sobol(
    problem: dict,
    samples: np.ndarray,
    Y: np.ndarray,
    calc_second_order: bool = True,
) -> dict:
    """
    Run SALib Sobol' analysis and return the full indices dictionary.

    Parameters
    ----------
    problem : SALib problem dict
    samples : Saltelli sample matrix
    Y       : model output vector
    calc_second_order : bool

    Returns
    -------
    SALib Si dict with keys: S1, S1_conf, ST, ST_conf, (S2, S2_conf)
    """
    try:
        from SALib.analyze import sobol as sobol_analyzer
    except ImportError as e:
        raise ImportError("SALib is required. pip install SALib>=1.4.7") from e

    Si = sobol_analyzer.analyze(
        problem,
        Y,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )
    return Si


def rank_parameters(
    problem: dict,
    Si: dict,
    fix_threshold: float = _FIX_THRESHOLD_ST,
) -> pd.DataFrame:
    """
    Produce a ranked DataFrame of Sobol' indices.

    Returns
    -------
    pd.DataFrame with columns:
        parameter, S1, S1_conf, ST, ST_conf, rank_ST, recommendation
    """
    names  = problem["names"]
    s1     = np.asarray(Si["S1"])
    s1_c   = np.asarray(Si["S1_conf"])
    st     = np.asarray(Si["ST"])
    st_c   = np.asarray(Si["ST_conf"])

    df = pd.DataFrame({
        "parameter": names,
        "S1":        s1.round(6),
        "S1_conf":   s1_c.round(6),
        "ST":        st.round(6),
        "ST_conf":   st_c.round(6),
    })
    df["rank_ST"] = df["ST"].rank(ascending=False).astype(int)
    df["recommendation"] = df["ST"].apply(
        lambda x: "FIX (insensitive)" if x < fix_threshold else "TUNE (sensitive)"
    )
    df = df.sort_values("ST", ascending=False).reset_index(drop=True)
    return df


def save_results(
    Si: dict,
    ranked: pd.DataFrame,
    problem: dict,
    output_dir: str | Path,
) -> None:
    """Save Sobol indices CSV and a human-readable sensitivity report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / "sobol_indices.csv"
    ranked.to_csv(csv_path, index=False)
    logger.info("[analyzer] Sobol indices saved → %s", csv_path)

    # Text report
    report_path = output_dir / "sensitivity_report.txt"
    with report_path.open("w") as f:
        f.write("=" * 60 + "\n")
        f.write("CA-CRS+ Sobol' Sensitivity Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Fixed crowd state used for evaluation:\n")
        f.write(f"  D_norm={_D_NORM}, S_norm={_S_NORM}, C_norm={_C_NORM}\n\n")
        f.write("Parameter Rankings by Total-Order Index (ST)\n")
        f.write("-" * 60 + "\n")
        f.write(ranked.to_string(index=False))
        f.write("\n\n")

        tune_params = ranked[ranked["recommendation"].str.startswith("TUNE")]["parameter"].tolist()
        fix_params  = ranked[ranked["recommendation"].str.startswith("FIX")]["parameter"].tolist()

        f.write("Recommendations\n")
        f.write("-" * 60 + "\n")
        f.write(f"Tune  (ST ≥ {_FIX_THRESHOLD_ST}): {', '.join(tune_params) or 'None'}\n")
        f.write(f"Fix   (ST < {_FIX_THRESHOLD_ST}): {', '.join(fix_params)  or 'None'}\n\n")
        f.write(
            "Parameters classified as FIX contribute less than "
            f"{_FIX_THRESHOLD_ST*100:.0f}% of total output variance.\n"
            "They can be held at their nominal defaults without meaningful\n"
            "loss of model accuracy.\n"
        )

    logger.info("[analyzer] Sensitivity report saved → %s", report_path)
    print(f"\n── Sensitivity Report saved to {report_path}\n")
    print(ranked.to_string(index=False))
