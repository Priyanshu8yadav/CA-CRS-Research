"""
plots.py  (sensitivity)
────────────────────────
Matplotlib plotting helpers for Sobol' sensitivity analysis.

Generates:
  - Horizontal bar chart  : S1 and ST per parameter
  - Second-order heatmap  : S2 interaction strengths (if available)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


_STYLE = {
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
}


def plot_sobol_bar(
    ranked: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """
    Horizontal bar chart showing S1 (first-order) and ST (total-order)
    Sobol' indices per parameter, sorted by ST descending.

    Parameters
    ----------
    ranked : pd.DataFrame
        Output of sensitivity.analyzer.rank_parameters().
        Must contain: parameter, S1, S1_conf, ST, ST_conf.
    out_path : str | Path
    """
    plt.rcParams.update(_STYLE)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = ranked["parameter"].tolist()
    s1    = ranked["S1"].clip(lower=0).tolist()
    st    = ranked["ST"].clip(lower=0).tolist()
    s1_c  = ranked["S1_conf"].tolist()
    st_c  = ranked["ST_conf"].tolist()

    y     = np.arange(len(names))
    height = 0.35

    fig, ax = plt.subplots(figsize=(8, 4 + len(names) * 0.4))
    bars_st = ax.barh(
        y + height / 2, st, height,
        xerr=st_c, color="#DD4949", alpha=0.85,
        label="Total-order (ST)", capsize=4,
    )
    bars_s1 = ax.barh(
        y - height / 2, s1, height,
        xerr=s1_c, color="#4C72B0", alpha=0.85,
        label="First-order (S1)", capsize=4,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Sobol' Index", fontsize=12)
    ax.set_title(
        "CA-CRS+ Risk Parameter Sensitivity (Sobol')",
        fontsize=13, fontweight="bold",
    )
    ax.axvline(0.05, color="grey", linestyle="--", linewidth=1.0, alpha=0.6,
               label="Fix threshold (0.05)")
    ax.legend(fontsize=10)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_sobol_heatmap(
    Si: dict[str, Any],
    problem: dict[str, Any],
    out_path: str | Path,
) -> Path | None:
    """
    Heatmap of second-order Sobol' interaction indices (S2).

    Returns None if second-order indices are not present in Si.
    """
    plt.rcParams.update(_STYLE)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if "S2" not in Si or Si["S2"] is None:
        return None

    names = problem["names"]
    s2    = np.array(Si["S2"])

    # S2 is upper-triangular; mirror for display
    n = len(names)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            val = max(float(s2[i, j]), 0.0)
            matrix[i, j] = val
            matrix[j, i] = val

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        xticklabels=names,
        yticklabels=names,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        vmin=0,
    )
    ax.set_title(
        "Second-Order Sobol' Interactions (S2)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
