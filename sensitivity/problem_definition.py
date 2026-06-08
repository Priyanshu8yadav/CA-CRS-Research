"""
problem_definition.py
──────────────────────
Defines the SALib problem dictionary for CA-CRS+ Sobol' sensitivity analysis.

CA-CRS+ Risk Formula (from risk_scoring.py)
────────────────────────────────────────────
  CRS(k) = w1·D_norm + Φ(D,S) + w3·C_norm
  Φ(D,S) = w2·S·(1-D) + γ_exp · exp(λ·(D-S))

Five tunable parameters:
  w1       : density weight
  w2       : speed / gridlock weight
  w3       : directional conflict weight
  gamma_exp: amplitude of the exponential crush penalty
  lambda   : sharpness of the exponential crush penalty

The constraint w1 + w2 + w3 = 1 is NOT enforced during sampling
(SALib assumes independent parameters). Instead, a normalisation step
is applied inside the model evaluator so the formula always sums correctly.

Bounds are loaded from config.yaml but can be overridden at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# ── Default bounds (used if config.yaml is unavailable) ─────────────────────

_DEFAULT_PROBLEM: dict[str, Any] = {
    "num_vars": 5,
    "names": ["w1", "w2", "w3", "gamma_exp", "lambda"],
    "bounds": [
        [0.20, 0.60],   # w1
        [0.10, 0.50],   # w2
        [0.10, 0.50],   # w3
        [0.01, 0.20],   # gamma_exp
        [1.0,  8.0],    # lambda
    ],
}


def load_problem(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Build and return the SALib problem dictionary.

    If config_path is provided and contains a 'sensitivity.problem' section,
    those bounds override the defaults.

    Returns
    -------
    dict with keys: num_vars, names, bounds
    """
    problem = _DEFAULT_PROBLEM.copy()
    problem["bounds"] = [list(b) for b in _DEFAULT_PROBLEM["bounds"]]

    if config_path is not None:
        cfg_path = Path(config_path)
        if cfg_path.exists():
            with cfg_path.open() as f:
                cfg = yaml.safe_load(f)

            sa_cfg = cfg.get("sensitivity", {}).get("problem", {})
            if sa_cfg:
                names  = sa_cfg.get("names",  problem["names"])
                bounds_dict = sa_cfg.get("bounds", {})
                bounds = [
                    bounds_dict.get(n, problem["bounds"][i])
                    for i, n in enumerate(names)
                ]
                problem["names"]    = names
                problem["bounds"]   = bounds
                problem["num_vars"] = len(names)

    return problem


def print_problem(problem: dict[str, Any]) -> None:
    """Pretty-print the problem definition."""
    print("\n── Sobol' Sensitivity Analysis — Problem Definition ──────────────")
    print(f"  num_vars : {problem['num_vars']}")
    print(f"  {'Parameter':<14} {'Lower':>8}  {'Upper':>8}")
    print("  " + "-" * 34)
    for name, (lo, hi) in zip(problem["names"], problem["bounds"]):
        print(f"  {name:<14} {lo:>8.4f}  {hi:>8.4f}")
    print()
