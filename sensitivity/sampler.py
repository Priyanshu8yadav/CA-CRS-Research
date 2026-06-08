"""
sampler.py
──────────
Generate Saltelli quasi-random samples for Sobol' sensitivity analysis.

Uses SALib.sample.saltelli which produces N·(2D+2) samples for D parameters.
With N=1024 and D=5: 1024 × 12 = 12 288 model evaluations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def saltelli_sample(
    problem: dict[str, Any],
    n: int = 1024,
    calc_second_order: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """
    Draw Saltelli samples from the SALib problem space.

    Parameters
    ----------
    problem : dict
        SALib problem dict with keys: num_vars, names, bounds.
    n : int
        Base sample count N. Total evaluations = N·(2D+2) with second-order,
        or N·(D+2) without.
    calc_second_order : bool
        Whether to include second-order Sobol' indices in the analysis.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (total_samples, num_vars)
    """
    try:
        # SALib >= 1.5: saltelli was renamed to sobol
        from SALib.sample import sobol as sobol_sampler
        _sample_fn = sobol_sampler.sample
    except ImportError:
        try:
            from SALib.sample import saltelli as sobol_sampler  # type: ignore[no-redef]
            _sample_fn = sobol_sampler.sample
        except ImportError as e:
            raise ImportError(
                "SALib is required. Install with: pip install SALib>=1.4.7"
            ) from e

    param = {**problem, "seed": seed} if "seed" not in problem else problem

    samples = _sample_fn(
        param,
        N=n,
        calc_second_order=calc_second_order,
    )
    logger.info(
        "[sampler] Saltelli samples: N=%d, D=%d → %d rows",
        n, problem["num_vars"], len(samples),
    )
    return samples


def save_samples(
    samples: np.ndarray,
    problem: dict[str, Any],
    out_path: str | Path,
) -> Path:
    """Save the sample matrix to a CSV with named columns."""
    import pandas as pd

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(samples, columns=problem["names"])
    df.to_csv(out_path, index=False)
    logger.info("[sampler] Samples saved → %s  (%d rows)", out_path, len(df))
    return out_path
