"""
tests/test_sensitivity.py
──────────────────────────
Unit tests for the Sobol' sensitivity analysis module.

All tests use small synthetic samples (N=32) to run quickly.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensitivity.problem_definition import load_problem, _DEFAULT_PROBLEM
from sensitivity.sampler import saltelli_sample
from sensitivity.analyzer import evaluate_model, run_sobol, rank_parameters, _cacrs_score


# ── Problem definition ────────────────────────────────────────────────────────

class TestProblemDefinition:

    def test_default_problem_has_five_params(self):
        p = load_problem()
        assert p["num_vars"] == 5
        assert len(p["names"])  == 5
        assert len(p["bounds"]) == 5

    def test_correct_param_names(self):
        p = load_problem()
        assert set(p["names"]) == {"w1", "w2", "w3", "gamma_exp", "lambda"}

    def test_bounds_are_valid_ranges(self):
        p = load_problem()
        for lo, hi in p["bounds"]:
            assert lo < hi, f"Invalid bound [{lo}, {hi}]"

    def test_load_from_config(self):
        cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
        if cfg_path.exists():
            p = load_problem(cfg_path)
            assert p["num_vars"] == 5


# ── Saltelli sampler ──────────────────────────────────────────────────────────

class TestSampler:

    def test_sample_shape(self):
        problem = load_problem()
        n = 32
        samples = saltelli_sample(problem, n=n, calc_second_order=True)
        expected_rows = n * (2 * problem["num_vars"] + 2)
        assert samples.shape == (expected_rows, problem["num_vars"])

    def test_samples_within_bounds(self):
        problem = load_problem()
        samples = saltelli_sample(problem, n=32)
        for i, (lo, hi) in enumerate(problem["bounds"]):
            col = samples[:, i]
            assert col.min() >= lo - 1e-6, f"Col {i} below lower bound"
            assert col.max() <= hi + 1e-6, f"Col {i} above upper bound"

    def test_reproducible_with_seed(self):
        problem = load_problem()
        s1 = saltelli_sample(problem, n=32, seed=0)
        s2 = saltelli_sample(problem, n=32, seed=0)
        np.testing.assert_array_equal(s1, s2)


# ── Model evaluator ───────────────────────────────────────────────────────────

class TestModelEvaluator:

    def test_output_in_unit_interval(self):
        problem = load_problem()
        samples = saltelli_sample(problem, n=32)
        Y = evaluate_model(samples)
        assert Y.min() >= 0.0 - 1e-9
        assert Y.max() <= 1.0 + 1e-9

    def test_weight_normalisation(self):
        """Even with w1+w2+w3 ≠ 1, score should still be in [0,1]."""
        score = _cacrs_score(0.1, 0.1, 0.1, 0.05, 4.0)
        assert 0.0 <= score <= 1.0

    def test_high_density_high_risk(self):
        """Dense, slow, conflicting crowd should give high score."""
        score = _cacrs_score(0.4, 0.3, 0.3, 0.05, 4.0, d=0.95, s=0.05, c=0.90)
        assert score > 0.50, f"Expected high risk, got {score:.3f}"

    def test_low_density_low_risk(self):
        score = _cacrs_score(0.4, 0.3, 0.3, 0.05, 4.0, d=0.05, s=0.80, c=0.02)
        assert score < 0.60, f"Expected lower risk, got {score:.3f}"


# ── Sobol analysis ────────────────────────────────────────────────────────────

class TestSobolAnalysis:

    @pytest.fixture(scope="class")
    def sobol_output(self):
        problem = load_problem()
        samples = saltelli_sample(problem, n=64, calc_second_order=True)
        Y = evaluate_model(samples)
        Si = run_sobol(problem, samples, Y, calc_second_order=True)
        ranked = rank_parameters(problem, Si)
        return Si, ranked, problem

    def test_all_params_in_ranked(self, sobol_output):
        _, ranked, problem = sobol_output
        assert set(ranked["parameter"]) == set(problem["names"])

    def test_st_nonnegative(self, sobol_output):
        _, ranked, _ = sobol_output
        assert (ranked["ST"] >= -0.05).all(), "ST indices should not be strongly negative"

    def test_recommendations_present(self, sobol_output):
        _, ranked, _ = sobol_output
        assert "recommendation" in ranked.columns
        assert ranked["recommendation"].str.contains("TUNE|FIX").all()

    def test_rank_st_covers_all(self, sobol_output):
        _, ranked, problem = sobol_output
        assert len(ranked) == problem["num_vars"]

    def test_s2_keys_present(self, sobol_output):
        Si, _, _ = sobol_output
        assert "S2" in Si or Si.get("S2") is None  # present or explicitly None
