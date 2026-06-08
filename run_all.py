"""
run_all.py
───────────
Top-level orchestrator for the CA-CRS-Research pipeline.

Runs all tasks in order:
  A. Parse all three datasets → outputs/tables/
  B. Build combined evaluation table
  C. Fit κ(β) → outputs/calibration/
  D. Run Sobol sensitivity analysis → outputs/sensitivity/
  E. All plots and reports are saved automatically.

Usage
─────
    python run_all.py
    python run_all.py --config my_config.yaml
    python run_all.py --skip-parsers          # skip A/B if CSVs already exist
    python run_all.py --skip-calibration      # skip C
    python run_all.py --skip-sensitivity      # skip D
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/run_all.log", mode="a"),
    ],
)
logger = logging.getLogger("run_all")

# ── Imports ───────────────────────────────────────────────────────────────────
from parsers import ShanghaiTechParser, UCFQNRFParser, PETS2009Parser
from calibration.build_dataset import build_from_parser_csv
from calibration.fit_kappa import fit_kappa_model
from sensitivity.problem_definition import load_problem, print_problem
from sensitivity.sampler import saltelli_sample, save_samples
from sensitivity.analyzer import evaluate_model, run_sobol, rank_parameters, save_results
from sensitivity import plots as sa_plots


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Task A+B: Parse datasets ──────────────────────────────────────────────────

def task_parse_datasets(cfg: dict) -> Path:
    """Run all parsers, merge to all_datasets.csv, return path."""
    ds_cfg = cfg["datasets"]
    out_tables = Path(cfg["outputs"]["tables"])
    out_tables.mkdir(parents=True, exist_ok=True)

    all_dfs: list[pd.DataFrame] = []

    # ShanghaiTech
    try:
        parser = ShanghaiTechParser(root=ds_cfg["shanghaitech"], part="both")
        df_st = parser.parse()
        parser.save_csv(df_st, out_tables)
        parser.save_json_summary(df_st, out_tables)
        all_dfs.append(df_st)
        logger.info("✓ ShanghaiTech parsed: %d rows", len(df_st))
    except Exception as exc:
        logger.error("✗ ShanghaiTech parser failed: %s", exc)

    # UCF-QNRF
    try:
        parser = UCFQNRFParser(root=ds_cfg["ucf_qnrf"])
        df_ucf = parser.parse()
        parser.save_csv(df_ucf, out_tables)
        parser.save_json_summary(df_ucf, out_tables)
        all_dfs.append(df_ucf)
        logger.info("✓ UCF-QNRF parsed: %d rows", len(df_ucf))
    except Exception as exc:
        logger.error("✗ UCF-QNRF parser failed: %s", exc)

    # PETS2009
    try:
        parser = PETS2009Parser(root=ds_cfg["pets2009"])
        df_pets = parser.parse()
        parser.save_csv(df_pets, out_tables)
        parser.save_json_summary(df_pets, out_tables)
        all_dfs.append(df_pets)
        logger.info("✓ PETS2009 parsed: %d rows", len(df_pets))
    except Exception as exc:
        logger.error("✗ PETS2009 parser failed: %s", exc)

    if not all_dfs:
        raise RuntimeError("All parsers failed. Check dataset paths in config.yaml.")

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = out_tables / "all_datasets.csv"
    combined.to_csv(out_path, index=False)
    logger.info("Task A+B complete → %s  (%d total rows)", out_path, len(combined))
    return out_path


# ── Task C: Calibrate κ(β) ────────────────────────────────────────────────────

def task_calibrate_kappa(cfg: dict, all_csv: Path) -> dict:
    """Build calibration dataset and fit κ(β)."""
    kappa_cfg = cfg["kappa"]
    out_cal   = Path(cfg["outputs"]["calibration"])

    logger.info("Building calibration dataset from %s …", all_csv)
    cal_df = build_from_parser_csv(all_csv)

    # Save calibration dataset for inspection
    cal_csv = out_cal / "calibration_dataset.csv"
    out_cal.mkdir(parents=True, exist_ok=True)
    cal_df.to_csv(cal_csv, index=False)
    logger.info("Calibration dataset saved → %s  (%d rows)", cal_csv, len(cal_df))

    result = fit_kappa_model(
        df           = cal_df,
        alpha0_init  = kappa_cfg["alpha0_init"],
        gamma_init   = kappa_cfg["gamma_init"],
        alpha0_bounds= tuple(kappa_cfg["alpha0_bounds"]),
        gamma_bounds = tuple(kappa_cfg["gamma_bounds"]),
        output_dir   = out_cal,
    )
    logger.info(
        "Task C complete — α₀=%.6f  γ=%.6f  R²=%.4f",
        result["alpha0"], result["gamma"], result["metrics"]["r2"],
    )
    return result


# ── Task D: Sobol sensitivity analysis ───────────────────────────────────────

def task_sensitivity(cfg: dict, config_path: str | Path) -> None:
    """Sample, evaluate, analyse, plot."""
    sa_cfg    = cfg.get("sensitivity", {})
    n_samples = int(sa_cfg.get("n_samples", 1024))
    out_sa    = Path(cfg["outputs"]["sensitivity"])
    out_sa.mkdir(parents=True, exist_ok=True)

    problem = load_problem(config_path)
    print_problem(problem)

    logger.info("Sampling with Saltelli N=%d …", n_samples)
    samples = saltelli_sample(problem, n=n_samples, calc_second_order=True)
    save_samples(samples, problem, out_sa / "saltelli_samples.csv")

    logger.info("Evaluating CA-CRS+ model on %d samples …", len(samples))
    Y = evaluate_model(samples)

    logger.info("Running Sobol' analysis …")
    Si = run_sobol(problem, samples, Y, calc_second_order=True)

    fix_thresh = float(sa_cfg.get("fix_threshold_ST", 0.05))
    ranked = rank_parameters(problem, Si, fix_threshold=fix_thresh)
    save_results(Si, ranked, problem, out_sa)

    # Plots
    sa_plots.plot_sobol_bar(ranked, out_sa / "sobol_bar_chart.png")
    hm = sa_plots.plot_sobol_heatmap(Si, problem, out_sa / "sobol_heatmap.png")
    if hm:
        logger.info("S2 heatmap saved → %s", hm)

    logger.info("Task D complete — sensitivity results in %s", out_sa)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CA-CRS-Research pipeline orchestrator")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument("--skip-parsers",     action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    cfg = load_config(config_path)
    Path("outputs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("CA-CRS-Research Pipeline — START")
    logger.info("=" * 60)

    all_csv: Path | None = None

    # ── A + B ────────────────────────────────────────────────────────────────
    if not args.skip_parsers:
        logger.info("── Task A+B: Parsing datasets …")
        all_csv = task_parse_datasets(cfg)
    else:
        all_csv = Path(cfg["outputs"]["tables"]) / "all_datasets.csv"
        if not all_csv.exists():
            logger.error("--skip-parsers set but %s not found. Run parsers first.", all_csv)
            sys.exit(1)
        logger.info("Skipping parsers — using %s", all_csv)

    # ── C ─────────────────────────────────────────────────────────────────────
    if not args.skip_calibration:
        logger.info("── Task C: Calibrating κ(β) …")
        task_calibrate_kappa(cfg, all_csv)
    else:
        logger.info("Skipping calibration.")

    # ── D ─────────────────────────────────────────────────────────────────────
    if not args.skip_sensitivity:
        logger.info("── Task D: Sobol' sensitivity analysis …")
        task_sensitivity(cfg, config_path)
    else:
        logger.info("Skipping sensitivity analysis.")

    logger.info("=" * 60)
    logger.info("CA-CRS-Research Pipeline — DONE")
    logger.info("All outputs in: outputs/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
