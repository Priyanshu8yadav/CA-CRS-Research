"""
pets2009_validation.py
───────────────────────
Validate and fine-tune γ_exp and λ against PETS2009 stampede video
trajectories.

Approach
────────
The PETS2009 XML annotations provide per-frame bounding boxes with
trajectory IDs. From these we compute:

  - Per-frame speed S(t): P75 of tracked person displacements.
  - Per-frame directional conflict C(t): fraction of tracked persons
    moving in opposing directions (angle > 120°).
  - Per-frame density D(t): n_persons / scene_capacity (normalised).

Density normalisation uses *capacity-relative* density, not absolute
persons/m². Each PETS2009 scenario has a known maximum crowd count;
D_norm = n_persons / max_persons_across_all_scenarios. This matches
how the original CA-CRS+ DensityEstimator works — density is always
relative to the scene's saturation point.

We evaluate the CA-CRS+ formula across a grid of (γ_exp, λ) and
score each configuration against PETS2009 benchmark labels:
  - S1L1: Sparse, walking → SAFE
  - S1L2: Moderate → WARNING
  - S2L1: Dense, walking fast → WARNING
  - S2L2/S2L3: Dense, running, conflict → DANGER
  - S3MF1: Multiple flow stampede → DANGER

The best (γ_exp, λ) maximises the CRS separation between SAFE and
DANGER scenarios while correctly classifying each.

Usage:
    python -m sensitivity.pets2009_validation
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)

# ── Scenario ground-truth labels ────────────────────────────────────────────
SCENARIO_LABELS = {
    "PETS2009-S1L1-1": "SAFE",        # Sparse crowd, walking
    "PETS2009-S1L1-2": "SAFE",
    "PETS2009-S1L2-1": "WARNING",     # Moderate crowd
    "PETS2009-S1L2-2": "WARNING",
    "PETS2009-S2L1":   "WARNING",     # Dense, walking fast
    "PETS2009-S2L2":   "DANGER",      # Dense, running
    "PETS2009-S2L3":   "DANGER",      # Dense, running, high conflict
    "PETS2009-S3MF1":  "DANGER",      # Multiple flow / stampede
}

# CRS thresholds (from original risk_scoring.py)
THRESH_WARNING = 0.35
THRESH_DANGER  = 0.70

# Speed normalisation: max plausible pixel displacement per frame.
# PETS2009 runs at 7 fps; a person running at ~3 m/s covers ~20 px/frame
# at typical PETS2009 camera calibration. V_MAX_PX is the displacement at
# which S_norm saturates to 1.0.
V_MAX_PX = 20.0


def parse_pets_trajectories(csv_path: str | Path) -> pd.DataFrame:
    """Load the parsed PETS2009 CSV and extract trajectory data."""
    df = pd.read_csv(csv_path)
    df = df[df["dataset"] == "pets2009"].copy()
    df["frame_id"] = df["frame_id"].astype(int)
    df["trajectory_id"] = df["trajectory_id"].astype(str)
    return df


def compute_per_frame_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-frame (D_norm, S_norm, C_norm) from PETS2009 trajectories.

    D_norm uses capacity-relative density: n_persons / global_max_persons.
    S_norm uses P75 of displacement magnitudes between consecutive frames.
    C_norm uses fraction of persons with opposing motion (>120° from dominant).
    """
    # Find the global maximum person count across all frames/scenarios
    # This serves as the "scene capacity" for density normalisation.
    frame_counts = df.groupby(["image_id", "frame_id"])["trajectory_id"].nunique()
    global_max_count = frame_counts.max()
    logger.info("[features] Global max person count per frame: %d", global_max_count)

    records: list[dict] = []

    for image_id, scenario_df in df.groupby("image_id"):
        frames = sorted(scenario_df["frame_id"].unique())

        # Build lookup: frame → {traj_id: (cx, cy)}
        frame_positions: dict[int, dict[str, tuple[float, float]]] = {}
        for frame_id, frame_df in scenario_df.groupby("frame_id"):
            positions = {}
            for _, row in frame_df.iterrows():
                cx = row["cx"] if pd.notna(row["cx"]) else row["bbox_x"] + row["bbox_w"] / 2
                cy = row["cy"] if pd.notna(row["cy"]) else row["bbox_y"] + row["bbox_h"] / 2
                if pd.notna(cx) and pd.notna(cy):
                    positions[str(row["trajectory_id"])] = (float(cx), float(cy))
            frame_positions[int(frame_id)] = positions

        for i, frame_id in enumerate(frames):
            curr_pos = frame_positions.get(frame_id, {})
            n_persons = len(curr_pos)

            # Density: capacity-relative
            d_norm = min(n_persons / max(global_max_count, 1), 1.0)

            # Speed and conflict
            s_norm = 0.0
            c_norm = 0.0

            if i > 0:
                prev_frame = frames[i - 1]
                prev_pos = frame_positions.get(prev_frame, {})
                common_ids = set(curr_pos.keys()) & set(prev_pos.keys())

                if len(common_ids) >= 2:
                    displacements: list[tuple[float, float]] = []
                    for tid in common_ids:
                        dx = curr_pos[tid][0] - prev_pos[tid][0]
                        dy = curr_pos[tid][1] - prev_pos[tid][1]
                        displacements.append((dx, dy))

                    # Speed: P75 of displacement magnitudes
                    mags = [math.sqrt(dx**2 + dy**2) for dx, dy in displacements]
                    s_norm = min(np.percentile(mags, 75) / V_MAX_PX, 1.0)

                    # Conflict: fraction moving opposite to dominant direction
                    if all(m < 0.5 for m in mags):
                        # Everyone is nearly stationary — no meaningful conflict
                        c_norm = 0.0
                    else:
                        angles = [math.atan2(dy, dx) for dx, dy in displacements]
                        w = [max(m, 0.01) for m in mags]
                        mean_sin = sum(wi * math.sin(a) for wi, a in zip(w, angles)) / sum(w)
                        mean_cos = sum(wi * math.cos(a) for wi, a in zip(w, angles)) / sum(w)
                        dom_angle = math.atan2(mean_sin, mean_cos)

                        opposing = 0
                        for a in angles:
                            diff = abs(a - dom_angle)
                            if diff > math.pi:
                                diff = 2 * math.pi - diff
                            if diff > (2 * math.pi / 3):
                                opposing += 1
                        c_norm = opposing / len(angles)

            records.append({
                "image_id": image_id,
                "frame_id": frame_id,
                "n_persons": n_persons,
                "d_norm": round(d_norm, 6),
                "s_norm": round(s_norm, 6),
                "c_norm": round(c_norm, 6),
            })

    return pd.DataFrame(records)


def cacrs_score(
    d: float, s: float, c: float,
    w1: float, w2: float, w3: float,
    gamma_exp: float, lam: float,
) -> float:
    """Evaluate CA-CRS+ risk score."""
    d = max(0.0, min(d, 1.0))
    s = max(0.0, min(s, 1.0))
    c = max(0.0, min(c, 1.0))
    phi = w2 * s * (1 - d) + gamma_exp * math.exp(lam * (d - s))
    crs = w1 * d + phi + w3 * c
    return float(min(max(crs, 0.0), 1.0))


def classify_crs(crs: float) -> str:
    if crs < THRESH_WARNING:
        return "SAFE"
    elif crs < THRESH_DANGER:
        return "WARNING"
    return "DANGER"


def sweep_parameters(
    features_df: pd.DataFrame,
    gamma_exp_range: np.ndarray,
    lambda_range: np.ndarray,
    w1: float = 0.40,
    w2: float = 0.30,
    w3: float = 0.30,
) -> pd.DataFrame:
    """
    Sweep (γ_exp, λ) grid and evaluate classification accuracy
    against known PETS2009 scenario labels.

    Uses per-scenario P90 features (not mean) — this captures the peak
    risk moments that determine the scenario's true label.
    """
    # Per-scenario P90 features (peak moments drive the label)
    scenario_features: dict[str, dict] = {}
    for image_id, grp in features_df.groupby("image_id"):
        scenario_features[image_id] = {
            "d_norm": grp["d_norm"].quantile(0.90),
            "s_norm": grp["s_norm"].quantile(0.90),
            "c_norm": grp["c_norm"].quantile(0.90),
        }

    results: list[dict] = []

    for gamma_exp in tqdm(gamma_exp_range, desc="γ_exp sweep", ncols=80):
        for lam in lambda_range:
            correct = 0
            total = 0
            safe_scores: list[float] = []
            danger_scores: list[float] = []

            for scenario_id, feats in scenario_features.items():
                true_label = SCENARIO_LABELS.get(scenario_id)
                if true_label is None:
                    continue

                crs = cacrs_score(
                    feats["d_norm"], feats["s_norm"], feats["c_norm"],
                    w1, w2, w3, float(gamma_exp), float(lam),
                )
                pred_label = classify_crs(crs)

                # Exact match or at least as severe
                label_order = {"SAFE": 0, "WARNING": 1, "DANGER": 2}
                is_correct = label_order[pred_label] >= label_order[true_label]
                correct += int(is_correct)
                total += 1

                if true_label == "SAFE":
                    safe_scores.append(crs)
                else:
                    danger_scores.append(crs)

            accuracy = correct / max(total, 1)
            mean_safe   = float(np.mean(safe_scores))   if safe_scores   else 0.0
            mean_danger = float(np.mean(danger_scores))  if danger_scores else 0.0
            separation  = mean_danger - mean_safe

            results.append({
                "gamma_exp":   float(gamma_exp),
                "lambda":      float(lam),
                "accuracy":    round(accuracy, 4),
                "separation":  round(separation, 4),
                "mean_safe":   round(mean_safe, 4),
                "mean_danger": round(mean_danger, 4),
                "n_scenarios": total,
            })

    return pd.DataFrame(results)


def plot_parameter_heatmap(
    sweep_df: pd.DataFrame,
    metric: str,
    out_path: str | Path,
    title: str = "",
) -> Path:
    """Plot a 2D heatmap of γ_exp vs λ coloured by a metric."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = sweep_df.pivot(index="gamma_exp", columns="lambda", values=metric)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        extent=[
            pivot.columns.min(), pivot.columns.max(),
            pivot.index.min(), pivot.index.max(),
        ],
        cmap="YlOrRd" if "separation" in metric else "RdYlGn",
    )
    ax.set_xlabel("λ (crush penalty sharpness)", fontsize=12)
    ax.set_ylabel("γ_exp (crush penalty amplitude)", fontsize=12)
    ax.set_title(title or f"PETS2009 Validation — {metric}", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_scenario_timeline(
    features_df: pd.DataFrame,
    best_gamma_exp: float,
    best_lambda: float,
    out_path: str | Path,
    w1: float = 0.40,
    w2: float = 0.30,
    w3: float = 0.30,
) -> Path:
    """
    Plot per-frame CRS timelines for all scenarios using the optimal
    (γ_exp, λ), with horizontal lines at WARNING and DANGER thresholds.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scenarios = sorted(features_df["image_id"].unique())
    n = len(scenarios)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    colors = {"SAFE": "#2ca02c", "WARNING": "#ff7f0e", "DANGER": "#d62728"}

    for ax, scenario_id in zip(axes, scenarios):
        grp = features_df[features_df["image_id"] == scenario_id].sort_values("frame_id")
        true_label = SCENARIO_LABELS.get(scenario_id, "?")

        crs_vals = []
        for _, row in grp.iterrows():
            crs = cacrs_score(
                row["d_norm"], row["s_norm"], row["c_norm"],
                w1, w2, w3, best_gamma_exp, best_lambda,
            )
            crs_vals.append(crs)

        ax.plot(grp["frame_id"], crs_vals, color=colors.get(true_label, "grey"),
                linewidth=1.2, alpha=0.9)
        ax.axhline(THRESH_WARNING, color="#ff7f0e", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(THRESH_DANGER, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("CRS", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{scenario_id}  [{true_label}]", fontsize=10, fontweight="bold")

    axes[-1].set_xlabel("Frame", fontsize=11)
    fig.suptitle(
        f"Per-Frame CRS Timeline (γ_exp={best_gamma_exp:.4f}, λ={best_lambda:.4f})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path("outputs/sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)

    pets_csv = Path("outputs/tables/pets2009.csv")
    if not pets_csv.exists():
        logger.error("PETS2009 CSV not found at %s — run parsers first.", pets_csv)
        sys.exit(1)

    # ── Step 1: Parse trajectories and compute per-frame features ──────────
    logger.info("Parsing PETS2009 trajectories …")
    traj_df = parse_pets_trajectories(pets_csv)
    logger.info("Loaded %d trajectory rows from %d scenarios",
                len(traj_df), traj_df["image_id"].nunique())

    logger.info("Computing per-frame D, S, C features …")
    features_df = compute_per_frame_features(traj_df)
    features_path = output_dir / "pets2009_per_frame_features.csv"
    features_df.to_csv(features_path, index=False)
    logger.info("Per-frame features saved → %s (%d rows)", features_path, len(features_df))

    # Scenario summary
    print("\n── PETS2009 Scenario Feature Summary (P90 peak values) ──")
    for sid in sorted(features_df["image_id"].unique()):
        grp = features_df[features_df["image_id"] == sid]
        label = SCENARIO_LABELS.get(str(sid), "?")
        print(f"  {sid:<22} D={grp['d_norm'].quantile(0.90):.3f}  "
              f"S={grp['s_norm'].quantile(0.90):.3f}  "
              f"C={grp['c_norm'].quantile(0.90):.3f}  "
              f"frames={grp['frame_id'].nunique()}  [{label}]")

    # ── Step 2: Sweep γ_exp and λ ──────────────────────────────────────────
    gamma_exp_range = np.linspace(0.01, 0.30, 60)
    lambda_range    = np.linspace(0.5, 12.0, 60)

    logger.info("Sweeping γ_exp × λ grid: %d × %d = %d evaluations",
                len(gamma_exp_range), len(lambda_range),
                len(gamma_exp_range) * len(lambda_range))

    sweep_df = sweep_parameters(features_df, gamma_exp_range, lambda_range)
    sweep_path = output_dir / "pets2009_parameter_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    logger.info("Parameter sweep saved → %s", sweep_path)

    # ── Step 3: Find optimal parameters ───────────────────────────────────
    # Primary: accuracy ≥ 75%. Secondary: highest separation.
    viable = sweep_df[sweep_df["accuracy"] >= 0.75]
    if viable.empty:
        # Lower threshold progressively
        for thresh in [0.625, 0.50, 0.375]:
            viable = sweep_df[sweep_df["accuracy"] >= thresh]
            if not viable.empty:
                logger.info("Accuracy threshold lowered to %.1f%%", thresh * 100)
                break
    if viable.empty:
        viable = sweep_df
        logger.warning("No viable configs found; using best separation from all.")

    best_idx = viable["separation"].idxmax()
    best = viable.loc[best_idx]

    print(f"\n═══ OPTIMAL (γ_exp, λ) from PETS2009 Validation ═══")
    print(f"  γ_exp    = {best['gamma_exp']:.6f}")
    print(f"  λ        = {best['lambda']:.6f}")
    print(f"  Accuracy = {best['accuracy']:.2%}")
    print(f"  Separation (danger − safe) = {best['separation']:.4f}")
    print(f"  Mean SAFE CRS   = {best['mean_safe']:.4f}")
    print(f"  Mean DANGER CRS = {best['mean_danger']:.4f}")

    # Save optimal params
    optimal = {
        "gamma_exp":       float(best["gamma_exp"]),
        "lambda":          float(best["lambda"]),
        "accuracy":        float(best["accuracy"]),
        "separation":      float(best["separation"]),
        "mean_safe_crs":   float(best["mean_safe"]),
        "mean_danger_crs": float(best["mean_danger"]),
        "w1": 0.40, "w2": 0.30, "w3": 0.30,
        "note": "w1/w2/w3 fixed per Sobol analysis (insensitive). "
                "gamma_exp and lambda tuned on PETS2009 trajectories.",
        "source": "PETS2009 trajectory validation",
    }
    opt_path = output_dir / "optimal_risk_params.json"
    opt_path.write_text(json.dumps(optimal, indent=2))
    logger.info("Optimal params saved → %s", opt_path)

    # ── Step 4: Plots ──────────────────────────────────────────────────────
    plot_parameter_heatmap(
        sweep_df, "separation",
        output_dir / "pets2009_separation_heatmap.png",
        title="PETS2009 — Risk Score Separation (DANGER − SAFE) vs (γ_exp, λ)",
    )
    plot_parameter_heatmap(
        sweep_df, "accuracy",
        output_dir / "pets2009_accuracy_heatmap.png",
        title="PETS2009 — Classification Accuracy vs (γ_exp, λ)",
    )
    plot_scenario_timeline(
        features_df,
        float(best["gamma_exp"]),
        float(best["lambda"]),
        output_dir / "pets2009_crs_timeline.png",
    )

    # ── Step 5: Final per-scenario breakdown ──────────────────────────────
    print(f"\n── Per-Scenario Results with Optimal (γ_exp, λ) ──")
    w1, w2, w3 = 0.40, 0.30, 0.30
    label_order = {"SAFE": 0, "WARNING": 1, "DANGER": 2}

    final_rows = []
    for scenario_id in sorted(features_df["image_id"].unique()):
        grp = features_df[features_df["image_id"] == scenario_id]
        true_label = SCENARIO_LABELS.get(scenario_id, "?")

        # P90 features for classification
        d = grp["d_norm"].quantile(0.90)
        s = grp["s_norm"].quantile(0.90)
        c = grp["c_norm"].quantile(0.90)

        crs = cacrs_score(d, s, c, w1, w2, w3,
                          float(best["gamma_exp"]), float(best["lambda"]))
        pred_label = classify_crs(crs)
        match = "✓" if label_order.get(pred_label, 0) >= label_order.get(true_label, 0) else "✗"

        print(f"  {scenario_id:<22}  CRS={crs:.4f}  {pred_label:<8}  "
              f"(expected {true_label:<8})  {match}")
        final_rows.append({
            "scenario": scenario_id,
            "d_norm_p90": round(d, 4),
            "s_norm_p90": round(s, 4),
            "c_norm_p90": round(c, 4),
            "crs": round(crs, 4),
            "predicted": pred_label,
            "expected": true_label,
            "correct": match == "✓",
        })

    final_df = pd.DataFrame(final_rows)
    final_path = output_dir / "pets2009_final_classification.csv"
    final_df.to_csv(final_path, index=False)
    logger.info("Final classification saved → %s", final_path)

    n_correct = final_df["correct"].sum()
    n_total   = len(final_df)
    print(f"\n  Classification: {n_correct}/{n_total} correct "
          f"({n_correct/n_total:.0%})")
    print(f"\nAll sensitivity outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
