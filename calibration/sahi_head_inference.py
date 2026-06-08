"""
sahi_head_inference.py
──────────────────────
Run SAHI-wrapped YOLOv8 head detection over ShanghaiTech and UCF-QNRF
images for κ(β) calibration.

Key differences from yolo_inference.py:
  1. Uses a SCUT-HEAD pre-trained YOLOv8 model (head-only detection)
  2. Wraps inference in SAHI's slicing pipeline: the image is split into
     overlapping patches, inference runs on each patch, and boxes are
     merged back via NMS — dramatically boosting recall on dense crowds.

Usage:
    python -m calibration.sahi_head_inference
    python -m calibration.sahi_head_inference --slice-height 512 --slice-width 512
    python -m calibration.sahi_head_inference --model models/yolov8m_head_scut.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.build_dataset import compute_beta_from_boxes

logger = logging.getLogger(__name__)


def run_sahi_inference(
    image_dirs: list[dict],
    model_path: str,
    conf: float = 0.25,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_ratio: float = 0.25,
    output_dir: str | Path = "outputs/calibration",
    max_images_per_split: int = 0,
) -> pd.DataFrame:
    """
    Run SAHI + head-detection YOLOv8 over dataset images.

    Parameters
    ----------
    image_dirs : list[dict]
        Each dict: {path, dataset, split}
    model_path : str
        Path to SCUT-HEAD trained YOLOv8 weights.
    conf : float
        Detection confidence threshold.
    slice_height, slice_width : int
        SAHI slice dimensions in pixels.
    overlap_ratio : float
        Overlap between adjacent slices (0.0–0.5).
    output_dir : str | Path
    max_images_per_split : int
        0 = process all images.

    Returns
    -------
    pd.DataFrame
    """
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[sahi] Loading head detection model: %s", model_path)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=conf,
        device="cpu",
    )

    all_records: list[dict] = []
    total_images = 0
    start_time = time.time()

    # Check for pre-existing partial results so we can resume
    raw_path = output_dir / "sahi_head_detections.csv"
    done_ids: set[str] = set()
    if raw_path.exists():
        prev = pd.read_csv(raw_path)
        done_ids = set(prev["image_id"].astype(str))
        all_records = prev.to_dict("records")
        total_images = len(all_records)
        logger.info("[sahi] Resuming — %d images already done", total_images)

    for dir_info in image_dirs:
        img_dir = Path(dir_info["path"])
        dataset = dir_info["dataset"]
        split   = dir_info["split"]
        use_sahi = dir_info.get("use_sahi", True)

        if not img_dir.exists():
            logger.warning("[sahi] Dir not found: %s", img_dir)
            continue

        all_imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        if max_images_per_split > 0:
            all_imgs = all_imgs[:max_images_per_split]
        # Skip already-processed images
        all_imgs = [p for p in all_imgs if p.stem not in done_ids]
        if not all_imgs:
            logger.info("[sahi] %s/%s — all images already done, skipping",
                        dataset, split)
            continue

        mode = "SAHI" if use_sahi else "standard"
        logger.info("[sahi] %s/%s — %d images [%s] from %s",
                    dataset, split, len(all_imgs), mode, img_dir)

        # Load standard YOLO model for non-SAHI splits
        yolo_model = None
        if not use_sahi:
            from ultralytics import YOLO
            yolo_model = YOLO(model_path)

        split_records: list[dict] = []
        for img_path in tqdm(all_imgs, desc=f"{dataset}/{split}",
                             leave=True, ncols=80):
            image_id = img_path.stem

            try:
                if use_sahi:
                    result = get_sliced_prediction(
                        image=str(img_path),
                        detection_model=detection_model,
                        slice_height=slice_height,
                        slice_width=slice_width,
                        overlap_height_ratio=overlap_ratio,
                        overlap_width_ratio=overlap_ratio,
                        verbose=0,
                    )
                    boxes_list: list[list[float]] = []
                    for pred in result.object_prediction_list:
                        bbox = pred.bbox
                        boxes_list.append([
                            float(bbox.minx), float(bbox.miny),
                            float(bbox.maxx), float(bbox.maxy),
                        ])
                else:
                    # Standard full-image inference (head model, no slicing)
                    res = yolo_model.predict(
                        str(img_path), conf=conf, verbose=False,
                        imgsz=1280,
                    )[0]
                    boxes_list = []
                    for box in res.boxes.xyxy.cpu().numpy():
                        boxes_list.append([float(box[0]), float(box[1]),
                                           float(box[2]), float(box[3])])
            except Exception as exc:
                logger.error("[sahi] Failed on %s: %s", img_path.name, exc)
                continue

            detected = len(boxes_list)

            # Compute β from pairwise IoU of head bounding boxes
            if detected >= 2:
                boxes_arr = np.array(boxes_list, dtype=float)
                beta = compute_beta_from_boxes(boxes_arr, iou_thresh=0.15)
            else:
                beta = 0.0

            rec = {
                "image_id": image_id,
                "dataset":  dataset,
                "split":    split,
                "detected": detected,
                "beta":     round(beta, 6),
                "boxes_json": json.dumps(boxes_list),
            }
            all_records.append(rec)
            split_records.append(rec)
            done_ids.add(image_id)
            total_images += 1

        # ── Flush to CSV after each split (safe against kills) ──────────────
        if split_records:
            df_cur = pd.DataFrame(all_records)
            df_cur.to_csv(raw_path, index=False)
            logger.info("[sahi] Flushed %d new rows → %s  (total=%d)",
                        len(split_records), raw_path, total_images)

    elapsed = time.time() - start_time
    logger.info("[sahi] Done — %d images in %.1fs (%.2f img/s)",
                total_images, elapsed, total_images / max(elapsed, 0.01))

    df = pd.DataFrame(all_records)
    df.to_csv(raw_path, index=False)
    logger.info("[sahi] Raw detections saved → %s", raw_path)

    return df


def merge_with_gt(
    sahi_df: pd.DataFrame,
    gt_csv: str | Path,
) -> pd.DataFrame:
    """Merge SAHI detections with GT counts."""
    gt = pd.read_csv(gt_csv)
    gt_per_image = (
        gt.groupby(["image_id", "dataset", "split"])["gt_count"]
        .first()
        .reset_index()
    )

    merged = pd.merge(
        sahi_df[["image_id", "dataset", "split", "detected", "beta"]],
        gt_per_image,
        on=["image_id", "dataset", "split"],
        how="inner",
    )

    merged["kappa_obs"] = merged["gt_count"] / merged["detected"].clip(lower=1)
    merged["detection_rate"] = merged["detected"] / merged["gt_count"].clip(lower=1)

    logger.info(
        "[merge] %d images — det_rate=%.3f, β mean=%.3f, κ_obs mean=%.3f",
        len(merged),
        merged["detection_rate"].mean(),
        merged["beta"].mean(),
        merged["kappa_obs"].mean(),
    )

    # Per-dataset summary
    print("\n── SAHI Head Detection Stats (per dataset) ──")
    for ds, grp in merged.groupby("dataset"):
        print(f"  {ds:<22} n={len(grp):4d}  det_rate={grp['detection_rate'].mean():.3f}  "
              f"β_mean={grp['beta'].mean():.3f}  κ_obs_mean={grp['kappa_obs'].mean():.3f}  "
              f"detected_mean={grp['detected'].mean():.1f}  gt_mean={grp['gt_count'].mean():.1f}")

    return merged


def fit_and_report(
    cal_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Run both β-based and density-based κ fits on the SAHI data."""
    from calibration.recalibrate import (
        kappa_fn, MIN_DETECTION_RATE, MAX_KAPPA, MIN_BETA,
    )
    from calibration.metrics import mae, rmse, mape, r2
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt

    # Filter
    mask = (
        (cal_df["detection_rate"] >= MIN_DETECTION_RATE) &
        (cal_df["kappa_obs"]      <= MAX_KAPPA) &
        (cal_df["detected"]       >= 2) &
        (cal_df["beta"]           >= MIN_BETA)
    )
    filt = cal_df[mask].copy()
    logger.info("[fit] Filtered: %d → %d rows", len(cal_df), len(filt))

    if len(filt) < 10:
        logger.error("[fit] Too few samples (%d) after filter!", len(filt))
        return

    # ── (A) β-based fit ────────────────────────────────────────────────────
    beta    = filt["beta"].values
    kap_obs = filt["kappa_obs"].values

    try:
        popt, pcov = curve_fit(
            kappa_fn, beta, kap_obs,
            p0=[1.5, 0.8],
            bounds=([0.05, 0.2], [10.0, 3.0]),
            method="trf", maxfev=10000,
        )
        a0_b, g_b = float(popt[0]), float(popt[1])
        perr_b = np.sqrt(np.diag(pcov))
    except Exception as exc:
        logger.error("[fit] β-fit failed: %s", exc)
        a0_b, g_b, perr_b = 1.5, 0.8, [0, 0]

    kpred_b = kappa_fn(beta, a0_b, g_b)
    r2_b = r2(kap_obs, kpred_b)
    corr_b = float(np.corrcoef(beta, kap_obs)[0, 1])

    print(f"\n{'═'*60}")
    print("(A) β-BASED FIT:  κ(β) = 1 + α₀·β^γ   [SAHI Head Detection]")
    print(f"{'═'*60}")
    print(f"  α₀={a0_b:.4f} (±{perr_b[0]:.4f}), γ={g_b:.4f} (±{perr_b[1]:.4f})")
    print(f"  R²={r2_b:.4f}  |  β-κ correlation={corr_b:.4f}")
    print(f"  MAE={mae(kap_obs, kpred_b):.4f}  RMSE={rmse(kap_obs, kpred_b):.4f}")

    # ── (B) Density-based fit ──────────────────────────────────────────────
    max_gt = filt["gt_count"].max()
    rho    = filt["gt_count"].values / max_gt

    try:
        popt2, pcov2 = curve_fit(
            kappa_fn, rho, kap_obs,
            p0=[2.0, 0.5],
            bounds=([0.05, 0.1], [15.0, 3.0]),
            method="trf", maxfev=10000,
        )
        a0_d, g_d = float(popt2[0]), float(popt2[1])
        perr_d = np.sqrt(np.diag(pcov2))
    except Exception as exc:
        logger.error("[fit] density-fit failed: %s", exc)
        a0_d, g_d, perr_d = 2.0, 0.5, [0, 0]

    kpred_d = kappa_fn(rho, a0_d, g_d)
    r2_d = r2(kap_obs, kpred_d)
    corr_d = float(np.corrcoef(rho, kap_obs)[0, 1])

    print(f"\n{'═'*60}")
    print("(B) ρ-BASED FIT:  κ(ρ) = 1 + α₀·ρ^γ   [SAHI Head Detection]")
    print(f"{'═'*60}")
    print(f"  α₀={a0_d:.4f} (±{perr_d[0]:.4f}), γ={g_d:.4f} (±{perr_d[1]:.4f})")
    print(f"  R²={r2_d:.4f}  |  ρ-κ correlation={corr_d:.4f}")
    print(f"  MAE={mae(kap_obs, kpred_d):.4f}  RMSE={rmse(kap_obs, kpred_d):.4f}")

    # ── Summary ────────────────────────────────────────────────────────────
    better = "β-based" if r2_b > r2_d else "ρ-based"
    best_a0 = a0_b if r2_b > r2_d else a0_d
    best_g  = g_b  if r2_b > r2_d else g_d
    best_r2 = max(r2_b, r2_d)

    print(f"\n{'═'*60}")
    print("SUMMARY — SAHI Head Detection κ Calibration")
    print(f"{'═'*60}")
    print(f"  Better fit: {better}  (R²={best_r2:.4f})")
    print(f"  α₀ = {best_a0:.4f}")
    print(f"  γ  = {best_g:.4f}")
    print(f"  Detection rate: {filt['detection_rate'].mean():.1%} "
          f"(vs {cal_df['detection_rate'].mean():.1%} unfiltered)")

    # Save
    result = {
        "beta_fit": {
            "alpha0": a0_b, "gamma": g_b,
            "alpha0_stderr": float(perr_b[0]), "gamma_stderr": float(perr_b[1]),
            "r2": r2_b, "correlation": corr_b,
        },
        "density_fit": {
            "alpha0": a0_d, "gamma": g_d,
            "alpha0_stderr": float(perr_d[0]), "gamma_stderr": float(perr_d[1]),
            "r2": r2_d, "correlation": corr_d,
            "max_gt_count": int(max_gt),
        },
        "recommended": better,
        "n_images_total": len(cal_df),
        "n_images_filtered": len(filt),
        "detection_rate_mean": round(float(filt["detection_rate"].mean()), 4),
        "model": "YOLOv8m SCUT-HEAD + SAHI",
        "source": "SAHI sliced inference on ShanghaiTech + UCF-QNRF",
    }

    params_path = output_dir / "kappa_params_sahi.json"
    params_path.write_text(json.dumps(result, indent=2))
    logger.info("[fit] SAHI params saved → %s", params_path)

    # ── Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"shanghaitech_A": "#4C72B0", "shanghaitech_B": "#55A868",
              "ucf_qnrf": "#DD4949"}

    # Left: β-based
    b_line = np.linspace(0, max(beta.max(), 0.8), 200)
    for ds, grp in filt.groupby("dataset"):
        axes[0].scatter(grp["beta"], grp["kappa_obs"],
                        s=8, alpha=0.4, label=ds, color=colors.get(ds, "grey"))
    axes[0].plot(b_line, kappa_fn(b_line, a0_b, g_b), "k-", linewidth=2,
                 label=f"1+{a0_b:.3f}·β^{g_b:.3f}")
    axes[0].set_xlabel("β (head bbox overlap)", fontsize=11)
    axes[0].set_ylabel("κ observed", fontsize=11)
    axes[0].set_title(f"β-based (R²={r2_b:.3f})", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=7)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right: density-based
    r_line = np.linspace(0, 1, 200)
    for ds, grp in filt.groupby("dataset"):
        axes[1].scatter(grp["gt_count"] / max_gt, grp["kappa_obs"],
                        s=8, alpha=0.4, label=ds, color=colors.get(ds, "grey"))
    axes[1].plot(r_line, kappa_fn(r_line, a0_d, g_d), "k-", linewidth=2,
                 label=f"1+{a0_d:.3f}·ρ^{g_d:.3f}")
    axes[1].set_xlabel("ρ_norm (gt/max_gt)", fontsize=11)
    axes[1].set_ylabel("κ observed", fontsize=11)
    axes[1].set_title(f"ρ-based (R²={r2_d:.3f})", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=7)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("κ Calibration — SAHI + SCUT-HEAD Model", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "kappa_fit_sahi.png", dpi=150)
    plt.close(fig)
    logger.info("[fit] Plot saved → %s", output_dir / "kappa_fit_sahi.png")

    # ── Comparison with previous YOLO person-detection ─────────────────────
    prev_yolo = output_dir / "kappa_params_yolo.json"
    prev_dens = output_dir / "kappa_params_density.json"
    if prev_yolo.exists() and prev_dens.exists():
        old_b = json.loads(prev_yolo.read_text())
        old_d = json.loads(prev_dens.read_text())
        print(f"\n── Comparison: YOLO Person vs SAHI Head ──")
        print(f"  {'Method':<30} {'α₀':>8} {'γ':>8} {'R²':>8}")
        print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8}")
        print(f"  {'YOLO person (β-based)':<30} {old_b['alpha0']:8.4f} {old_b['gamma']:8.4f} {old_b['metrics']['r2']:8.4f}")
        print(f"  {'YOLO person (ρ-based)':<30} {old_d['alpha0']:8.4f} {old_d['gamma']:8.4f} {old_d['metrics']['r2']:8.4f}")
        print(f"  {'SAHI head (β-based)':<30} {a0_b:8.4f} {g_b:8.4f} {r2_b:8.4f}")
        print(f"  {'SAHI head (ρ-based)':<30} {a0_d:8.4f} {g_d:8.4f} {r2_d:8.4f}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="SAHI + head detection inference for κ(β) calibration")
    parser.add_argument("--model",
                        default="models/yolov8m_head_scut.pt",
                        help="Path to SCUT-HEAD YOLOv8 weights")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default 0.25)")
    parser.add_argument("--slice-height", type=int, default=512)
    parser.add_argument("--slice-width",  type=int, default=512)
    parser.add_argument("--overlap",      type=float, default=0.25,
                        help="SAHI slice overlap ratio (default 0.25)")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max images per split (0=all)")
    parser.add_argument("--output-dir", default="outputs/calibration")
    args = parser.parse_args()

    # Dataset image directories
    # use_sahi=True  → SAHI sliced inference (dense crowds, smaller images)
    # use_sahi=False → standard full-image inference at imgsz=1280 (UCF-QNRF
    #                  has very large images; SAHI takes 5-7s/image there)
    image_dirs = [
        {"path": "/Users/riyansh/Downloads/archive/ShanghaiTech/part_A/train_data/images",
         "dataset": "shanghaitech_A", "split": "train", "use_sahi": True},
        {"path": "/Users/riyansh/Downloads/archive/ShanghaiTech/part_A/test_data/images",
         "dataset": "shanghaitech_A", "split": "test",  "use_sahi": True},
        {"path": "/Users/riyansh/Downloads/archive/ShanghaiTech/part_B/train_data/images",
         "dataset": "shanghaitech_B", "split": "train", "use_sahi": True},
        {"path": "/Users/riyansh/Downloads/archive/ShanghaiTech/part_B/test_data/images",
         "dataset": "shanghaitech_B", "split": "test",  "use_sahi": True},
        {"path": "/Users/riyansh/Downloads/UCF-QNRF_ECCV18/Train",
         "dataset": "ucf_qnrf", "split": "train", "use_sahi": False},
        {"path": "/Users/riyansh/Downloads/UCF-QNRF_ECCV18/Test",
         "dataset": "ucf_qnrf", "split": "test",  "use_sahi": False},
    ]

    out_dir = Path(args.output_dir)

    # Step 1: SAHI inference
    sahi_df = run_sahi_inference(
        image_dirs=image_dirs,
        model_path=args.model,
        conf=args.conf,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_ratio=args.overlap,
        output_dir=out_dir,
        max_images_per_split=args.max_images,
    )

    # Step 2: Merge with GT
    gt_csv = Path("outputs/tables/all_datasets.csv")
    if not gt_csv.exists():
        logger.error("GT CSV not found at %s — run parsers first.", gt_csv)
        sys.exit(1)

    cal_df = merge_with_gt(sahi_df, gt_csv)
    cal_path = out_dir / "sahi_calibration_dataset.csv"
    cal_df.to_csv(cal_path, index=False)
    logger.info("[main] SAHI calibration dataset saved → %s", cal_path)

    # Step 3: Fit κ and compare
    fit_and_report(cal_df, out_dir)

    print(f"\nAll SAHI outputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
