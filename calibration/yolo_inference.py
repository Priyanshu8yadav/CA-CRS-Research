"""
yolo_inference.py
──────────────────
Run YOLOv8 over ShanghaiTech and UCF-QNRF images to extract real
detection bounding boxes and raw counts for κ(β) calibration.

For each image, records:
  - image_id, dataset, split
  - raw YOLO detection count
  - all bounding boxes (x1, y1, x2, y2)
  - computed β from pairwise IoU overlap

Saves results as a CSV ready for fit_kappa.py.

Usage:
    python -m calibration.yolo_inference
    python -m calibration.yolo_inference --model yolov8m.pt --conf 0.04
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


def run_yolo_inference(
    image_dirs: list[dict],
    model_path: str,
    conf: float = 0.04,
    imgsz: int = 640,
    output_dir: str | Path = "outputs/calibration",
    max_images_per_split: int = 0,
) -> pd.DataFrame:
    """
    Run YOLO over multiple dataset image directories.

    Parameters
    ----------
    image_dirs : list[dict]
        Each dict has keys: 'path' (dir of .jpg), 'dataset', 'split'.
    model_path : str
        Path to YOLO weights.
    conf : float
        Detection confidence threshold.
    imgsz : int
        YOLO inference image size.
    output_dir : str | Path
        Where to save outputs.
    max_images_per_split : int
        Limit per split (0 = all images).

    Returns
    -------
    pd.DataFrame with columns: image_id, dataset, split, detected,
                                beta, boxes_json
    """
    from ultralytics import YOLO

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[yolo] Loading model: %s", model_path)
    model = YOLO(model_path)

    all_records: list[dict] = []
    total_images = 0
    start_time = time.time()

    for dir_info in image_dirs:
        img_dir  = Path(dir_info["path"])
        dataset  = dir_info["dataset"]
        split    = dir_info["split"]

        if not img_dir.exists():
            logger.warning("[yolo] Image directory not found: %s", img_dir)
            continue

        jpg_files = sorted(img_dir.glob("*.jpg"))
        png_files = sorted(img_dir.glob("*.png"))
        all_imgs = jpg_files + png_files

        if max_images_per_split > 0:
            all_imgs = all_imgs[:max_images_per_split]

        if not all_imgs:
            logger.warning("[yolo] No images found in %s", img_dir)
            continue

        logger.info(
            "[yolo] %s/%s — processing %d images from %s",
            dataset, split, len(all_imgs), img_dir,
        )

        for img_path in tqdm(
            all_imgs,
            desc=f"{dataset}/{split}",
            leave=True,
            ncols=80,
        ):
            image_id = img_path.stem

            try:
                results = model(
                    str(img_path),
                    classes=[0],        # person class only
                    conf=conf,
                    imgsz=imgsz,
                    verbose=False,
                )
            except Exception as exc:
                logger.error("[yolo] Failed on %s: %s", img_path.name, exc)
                continue

            # Extract boxes
            boxes_list: list[list[float]] = []
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                xyxy = result.boxes.xyxy.detach().cpu().numpy()
                for box in xyxy:
                    boxes_list.append([
                        float(box[0]), float(box[1]),
                        float(box[2]), float(box[3]),
                    ])

            detected = len(boxes_list)

            # Compute β from real YOLO bbox overlaps
            if detected >= 2:
                boxes_arr = np.array(boxes_list, dtype=float)
                beta = compute_beta_from_boxes(boxes_arr, iou_thresh=0.30)
            else:
                beta = 0.0

            all_records.append({
                "image_id": image_id,
                "dataset":  dataset,
                "split":    split,
                "detected": detected,
                "beta":     round(beta, 6),
                "boxes_json": json.dumps(boxes_list),
            })
            total_images += 1

    elapsed = time.time() - start_time
    logger.info(
        "[yolo] Done — %d images in %.1fs (%.2f img/s)",
        total_images, elapsed, total_images / max(elapsed, 0.01),
    )

    df = pd.DataFrame(all_records)

    # Save raw YOLO results
    raw_path = output_dir / "yolo_detections.csv"
    df.to_csv(raw_path, index=False)
    logger.info("[yolo] Raw detections saved → %s", raw_path)

    return df


def merge_with_gt(
    yolo_df: pd.DataFrame,
    gt_csv: str | Path,
) -> pd.DataFrame:
    """
    Merge YOLO detection results with GT counts from parser CSVs.

    Returns a calibration-ready DataFrame with:
        image_id, dataset, split, gt_count, detected, beta, kappa_obs
    """
    gt = pd.read_csv(gt_csv)

    # One GT count per image
    gt_per_image = (
        gt.groupby(["image_id", "dataset", "split"])["gt_count"]
        .first()
        .reset_index()
    )

    merged = pd.merge(
        yolo_df[["image_id", "dataset", "split", "detected", "beta"]],
        gt_per_image,
        on=["image_id", "dataset", "split"],
        how="inner",
    )

    merged["kappa_obs"] = merged["gt_count"] / merged["detected"].clip(lower=1)

    logger.info(
        "[merge] Merged %d images — β mean=%.3f, κ_obs mean=%.3f, "
        "detected mean=%.1f, gt mean=%.1f",
        len(merged),
        merged["beta"].mean(),
        merged["kappa_obs"].mean(),
        merged["detected"].mean(),
        merged["gt_count"].mean(),
    )
    return merged


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="YOLO inference for κ(β) calibration")
    parser.add_argument("--model", default="/Users/riyansh/Documents/New project/cacrs_plus/yolov8m.pt")
    parser.add_argument("--conf", type=float, default=0.04)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max images per split (0=all)")
    parser.add_argument("--output-dir", default="outputs/calibration")
    args = parser.parse_args()

    # Define all image directories to process
    image_dirs = [
        # ShanghaiTech Part A
        {"path": "/Users/riyansh/Downloads/archive/ShanghaiTech/part_A/train_data/images",
         "dataset": "shanghaitech_A", "split": "train"},
        {"path": "/Users/riyansh/Downloads/archive/ShanghaiTech/part_A/test_data/images",
         "dataset": "shanghaitech_A", "split": "test"},
        # ShanghaiTech Part B
        {"path": "/Users/riyansh/Downloads/archive/ShanghaiTech/part_B/train_data/images",
         "dataset": "shanghaitech_B", "split": "train"},
        {"path": "/Users/riyansh/Downloads/archive/ShanghaiTech/part_B/test_data/images",
         "dataset": "shanghaitech_B", "split": "test"},
        # UCF-QNRF
        {"path": "/Users/riyansh/Downloads/UCF-QNRF_ECCV18/Train",
         "dataset": "ucf_qnrf", "split": "train"},
        {"path": "/Users/riyansh/Downloads/UCF-QNRF_ECCV18/Test",
         "dataset": "ucf_qnrf", "split": "test"},
    ]

    yolo_df = run_yolo_inference(
        image_dirs=image_dirs,
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        output_dir=args.output_dir,
        max_images_per_split=args.max_images,
    )

    # Merge with GT and save calibration dataset
    gt_csv = Path("outputs/tables/all_datasets.csv")
    if gt_csv.exists():
        cal_df = merge_with_gt(yolo_df, gt_csv)
        cal_path = Path(args.output_dir) / "yolo_calibration_dataset.csv"
        cal_df.to_csv(cal_path, index=False)
        logger.info("[main] Calibration dataset saved → %s", cal_path)

        # Now fit κ(β) with real data
        from calibration.fit_kappa import fit_kappa_model
        result = fit_kappa_model(
            df=cal_df,
            alpha0_init=0.80,
            gamma_init=1.20,
            alpha0_bounds=(0.1, 5.0),
            gamma_bounds=(0.3, 3.0),
            output_dir=args.output_dir,
        )
        print(f"\n═══ FINAL κ(β) PARAMETERS (from YOLO data) ═══")
        print(f"  α₀ = {result['alpha0']:.6f}")
        print(f"  γ  = {result['gamma']:.6f}")
        print(f"  R² = {result['metrics']['r2']:.6f}")
        print(f"  MAE  = {result['metrics']['mae']:.6f}")
        print(f"  RMSE = {result['metrics']['rmse']:.6f}")
        print(f"  MAPE = {result['metrics']['mape']:.4f}%")
    else:
        logger.warning("[main] GT CSV not found at %s — run parsers first.", gt_csv)


if __name__ == "__main__":
    main()
