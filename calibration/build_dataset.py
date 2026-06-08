"""
build_dataset.py
─────────────────
Build the calibration dataset used to fit κ(β) = 1 + α₀ · β^γ.

What is β?
──────────
β (occlusion ratio) = fraction of detected bounding boxes that are
classified as "occluded" in a given image/frame.

Since the raw crowd datasets do not always label occlusion explicitly,
we approximate β from geometry:
  - For each image, compute pairwise IoU among all detected boxes.
  - A box is considered occluded if it overlaps any other box with IoU > iou_threshold.
  - β = n_occluded / n_detected

If YOLO detection .txt files are present alongside ground-truth counts,
they are loaded directly.  Otherwise, the parser CSVs are used as a proxy
(bounding boxes available for PETS2009; dot annotations for ShanghaiTech/UCF-QNRF).

Output schema
─────────────
image_id      : str    — unique identifier
dataset       : str    — source dataset name
split         : str    — train | test
gt_count      : int    — ground-truth person count
detected      : int    — YOLO/GT raw detection count (pre-correction)
beta          : float  — estimated occlusion ratio [0, 1]
kappa_obs     : float  — observed κ = gt_count / max(detected, 1)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# IoU threshold for classifying a box as occluded
DEFAULT_IOU_THRESH = 0.30


def iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """
    Compute all-pairs IoU for an (N, 4) array of [x1, y1, x2, y2] boxes.

    Parameters
    ----------
    boxes : np.ndarray, shape (N, 4)
        Bounding boxes in [x1, y1, x2, y2] format.

    Returns
    -------
    np.ndarray, shape (N, N)
    """
    if len(boxes) == 0:
        return np.zeros((0, 0))

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    # Vectorised
    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = areas[:, None] + areas[None, :] - inter + 1e-6
    iou   = inter / union
    np.fill_diagonal(iou, 0.0)  # exclude self-overlap
    return iou


def compute_beta_from_boxes(
    boxes: np.ndarray,
    iou_thresh: float = DEFAULT_IOU_THRESH,
) -> float:
    """
    Estimate occlusion ratio β from a set of bounding boxes.

    A box is "occluded" if any other box overlaps it with IoU > iou_thresh.

    Parameters
    ----------
    boxes : np.ndarray, shape (N, 4)  [x1, y1, x2, y2]
    iou_thresh : float

    Returns
    -------
    float in [0, 1]
    """
    if len(boxes) < 2:
        return 0.0
    iou = iou_matrix(boxes)
    occluded = (iou > iou_thresh).any(axis=1)
    return float(occluded.sum()) / len(boxes)


def build_from_parser_csv(
    csv_path: str | Path,
    iou_thresh: float = DEFAULT_IOU_THRESH,
) -> pd.DataFrame:
    """
    Build the calibration DataFrame from a normalised parser CSV.

    Uses bbox columns where available (PETS2009); otherwise approximates
    β from the dot-annotation density (used for ShanghaiTech & UCF-QNRF
    when no YOLO detections are present).

    Parameters
    ----------
    csv_path : str | Path
        Path to a CSV produced by one of the dataset parsers.
    iou_thresh : float
        IoU threshold for occlusion classification.

    Returns
    -------
    pd.DataFrame with columns: image_id, dataset, split, gt_count,
                                detected, beta, kappa_obs
    """
    df = pd.read_csv(csv_path)
    records: list[dict] = []

    has_bbox = df["bbox_x"].notna().any()

    for (image_id, dataset, split), grp in df.groupby(["image_id", "dataset", "split"]):
        gt_count = int(grp["gt_count"].iloc[0])
        n_points = len(grp)

        if has_bbox and grp["bbox_x"].notna().all():
            # Use actual bounding boxes
            boxes_df = grp[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].dropna()
            if len(boxes_df):
                # Convert xywh → x1y1x2y2
                x1 = boxes_df["bbox_x"].values
                y1 = boxes_df["bbox_y"].values
                x2 = x1 + boxes_df["bbox_w"].values
                y2 = y1 + boxes_df["bbox_h"].values
                boxes = np.stack([x1, y1, x2, y2], axis=1)
                beta = compute_beta_from_boxes(boxes, iou_thresh)
                detected = len(boxes)
            else:
                beta = 0.0
                detected = 0
        else:
            # Proxy: use dot density as β surrogate
            # β_proxy = clamp(count / 500, 0, 1)  — heuristic for dense crowds
            beta     = float(min(gt_count / 500.0, 1.0))
            detected = n_points

        kappa_obs = gt_count / max(detected, 1)

        records.append({
            "image_id":  image_id,
            "dataset":   dataset,
            "split":     split,
            "gt_count":  gt_count,
            "detected":  detected,
            "beta":      round(beta, 6),
            "kappa_obs": round(kappa_obs, 6),
        })

    result = pd.DataFrame(records)
    logger.info(
        "[build_dataset] Built calibration dataset: %d rows, "
        "β mean=%.3f, κ_obs mean=%.3f",
        len(result),
        result["beta"].mean(),
        result["kappa_obs"].mean(),
    )
    return result


def load_yolo_detections(
    yolo_dir: str | Path,
    gt_csv: str | Path,
    iou_thresh: float = DEFAULT_IOU_THRESH,
) -> pd.DataFrame:
    """
    Load YOLO detection .txt files (one per image) and pair with GT counts.

    YOLO label format per line: class cx cy w h [conf]   (normalised [0,1])
    The image dimensions must be known; they are inferred from the GT CSV
    (or set to 1.0 to keep normalised coordinates for IoU computation).

    Parameters
    ----------
    yolo_dir : str | Path
        Directory containing YOLO *.txt detection files.
    gt_csv : str | Path
        Normalised parser CSV with gt_count per image_id.
    iou_thresh : float

    Returns
    -------
    pd.DataFrame with columns: image_id, dataset, split, gt_count,
                                detected, beta, kappa_obs
    """
    yolo_dir = Path(yolo_dir)
    gt_df    = pd.read_csv(gt_csv)

    # One GT count per image
    gt_counts = (
        gt_df.groupby("image_id")
             .agg(gt_count=("gt_count", "first"),
                  dataset=("dataset", "first"),
                  split=("split", "first"))
             .reset_index()
    )

    records: list[dict] = []

    for _, row in gt_counts.iterrows():
        image_id = row["image_id"]
        txt_path = yolo_dir / f"{image_id}.txt"

        if not txt_path.exists():
            logger.debug("YOLO file not found for %s, skipping.", image_id)
            continue

        lines = txt_path.read_text().strip().splitlines()
        detected = len(lines)

        if detected < 2:
            beta = 0.0
        else:
            # Parse normalised cx,cy,w,h → x1,y1,x2,y2 (still normalised)
            boxes_list: list[list[float]] = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = (float(p) for p in parts[1:5])
                boxes_list.append([
                    cx - bw / 2, cy - bh / 2,
                    cx + bw / 2, cy + bh / 2,
                ])
            boxes = np.array(boxes_list, dtype=float)
            beta  = compute_beta_from_boxes(boxes, iou_thresh)

        kappa_obs = int(row["gt_count"]) / max(detected, 1)

        records.append({
            "image_id":  image_id,
            "dataset":   row["dataset"],
            "split":     row["split"],
            "gt_count":  int(row["gt_count"]),
            "detected":  detected,
            "beta":      round(beta, 6),
            "kappa_obs": round(kappa_obs, 6),
        })

    result = pd.DataFrame(records)
    logger.info(
        "[load_yolo_detections] Loaded %d YOLO-paired images.", len(result)
    )
    return result
