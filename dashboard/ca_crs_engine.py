"""
ca_crs_engine.py
────────────────
Core CA-CRS⁺ risk scoring engine used by the dashboard.
Wraps SAHI + YOLOv8 head detection and computes per-zone risk scores.

Production features:
  - ThreadedStream for non-blocking frame reads (file + RTSP)
  - Empirically calibrated κ(ρ) occlusion correction
  - Non-linear CRS formula with exponential crush penalty
  - Speed normalization from config (v_max = 15.0 px/frame)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from dashboard.threaded_stream import ThreadedStream

# ── Calibrated constants (from SAHI calibration 2026-06-08) ──────────────────
KAPPA_ALPHA0  = 8.2442
KAPPA_GAMMA   = 1.1333
KAPPA_MAX_GT  = 3138       # max GT count seen during calibration

RISK_W1       = 0.40       # density weight
RISK_W2       = 0.30       # speed/gridlock weight
RISK_W3       = 0.30       # conflict weight
RISK_GAMMA_EXP = 0.01      # exponential crush amplitude
RISK_LAMBDA   = 8.4915     # exponential crush sharpness

RHO_MAX       = 7.0        # persons/m² saturation
AREA_M2       = 50.0       # assumed zone area in m²
CAPACITY      = int(RHO_MAX * AREA_M2)   # ≈ 350 heads → D_norm saturation

V_MAX         = 15.0       # px/frame speed saturation (from config.yaml physics.v_max)

THRESHOLD_WARN   = 0.35
THRESHOLD_DANGER = 0.70

GATE_COMMANDS = {
    "SAFE":    ("HOLD",     "#22c55e"),
    "WARNING": ("REDIRECT", "#f59e0b"),
    "DANGER":  ("OPEN",     "#ef4444"),
}

MARSHAL_BASE      = 2
MARSHAL_LOG_SCALE = 3.0    # marshals = base + log_scale * ln(1 + count/50)


@dataclass
class ZoneState:
    zone_id: str
    label:   str
    color:   str = "#6b7280"   # current border color
    risk:    float = 0.0
    density: float = 0.0       # D_norm
    speed:   float = 0.0       # S_norm
    conflict: float = 0.0      # C_norm
    head_count: int = 0
    corrected_count: int = 0
    kappa:   float = 1.0
    gate_cmd: str = "HOLD"
    gate_color: str = "#22c55e"
    status:  str = "SAFE"
    fps:     float = 0.0
    risk_history: list[float]   = field(default_factory=list)
    density_history: list[float]= field(default_factory=list)
    speed_history: list[float]  = field(default_factory=list)
    conflict_history: list[float]=field(default_factory=list)
    frame_rgb: Optional[np.ndarray] = None   # latest annotated frame


def kappa(rho_norm: float) -> float:
    """κ(ρ) = 1 + α₀·ρ^γ  — empirically calibrated occlusion correction."""
    return 1.0 + KAPPA_ALPHA0 * (max(rho_norm, 0.0) ** KAPPA_GAMMA)


def compute_risk(D: float, S: float, C: float) -> float:
    """CA-CRS⁺ non-linear risk score.
    CRS = w₁·D + w₂·S·(1−D) + γ_exp·exp(λ·(D−S)) + w₃·C
    """
    phi = RISK_W2 * S * (1 - D) + RISK_GAMMA_EXP * np.exp(RISK_LAMBDA * (D - S))
    crs = RISK_W1 * D + phi + RISK_W3 * C
    return float(np.clip(crs, 0.0, 1.0))


def classify(risk: float) -> tuple[str, str]:
    """Return (status, hex_color) for a risk value."""
    if risk >= THRESHOLD_DANGER:
        return "DANGER",  "#ef4444"
    if risk >= THRESHOLD_WARN:
        return "WARNING", "#f59e0b"
    return "SAFE", "#22c55e"


def marshal_demand(total_heads: int) -> tuple[int, str]:
    """Returns (n_marshals, status_label).
    Formula: n = ⌈base + scale · ln(1 + N/50)⌉
    """
    n = int(MARSHAL_BASE + MARSHAL_LOG_SCALE * np.log1p(total_heads / 50.0))
    if total_heads < 100:
        return n, "ADEQUATE"
    if total_heads < 300:
        return n, "STRAINED"
    return n, "CRITICAL"


class ZoneProcessor:
    """
    Per-zone video processor with threaded frame reading.
    Reads frames via ThreadedStream, runs head detection, computes risk.
    Supports both file paths and RTSP URLs.
    """

    def __init__(
        self,
        zone_id: str,
        label: str,
        video_source: str,
        model,               # ultralytics YOLO model
        conf: float = 0.25,
        imgsz: int  = 640,
        target_fps: int = 8,
    ):
        self.zone_id = zone_id
        self.label   = label
        self.source  = video_source

        # Threaded stream handles both files and RTSP
        self._stream = ThreadedStream(video_source, queue_size=2)

        self.model   = model
        self.conf    = conf
        self.imgsz   = imgsz

        # Frame skipping for target FPS
        src_fps = self._stream.fps
        self._frame_skip = max(1, int(src_fps / target_fps))
        self._frame_count = 0

        self._prev_positions: np.ndarray = np.zeros((0, 2))
        self._prev_time = time.time()
        self.state = ZoneState(zone_id=zone_id, label=label)

    def _estimate_speed(self, curr_boxes: np.ndarray) -> float:
        """Frame-to-frame centroid displacement → speed norm.
        Normalized by V_MAX (15.0 px/frame from config).
        """
        if len(curr_boxes) == 0:
            self._prev_positions = np.zeros((0, 2))
            return 0.0
        # Compute centroids from boxes [x1,y1,x2,y2] → [cx,cy]
        curr_c = np.stack([
            (curr_boxes[:, 0] + curr_boxes[:, 2]) / 2,
            (curr_boxes[:, 1] + curr_boxes[:, 3]) / 2,
        ], axis=1)   # shape (N, 2)
        if len(self._prev_positions) == 0:
            self._prev_positions = curr_c
            return 0.0
        n = min(len(curr_c), len(self._prev_positions))
        dists = np.linalg.norm(curr_c[:n] - self._prev_positions[:n], axis=1)
        speed_px = float(dists.mean()) if n > 0 else 0.0
        self._prev_positions = curr_c
        return float(np.clip(speed_px / V_MAX, 0.0, 1.0))

    def _estimate_conflict(self, boxes: np.ndarray) -> float:
        """Rough conflict: x-spread of centroids as proxy for bidirectional flow."""
        if len(boxes) < 4:
            return 0.0
        cx = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
        spread = float(cx.std() / (cx.max() - cx.min() + 1e-3))
        return float(np.clip(spread * 1.5, 0.0, 1.0))

    def process_frame(self) -> ZoneState:
        t0 = time.time()

        # Non-blocking read from threaded stream
        ok, frame = self._stream.read()
        if not ok or frame is None:
            return self.state

        # Frame skipping for target FPS
        self._frame_count += 1
        if self._frame_count % self._frame_skip != 0:
            return self.state

        # Run YOLO head detection (skip corrupt frames gracefully)
        try:
            results = self.model.predict(
                frame, conf=self.conf, imgsz=self.imgsz, verbose=False
            )[0]
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else np.zeros((0, 4))
        except Exception:
            boxes = np.zeros((0, 4))
        head_count = len(boxes)

        # Compute normalised inputs
        rho      = head_count / KAPPA_MAX_GT
        kap      = kappa(rho)
        corrected_count = head_count * kap
        D_norm   = float(np.clip(corrected_count / CAPACITY, 0.0, 1.0))
        S_norm   = self._estimate_speed(boxes)
        C_norm   = self._estimate_conflict(boxes)
        risk_val = compute_risk(D_norm, S_norm, C_norm)
        status, color = classify(risk_val)
        gate_cmd, gate_color = GATE_COMMANDS[status][0], GATE_COMMANDS[status][1]

        # Annotate frame
        frame_ann = self._annotate(frame.copy(), boxes, risk_val, status, color,
                                   head_count, corrected_count)
        frame_rgb = cv2.cvtColor(frame_ann, cv2.COLOR_BGR2RGB)

        elapsed = time.time() - t0
        fps = 1.0 / max(elapsed, 0.001)

        # Update state
        s = self.state
        s.risk     = round(risk_val, 4)
        s.density  = round(D_norm, 4)
        s.speed    = round(S_norm, 4)
        s.conflict = round(C_norm, 4)
        s.head_count = head_count
        s.corrected_count = int(corrected_count)
        s.kappa    = round(kap, 3)
        s.status   = status
        s.color    = color
        s.gate_cmd = gate_cmd
        s.gate_color = gate_color
        s.fps      = round(fps, 1)
        s.frame_rgb = frame_rgb

        MAXHIST = 150
        s.risk_history    = (s.risk_history    + [risk_val])[-MAXHIST:]
        s.density_history = (s.density_history + [D_norm  ])[-MAXHIST:]
        s.speed_history   = (s.speed_history   + [S_norm  ])[-MAXHIST:]
        s.conflict_history= (s.conflict_history+ [C_norm  ])[-MAXHIST:]

        return s

    def _annotate(self, frame, boxes, risk, status, color, raw_count, corr_count):
        h, w = frame.shape[:2]
        # Draw coloured border
        c = {"SAFE": (34,197,94), "WARNING": (245,158,11), "DANGER": (239,68,68)}[status]
        thick = 10
        cv2.rectangle(frame, (0, 0), (w, h), c, thick * 2)

        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

        # HUD overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"Heads: {raw_count}  Corrected: {corr_count:.0f}",
                    (12, 30), font, 0.7, (255,255,255), 1)
        cv2.putText(frame, f"CRS: {risk:.3f}  [{status}]",
                    (12, 62), font, 0.85, c, 2)
        return frame

    def stop(self):
        """Cleanly stop the threaded stream."""
        self._stream.stop()


def compute_grs(zone_states: list[ZoneState]) -> float:
    """Global Risk Score = weighted max + mean blend."""
    if not zone_states:
        return 0.0
    risks = [z.risk for z in zone_states]
    return float(np.clip(0.6 * max(risks) + 0.4 * np.mean(risks), 0.0, 1.0))
