"""
app.py — Crowd Safety Watch Dashboard
================================================
Run with:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import plotly.graph_objects as go
import psutil
import streamlit as st

# ── Path fix so we can import project modules ─────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.ca_crs_engine import (
    ZoneProcessor, ZoneState, compute_grs,
    THRESHOLD_WARN, THRESHOLD_DANGER,
    marshal_demand, classify,
    KAPPA_ALPHA0, KAPPA_GAMMA,
)
from dashboard.virtual_plc import (
    start_plc_server, write_gate, write_emergency_open,
    read_all_gates, is_plc_running,
)

import math

logging.basicConfig(level=logging.WARNING)


@st.cache_resource
def _get_executor() -> ThreadPoolExecutor:
    """Cached executor — created once, truly reused across all Streamlit reruns."""
    return ThreadPoolExecutor(max_workers=3)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crowd Safety Watch",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject premium CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark theme overrides */
.stApp { background: #0a0e1a; }

.metric-card {
    background: linear-gradient(135deg, #0f1628 0%, #1a2035 100%);
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}

.zone-card {
    border-radius: 12px;
    padding: 14px;
    margin: 6px 0;
    transition: all 0.3s ease;
}

.risk-gauge-container {
    background: linear-gradient(135deg, #0f1628, #1a2035);
    border-radius: 20px;
    padding: 24px;
    border: 1px solid #1e2d4a;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}

.grs-value {
    font-size: 72px;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin: 8px 0;
}

.gate-indicator {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #0f1628;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    border: 1px solid #1e2d4a;
}

.gate-dot {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.2); }
}

.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.telemetry-row {
    display: flex;
    justify-content: space-between;
    background: #0f1628;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 4px 0;
    border: 1px solid #1e2d4a;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}

.causal-bar-container {
    margin: 6px 0;
}

.causal-bar {
    height: 8px;
    border-radius: 4px;
    transition: width 0.5s ease;
}

.section-header {
    color: #94a3b8;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 18px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e2d4a;
}

/* Streamlit element overrides */
[data-testid="stSidebar"] { background: #070c18 !important; }
[data-testid="stMetric"] { background: transparent; }
.stPlotlyChart { border-radius: 12px; overflow: hidden; }
div[data-testid="stImage"] img { border-radius: 10px; }

/* Remove default padding */
.block-container { padding-top: 1rem; padding-bottom: 0; }

@keyframes alert-pulse {
    0%, 100% { background: linear-gradient(90deg, #ef4444, #b91c1c); box-shadow: 0 0 15px rgba(239,68,68,0.5); }
    50% { background: linear-gradient(90deg, #b91c1c, #991b1b); box-shadow: 0 0 30px rgba(239,68,68,0.85); }
}
.danger-banner {
    animation: alert-pulse 1.5s infinite;
}

/* Gradient headline */
.hero-title {
    background: linear-gradient(90deg, #60a5fa, #34d399, #fbbf24);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Soft hover lift on info cards */
.info-card {
    background: #0f1628;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.info-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.45);
}

/* Emergency button styling */
.emergency-btn {
    background: linear-gradient(135deg, #dc2626, #991b1b) !important;
    border: 2px solid #ef4444 !important;
    animation: emergency-pulse 1s infinite;
}
@keyframes emergency-pulse {
    0%, 100% { box-shadow: 0 0 8px rgba(239,68,68,0.4); }
    50% { box-shadow: 0 0 24px rgba(239,68,68,0.8); }
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def init_session():
    if "processors" not in st.session_state:
        st.session_state.processors = None
    if "plc_started" not in st.session_state:
        st.session_state.plc_started = False
    if "gate_states" not in st.session_state:
        st.session_state.gate_states = {0: False, 1: False, 2: False}
    if "gate_log" not in st.session_state:
        st.session_state.gate_log = []
    if "tick" not in st.session_state:
        st.session_state["tick"] = 0
    if "auto_gate" not in st.session_state:
        st.session_state.auto_gate = True
    if "master_control" not in st.session_state:
        st.session_state.master_control = "Auto"
    if "emergency_active" not in st.session_state:
        st.session_state.emergency_active = False
    if "manual_staff" not in st.session_state:
        st.session_state.manual_staff = {"zone_a": 2, "zone_b": 2, "zone_c": 2}


init_session()


# Resolve default paths dynamically based on active user's Home / Downloads directory
downloads_dir = Path.home() / "Downloads"
def get_default_path(filename: str, fallback: str) -> str:
    local_file = downloads_dir / filename
    if local_file.exists():
        return str(local_file)
    return fallback

VIDEO_A = get_default_path("scen_a.mp4", "C:/Users/pp/Downloads/scen_a.mp4")
VIDEO_B = get_default_path("scen_b.mp4", "C:/Users/pp/Downloads/scen_b.mp4")
VIDEO_C = get_default_path("scen_c.mp4", "C:/Users/pp/Downloads/scen_c.mp4")

ZONE_LABELS = {
    "zone_a": "Entry Corridor",
    "zone_b": "Central Plaza",
    "zone_c": "Exit Corridor",
}


# ── Helper: plain-English wording ─────────────────────────────────────────────
STATUS_WORDS = {
    "SAFE":    "✅ All Good",
    "WARNING": "⚠️ Getting Busy",
    "DANGER":  "🚨 Too Crowded",
}

GATE_WORDS = {
    "OPEN":     "🔓 Open — letting people out",
    "CLOSE":    "🔒 Closed",
    "REDIRECT": "↪️ Sending people another way",
    "HOLD":     "⏸️ Keeping closed for now",
}

def friendly_status(status: str) -> str:
    return STATUS_WORDS.get(status, status)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Crowd Safety Watch")
    st.markdown("**Keeps an eye on the crowd and warns you before it gets too packed**")
    st.markdown("---")

    st.markdown('<div class="section-header">Who is using this?</div>',
                unsafe_allow_html=True)
    app_mode = st.radio("Select View", ["Simple View (for everyone)", "Expert View (technical details)"], label_visibility="collapsed")

    if app_mode == "Expert View (technical details)":
        st.caption("💡 Supports local files & RTSP URLs")
        VIDEO_A = st.text_input("Zone A Source", value=VIDEO_A)
        VIDEO_B = st.text_input("Zone B Source", value=VIDEO_B)
        VIDEO_C = st.text_input("Zone C Source", value=VIDEO_C)

        st.markdown('<div class="section-header">Detection Settings</div>',
                    unsafe_allow_html=True)
        conf_thresh = st.slider("Confidence Threshold", 0.10, 0.60, 0.25, 0.05)
        imgsz       = st.select_slider("Inference Resolution", [320, 480, 640], value=480)
        target_fps  = st.slider("Target FPS", 2, 30, 6)
        
        st.markdown('<div class="section-header">Gate Control</div>',
                    unsafe_allow_html=True)
        auto_gate = st.toggle("Auto Gate Control", value=True)
        st.session_state.auto_gate = auto_gate

        st.markdown('<div class="section-header">Model Info</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size:12px; color:#64748b; font-family: JetBrains Mono, monospace;'>
        κ(ρ) = 1 + {KAPPA_ALPHA0}·ρ^{KAPPA_GAMMA}<br>
        R² = 0.615 &nbsp;|&nbsp; n=1,601 imgs<br>
        Model: YOLOv8n + SAHI<br>
        Datasets: SHTech A/B + UCF-QNRF
        </div>""", unsafe_allow_html=True)
    else:
        # Default settings for Simple View
        conf_thresh = 0.25
        imgsz       = 480
        target_fps  = 6

    st.markdown("---")
    start_btn = st.button("▶ Start Watching", width="stretch", type="primary")
    stop_btn  = st.button("⏹ Stop", width="stretch")


# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLOv8 head detector…")
def load_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)

MODEL_PATH = str(ROOT / "models" / "yolov8n_crowd_head.pt")


# ── Helpers: source validation ─────────────────────────────────────────────────
_RTSP_PREFIXES = ("rtsp://", "rtsp_tcp://", "rtsps://", "http://", "https://")

def is_valid_source(source: str) -> bool:
    if any(source.lower().startswith(p) for p in _RTSP_PREFIXES):
        return True
    return Path(source).exists()


# ── Helpers: Plotly charts ──────────────────────────────────────────────────────
def make_gauge(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 28, "family": "JetBrains Mono"},
                "valueformat": ".1f"},
        title={"text": title, "font": {"size": 13, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155",
                     "tickfont": {"color": "#475569", "size": 10}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#0f1628",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  35], "color": "#0d2218"},
                {"range": [35, 70], "color": "#1f1a08"},
                {"range": [70, 100],"color": "#200d0d"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 2},
                "thickness": 0.8,
                "value": value * 100,
            },
        },
    ))
    fig.update_layout(
        height=180, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#0a0e1a", font_color="#e2e8f0",
    )
    return fig


def make_risk_chart(states: list[ZoneState]) -> go.Figure:
    fig = go.Figure()
    FILL_COLORS = {
        "zone_a": "rgba(99,102,241,0.08)",
        "zone_b": "rgba(6,182,212,0.08)",
        "zone_c": "rgba(245,158,11,0.08)",
    }
    LINE_COLORS = {
        "zone_a": "#6366f1",
        "zone_b": "#06b6d4",
        "zone_c": "#f59e0b",
    }
    for s in states:
        if not s.risk_history:
            continue
        x = list(range(len(s.risk_history)))
        fig.add_trace(go.Scatter(
            x=x, y=s.risk_history, mode="lines",
            name=s.label,
            line=dict(color=LINE_COLORS.get(s.zone_id, "#6366f1"), width=2.5),
            fill="tozeroy",
            fillcolor=FILL_COLORS.get(s.zone_id, "rgba(99,102,241,0.08)"),
        ))

    fig.add_hline(y=THRESHOLD_WARN,   line=dict(color="#f59e0b", dash="dash", width=1.5),
                  annotation_text="⚠️ Getting Busy", annotation_position="top left",
                  annotation_font=dict(color="#f59e0b", size=10))
    fig.add_hline(y=THRESHOLD_DANGER, line=dict(color="#ef4444", dash="dash", width=1.5),
                  annotation_text="🚨 Too Crowded", annotation_position="top left",
                  annotation_font=dict(color="#ef4444", size=10))

    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1628",
        xaxis=dict(showgrid=False, color="#334155", title="Time →"),
        yaxis=dict(range=[0, 1], showgrid=True, gridcolor="#1e2d4a",
                   color="#334155", title="Crowd Level", tickformat=".0%"),
        legend=dict(bgcolor="#0f1628", bordercolor="#1e2d4a", borderwidth=1,
                    font=dict(size=11, color="#94a3b8")),
        font=dict(family="Inter"),
    )
    return fig


def make_causal_chart(states: list[ZoneState]) -> go.Figure:
    """Causal Factor Breakdown using weighted attribution ratios r_f.
    r_f^(k) = (w_f · X_f) / Σ_j(w_j · X_j)  — Module 3 paper formula.
    Bars show proportion of risk attributable to each causal factor.
    """
    labels   = [s.label for s in states]
    density  = [s.causal_ratios.get("Density",  0.0) for s in states]
    speed    = [s.causal_ratios.get("Speed",    0.0) for s in states]
    conflict = [s.causal_ratios.get("Conflict", 0.0) for s in states]

    # Build dominant-cause annotation labels
    dom_labels = [f"★ {s.dominant_cause}" for s in states]

    fig = go.Figure()
    fig.add_bar(
        name="Density (r_D)", x=labels, y=density,
        marker_color="#6366f1",
        text=[f"{v*100:.0f}%" for v in density],
        textposition="inside", textfont=dict(size=10),
    )
    fig.add_bar(
        name="Speed (r_S)", x=labels, y=speed,
        marker_color="#06b6d4",
        text=[f"{v*100:.0f}%" for v in speed],
        textposition="inside", textfont=dict(size=10),
    )
    fig.add_bar(
        name="Conflict (r_C)", x=labels, y=conflict,
        marker_color="#f59e0b",
        text=[f"{v*100:.0f}%" for v in conflict],
        textposition="inside", textfont=dict(size=10),
    )

    # Dominant cause threshold line
    fig.add_hline(
        y=0.40, line=dict(color="#94a3b8", dash="dot", width=1),
        annotation_text="Dominance threshold (0.40)",
        annotation_position="top right",
        annotation_font=dict(color="#94a3b8", size=9),
    )

    fig.update_layout(
        barmode="stack", height=220,
        margin=dict(l=10, r=10, t=30, b=30),
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1628",
        xaxis=dict(color="#334155"),
        yaxis=dict(range=[0, 1.05], gridcolor="#1e2d4a", color="#334155",
                   title="Attribution Ratio r_f"),
        legend=dict(bgcolor="#0f1628", bordercolor="#1e2d4a",
                    font=dict(size=11, color="#94a3b8"),
                    orientation="h", y=1.12),
        font=dict(family="Inter", color="#e2e8f0"),
        title=dict(
            text=" | ".join(dom_labels),
            font=dict(size=11, color="#94a3b8"), x=0.5,
        ),
    )
    return fig


# ── Render: zone info card (below video) ──────────────────────────────────────
def zone_info_html(s: ZoneState) -> str:
    border = s.color
    dom_pct = s.causal_ratios.get(s.dominant_cause, 0.0) * 100
    peak_color = "#ef4444" if s.peak_risk >= 0.70 else "#f59e0b" if s.peak_risk >= 0.35 else "#22c55e"
    # Compact HTML — no leading spaces on any line to prevent markdown code-block rendering
    mixed_pct = ""
    if s.dominant_cause == "MIXED":
        cause_str = "MIXED"
        pct_str   = "(no factor &gt;40%)"
    else:
        dom_pct   = s.causal_ratios.get(s.dominant_cause, 0.0) * 100
        cause_str = s.dominant_cause
        pct_str   = f"({dom_pct:.0f}%)"
    fs = friendly_status(s.status)
    return (
        f"<div style='display:flex;justify-content:space-between;align-items:center;padding:6px 2px'>"
        f"<span style='font-weight:700;font-size:14px;color:#e2e8f0'>{s.label}</span>"
        f"<div style='display:flex;gap:6px;align-items:center'>"
        f"<span style='background:{border}22;color:{border};padding:2px 10px;border-radius:12px;"
        f"font-size:11px;font-weight:700;border:1px solid {border}55'>{fs}</span>"
        f"<span style='background:#1e2d4a;color:{peak_color};padding:2px 8px;border-radius:10px;"
        f"font-size:10px;font-family:JetBrains Mono;border:1px solid {peak_color}44' "
        f"title='Session peak CRS score'>&#9650;{s.peak_risk:.3f}</span>"
        f"</div></div>"
        f"<div style='display:flex;gap:8px;margin:4px 0'>"
        f"<div style='flex:1;background:#0f1628;border-radius:8px;padding:8px 10px;border:1px solid #1e2d4a'>"
        f"<div style='font-size:10px;color:#64748b'>CRS Score</div>"
        f"<div style='font-size:20px;font-weight:800;font-family:JetBrains Mono;color:{border}'>{s.risk:.3f}</div>"
        f"</div>"
        f"<div style='flex:1;background:#0f1628;border-radius:8px;padding:8px 10px;border:1px solid #1e2d4a'>"
        f"<div style='font-size:10px;color:#64748b'>Heads (raw&rarr;corr)</div>"
        f"<div style='font-size:14px;font-weight:700;font-family:JetBrains Mono;color:#e2e8f0'>"
        f"{s.head_count} &rarr; {s.corrected_count}</div>"
        f"</div>"
        f"<div style='flex:1;background:#0f1628;border-radius:8px;padding:8px 10px;border:1px solid #1e2d4a'>"
        f"<div style='font-size:10px;color:#64748b'>&kappa;(&rho;)</div>"
        f"<div style='font-size:20px;font-weight:800;font-family:JetBrains Mono;color:#94a3b8'>{s.kappa:.2f}</div>"
        f"</div></div>"
        f"<div style='background:#0f1628;border-radius:6px;padding:6px 10px;"
        f"border:1px solid #1e2d4a;margin-top:4px;font-size:11px;color:#94a3b8;"
        f"display:flex;gap:6px;align-items:center'>"
        f"<span style='color:#64748b'>Cause:</span>"
        f"<span style='color:{border};font-weight:700'>{cause_str}</span>"
        f"<span style='color:#475569'>{pct_str}</span>"
        f"<span style='margin-left:auto;color:#64748b'>&rarr; Gate:</span>"
        f"<span style='color:#e2e8f0;font-family:JetBrains Mono;font-weight:700'>{s.gate_cmd}</span>"
        f"</div>"
    )


# ── Main dashboard loop (flicker-free with st.empty) ─────────────────────────
def run_dashboard(processors: list[ZoneProcessor]):
    """
    Main rendering loop using st.empty() placeholders.
    Static layout is created ONCE. Only dynamic data updates in-place.
    """

    # ── Static header ────────────────────────────────────────────────────────
    col_title, col_tick = st.columns([8, 2])
    with col_title:
        st.markdown(f"""
        <h1 class='hero-title' style='margin:0;font-size:28px;font-weight:800;'>
            🛡️ Crowd Safety Watch
        </h1>
        <p style='margin:2px 0 0 0;font-size:13px;color:#64748b;'>
            Live view of how crowded each area is — with automatic gates and clear advice
        </p>""", unsafe_allow_html=True)
    ph_tick = col_tick.empty()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    
    # Placeholder for Danger Banner
    ph_danger_banner = st.empty()

    # ── Video feed row: create placeholders ──────────────────────────────────
    vid_cols = st.columns(3)
    ph_videos = []
    ph_zone_info = []
    for col in vid_cols:
        with col:
            ph_videos.append(col.empty())
            ph_zone_info.append(col.empty())

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Middle row: GRS + Charts + Gates ─────────────────────────────────────
    left, mid, right = st.columns([2, 4, 2])

    with left:
        st.markdown('<div class="section-header">Global Risk Score</div>',
                    unsafe_allow_html=True)
        ph_gauge = left.empty()

        st.markdown('<div class="section-header">Zone Triage (↑ Risk)</div>',
                    unsafe_allow_html=True)
        ph_triage = left.empty()

        st.markdown('<div class="section-header">Resource Demand</div>',
                    unsafe_allow_html=True)
        ph_marshal = left.empty()

    with mid:
        st.markdown('<div class="section-header">CA-CRS⁺ Risk Score — Live Timeline</div>',
                    unsafe_allow_html=True)
        ph_timeline = mid.empty()

        st.markdown('<div class="section-header">Causal Factor Breakdown</div>',
                    unsafe_allow_html=True)
        ph_causal = mid.empty()

        # Static calibration expander
        with st.expander("📊 Calibration & Sensitivity Results", expanded=False):
            m1, m2, m3 = st.columns(3)
            m1.metric("κ(ρ) R²", "0.615", "▲ vs YOLO-person")
            m2.metric("Head Det. Rate", "83.8%", "ShanghaiTech A")
            m3.metric("PETS2009 Acc.", "75%", "DANGER classification")
            try:
                kappa_params = json.loads(
                    (ROOT / "outputs/calibration/kappa_params_density.json").read_text()
                )
                st.markdown(f"""
                **κ(ρ) = 1 + {kappa_params['alpha0']} · ρ^{kappa_params['gamma']}**
                — calibrated on {kappa_params['n_samples_total']:,} images (ShanghaiTech A/B + UCF-QNRF).
                β-based fitting R² = {kappa_params['metrics']['beta_r2']:.3f} (useless);
                ρ-based R² = {kappa_params['metrics']['r2']:.3f} ✓
                """)
            except Exception:
                st.caption("Calibration params file not found.")

    with right:
        plc_online = is_plc_running()
        plc_status_txt = "🟢 PLC ONLINE" if plc_online else "🟡 PLC CONNECTING"
        st.markdown(f'<div class="section-header">Gate Control &nbsp; <span style="float:right;font-size:10px;color:{"#22c55e" if plc_online else "#f59e0b"}">{plc_status_txt}</span></div>',
                    unsafe_allow_html=True)
        ph_gates = right.empty()

        # Manual override section
        st.markdown('<div class="section-header">Manual Override</div>',
                    unsafe_allow_html=True)
        for i, lbl in enumerate(["A", "B", "C"]):
            c1, c2 = st.columns(2)
            with c1:
                if st.button(f"🔓 Z{lbl}", key=f"open_{i}", use_container_width=True):
                    write_gate(i, True)
                    st.session_state.gate_states[i] = True
            with c2:
                if st.button(f"🔒 Z{lbl}", key=f"close_{i}", use_container_width=True):
                    write_gate(i, False)
                    st.session_state.gate_states[i] = False

        # Emergency lockdown
        st.markdown('<div class="section-header">Emergency</div>',
                    unsafe_allow_html=True)
        if st.button("🚨 EMERGENCY — OPEN ALL GATES", key="emergency",
                     use_container_width=True, type="primary"):
            write_emergency_open()
            for i in range(3):
                st.session_state.gate_states[i] = True
            st.session_state.emergency_active = True
            st.session_state.gate_log = (
                [f"[{time.strftime('%H:%M:%S')}] 🚨 EMERGENCY LOCKDOWN — ALL GATES OPENED"]
                + st.session_state.gate_log
            )[:20]

        # Gate event log
        st.markdown('<div class="section-header">Gate Event Log</div>',
                    unsafe_allow_html=True)
        ph_gate_log = right.empty()

    # ── Bottom: telemetry ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Edge Hardware Telemetry</div>',
                unsafe_allow_html=True)
    telem_cols = st.columns(6)
    ph_telemetry = [col.empty() for col in telem_cols]

    # Persistent ThreadPoolExecutor — cached so it's truly reused across ticks
    executor = _get_executor()

    # ═══════════════════════════════════════════════════════════════════════════
    #  LIVE UPDATE LOOP — only placeholders are updated, no DOM rebuild
    # ═══════════════════════════════════════════════════════════════════════════
    while st.session_state.processors is not None:
        
        # Parallel frame processing across all cameras using ThreadPoolExecutor
        futures = [executor.submit(p.process_frame) for p in processors]
        states: list[ZoneState] = [f.result() for f in futures]
        
        # Module 6: Urgency-weighted GRS
        grs = compute_grs(states)
        grs_status, grs_color = classify(grs)
        total_heads = sum(s.corrected_count for s in states)
        # Module 5: Per-zone log10 marshal demand
        n_marshals, marshal_status = marshal_demand(states)
        st.session_state.tick += 1

        # ── Auto gate control + ripple logic ─────────────────────────────────
        gate_cmds = [s.gate_cmd for s in states]
        gate_cmds_final = list(gate_cmds)
        ripple_notes = ["", "", ""]

        if st.session_state.auto_gate and not st.session_state.emergency_active:
            for i, s in enumerate(states):
                # Module 4: Ripple-effect prevention
                # If gate_cmd is REDIRECT, check downstream zone risk > 0.50
                # (not just DANGER threshold of 0.70 — paper spec: strictly > 0.50)
                if gate_cmds_final[i] == "REDIRECT":
                    downstream_idx = (i + 1) % len(states)
                    if states[downstream_idx].risk > 0.50:
                        gate_cmds_final[i] = "HOLD"
                        ripple_notes[i] = f"⚠ Ripple: downstream risk {states[downstream_idx].risk:.2f} > 0.50 → HOLD"

                new_open = (gate_cmds_final[i] == "OPEN")
                if new_open != st.session_state.gate_states.get(i, False):
                    ok = write_gate(i, new_open)
                    if ok:
                        action = "OPENED" if new_open else "CLOSED"
                        cause_note = f" [{s.dominant_cause}-driven]"
                        log_entry = f"[{time.strftime('%H:%M:%S')}] {s.label} gate {action}{cause_note}"
                        st.session_state.gate_log = (
                            [log_entry] + st.session_state.gate_log
                        )[:20]
                    st.session_state.gate_states[i] = new_open

        # ── Update placeholders (in-place, no DOM rebuild) ───────────────────

        # Tick counter
        ph_tick.markdown(f"""
        <div style='text-align:right;padding-top:8px'>
        <span style='font-family:JetBrains Mono;font-size:12px;color:#475569'>
        🔄 Live &nbsp;·&nbsp; {time.strftime("%H:%M:%S")}
        </span></div>""", unsafe_allow_html=True)

        # Danger Banner — shows dominant cause and gate action (Module 3/4)
        danger_zones = [s for s in states if s.status == "DANGER"]
        if danger_zones:
            parts = []
            for dz in danger_zones:
                gate_action_map = {
                    "OPEN":     "exit gates OPENED (density relief)",
                    "CLOSE":    "entry gates CLOSED (inflow stop)",
                    "REDIRECT": "crowd REDIRECTED (flow separation)",
                    "HOLD":     "gates HELD (ripple prevention / ambiguous cause)",
                }
                action_text = gate_action_map.get(dz.gate_cmd, dz.gate_cmd)
                if dz.dominant_cause == "MIXED":
                    cause_text = "MIXED — no factor &gt; 40% (HOLD)"
                else:
                    dom_pct = dz.causal_ratios.get(dz.dominant_cause, 0.0) * 100
                    cause_text = f"{dz.dominant_cause} ({dom_pct:.0f}%)"
                parts.append(
                    f"<strong>{dz.label}</strong> — Cause: {cause_text} → {action_text}"
                )
            alert_msg = "🚨 &nbsp;DANGER: " + " &nbsp;|&nbsp; ".join(parts)
            ph_danger_banner.markdown(f"""
            <div class="danger-banner" style="
                color: #ffffff; padding: 14px 20px; border-radius: 10px;
                font-weight: 700; font-size: 13px; text-align: center;
                margin-bottom: 16px; border: 1px solid rgba(255,255,255,0.15);
            ">
                {alert_msg}
            </div>
            """, unsafe_allow_html=True)
        else:
            ph_danger_banner.empty()

        # Video feeds
        for i, s in enumerate(states):
            border = s.color
            if s.frame_rgb is not None:
                thumb = cv2.resize(s.frame_rgb, (400, 225))
                ph_videos[i].image(thumb, width="stretch")
            ph_zone_info[i].markdown(zone_info_html(s), unsafe_allow_html=True)

        # GRS gauge
        tick = st.session_state.tick
        with ph_gauge.container():
            st.plotly_chart(make_gauge(grs, "GRS", grs_color),
                            width="stretch", config={"displayModeBar": False},
                            key=f"grs_gauge_{tick}")

        # Zone triage — compact HTML, no leading whitespace (avoids markdown code-block bug)
        triage_html = ""
        sorted_states = sorted(states, key=lambda x: x.risk, reverse=True)
        for i, s in enumerate(sorted_states):
            bar_w = int(s.risk * 100)
            triage_html += (f"<div class='zone-card' style='background:#0f1628;border:1px solid #1e2d4a'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                            f"<span style='color:#64748b;font-size:11px'>#{i+1}</span>"
                            f"<span style='font-weight:600;font-size:13px;flex:1;margin-left:12px;color:#e2e8f0'>{s.label}</span>"
                            f"<div style='width:60px;height:4px;background:#1e2d4a;border-radius:2px;margin:0 12px'>"
                            f"<div style='width:{bar_w}%;height:100%;background:{s.color};border-radius:2px'></div></div>"
                            f"<span style='font-family:JetBrains Mono;font-weight:800;color:{s.color}'>{s.risk:.3f}</span>"
                            f"</div></div>")
        ph_triage.markdown(triage_html, unsafe_allow_html=True)

        # Resource Demand — per-zone breakdown (Module 5)
        import math as _math
        ALPHA_MAP = {"SAFE": 0, "WARNING": 4, "DANGER": 10}
        m_color = "#ef4444" if marshal_status == "CRITICAL" else "#f59e0b" if marshal_status == "STRAINED" else "#22c55e"
        per_zone_rows = ""
        for s in states:
            alpha = ALPHA_MAP[s.status]
            per_demand = _math.ceil(alpha * _math.log10(1 + s.corrected_count)) if s.corrected_count > 0 else 0
            row_color = "#ef4444" if s.status == "DANGER" else "#f59e0b" if s.status == "WARNING" else "#22c55e"
            # Compact single-line HTML — no leading spaces to avoid markdown code-block interpretation
            per_zone_rows += (f"<div style='display:flex;justify-content:space-between;align-items:center;padding:4px 8px;border-bottom:1px solid #1e2d4a;font-size:11px'>"
                              f"<span style='color:#94a3b8'>{s.label}</span>"
                              f"<span style='color:#64748b;font-family:JetBrains Mono'>&alpha;={alpha}&middot;log&#8321;&#8320;(1+{s.corrected_count})</span>"
                              f"<span style='color:{row_color};font-family:JetBrains Mono;font-weight:800'>{per_demand}</span>"
                              f"</div>")
        marshal_html = (f"<div class='metric-card' style='padding:16px 20px'>"
                        f"<div style='text-align:center;padding-bottom:10px'>"
                        f"<div style='font-size:10px;color:#94a3b8;letter-spacing:1px;text-transform:uppercase'>Active Marshals Required</div>"
                        f"<div style='font-size:42px;font-weight:800;font-family:JetBrains Mono;color:{m_color};margin:4px 0'>{n_marshals}</div>"
                        f"<div style='font-size:11px;color:{m_color};font-weight:600'>{marshal_status} &nbsp;&middot;&nbsp; D&#8347;&#8336;&#8319; = &Sigma;&lceil;&alpha;&#8343;&middot;log&#8321;&#8320;(1+N&#8342;)&rceil;</div>"
                        f"</div>"
                        f"{per_zone_rows}"
                        f"</div>")
        ph_marshal.markdown(marshal_html, unsafe_allow_html=True)

        # Charts
        with ph_timeline.container():
            st.plotly_chart(make_risk_chart(states), width="stretch",
                            config={"displayModeBar": False}, key=f"timeline_{tick}")
        with ph_causal.container():
            st.plotly_chart(make_causal_chart(states), width="stretch",
                            config={"displayModeBar": False}, key=f"causal_{tick}")

        # Gate status panel — compact single-line HTML, no leading spaces
        gate_html = ""
        for i, s in enumerate(states):
            is_open = st.session_state.gate_states.get(i, False)
            cmd_text = GATE_WORDS.get(gate_cmds_final[i], gate_cmds_final[i])
            g_color = "#22c55e" if is_open else "#ef4444"
            rip_part = (f"<div style='font-size:10px;color:#f59e0b;margin-top:2px'>&#8618; {ripple_notes[i]}</div>"
                        if ripple_notes[i] else "")
            gate_html += (f"<div class='gate-indicator'>"
                          f"<div class='gate-dot' style='background:{g_color};box-shadow:0 0 10px {g_color}'></div>"
                          f"<div style='flex:1'>"
                          f"<div style='font-weight:600;font-size:13px;color:#e2e8f0'>{s.label} Gate</div>"
                          f"<div style='font-family:JetBrains Mono;font-size:12px;color:#94a3b8'>{cmd_text}{rip_part}</div>"
                          f"</div>"
                          f"</div>")
        ph_gates.markdown(gate_html, unsafe_allow_html=True)

        # Gate Event Log
        log_html = "<div style='background:#0f1628;border-radius:8px;padding:12px;border:1px solid #1e2d4a;height:160px;overflow-y:auto;font-family:JetBrains Mono;font-size:11px;color:#94a3b8'>"
        for entry in st.session_state.gate_log:
            color = "#ef4444" if "EMERGENCY" in entry else "#22c55e" if "OPENED" in entry else "#e2e8f0"
            log_html += f"<div style='margin-bottom:4px;color:{color}'>{entry}</div>"
        if not st.session_state.gate_log:
            log_html += "<div>No events yet...</div>"
        log_html += "</div>"
        ph_gate_log.markdown(log_html, unsafe_allow_html=True)

        # Telemetry
        proc = psutil.Process()
        mem_mb = proc.memory_info().rss / 1024 / 1024
        cpu_pct = psutil.cpu_percent(interval=None)
        sys_mem = psutil.virtual_memory()
        avg_fps = np.mean([s.fps for s in states])

        telems = [
            (ph_telemetry[0], "Camera Speed", f"{avg_fps:.1f}/sec", "#6366f1"),
            (ph_telemetry[1], "Computer Load", f"{cpu_pct:.0f}%", "#06b6d4"),
            (ph_telemetry[2], "App Memory", f"{mem_mb:.0f} MB", "#f59e0b"),
            (ph_telemetry[3], "PC Memory Used", f"{sys_mem.percent:.0f}%", "#ec4899"),
            (ph_telemetry[4], "Cameras Running", f"{len(states)}", "#22c55e"),
            (ph_telemetry[5], "People in View", f"~{total_heads}", "#94a3b8"),
        ]

        for ph, label, val, color in telems:
            ph.markdown(f"""
            <div class='info-card' style='padding:12px;text-align:center'>
                <div style='font-size:10px;color:#64748b;letter-spacing:1px;text-transform:uppercase'>{label}</div>
                <div style='font-size:22px;font-weight:800;font-family:JetBrains Mono;color:{color};margin-top:4px'>{val}</div>
            </div>""", unsafe_allow_html=True)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    if not st.session_state.plc_started:
        start_plc_server()
        st.session_state.plc_started = True

    if start_btn:
        with st.spinner("Initializing neural engine and allocating threaded streams..."):
            try:
                model = load_model(MODEL_PATH)
                zones = [
                    ("zone_a", ZONE_LABELS["zone_a"], VIDEO_A),
                    ("zone_b", ZONE_LABELS["zone_b"], VIDEO_B),
                    ("zone_c", ZONE_LABELS["zone_c"], VIDEO_C),
                ]
                processors = []
                for zid, zlabel, vpath in zones:
                    if is_valid_source(vpath):
                        processors.append(ZoneProcessor(
                            zid, zlabel, vpath, model,
                            conf=conf_thresh, imgsz=imgsz, target_fps=target_fps,
                        ))
                    else:
                        st.sidebar.warning(f"⚠ {zlabel}: source not found — {vpath}")
                if processors:
                    st.session_state.processors = processors
            except Exception as e:
                st.error(f"Failed to start: {e}")
                return

    if stop_btn:
        if st.session_state.processors:
            for p in st.session_state.processors:
                p.stop()
        st.session_state.processors = None
        st.info("Stopped. Press ▶ Start Watching in the sidebar to begin again.")
        return

    if st.session_state.processors is None:
        # Welcome screen
        st.markdown("""
        <div style='text-align:center;padding:80px 40px'>
            <div style='font-size:64px'>🛡️</div>
            <h1 class='hero-title' style='font-size:38px;font-weight:800;margin:16px 0'>
                Crowd Safety Watch
            </h1>
            <p style='color:#94a3b8;font-size:17px;max-width:620px;margin:0 auto;line-height:1.6'>
                This dashboard watches your camera feeds, counts how many people
                are in each area, and warns you <strong>before</strong> a spot gets
                dangerously packed. It can even open and close gates on its own.<br><br>
                Set up your cameras in the sidebar, then press
                <strong style='color:#34d399'>▶ Start Watching</strong>.
            </p>
            <div style='margin-top:36px;display:flex;gap:24px;
                        justify-content:center;flex-wrap:wrap'>
                <div class='info-card' style='padding:20px 28px;min-width:170px'>
                    <div style='font-size:34px'>👀</div>
                    <div style='font-size:14px;font-weight:700;color:#e2e8f0;margin-top:6px'>Counts People</div>
                    <div style='font-size:12px;color:#64748b;margin-top:4px'>Watches 3 areas live and counts everyone, even people hidden behind others</div>
                </div>
                <div class='info-card' style='padding:20px 28px;min-width:170px'>
                    <div style='font-size:34px'>⚠️</div>
                    <div style='font-size:14px;font-weight:700;color:#e2e8f0;margin-top:6px'>Warns You Early</div>
                    <div style='font-size:12px;color:#64748b;margin-top:4px'>Shows a simple color code — green is fine, yellow is busy, red is danger</div>
                </div>
                <div class='info-card' style='padding:20px 28px;min-width:170px'>
                    <div style='font-size:34px'>🚪</div>
                    <div style='font-size:14px;font-weight:700;color:#e2e8f0;margin-top:6px'>Controls Gates</div>
                    <div style='font-size:12px;color:#64748b;margin-top:4px'>Opens and closes gates automatically to keep people flowing safely</div>
                </div>
                <div class='info-card' style='padding:20px 28px;min-width:170px'>
                    <div style='font-size:34px'>🦺</div>
                    <div style='font-size:14px;font-weight:700;color:#e2e8f0;margin-top:6px'>Suggests Staff</div>
                    <div style='font-size:12px;color:#64748b;margin-top:4px'>Tells you how many safety staff you need and where to send them</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Main render loop (flicker-free)
    run_dashboard(st.session_state.processors)


if __name__ == "__main__":
    main()
