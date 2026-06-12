"""
app.py — CA-CRS⁺ Crowd Safety Command Dashboard (Production)
==============================================================
Run with:  streamlit run dashboard/app.py

Production features:
  - Flicker-free rendering via st.empty() placeholders
  - Threaded video stream (non-blocking, RTSP-ready)
  - Non-blocking Modbus gate writes
  - Emergency lockdown button
  - RTSP URL support in sidebar
"""
from __future__ import annotations

import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional

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

logging.basicConfig(level=logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CA-CRS⁺ | Crowd Safety Command",
    page_icon="🚨",
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
        st.session_state.tick = 0
    if "auto_gate" not in st.session_state:
        st.session_state.auto_gate = True
    if "emergency_active" not in st.session_state:
        st.session_state.emergency_active = False


init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚨 CA-CRS⁺ Command")
    st.markdown("**Crowd Risk Scoring System**")
    st.markdown("---")

    st.markdown('<div class="section-header">Zone Configuration</div>',
                unsafe_allow_html=True)

    st.caption("💡 Supports local files & RTSP URLs (e.g. `rtsp://user:pass@192.168.1.100:554/stream1`)")

    VIDEO_A = st.text_input("Zone A — Video Source",
                             value="/Users/riyansh/Downloads/scen_a.mp4")
    VIDEO_B = st.text_input("Zone B — Video Source",
                             value="/Users/riyansh/Downloads/scen_b.mp4")
    VIDEO_C = st.text_input("Zone C — Video Source",
                             value="/Users/riyansh/Downloads/scen_c.mp4")

    ZONE_LABELS = {
        "zone_a": st.text_input("Zone A Label", "Entry Corridor"),
        "zone_b": st.text_input("Zone B Label", "Central Plaza"),
        "zone_c": st.text_input("Zone C Label", "Exit Gate"),
    }

    st.markdown('<div class="section-header">Detection Settings</div>',
                unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.60, 0.25, 0.05)
    imgsz       = st.select_slider("Inference Resolution", [320, 480, 640], value=480)
    target_fps  = st.slider("Target FPS", 2, 15, 6)

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

    start_btn = st.button("▶ Start Dashboard", width="stretch", type="primary")
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
    """Check if a video source is a valid file path or RTSP URL."""
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
                  annotation_text="WARNING 0.35", annotation_position="top left",
                  annotation_font=dict(color="#f59e0b", size=10))
    fig.add_hline(y=THRESHOLD_DANGER, line=dict(color="#ef4444", dash="dash", width=1.5),
                  annotation_text="DANGER 0.70", annotation_position="top left",
                  annotation_font=dict(color="#ef4444", size=10))

    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1628",
        xaxis=dict(showgrid=False, color="#334155", title="Frame"),
        yaxis=dict(range=[0, 1], showgrid=True, gridcolor="#1e2d4a",
                   color="#334155", title="CRS"),
        legend=dict(bgcolor="#0f1628", bordercolor="#1e2d4a", borderwidth=1,
                    font=dict(size=11, color="#94a3b8")),
        font=dict(family="Inter"),
    )
    return fig


def make_causal_chart(states: list[ZoneState]) -> go.Figure:
    labels = [s.label for s in states]
    density  = [s.density  for s in states]
    speed    = [s.speed    for s in states]
    conflict = [s.conflict for s in states]

    fig = go.Figure()
    fig.add_bar(name="Density",  x=labels, y=density,
                marker_color="#6366f1", text=[f"{v:.2f}" for v in density],
                textposition="inside", textfont=dict(size=10))
    fig.add_bar(name="Speed",    x=labels, y=speed,
                marker_color="#06b6d4", text=[f"{v:.2f}" for v in speed],
                textposition="inside", textfont=dict(size=10))
    fig.add_bar(name="Conflict", x=labels, y=conflict,
                marker_color="#f59e0b", text=[f"{v:.2f}" for v in conflict],
                textposition="inside", textfont=dict(size=10))

    fig.update_layout(
        barmode="stack", height=200,
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1628",
        xaxis=dict(color="#334155"),
        yaxis=dict(range=[0, 1.05], gridcolor="#1e2d4a", color="#334155"),
        legend=dict(bgcolor="#0f1628", bordercolor="#1e2d4a",
                    font=dict(size=11, color="#94a3b8"),
                    orientation="h", y=1.1),
        font=dict(family="Inter", color="#e2e8f0"),
    )
    return fig


# ── Render: zone info card (below video) ──────────────────────────────────────
def zone_info_html(s: ZoneState) -> str:
    border = s.color
    return f"""
    <div style='display:flex;justify-content:space-between;
                align-items:center;padding:6px 2px'>
        <span style='font-weight:700;font-size:14px;color:#e2e8f0'>
            {s.label}
        </span>
        <span style='background:{border}22;color:{border};
                     padding:2px 10px;border-radius:12px;
                     font-size:11px;font-weight:700;
                     border:1px solid {border}55'>
            {s.status}
        </span>
    </div>
    <div style='display:flex;gap:8px;margin:4px 0'>
        <div style='flex:1;background:#0f1628;border-radius:8px;
                    padding:8px 10px;border:1px solid #1e2d4a'>
            <div style='font-size:10px;color:#64748b'>CRS Score</div>
            <div style='font-size:20px;font-weight:800;
                        font-family:JetBrains Mono;color:{border}'>
                {s.risk:.3f}
            </div>
        </div>
        <div style='flex:1;background:#0f1628;border-radius:8px;
                    padding:8px 10px;border:1px solid #1e2d4a'>
            <div style='font-size:10px;color:#64748b'>Heads (raw→corr)</div>
            <div style='font-size:14px;font-weight:700;
                        font-family:JetBrains Mono;color:#e2e8f0'>
                {s.head_count} → {s.corrected_count}
            </div>
        </div>
        <div style='flex:1;background:#0f1628;border-radius:8px;
                    padding:8px 10px;border:1px solid #1e2d4a'>
            <div style='font-size:10px;color:#64748b'>κ (ρ-corr)</div>
            <div style='font-size:20px;font-weight:800;
                        font-family:JetBrains Mono;color:#94a3b8'>
                {s.kappa:.2f}
            </div>
        </div>
    </div>
    """


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
        <h1 style='margin:0;font-size:26px;font-weight:800;color:#f1f5f9;'>
            🚨 CA-CRS<sup style='font-size:14px'>+</sup> Crowd Safety Command
        </h1>
        <p style='margin:2px 0 0 0;font-size:13px;color:#475569;'>
            Real-time multi-zone crowd risk scoring &amp; gate control
        </p>""", unsafe_allow_html=True)
    ph_tick = col_tick.empty()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

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

    # ═══════════════════════════════════════════════════════════════════════════
    #  LIVE UPDATE LOOP — only placeholders are updated, no DOM rebuild
    # ═══════════════════════════════════════════════════════════════════════════
    while st.session_state.processors is not None:
        # Process one frame per zone
        states: list[ZoneState] = [p.process_frame() for p in processors]
        grs = compute_grs(states)
        grs_status, grs_color = classify(grs)
        total_heads = sum(s.head_count for s in states)
        n_marshals, marshal_status = marshal_demand(total_heads)
        st.session_state.tick += 1

        # ── Auto gate control + ripple logic ─────────────────────────────────
        gate_cmds = [s.gate_cmd for s in states]
        gate_cmds_final = list(gate_cmds)
        ripple_notes = ["", "", ""]

        if st.session_state.auto_gate:
            for i, s in enumerate(states):
                # Ripple: if REDIRECT but downstream zone is DANGER → HOLD
                if s.status == "WARNING":
                    downstream_idx = (i + 1) % len(states)
                    if states[downstream_idx].status == "DANGER":
                        gate_cmds_final[i] = "HOLD"
                        ripple_notes[i] = "⚠ Ripple: downstream congested → HOLD"

                new_open = (gate_cmds_final[i] == "OPEN")
                if new_open != st.session_state.gate_states.get(i, False):
                    ok = write_gate(i, new_open)
                    if ok:
                        action = "OPENED" if new_open else "CLOSED"
                        log_entry = f"[{time.strftime('%H:%M:%S')}] {s.label} gate {action}"
                        st.session_state.gate_log = (
                            [log_entry] + st.session_state.gate_log
                        )[:20]
                    st.session_state.gate_states[i] = new_open

        # ── Update placeholders (in-place, no DOM rebuild) ───────────────────

        # Tick counter
        ph_tick.markdown(f"""
        <div style='text-align:right;padding-top:8px'>
        <span style='font-family:JetBrains Mono;font-size:12px;color:#475569'>
        TICK #{st.session_state.tick:05d} &nbsp;·&nbsp; {time.strftime("%H:%M:%S")}
        </span></div>""", unsafe_allow_html=True)

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

        # Zone triage
        sorted_states = sorted(states, key=lambda s: s.risk, reverse=True)
        triage_html = ""
        for rank, s in enumerate(sorted_states, 1):
            bar_w = int(s.risk * 100)
            triage_html += f"""
            <div style='display:flex;align-items:center;gap:8px;margin:4px 0;
                        background:#0f1628;border-radius:8px;padding:8px 12px;
                        border:1px solid #1e2d4a'>
                <span style='color:#475569;font-size:11px;width:16px'>#{rank}</span>
                <span style='font-size:13px;font-weight:600;color:#e2e8f0;
                             flex:1'>{s.label}</span>
                <div style='width:60px;background:#1e2d4a;border-radius:4px;height:6px'>
                    <div style='width:{bar_w}%;background:{s.color};
                                border-radius:4px;height:6px'></div>
                </div>
                <span style='font-family:JetBrains Mono;font-size:12px;
                             color:{s.color};width:40px;text-align:right'>
                    {s.risk:.3f}
                </span>
            </div>"""
        ph_triage.markdown(triage_html, unsafe_allow_html=True)

        # Marshal demand
        marshal_col = {"ADEQUATE": "#22c55e", "STRAINED": "#f59e0b",
                       "CRITICAL": "#ef4444"}[marshal_status]
        ph_marshal.markdown(f"""
        <div style='background:#0f1628;border-radius:12px;padding:16px;
                    border:1px solid #1e2d4a;text-align:center'>
            <div style='font-size:40px;font-weight:800;font-family:JetBrains Mono;
                        color:{marshal_col}'>{n_marshals}</div>
            <div style='font-size:11px;color:#64748b;margin-top:2px'>
                Marshals Required
            </div>
            <div style='margin-top:8px;background:{marshal_col}22;color:{marshal_col};
                        padding:4px 12px;border-radius:12px;font-weight:700;
                        font-size:11px;display:inline-block;border:1px solid {marshal_col}44'>
                {marshal_status}
            </div>
            <div style='font-size:11px;color:#475569;margin-top:6px'>
                Total heads: {total_heads}
            </div>
        </div>""", unsafe_allow_html=True)

        # Risk timeline
        with ph_timeline.container():
            st.plotly_chart(make_risk_chart(states),
                            width="stretch",
                            config={"displayModeBar": False},
                            key=f"risk_timeline_{tick}")

        # Causal chart
        with ph_causal.container():
            st.plotly_chart(make_causal_chart(states),
                            width="stretch",
                            config={"displayModeBar": False},
                            key=f"causal_chart_{tick}")

        # Gate status panel
        gate_labels = ["Entry Corridor Gate", "Central Plaza Gate", "Exit Gate"]
        gates_html = ""
        cmd_emoji = {"OPEN": "🔓", "CLOSE": "🔒", "REDIRECT": "↪", "HOLD": "⏸"}
        cmd_colors = {
            "OPEN":     ("#22c55e", "rgba(34,197,94,0.12)"),
            "CLOSE":    ("#ef4444", "rgba(239,68,68,0.12)"),
            "REDIRECT": ("#f59e0b", "rgba(245,158,11,0.12)"),
            "HOLD":     ("#64748b", "rgba(100,116,139,0.12)"),
        }
        for i, (s, label, cmd, note) in enumerate(
            zip(states, gate_labels, gate_cmds_final, ripple_notes)
        ):
            is_open = st.session_state.gate_states.get(i, False)
            dot_color = "#22c55e" if is_open else "#ef4444"
            cc, cbg = cmd_colors.get(cmd, cmd_colors["HOLD"])
            ripple = (f"<br><span style='font-size:10px;color:#f59e0b'>{note}</span>"
                      if note else "")
            gates_html += f"""
            <div style='display:flex;align-items:center;gap:10px;
                        background:#0f1628;border-radius:10px;padding:10px 14px;
                        margin:5px 0;border:1px solid #1e2d4a'>
                <div style='width:12px;height:12px;border-radius:50%;
                            background:{dot_color};box-shadow:0 0 8px {dot_color};
                            flex-shrink:0'></div>
                <div style='flex:1'>
                    <div style='font-weight:600;font-size:12px;color:#e2e8f0'>{label}</div>
                    {ripple}
                </div>
                <div style='font-family:JetBrains Mono;font-weight:700;
                            font-size:12px;color:{cc};background:{cbg};
                            padding:3px 8px;border-radius:6px'>
                    {cmd_emoji.get(cmd,'⏸')} {cmd}
                </div>
            </div>"""
        ph_gates.markdown(gates_html, unsafe_allow_html=True)

        # Gate log
        if st.session_state.gate_log:
            log_html = "".join(
                f"<div style='font-size:11px;color:#64748b;padding:2px 0;font-family:JetBrains Mono'>{e}</div>"
                for e in st.session_state.gate_log[:6]
            )
            ph_gate_log.markdown(log_html, unsafe_allow_html=True)

        # Telemetry
        proc = psutil.Process()
        mem_mb = proc.memory_info().rss / 1024 / 1024
        cpu_pct = psutil.cpu_percent(interval=None)
        sys_mem = psutil.virtual_memory()
        avg_fps = np.mean([s.fps for s in states])

        telem_data = [
            ("Avg FPS",      f"{avg_fps:.1f}",        "#6366f1"),
            ("CPU",          f"{cpu_pct:.0f}%",        "#06b6d4"),
            ("Process RAM",  f"{mem_mb:.0f} MB",       "#f59e0b"),
            ("Sys RAM Used", f"{sys_mem.percent:.0f}%","#ec4899"),
            ("Active Zones", f"{len(states)}",         "#22c55e"),
            ("Total Heads",  f"{total_heads}",         "#94a3b8"),
        ]
        for ph, (label, val, color) in zip(ph_telemetry, telem_data):
            ph.markdown(f"""
            <div style='background:#0f1628;border-radius:10px;padding:12px;
                        border:1px solid #1e2d4a;text-align:center'>
                <div style='font-size:10px;color:#64748b;
                            letter-spacing:1px;text-transform:uppercase'>{label}</div>
                <div style='font-size:22px;font-weight:800;font-family:JetBrains Mono;
                            color:{color};margin-top:4px'>{val}</div>
            </div>""", unsafe_allow_html=True)

        # Frame pacing
        time.sleep(0.05)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    # Start virtual PLC
    if not st.session_state.plc_started:
        try:
            start_plc_server()
            st.session_state.plc_started = True
        except Exception:
            pass

    # Initialize processors
    if start_btn or (st.session_state.processors is None and not stop_btn):
        if start_btn or st.session_state.processors is None:
            try:
                model = load_model(MODEL_PATH)
                zones = [
                    ("zone_a", ZONE_LABELS["zone_a"], VIDEO_A),
                    ("zone_b", ZONE_LABELS["zone_b"], VIDEO_B),
                    ("zone_c", ZONE_LABELS["zone_c"], VIDEO_C),
                ]
                processors = []
                for zid, zlabel, vsource in zones:
                    if is_valid_source(vsource):
                        processors.append(ZoneProcessor(
                            zid, zlabel, vsource, model,
                            conf=conf_thresh, imgsz=imgsz, target_fps=target_fps,
                        ))
                    else:
                        st.sidebar.warning(f"⚠ {zlabel}: source not found — {vsource}")
                if processors:
                    st.session_state.processors = processors
            except Exception as e:
                st.error(f"Failed to initialize: {e}")
                return

    if stop_btn:
        # Clean shutdown of threaded streams
        if st.session_state.processors:
            for p in st.session_state.processors:
                p.stop()
        st.session_state.processors = None
        st.info("Dashboard stopped. Press ▶ Start Dashboard to resume.")
        return

    if st.session_state.processors is None:
        # Welcome screen
        st.markdown("""
        <div style='text-align:center;padding:80px 40px'>
            <div style='font-size:64px'>🚨</div>
            <h1 style='color:#f1f5f9;font-size:36px;font-weight:800;margin:16px 0'>
                CA-CRS⁺ Crowd Safety Command
            </h1>
            <p style='color:#64748b;font-size:16px;max-width:600px;margin:0 auto'>
                Real-time multi-zone crowd risk scoring with SAHI head detection,
                empirically calibrated κ(ρ) correction, and Modbus TCP gate control.
                Configure your zones in the sidebar and press <strong>▶ Start</strong>.
            </p>
            <div style='margin-top:32px;display:flex;gap:24px;
                        justify-content:center;flex-wrap:wrap'>
                <div style='background:#0f1628;border-radius:12px;padding:20px 28px;
                            border:1px solid #1e2d4a;min-width:160px'>
                    <div style='font-size:28px;font-weight:800;color:#6366f1;
                                font-family:JetBrains Mono'>0.615</div>
                    <div style='font-size:12px;color:#64748b'>κ(ρ) R²</div>
                </div>
                <div style='background:#0f1628;border-radius:12px;padding:20px 28px;
                            border:1px solid #1e2d4a;min-width:160px'>
                    <div style='font-size:28px;font-weight:800;color:#06b6d4;
                                font-family:JetBrains Mono'>1,601</div>
                    <div style='font-size:12px;color:#64748b'>Training Images</div>
                </div>
                <div style='background:#0f1628;border-radius:12px;padding:20px 28px;
                            border:1px solid #1e2d4a;min-width:160px'>
                    <div style='font-size:28px;font-weight:800;color:#22c55e;
                                font-family:JetBrains Mono'>83.8%</div>
                    <div style='font-size:12px;color:#64748b'>Head Det. Rate</div>
                </div>
                <div style='background:#0f1628;border-radius:12px;padding:20px 28px;
                            border:1px solid #1e2d4a;min-width:160px'>
                    <div style='font-size:28px;font-weight:800;color:#f59e0b;
                                font-family:JetBrains Mono'>Modbus</div>
                    <div style='font-size:12px;color:#64748b'>TCP Gate Control</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Main render loop (flicker-free)
    run_dashboard(st.session_state.processors)


if __name__ == "__main__":
    main()
