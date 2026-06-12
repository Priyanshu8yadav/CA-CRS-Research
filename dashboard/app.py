"""
app.py — CA-CRS+ Crowd Safety Command Dashboard
================================================
Run with:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
import time
import math
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
    start_plc_server, write_gate, read_all_gates, is_plc_running,
)

logging.basicConfig(level=logging.WARNING)

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
        st.session_state.auto_gate = False
    if "master_control" not in st.session_state:
        st.session_state.master_control = "Human"
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Crowd Safety Watch")
    st.markdown("**Keeps an eye on the crowd and warns you before it gets too packed**")
    st.markdown("---")

    st.markdown('<div class="section-header">Who is using this?</div>',
                unsafe_allow_html=True)
    app_mode = st.radio("Select View", ["Simple View (for everyone)", "Expert View (technical details)"], label_visibility="collapsed")

    if app_mode == "Expert View (technical details)":
        st.markdown('<div class="section-header">Detection Settings</div>',
                    unsafe_allow_html=True)
        conf_thresh = st.slider("Confidence Threshold", 0.10, 0.60, 0.25, 0.05)
        imgsz       = st.select_slider("Inference Resolution", [320, 480, 640], value=480)
        target_fps  = st.slider("Target FPS", 2, 15, 6)

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


# ── Helper: Plotly gauge ──────────────────────────────────────────────────────
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


# ── Helper: Risk timeline ─────────────────────────────────────────────────────
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

    # Threshold lines
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


# ── Helper: Causal breakdown stacked bar ─────────────────────────────────────
def make_causal_chart(states: list[ZoneState]) -> go.Figure:
    labels = [s.label for s in states]
    density  = [s.density  for s in states]
    speed    = [s.speed    for s in states]
    conflict = [s.conflict for s in states]

    fig = go.Figure()
    fig.add_bar(name="How tightly packed",  x=labels, y=density,
                marker_color="#6366f1", text=[f"{v:.2f}" for v in density],
                textposition="inside", textfont=dict(size=10))
    fig.add_bar(name="How fast people move",    x=labels, y=speed,
                marker_color="#06b6d4", text=[f"{v:.2f}" for v in speed],
                textposition="inside", textfont=dict(size=10))
    fig.add_bar(name="People crossing paths", x=labels, y=conflict,
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


# ── Helper: Gate status HTML ──────────────────────────────────────────────────
def gate_html(label: str, cmd: str, is_open: bool, color: str,
              ripple_note: str = "") -> str:
    dot_color = "#22c55e" if is_open else "#ef4444"
    cmd_colors = {
        "OPEN":     ("#22c55e", "rgba(34,197,94,0.09)",  "rgba(34,197,94,0.27)"),
        "CLOSE":    ("#ef4444", "rgba(239,68,68,0.09)",  "rgba(239,68,68,0.27)"),
        "REDIRECT": ("#f59e0b", "rgba(245,158,11,0.09)", "rgba(245,158,11,0.27)"),
        "HOLD":     ("#64748b", "rgba(100,116,139,0.09)","rgba(100,116,139,0.27)"),
    }
    cc, cbg, cborder = cmd_colors.get(cmd, cmd_colors["HOLD"])
    ripple = (f"<br><span style='font-size:10px;color:#64748b'>{ripple_note}</span>"
              if ripple_note else "")
    return f"""<div style="display:flex;align-items:center;gap:12px;
        background:#0f1628;border-radius:10px;padding:12px 16px;
        margin:6px 0;border:1px solid #1e2d4a">
        <div style="width:14px;height:14px;border-radius:50%;
                    background:{dot_color};box-shadow:0 0 8px {dot_color};
                    flex-shrink:0"></div>
        <div style="flex:1">
            <div style="font-weight:600;font-size:13px;color:#e2e8f0">{label}</div>
            {ripple}
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-weight:700;
                    font-size:13px;color:{cc};background:{cbg};
                    padding:4px 10px;border-radius:6px;border:1px solid {cborder}">
            {cmd}
        </div>
    </div>"""



# ── Helper: Telemetry row ─────────────────────────────────────────────────────
def telem_html(label: str, value: str, color: str = "#94a3b8") -> str:
    return f"""
    <div class="telemetry-row">
        <span style="color:#64748b">{label}</span>
        <span style="color:{color};font-weight:600">{value}</span>
    </div>"""


# ── Helper: plain-English wording ─────────────────────────────────────────────
# Translate internal status / gate codes into everyday words for the screen.
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




# ── Main render loop ──────────────────────────────────────────────────────────
def render_dashboard(processors: list[ZoneProcessor]):
    # Process one frame per zone
    states: list[ZoneState] = [p.process_frame() for p in processors]
    grs = compute_grs(states)
    grs_status, grs_color = classify(grs)
    total_heads = sum(s.head_count for s in states)
    n_marshals, marshal_status = marshal_demand(total_heads)
    st.session_state["tick"] = st.session_state.get("tick", 0) + 1

    # ── Master control & Emergency logic ────────────────────────────────────
    if st.session_state.master_control == "Auto":
        st.session_state.auto_gate = True
        should_emergency = (grs >= THRESHOLD_DANGER)
        if should_emergency != st.session_state.emergency_active:
            st.session_state.emergency_active = should_emergency
            write_gate(3, should_emergency)
            status_text = "🚨 AUTO EMERGENCY TRIGGERED" if should_emergency else "✅ AUTO EMERGENCY CLEARED"
            log_entry = f"[{time.strftime('%H:%M:%S')}] {status_text} (GRS: {grs*100:.1f}%)"
            st.session_state.gate_log = ([log_entry] + st.session_state.gate_log)[:20]
    
    # ── Gate commands resolution ───────────────────────────────────────────
    gate_cmds = [s.gate_cmd for s in states]
    gate_cmds_final = list(gate_cmds)
    ripple_notes = [""] * len(states)

    if st.session_state.emergency_active:
        gate_cmds_final = ["OPEN"] * len(states)
        ripple_notes = ["EMERGENCY MODE ACTIVATED"] * len(states)
        for i in range(len(states)):
            if not st.session_state.gate_states.get(i, False):
                ok = write_gate(i, True)
                if ok:
                    log_entry = f"[{time.strftime('%H:%M:%S')}] {states[i].label} gate OPENED (Emergency)"
                    st.session_state.gate_log = ([log_entry] + st.session_state.gate_log)[:20]
                st.session_state.gate_states[i] = True
    else:
        if st.session_state.auto_gate:
            for i, s in enumerate(states):
                if s.status == "WARNING":
                    downstream_idx = (i + 1) % len(states)
                    if states[downstream_idx].status == "DANGER":
                        gate_cmds_final[i] = "HOLD"
                        ripple_notes[i] = "next area full"
                new_open = (gate_cmds_final[i] == "OPEN")
                if new_open != st.session_state.gate_states.get(i, False):
                    ok = write_gate(i, new_open)
                    if ok:
                        action = "OPENED" if new_open else "CLOSED"
                        log_entry = f"[{time.strftime('%H:%M:%S')}] {s.label} gate {action}"
                        st.session_state.gate_log = ([log_entry] + st.session_state.gate_log)[:20]
                    st.session_state.gate_states[i] = new_open
        else:
            for i in range(len(states)):
                gate_cmds_final[i] = "OPEN" if st.session_state.gate_states.get(i, False) else "CLOSE"

    # ── Layout ─────────────────────────────────────────────────────────────
    # Title bar
    col_title, col_tick = st.columns([8, 2])
    with col_title:
        st.markdown(f"""
        <h1 class='hero-title' style='margin:0;font-size:28px;font-weight:800;'>
            🛡️ Crowd Safety Watch
        </h1>
        <p style='margin:2px 0 0 0;font-size:13px;color:#64748b;'>
            Live view of how crowded each area is — with automatic gates and clear advice
        </p>""", unsafe_allow_html=True)
    with col_tick:
        st.markdown(f"""
        <div style='text-align:right;padding-top:8px'>
        <span style='font-family:JetBrains Mono;font-size:12px;color:#475569'>
        🔄 Live &nbsp;·&nbsp; {time.strftime("%H:%M:%S")}
        </span></div>""", unsafe_allow_html=True)

    # Flashing Alert Banner for DANGER status
    danger_zones = [s for s in states if s.status == "DANGER"]
    if danger_zones:
        labels_str = ", ".join([s.label for s in danger_zones])
        alert_msg = f"🚨 &nbsp;<strong>TOO CROWDED:</strong> {labels_str} is dangerously packed. Nearby gates were closed automatically so more people don't pile in."
        st.markdown(f"""
        <div class="danger-banner" style="
            color: #ffffff; padding: 14px 20px; border-radius: 10px;
            font-weight: 700; font-size: 14px; text-align: center;
            margin-bottom: 16px; border: 1px solid rgba(255,255,255,0.15);
        ">
            {alert_msg}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)



    # ── TOP ROW: Video feeds ─────────────────────────────────────────────────
    vid_cols = st.columns(len(states))
    for i, (col, s) in enumerate(zip(vid_cols, states)):
        with col:
            border = s.color
            st.markdown(f"""
            <div style='border:3px solid {border};border-radius:12px;
                        overflow:hidden;margin-bottom:6px;
                        box-shadow:0 0 20px {border}44'>""",
                        unsafe_allow_html=True)
            if s.frame_rgb is not None:
                thumb = cv2.resize(s.frame_rgb, (400, 225))
                st.image(thumb, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

            # Zone info
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;
                        align-items:center;padding:6px 2px'>
                <span style='font-weight:700;font-size:14px;color:#e2e8f0'>
                    {s.label}
                </span>
                <span style='background:{border}22;color:{border};
                             padding:2px 10px;border-radius:12px;
                             font-size:11px;font-weight:700;
                             border:1px solid {border}55'>
                    {friendly_status(s.status)}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Action advice helper text
            if s.status == "DANGER":
                advice = f"🚨 ACT NOW: Send 5 safety staff to {s.label} right away."
            elif s.status == "WARNING":
                advice = "⚠️ KEEP WATCHING: Have staff ready nearby and keep the path clear."
            else:
                advice = "✅ ALL GOOD: People are moving safely. Nothing to do."

            st.markdown(f"""
            <div style='display:flex;gap:8px;margin:4px 0'>
                <div class='info-card' style='flex:1;padding:8px 10px'>
                    <div style='font-size:10px;color:#64748b'>Crowding Level</div>
                    <div style='font-size:20px;font-weight:800;
                                font-family:JetBrains Mono;color:{border}'>
                        {s.risk * 100:.0f}%
                    </div>
                </div>
                <div class='info-card' style='flex:1;padding:8px 10px'>
                    <div style='font-size:10px;color:#64748b'>People Here (est.)</div>
                    <div style='font-size:20px;font-weight:800;
                                font-family:JetBrains Mono;color:#e2e8f0'>
                        ~{int(s.head_count * s.kappa)}
                    </div>
                </div>
                <div class='info-card' style='flex:1;padding:8px 10px'>
                    <div style='font-size:10px;color:#64748b'>Seen by Camera</div>
                    <div style='font-size:20px;font-weight:800;
                                font-family:JetBrains Mono;color:#94a3b8'>
                        {s.head_count}
                    </div>
                </div>
            </div>
            <div style='background:{border}11; color:{border}; border: 1px solid {border}33;
                        border-radius: 8px; padding: 10px 12px; margin-top: 6px;
                        font-size: 11px; font-weight: 600; line-height: 1.4;'>
                {advice}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── MIDDLE ROW: GRS + Charts + Gates ────────────────────────────────────
    left, mid, right = st.columns([2, 4, 2])

    with left:
        # Overall risk gauge
        st.markdown('<div class="section-header">Overall Crowd Level</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_gauge(grs, "How crowded is the whole venue?", grs_color),
                        width="stretch", config={"displayModeBar": False})

        # Most crowded areas first
        st.markdown('<div class="section-header">Busiest Areas First</div>',
                    unsafe_allow_html=True)
        sorted_states = sorted(states, key=lambda s: s.risk, reverse=True)
        for rank, s in enumerate(sorted_states, 1):
            bar_w = int(s.risk * 100)
            st.markdown(f"""
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
                    {s.risk * 100:.0f}%
                </span>
            </div>""", unsafe_allow_html=True)

        # Safety staff calculation & display
        marshal_col = {"ADEQUATE": "#22c55e", "STRAINED": "#f59e0b",
                       "CRITICAL": "#ef4444"}[marshal_status]
        marshal_word = {"ADEQUATE": "Enough staff on site",
                        "STRAINED": "Stretched — call backup",
                        "CRITICAL": "Not enough — urgent help"}[marshal_status]

        total_risk = sum(s.risk for s in states)
        allocated_staff = {}
        if len(states) > 0:
            if total_risk > 0:
                remaining = n_marshals
                for s in states[:-1]:
                    allocated = int(round(n_marshals * s.risk / total_risk))
                    allocated_staff[s.zone_id] = allocated
                    remaining -= allocated
                allocated_staff[states[-1].zone_id] = max(0, remaining)
            else:
                remaining = n_marshals
                div = n_marshals // len(states)
                for s in states[:-1]:
                    allocated_staff[s.zone_id] = div
                    remaining -= div
                allocated_staff[states[-1].zone_id] = max(0, remaining)

        total_staff = sum(st.session_state.manual_staff.get(s.zone_id, 2) for s in states)
        if total_staff >= n_marshals:
            staff_status = "ADEQUATE"
            staff_word = "Deployment meets recommended level"
            staff_color = "#22c55e"
        elif total_staff > 0:
            staff_status = "STRAINED"
            staff_word = "Deployment below recommendation"
            staff_color = "#f59e0b"
        else:
            staff_status = "CRITICAL"
            staff_word = "No safety staff deployed!"
            staff_color = "#ef4444"

        if st.session_state.master_control == "Auto":
            display_staff = n_marshals
            display_color = marshal_col
            display_word = marshal_word
            
            auto_lines = []
            for s in states:
                staff_count = allocated_staff.get(s.zone_id, 0)
                auto_lines.append(f"• {s.label}: {staff_count} staff")
            auto_lines_str = "<br>".join(auto_lines)
            
            breakdown_text = f"""
            <div style='font-size:11px;color:#94a3b8;margin-top:8px;text-align:left;line-height:1.6;border-top:1px solid #1e2d4a;padding-top:8px;'>
                🔹 <strong>Auto-allocation:</strong><br>
                {auto_lines_str}
            </div>
            """
        else:
            display_staff = total_staff
            display_color = staff_color
            display_word = staff_word
            
            manual_lines = []
            for s in states:
                staff_count = st.session_state.manual_staff.get(s.zone_id, 2)
                manual_lines.append(f"• {s.label}: {staff_count} staff")
            manual_lines_str = "<br>".join(manual_lines)
            
            breakdown_text = f"""
            <div style='font-size:11px;color:#94a3b8;margin-top:8px;text-align:left;line-height:1.6;border-top:1px solid #1e2d4a;padding-top:8px;'>
                👤 <strong>Manual dispatch:</strong><br>
                {manual_lines_str}<br>
                <span style='color:#64748b'>(Recommended total: {n_marshals})</span>
            </div>
            """

        st.markdown('<div class="section-header">Safety Staff Status</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-card' style='padding:16px;text-align:center'>
            <div style='font-size:40px;font-weight:800;font-family:JetBrains Mono;
                        color:{display_color}'>{display_staff}</div>
            <div style='font-size:11px;color:#64748b;margin-top:2px'>
                staff members currently deployed
            </div>
            <div style='margin-top:8px;background:{display_color}22;color:{display_color};
                        padding:4px 12px;border-radius:12px;font-weight:700;
                        font-size:11px;display:inline-block;border:1px solid {display_color}44'>
                {display_word}
            </div>
            {breakdown_text}
            <div style='font-size:10px;color:#475569;margin-top:8px;text-align:center'>
                People in all areas: ~{total_heads}
            </div>
        </div>""", unsafe_allow_html=True)

    with mid:
        # Risk timeline
        st.markdown('<div class="section-header">Crowd Level Over Time</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_risk_chart(states),
                        width="stretch", config={"displayModeBar": False})

        # What is making it crowded
        st.markdown('<div class="section-header">What is making each area risky?</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_causal_chart(states),
                        width="stretch", config={"displayModeBar": False})

        # Model performance (technical — Expert View only)
        if app_mode == "Expert View (technical details)":
            show_calibration_details()

    with right:
        render_control_center(states, gate_cmds_final, ripple_notes)

    render_system_health(states, total_heads)


def show_calibration_details():
    with st.expander("📊 Calibration & Sensitivity Results", expanded=False):
        m1, m2, m3 = st.columns(3)
        m1.metric("κ(ρ) R²", "0.615", "▲ vs YOLO-person")
        m2.metric("Head Det. Rate", "83.8%", "ShanghaiTech A")
        m3.metric("PETS2009 Acc.", "75%", "DANGER classification")
        try:
            kappa_params = json.loads(
                (ROOT / "outputs/calibration/kappa_params_density.json").read_text()
            )
        except Exception:
            kappa_params = {
                "alpha0": KAPPA_ALPHA0,
                "gamma": KAPPA_GAMMA,
                "n_samples_total": 1601,
                "metrics": {
                    "beta_r2": -0.040,
                    "r2": 0.6152
                }
            }
        st.markdown(f"""
        **κ(ρ) = 1 + {kappa_params['alpha0']} · ρ^{kappa_params['gamma']}**
        — calibrated on {kappa_params['n_samples_total']:,} images (ShanghaiTech A/B + UCF-QNRF).
        β-based fitting R² = {kappa_params['metrics']['beta_r2']:.3f} (useless);
        ρ-based R² = {kappa_params['metrics']['r2']:.3f} ✓
        """)


def render_control_center(states, gate_cmds_final, ripple_notes):
    plc_online = is_plc_running()
    plc_status_txt = "🟢 Connected" if plc_online else "🟡 Connecting…"
    st.markdown(f'<div class="section-header">Control Center &nbsp; <span style="float:right;font-size:10px;color:{"#22c55e" if plc_online else "#f59e0b"}">{plc_status_txt}</span></div>',
                unsafe_allow_html=True)

    # 1. Master Control Mode Selection
    st.markdown("<div style='font-size:12px;font-weight:600;color:#94a3b8;margin-bottom:6px'>Master Control Mode</div>", unsafe_allow_html=True)
    master_control = st.radio(
        "Master Control Mode",
        options=["Human", "Auto"],
        index=0 if st.session_state.master_control == "Human" else 1,
        horizontal=True,
        key="master_control_radio_btn",
        label_visibility="collapsed"
    )
    if master_control != st.session_state.master_control:
        st.session_state.master_control = master_control
        st.session_state.gate_log.append(f"[{time.strftime('%H:%M:%S')}] ⚙ Master Mode set to {master_control.upper()}")
        if master_control == "Auto":
            st.session_state.auto_gate = True
        st.rerun()

    st.markdown("<hr style='margin:10px 0;border-color:#1e2d4a'>", unsafe_allow_html=True)

    # 2. Gate Control Mode Toggle
    st.markdown("<div style='font-size:12px;font-weight:600;color:#94a3b8;margin-bottom:6px'>Gate Control Mode</div>", unsafe_allow_html=True)
    if st.session_state.master_control == "Auto":
        st.toggle("Automatic Gate Control", value=True, disabled=True, key="gate_auto_disabled")
        st.caption("🔒 Locked to Auto because Master Control is AUTO.")
    else:
        auto_gate_val = st.toggle("Automatic Gate Control", value=st.session_state.auto_gate, key="gate_auto_enabled")
        if auto_gate_val != st.session_state.auto_gate:
            st.session_state.auto_gate = auto_gate_val
            st.session_state.gate_log.append(f"[{time.strftime('%H:%M:%S')}] 🚪 Gate Mode set to {'AUTO' if auto_gate_val else 'MANUAL'}")
            st.rerun()

    # Gate Indicators
    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
    for i, (s, cmd, note) in enumerate(zip(states, gate_cmds_final, ripple_notes)):
        label = f"{s.label} Gate"
        is_open = st.session_state.gate_states.get(i, False)
        status_icon = "🟢" if is_open else "🔴"
        with st.container():
            col_dot, col_label, col_cmd = st.columns([1, 4, 3])
            with col_dot:
                st.markdown(f"<div style='font-size:18px;padding-top:8px'>{status_icon}</div>", unsafe_allow_html=True)
            with col_label:
                st.markdown(f"<div style='font-size:12px;font-weight:600;color:#e2e8f0;padding-top:10px'>{label}</div>", unsafe_allow_html=True)
                if note:
                    if note == "EMERGENCY MODE ACTIVATED":
                        st.caption("🚨 EMERGENCY OPEN")
                    else:
                        st.caption("⚠ Next area is full, gate stays closed")
            with col_cmd:
                cmd_txt = "OPEN" if is_open else "CLOSE"
                cmd_color = "#22c55e" if is_open else "#ef4444"
                st.markdown(
                    f"<div style='text-align:right;padding-top:6px'>"
                    f"<span style='font-weight:700;"
                    f"font-size:12px;color:{cmd_color};background:rgba(0,0,0,0.3);"
                    f"padding:3px 8px;border-radius:5px'>"
                    f"{cmd_txt}</span></div>",
                    unsafe_allow_html=True,
                )
        st.markdown("<hr style='margin:4px 0;border-color:#1e2d4a'>", unsafe_allow_html=True)

    # 3. Gate Operations (Buttons)
    if not st.session_state.auto_gate and not st.session_state.emergency_active:
        st.markdown("<div style='font-size:12px;font-weight:600;color:#94a3b8;margin:10px 0 6px 0'>Manual Gate Control</div>", unsafe_allow_html=True)
        for i, s in enumerate(states):
            c1, c2 = st.columns(2)
            with c1:
                if st.button(f"🔓 Open {s.label}", key=f"open_{i}", width="stretch"):
                    write_gate(i, True)
                    st.session_state.gate_states[i] = True
                    st.session_state.gate_log.append(f"[{time.strftime('%H:%M:%S')}] 🔓 {s.label} gate MANUAL OPEN")
                    st.rerun()
            with c2:
                if st.button(f"🔒 Close {s.label}", key=f"close_{i}", width="stretch"):
                    write_gate(i, False)
                    st.session_state.gate_states[i] = False
                    st.session_state.gate_log.append(f"[{time.strftime('%H:%M:%S')}] 🔒 {s.label} gate MANUAL CLOSE")
                    st.rerun()

    # 4. Emergency Control
    st.markdown("<hr style='margin:12px 0;border-color:#1e2d4a'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;font-weight:600;color:#94a3b8;margin-bottom:6px'>Emergency Protocol</div>", unsafe_allow_html=True)
    if st.session_state.master_control == "Auto":
        if st.session_state.emergency_active:
            st.markdown("""
            <div class="danger-banner" style="color:#ffffff;padding:8px 12px;border-radius:6px;font-weight:700;font-size:12px;text-align:center;">
                🚨 EMERGENCY EVACUATION ACTIVE
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#0d2218;color:#22c55e;padding:8px 12px;border-radius:6px;font-weight:700;font-size:12px;text-align:center;border:1px solid #1f3d24">
                🟢 EMERGENCY SYSTEM STANDBY
            </div>
            """, unsafe_allow_html=True)
        st.caption("Automatically triggers when Global Risk Score (GRS) reaches DANGER (70%).")
    else:
        if not st.session_state.emergency_active:
            if st.button("🚨 TRIGGER EMERGENCY EVACUATION", type="primary", use_container_width=True, key="manual_trigger_emergency"):
                st.session_state.emergency_active = True
                write_gate(3, True)
                for i in range(3):
                    write_gate(i, True)
                    st.session_state.gate_states[i] = True
                st.session_state.gate_log.append(f"[{time.strftime('%H:%M:%S')}] 🚨 MANUAL EMERGENCY ACTIVATED BY OPERATOR")
                st.rerun()
        else:
            st.markdown("""
            <div class="danger-banner" style="color:#ffffff;padding:8px 12px;border-radius:6px;font-weight:700;font-size:12px;text-align:center;margin-bottom:8px">
                🚨 EMERGENCY EVACUATION ACTIVE (MANUAL)
            </div>
            """, unsafe_allow_html=True)
            if st.button("✅ RESET EMERGENCY STATUS", type="secondary", use_container_width=True, key="manual_reset_emergency"):
                st.session_state.emergency_active = False
                write_gate(3, False)
                for i in range(3):
                    write_gate(i, False)
                    st.session_state.gate_states[i] = False
                st.session_state.gate_log.append(f"[{time.strftime('%H:%M:%S')}] ✅ MANUAL EMERGENCY CLEARED BY OPERATOR")
                st.rerun()

    # 5. Staff Control
    st.markdown("<hr style='margin:12px 0;border-color:#1e2d4a'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;font-weight:600;color:#94a3b8;margin-bottom:6px'>Safety Staff Control</div>", unsafe_allow_html=True)
    if st.session_state.master_control == "Auto":
        st.caption("Safety staff is automatically allocated based on area demand.")
    else:
        st.caption("Manually dispatch safety staff to each zone:")
        cols = st.columns(len(states))
        for col, s in zip(cols, states):
            with col:
                st.markdown(f"<div style='font-size:11px;color:#e2e8f0;text-align:center'>{s.label}</div>", unsafe_allow_html=True)
                val = st.number_input("", min_value=0, max_value=20, value=st.session_state.manual_staff.get(s.zone_id, 2), step=1, key=f"staff_input_{s.zone_id}", label_visibility="collapsed")
                if val != st.session_state.manual_staff.get(s.zone_id, 2):
                    st.session_state.manual_staff[s.zone_id] = val
                    st.session_state.gate_log.append(f"[{time.strftime('%H:%M:%S')}] 🦺 Deployed {val} staff to {s.label}")
                    st.rerun()

    # Gate event log
    if st.session_state.gate_log:
        st.markdown('<div class="section-header">What Happened Recently</div>',
                    unsafe_allow_html=True)
        for entry in st.session_state.gate_log[:6]:
            st.caption(entry)


def render_system_health(states, total_heads):
    # ── BOTTOM ROW: System health ────────────────────────────────────────────
    st.markdown('<div class="section-header">System Health</div>',
                unsafe_allow_html=True)
    t1, t2, t3, t4, t5, t6 = st.columns(6)

    proc = psutil.Process()
    mem_mb = proc.memory_info().rss / 1024 / 1024
    cpu_pct = psutil.cpu_percent(interval=None)
    sys_mem = psutil.virtual_memory()

    avg_fps = np.mean([s.fps for s in states])
    telems = [
        (t1, "Camera Speed", f"{avg_fps:.1f}/sec", "#6366f1"),
        (t2, "Computer Load", f"{cpu_pct:.0f}%", "#06b6d4"),
        (t3, "App Memory", f"{mem_mb:.0f} MB", "#f59e0b"),
        (t4, "PC Memory Used", f"{sys_mem.percent:.0f}%", "#ec4899"),
        (t5, "Cameras Running", f"{len(states)}", "#22c55e"),
        (t6, "People in View", f"~{total_heads}", "#94a3b8"),
    ]
    for col, label, val, color in telems:
        with col:
            st.markdown(f"""
            <div class='info-card' style='padding:12px;text-align:center'>
                <div style='font-size:10px;color:#64748b;
                            letter-spacing:1px;text-transform:uppercase'>{label}</div>
                <div style='font-size:22px;font-weight:800;font-family:JetBrains Mono;
                            color:{color};margin-top:4px'>{val}</div>
            </div>""", unsafe_allow_html=True)


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
                for zid, zlabel, vpath in zones:
                    if Path(vpath).exists():
                        processors.append(ZoneProcessor(
                            zid, zlabel, vpath, model,
                            conf=conf_thresh, imgsz=imgsz, target_fps=target_fps,
                        ))
                if processors:
                    st.session_state.processors = processors
            except Exception as e:
                st.error(f"Failed to initialize: {e}")
                return

    if stop_btn:
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

    # Main render + auto-refresh
    render_dashboard(st.session_state.processors)
    time.sleep(0.05)
    st.rerun()


if __name__ == "__main__":
    main()
