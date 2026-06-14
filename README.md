# 🛡️ CA-CRS⁺ — Crowd-Aware Crowd Risk Scoring System

> **Hackathon Project** | Real-time multi-zone crowd crush prevention using YOLO head detection, empirically calibrated occlusion correction, causal attribution, and Modbus TCP industrial gate control.

---

## 📥 Getting the Required Files

Before you run the dashboard, you need two things:

### 1. Model Weights
The fine-tuned `yolov8n_crowd_head.pt` model is published in the **[GitHub Releases](../../releases)** of this repository.

1. Go to the **Releases** page (right panel on GitHub)
2. Download `yolov8n_crowd_head.pt`
3. Place it in the `models/` folder inside the project

> ⚠️ The model is available in the Releases section. If it has been removed, contact the team.

### 2. Test Videos
Three crowd scenario videos are required to run the dashboard:

| File | Zone | Description |
|------|------|-------------|
| `scen_a.mp4` | Entry Corridor | Dense entry crowd |
| `scen_b.mp4` | Central Plaza | Central mass gathering |
| `scen_c.mp4` | Exit Corridor | Exit flow scenario |

Place them in the `public/` folder of this repository, or anywhere on your system and update the paths in the sidebar.

---

## 🚀 Setup & Run (Step-by-Step)

### Step 1 — Clone the repository
```bash
git clone https://github.com/Priyanshu8yadav/CA-CRS-Research.git
cd CA-CRS-Research
```

### Step 2 — Create a Python virtual environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (CMD)
python -m venv .venv
.venv\Scripts\activate.bat
```

### Step 3 — Install all dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add model weights
```
CA-CRS-Research/
└── models/
    └── yolov8n_crowd_head.pt   ← download from GitHub Releases
```

### Step 5 — Add video files
```
CA-CRS-Research/
└── public/
    ├── scen_a.mp4   ← Entry Corridor video
    ├── scen_b.mp4   ← Central Plaza video
    └── scen_c.mp4   ← Exit Corridor video
```

### Step 6 — Launch the dashboard

**macOS / Linux:**
```bash
bash run_dashboard.sh
```

**Windows PowerShell:**
```powershell
.\run_dashboard.ps1
```

**Windows CMD:**
```cmd
run_dashboard.bat
```

**Or directly:**
```bash
source .venv/bin/activate   # (if not already active)
streamlit run dashboard/app.py
```

### Step 7 — Use the dashboard
1. Open **http://localhost:8501** in your browser
2. In the **left sidebar**, verify the three video file paths are correct:
   - Zone A Source → path to `scen_a.mp4`
   - Zone B Source → path to `scen_b.mp4`
   - Zone C Source → path to `scen_c.mp4`
3. Click **▶ Start Watching**

---

## 📁 Project Structure

```
CA-CRS-Research/
├── dashboard/
│   ├── app.py               # Streamlit live dashboard
│   ├── ca_crs_engine.py     # Core: YOLO + κ correction + risk scoring + causal attribution
│   └── virtual_plc.py       # Modbus TCP virtual PLC for gate actuation
│
├── calibration/
│   ├── calibrate_kappa.py   # κ(ρ) empirical calibration pipeline
│   ├── metrics.py           # Evaluation metrics
│   └── sensitivity_analysis.py
│
├── models/                  # Place model weights here (not tracked by git)
├── public/                  # Place video files here
│
├── config.yaml              # All tunable parameters
├── requirements.txt
├── run_dashboard.sh         # macOS/Linux launcher
├── run_dashboard.ps1        # Windows PowerShell launcher
└── run_dashboard.bat        # Windows CMD launcher
```

---

## 🧠 What The System Does

### The Problem
Current crowd safety systems have four fatal flaws — they issue generic alerts without identifying the actual cause, they don't know the exact spatial location, they don't specify which gate to open, and they don't check if there are enough marshals available. These flaws contributed to tragedies at Kanjuruhan (2022), Astroworld (2021), and Itaewon (2022).

### CA-CRS⁺ solves all four in real-time:

```
3 Live Camera Zones (RTSP or MP4)
        │
        ▼  [ThreadPoolExecutor — 3 parallel workers]
  YOLOv8n + SAHI Head Detection
        │
  Module 2: κ(ρ) Occlusion Correction
        │
  Module 3: CRS Non-linear Risk Score + Causal Attribution r_f
        │
  Module 4: Cause-to-Gate Command Mapping
        │
  Module 5: Per-zone Marshal Demand log₁₀
        │
  Module 6: Urgency-weighted Global Risk Score (GRS)
        │
  Streamlit Dashboard + Modbus TCP Gate Control
```

---

## 🔬 Core Formulas

| Module | Formula | Purpose |
|--------|---------|---------|
| **M2** | `κ(ρ) = 1 + 8.2442·ρ^1.1333` | Correct for occluded heads |
| **M3** | `CRS = w₁D + w₂S(1−D) + γ·e^(λ(D−S)) + w₃C` | Non-linear risk score |
| **M3** | `r_f = (w_f·X_f) / Σ(w_j·X_j)` | Causal attribution ratio |
| **M4** | Density→OPEN, Speed→CLOSE, Conflict→REDIRECT, MIXED→HOLD | Gate mapping |
| **M5** | `D_mar = Σ⌈α·log₁₀(1+N)⌉` with α∈{0,4,10} | Marshal demand |
| **M6** | `GRS = Σ(w_k·CRS_k)/Σ(w_k)`, w=2 for DANGER | Global risk |

---

## 📊 Calibration Results

| Metric | Value |
|--------|-------|
| κ(ρ) R² | **0.615** (vs −0.04 for paper's original formula) |
| Head detection rate | **83.8%** (ShanghaiTech Part A) |
| DANGER classification accuracy | **75%** (PETS2009 benchmark) |
| Calibration images | **1,601** (ShanghaiTech A/B + UCF-QNRF) |
| Live processing speed | **11–22 FPS** (3 cameras simultaneously) |

---

## ⚙️ Risk Levels & Gate Logic

| CRS Score | Status | Color | Gate Action |
|-----------|--------|-------|-------------|
| 0.00 – 0.35 | SAFE | 🟢 Green | HOLD |
| 0.35 – 0.70 | WARNING | 🟡 Amber | REDIRECT / CLOSE |
| 0.70 – 1.00 | DANGER | 🔴 Red | OPEN / CLOSE |
| Any (MIXED) | Any | — | HOLD (no dominant cause) |

**Ripple-effect logic:** If upstream issues a REDIRECT command but downstream zone risk > 0.50, the command is automatically downgraded to HOLD to prevent funnel crush.

---

## 🔧 Requirements

```
Python 3.10+
streamlit>=1.35
ultralytics>=8.2
sahi>=0.11
opencv-python
pymodbus>=3.10
plotly
psutil
numpy
```

---

## 👥 Team

Built for the hackathon — CA-CRS⁺ (Crowd-Aware Crowd Risk Scoring Plus).
