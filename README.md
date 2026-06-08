# 🚨 CA-CRS⁺ — Crowd-Aware Crowd Risk Scoring System

> **Hackathon Project** | Real-time multi-zone crowd crush prevention with SAHI head detection, empirically calibrated occlusion correction, and Modbus TCP industrial gate control.

---

## 📸 Dashboard Preview

Live Streamlit command centre — 3 camera zones, Global Risk Score gauge, causal breakdown, and virtual PLC gate actuation.

---

## 🧠 What It Does

CA-CRS⁺ is an **edge AI safety system** that:

1. **Detects crowd density** using `YOLOv8n` fine-tuned for head detection + **SAHI** sliced inference for small/occluded heads
2. **Corrects for occlusion** using an empirically calibrated κ(ρ) function:
   ```
   κ(ρ) = 1 + 8.2442 · ρ^1.1333    (R² = 0.615)
   ```
   Calibrated on 1,601 images (ShanghaiTech A/B + UCF-QNRF)
3. **Scores crowd risk** via the CA-CRS⁺ non-linear formula:
   ```
   CRS = w₁·D + w₂·S·(1−D) + γ·exp(λ·(D−S)) + w₃·C
   ```
4. **Actuates industrial gates** via Modbus TCP (virtual PLC with ripple-lock logic)
5. **Shows a live dashboard** with per-zone risk scores, Global Risk Score, causal breakdown, and resource demand

---

## 🚀 Quick Start

### 1. Clone & set up environment
```bash
git clone https://github.com/YOUR_USERNAME/CA-CRS-Research.git
cd CA-CRS-Research

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the head detection model
```bash
mkdir -p models
# Download from HuggingFace:
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Ultralytics/assets', 'yolov8n.pt', local_dir='models/')
# or your fine-tuned head detector:
# hf_hub_download('arnabdhar/YOLOv8-Face-Detection', 'model.pt', local_dir='models/')
"
```
> The model file (`yolov8n_crowd_head.pt`) is excluded from git due to size. Ask the team for the fine-tuned weights.

### 3. Add your video feeds
Place three crowd video files (MP4):
```
~/Downloads/scen_a.mp4   ← Entry Corridor
~/Downloads/scen_b.mp4   ← Central Plaza  
~/Downloads/scen_c.mp4   ← Exit Gate
```
Or edit the paths in the dashboard sidebar.

### 4. Launch the dashboard
```bash
bash run_dashboard.sh
# Open: http://localhost:8501
# Click ▶ Start Dashboard
```

---

## 📁 Project Structure

```
CA-CRS-Research/
├── dashboard/
│   ├── app.py              # Streamlit command centre
│   ├── ca_crs_engine.py    # YOLO inference + κ correction + risk scoring
│   └── virtual_plc.py      # Modbus TCP virtual PLC (pymodbus 3.13+)
│
├── scripts/
│   ├── calibrate_kappa.py  # Empirical κ(ρ) calibration pipeline
│   ├── sensitivity_analysis.py
│   └── evaluate_pets2009.py
│
├── config.yaml             # All tunable parameters
├── requirements.txt        # Python dependencies
└── run_dashboard.sh        # One-command launcher
```

---

## 📊 Calibration Results

| Metric | Value |
|--------|-------|
| κ(ρ) R² | 0.615 |
| Head detection rate | 83.8% (ShanghaiTech A) |
| PETS2009 classification accuracy | 75% |
| Training images | 1,601 |
| Calibration datasets | ShanghaiTech A/B + UCF-QNRF |

---

## ⚙️ Risk Thresholds

| Score | Status | Gate Command |
|-------|--------|-------------|
| < 0.35 | 🟢 SAFE | HOLD |
| 0.35 – 0.70 | 🟡 WARNING | REDIRECT |
| > 0.70 | 🔴 DANGER | OPEN (evacuate) |

Ripple logic: if a downstream zone is also in DANGER when upstream is WARNING → override to HOLD to prevent funnel crush.

---

## 🔧 Dependencies

```
streamlit>=1.35
ultralytics>=8.2
sahi>=0.11
opencv-python
pymodbus>=3.10
plotly
psutil
numpy
huggingface_hub
```

---

## 👥 Team

Built for the hackathon — extend and improve as you go!
