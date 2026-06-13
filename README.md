# 🚨 CA-CRS⁺ — Crowd-Aware Crowd Risk Scoring System

> **Hackathon Project** | Real-time multi-zone crowd crush prevention with SAHI head detection, empirically calibrated occlusion correction, and Modbus TCP industrial gate control.

---

## 📥 Resources & Downloads

- **Video Datasets:** [Download the test video scenarios here (Google Drive)](https://drive.google.com/drive/u/1/folders/1JIDaOIGIhMhedQfAmlVUW6j5cHAvHEV-?usp=share_link)
- **Model Weights:** The fine-tuned `yolov8n_crowd_head.pt` model is available in the **GitHub Releases** page of this repository. Please download it and place it in the `models/` folder.

---

## 🧠 What It Does (The Core Problem)

Current crowd safety systems have **four fatal flaws** that have caused real disasters (Kanjuruhan 2022, Astroworld 2021, Seoul 2022):
1. **Generic alerts:** "High density detected" — but WHERE? Which gate to open?
2. **No causal diagnosis:** They can't tell you if the danger is *too many people*, *panic movement*, or *opposing flows*.
3. **Wrong intervention:** Opening a gate during a speed-driven panic makes people run faster and trample each other.
4. **No resource dispatch:** No calculation of how many marshals to send, or where.

**CA-CRS⁺ solves all four** in real-time, from standard CCTV cameras.

> *CA-CRS⁺ watches your crowd cameras, mathematically identifies the exact cause of a developing danger, and sends the correct physical gate command to prevent a crush — all without a human in the loop.*

---

## ⚙️ System Architecture

- **ThreadPoolExecutor:** Runs all 3 camera zones in parallel.
- **ThreadedStream:** Reads frames in a background daemon thread (non-blocking).
- **YOLOv8n + SAHI:** Detects visible heads even in dense crowds.
- **Live Dashboard:** Streamlit UI updates in-place (`st.empty()`) to maintain 11-22 FPS.
- **Modbus TCP:** Virtual PLC mapping actions directly to physical gate controllers.

---

## 🔬 The 6 Scientific Modules

### Module 1 — Head Detection & Occlusion Correction
Visible heads $\neq$ actual people. At 6 persons/m², ~60% of people are hidden. 
We use our empirically tuned density correction formula $\kappa(\rho)$:
$$\kappa(\rho) = 1 + 8.2442 \cdot \rho^{1.1333}$$
*Calibrated on 1,601 images (ShanghaiTech A/B + UCF-QNRF) achieving $R^2 = 0.615$ (compared to the paper's original formula which had $R^2 = -0.04$).*

### Module 2 — Non-Linear Risk Score (CRS)
$$CRS_k = w_1 D + \Phi(D,S) + w_3 C$$
Includes the **Gridlock Paradox** exponential term $\gamma \cdot e^{\lambda(D-S)}$. High density + low speed = the most dangerous state (crush verge).
We also implemented an improved **IQR-based Conflict metric** that doesn't self-cancel when opposing flows are equal, and a **Crush Reliability Gate** that zeroes out optical flow noise when density exceeds 6 persons/m².

### Module 3 — Causal Attribution
$$r_f^{(k)} = \frac{w_f \cdot X_f}{\sum_j w_j X_j}$$
Determines if the risk is driven by **Density**, **Speed**, or **Conflict**. If no factor dominates (>40%), it returns **MIXED**.

### Module 4 — Cause-to-Gate Mapping
Every command translates the physical intervention needed:
- **Density** $\rightarrow$ **OPEN** exit gates (relieve pressure)
- **Speed** $\rightarrow$ **CLOSE** entry gates (stop inflow)
- **Conflict** $\rightarrow$ **REDIRECT** (separate opposing flows)
- **MIXED** $\rightarrow$ **HOLD** (ambiguous, safest action is no action)
*Ripple-effect logic:* If downstream risk > 0.50, REDIRECT is downgraded to HOLD.

### Module 5 — Resource Demand
$$D_{mar} = \sum \lceil \alpha \cdot \log_{10}(1+N) \rceil$$
Calculates exact active marshals required per zone using base-10 logarithmic scaling, which matches real-world event staffing ratios better than linear scaling.

### Module 6 — Global Risk Score (GRS)
An urgency-weighted sum across all zones ($w_k=2$ for DANGER zones).

---

## 🚀 Quick Start

### 1. Clone & set up environment
```bash
git clone https://github.com/Priyanshu8yadav/CA-CRS-Research.git
cd CA-CRS-Research

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\Activate.ps1 # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Model and Videos
1. Download `yolov8n_crowd_head.pt` from the **GitHub Releases** page and place it in the `models/` directory.
2. Download the `scen_a.mp4`, `scen_b.mp4`, `scen_c.mp4` video files from the **Google Drive link** above and place them anywhere (e.g., `~/Downloads/`). Point the dashboard paths to them.

### 3. Launch the dashboard
* **On macOS/Linux**:
  ```bash
  bash run_dashboard.sh
  ```
* **On Windows (PowerShell)**:
  ```powershell
  .\run_dashboard.ps1
  ```

Once launched, open **http://localhost:8501** in your browser and click **▶ Start Watching**.

---

## 📊 Performance Results

| Metric | Value |
|--------|-------|
| Live FPS | **11–22 frames/sec** (3 cameras × YOLO simultaneously) |
| People tracked | **~1,085** simultaneously |
| $\kappa(\rho)$ calibration $R^2$ | **0.615** (vs -0.04 for paper's formula) |
| DANGER classification accuracy | **75%** (Validated on PETS2009) |
| Head detection rate | **83.8%** (ShanghaiTech Part A) |

---

## 👥 Team & License
Built as an advanced industrial-grade prototype for hackathon presentation.
