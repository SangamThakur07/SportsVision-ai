# SportsVision AI 🏟️
### AI-Based Sports Analytics System using Computer Vision & Deep Learning
**Lovely Professional University | B.Tech CSE Project**  
**Authors:** Sangam Thakur (12403852) · Viren (12409626)

---

## 📌 Project Overview

SportsVision AI is a real-time football match analytics system that processes broadcast video using **YOLOv8** object detection and **ByteTrack** multi-object tracking to:
- Detect and track all 22 players + ball across match frames
- Assign persistent tactical jersey IDs (White #1–#11, Red #1–#11)
- Compute calibrated FIFA pitch coordinates, speed (km/h), and distance (m)
- Generate 2D Gaussian heatmaps, possession statistics, and Grad-CAM XAI explanations
- Visualize everything in a professional **Streamlit** dashboard

---

## 🏗️ System Architecture

```
Football Broadcast Video (.mp4)
         │
         ▼
[1] OpenCV Frame Stream          ← memory-efficient generator, no full decode
         │
         ▼
[2] Image Preprocessor           ← letterbox resize, BGR→RGB, pixel normalization
         │
         ▼
[3] YOLOv8x Object Detection     ← Class 0: Person | Class 32: Sports Ball
         │
         ▼
[4] Pitch ROI Filter             ← drops sideline staff (ground_y_px / 720 > 0.78)
         │
         ▼
[5] ByteTrack MOT                ← persistent track_id across occlusions
         │
         ▼
[6] K-Means Team Classifier      ← HSV jersey color → Team White / Team Red
         │
         ▼
[7] Roster Mapper                ← track_id → tactical jersey + role (GK/CB/ST…)
         │
         ▼
[8] Kinematics & Analytics       ← speed (km/h), distance (m), heatmaps, possession
         │
         ▼
[9] Streamlit Dashboard          ← interactive 7-section web UI + video export
```

---

## 📁 Project Structure

```
SportsVision-ai/
│
├── app.py                          # Main Streamlit dashboard (7 analytical views)
├── run_pipeline.py                 # CLI pipeline runner (headless batch processing)
├── requirements.txt                # All Python dependencies with version pins
├── README.md                       # This file
│
├── config/
│   ├── config.yaml                 # Detection thresholds, colors, device settings
│   └── sports_bytetrack.yaml       # ByteTrack MOT hyperparameters
│
├── src/
│   ├── preprocessing/
│   │   └── preprocessor.py         # Letterbox resize, normalization
│   ├── detection/
│   │   └── detector.py             # YOLOv8 wrapper, NMS, ROI pitch filter
│   ├── tracking/
│   │   └── tracker.py              # ByteTrack integration, 78% ROI cutoff
│   ├── classification/
│   │   └── team_classifier.py      # K-Means HSV jersey color clustering
│   ├── analytics/
│   │   ├── roster.py               # Tactical jersey/role assignment (22-player)
│   │   ├── movement.py             # Kinematics: speed, distance, workrate
│   │   ├── heatmap.py              # 2D Gaussian KDE heatmap generation
│   │   ├── possession.py           # Nearest-neighbor ball possession estimation
│   │   ├── homography.py           # Pixel ↔ FIFA pitch metric calibration
│   │   └── team_analytics.py       # Formation spread, territory area
│   ├── evaluation/
│   │   ├── metrics.py              # mAP, IoU, Precision, Recall computation
│   │   └── gradcam.py              # Grad-CAM explainability saliency maps
│   └── visualization/
│       ├── annotator.py            # Bounding box, ID, trail overlay on frames
│       ├── video_loader.py         # OpenCV-based frame generator
│       └── video_writer.py         # Annotated output video writer
│
├── data/
│   ├── sample/                     # Sample match videos (not included in repo — see below)
│   └── sample_training_dataset/
│       ├── sample_training_dataset.csv   # 50 representative annotation records
│       ├── sample_training_dataset.json  # Same records with metadata header
│       └── DATA_DICTIONARY.md            # Full field-by-field description
│
├── models/
│   └── weights/
│       └── yolov8n.pt              # YOLOv8 nano weights (auto-downloaded if missing)
│
├── outputs/
│   ├── videos/                     # Processed annotated output videos
│   ├── heatmaps/                   # Generated PNG heatmap images
│   └── reports/                    # CSV exports, HTML/PDF reports, presentation
│
├── scripts/
│   ├── generate_sample_dataset.py  # Generates the 50-record training dataset
│   └── append_records.py           # Appends additional edge-case records
│
└── tests/
    ├── test_phase1.py              # Preprocessing & detection unit tests
    ├── test_phase2.py              # Tracking & classification tests
    ├── test_phase3.py              # Analytics & metrics tests
    ├── test_advanced.py            # Integration & pipeline tests
    └── test_roster.py              # Tactical roster mapping tests
```

---

## ⚙️ Installation & Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- GPU recommended (CUDA 11.8+) — CPU mode works but is slower

### Step 1 — Clone the Repository
```bash
git clone https://github.com/SangamThakur07/SportsVision-ai.git
cd SportsVision-ai
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

### Step 4 — Run CLI Pipeline (optional)
```bash
python run_pipeline.py --input data/sample/synthetic_match.mp4 \
                       --output outputs/videos/output_analytics.mp4 \
                       --mode tracking
```

### Step 5 — Run Tests
```bash
pytest tests/ -v
# Expected: 19 tests passing
```

---

## 🎛️ Configuration

Edit `config/config.yaml` to tune:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detection.confidence` | `0.40` | YOLO detection confidence threshold |
| `detection.iou_threshold` | `0.45` | NMS IoU threshold |
| `tracking.pitch_roi_y` | `0.78` | Vertical cutoff to exclude sideline persons |
| `device` | `cpu` | Set to `cuda` for GPU acceleration |
| `classes` | `[0, 32]` | COCO class IDs (Person + Sports Ball) |

---

## 📊 Dashboard Sections

| Section | Description |
|---------|-------------|
| 🏠 Home | Architecture overview, tech stack, live demo video |
| 📹 Video Analysis | Match video upload, ROI pitch filter slider, duration selector |
| 🎯 Tracking & Detection | Frame-by-frame tracking, persistent IDs, motion trails |
| 🏃 Player Analytics | 22-player roster, speed (km/h), distance (m), workrate categories |
| 👥 Team & Possession | Ball possession %, team formations, territory heatmap |
| 🔥 Movement Heatmaps | 2D Gaussian KDE overlaid on FIFA pitch diagram |
| 📊 Model Evaluation | mAP, IoU, Precision/Recall, Grad-CAM XAI saliency maps |

---

## 📐 Key Technical Concepts

### Coordinate Systems
- **Pixel space:** 1280 × 720 px, origin top-left
- **YOLO normalized:** `[cx, cy, w, h]` ∈ `[0.0, 1.0]`
- **FIFA metric:** `pitch_x_m = ground_x_px × (105 / 1280)`, `pitch_y_m = ground_y_px × (68 / 720)`

### Pitch ROI Filter (Sideline Exclusion)
Any detection where `ground_y_px / 720 > 0.78` is flagged as outside the active pitch zone, eliminating coaches, cameramen, and ball boys from tactical analysis.

### Speed Formula
$$v = \frac{\Delta d_{px} \times \text{scale}_{m/px}}{\Delta t} \times 3.6 \quad [\text{km/h}]$$

### Workrate Categories
| Category | Speed |
|----------|-------|
| ⚡ High Sprint | > 20 km/h |
| 🏃 Active Running | 10 – 20 km/h |
| 🚶 Positional Zone | 0 – 10 km/h |

---

## 🗃️ Sample Dataset

The `data/sample_training_dataset/` folder contains 50 representative annotation records across 4 video frames covering all object types, tactical roles, and edge cases (ball occlusion, ROI-filtered sideline persons). See `DATA_DICTIONARY.md` for full field descriptions.

---

## 🧪 Test Results

```
tests/test_phase1.py      5 passed   ✅  Preprocessing & detection
tests/test_phase2.py      4 passed   ✅  Tracking & team classification
tests/test_phase3.py      5 passed   ✅  Analytics & metrics
tests/test_advanced.py    3 passed   ✅  Integration tests
tests/test_roster.py      2 passed   ✅  Tactical roster mapping
─────────────────────────────────────
TOTAL                    19 passed   ✅
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8x (Ultralytics) |
| Multi-Object Tracking | ByteTrack MOT |
| Deep Learning Framework | PyTorch 2.0+ |
| Team Classification | K-Means clustering (scikit-learn) |
| Analytics | NumPy, SciPy, Pandas |
| Heatmaps | Matplotlib + Gaussian KDE |
| Dashboard | Streamlit |
| XAI | Grad-CAM |
| Video Processing | OpenCV |

---

## 👨‍💻 Authors

| Name | Registration No. | Institution |
|------|-----------------|-------------|
| Sangam Thakur | 12403852 | Lovely Professional University, Punjab |
| Viren | 12409626 | Lovely Professional University, Punjab |

**Department:** Computer Science & Engineering  
**Academic Year:** 2026  
**Repository:** https://github.com/SangamThakur07/SportsVision-ai.git

---

## 📜 License

This project is submitted as part of an academic course at Lovely Professional University.  
All code is original work by the authors listed above.

---

## 📝 Notes

- Video files in `data/sample/` are not included in the ZIP due to size (use your own `.mp4` football footage).
- YOLOv8 weights (`yolov8n.pt`) are auto-downloaded on first run if not present.
- No external API keys or credentials are required to run this project.
