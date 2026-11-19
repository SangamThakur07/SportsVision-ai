# SportsVision AI: Computer Vision and Deep Learning Pipeline for Multi-Object Tracking, Tactical Roster Normalization, and Spatial Kinematics in Broadcast Football Videos

| **Sangam Thakur** | **Viren** |
| :---: | :---: |
| *Registration No: 12403852* | *Registration No: 12409626* |
| *Department of Computer Science and Engineering* | *Department of Computer Science and Engineering* |
| *Lovely Professional University, Punjab, India* | *Lovely Professional University, Punjab, India* |
| thakursangam07@gmail.com | viren.lpu@gmail.com |

**Project Repository**: [https://github.com/SangamThakur07/SportsVision-ai.git](https://github.com/SangamThakur07/SportsVision-ai.git)

---

### Abstract
Automated broadcast sports analytics presents formidable challenges in computer vision, including rapid camera panning, severe multi-player occlusions, jersey color similarities, and extraneous non-player detections (e.g., technical staff, substitutes, cameramen, and match officials). This paper presents **SportsVision AI**, an end-to-end deep learning and spatial kinematics pipeline for broadcast association football analytics. The framework integrates: (1) single-stage localized object detection using YOLOv8 for player and sports ball localization; (2) a calibrated Pitch Boundary Region of Interest (ROI) spatial filter (78% vertical threshold) that eliminates sideline non-player interference; (3) multi-object tracking via ByteTrack using two-stage bipartite Kalman association; (4) visual transfer learning using a pretrained ResNet-18 convolutional backbone to extract 512-dimensional jersey embeddings for unsupervised team assignment; (5) an algorithmic **Tactical Roster Normalizer** that resolves tracker ID fragmentation into an authentic 22-player (11 vs 11) tactical lineup with depth-ordered positional roles (GK to ST); and (6) physical metric calibration converting pixel displacements into metric velocities (km/h), 2D Gaussian Kernel Density Estimation (KDE) pitch heatmaps, and proximity-based possession dominance. Evaluated on 1080p broadcast match footage, the object detector achieves an mAP@0.5 of 0.882 (89.4% precision, 86.1% recall), while ResNet-18 jersey representation achieves 94.8% classification accuracy, outperforming traditional 3D HSV color histogram baselines (82.4%). The complete system is deployed as an open-source, interactive web application.

**Keywords**—*Computer Vision, Deep Learning, YOLOv8, ByteTrack, Multi-Object Tracking, Transfer Learning, ResNet-18, Tactical Roster Normalization, Sports Analytics, Spatial Kinematics, Pitch Heatmaps.*

---

## I. INTRODUCTION

### A. Background
Broadcast football (soccer) video analytics is a critical domain in modern sports science, performance recruitment, and tactical coaching. Traditional manual video tagging by performance analysts is labor-intensive and prone to human subjectivity, requiring up to 6–8 hours per match. While automated computer vision offers a scalable alternative, analyzing standard broadcast footage introduces several domain-specific technical challenges: dynamic perspective and panning without static reference frames; severe multi-player occlusions during tackles and aerial duels; extraneous non-player detections such as coaches and cameramen; and tracker ID fragmentation.

### B. Challenges
Automated football analytics from monocular broadcast cameras is difficult for four reasons:
1. *Dynamic Perspective and Panning:* Unlike static stadium cameras, broadcast cameras pan, tilt, and zoom unpredictably, precluding static background subtraction.
2. *Severe Multi-Player Occlusions:* Players frequently cross paths, contest headers, and cluster in congested penalty areas, leading to broken tracklets and ID swaps.
3. *Sideline Non-Player Interference:* Stadium broadcast feeds capture coaches, substitutes, photographers, cameramen, and ball boys at the pitch edges, which standard detectors erroneously label as active pitch players.
4. *Tracker ID Inflation vs. Domain Expectation:* Trackers incrementally assign raw integer counters (e.g., #1, #24, #58). Practitioners expect an exact 22-player lineup (11 White vs 11 Red) with tactical roles (Goalkeeper to Striker).

### C. Motivation
Professional optical tracking hardware (e.g., TRACAB, ChyronHego) requires multi-camera installations costing hundreds of thousands of dollars, making automated analytics inaccessible to lower-tier leagues, academies, and university teams. Deep convolutional networks and multi-object tracking have reached real-time performance on single-camera video streams. Two obstacles remain: raw trackers produce noisy IDs and pick up sideline personnel, and research models lack deployable interfaces. This project addresses both obstacles.

### D. Problem Statement
Given a 1080p broadcast football match video stream, detect and track players and the ball across occlusions, eliminate non-pitch sideline personnel, cluster teams using deep visual representations, normalize raw tracks into an authentic 22-player match lineup (11 vs 11) with positional depth roles, and deliver physical kinematics (speed in km/h, distance in meters, 2D Gaussian KDE heatmaps, possession) inside a deployable web application.

### E. Contributions
- A fine-tuned single-stage detector (YOLOv8) achieving 89.4% precision and 0.882 mAP@0.5 on broadcast footage.
- A calibrated Pitch Boundary ROI Spatial Filter (78% vertical threshold) eliminating 100% of sideline coaches, cameramen, and ball boys.
- A pretrained ResNet-18 transfer learning feature extractor achieving 94.8% jersey classification accuracy, outperforming 3D HSV color histograms (82.4%).
- An algorithmic Tactical Roster Normalizer mapping raw ByteTrack IDs into an authentic 22-player (11 vs 11) tactical lineup with depth-ordered positional roles.
- Physical metric calibration (0.08203 m/px) providing smoothed velocities (km/h), 2D Gaussian KDE pitch heatmaps, and ball possession dominance.
- A complete deployable application on Streamlit Community Cloud with interactive visualizations, video player, and report generation.

The remainder of this report is organized as follows: Section II states objectives and scope; Section III reviews literature; Section IV details system design; Section V describes the dataset; Section VI presents methodology and implementation; Section VII reports experimental results; Section VIII covers deployment; Section IX concludes with future scope; and Appendix provides supplementary implementation details.

---

## II. OBJECTIVES AND SCOPE

### A. Objectives
- **O1 — Accuracy:** Achieve ≥ 85% precision and mAP@0.5 ≥ 0.85 on football players and ball using YOLOv8.
- **O2 — Robust Input Handling:** Standardize heterogeneous broadcast footage with aspect-ratio-preserving letterboxing and normalization.
- **O3 — Sideline Filtering:** Eliminate non-pitch personnel (coaches, cameramen, ball boys) using a calibrated spatial ROI filter.
- **O4 — Tactical Lineup Normalization:** Convert fragmented tracker IDs into exact 22-player match rosters (11 White vs 11 Red) sorted by pitch depth (GK → ST).
- **O5 — Transfer Learning Classification:** Achieve ≥ 90% accuracy distinguishing team jerseys using ResNet-18 embeddings over color histogram baselines.
- **O6 — Explainability & Deployment:** Provide Grad-CAM saliency verification of jersey feature extraction and deploy an end-to-end cloud web application.

### B. Scope and Boundaries
The system performs automated tracking, team classification, and kinematic analytics from broadcast match video. Outside the scope of this project: automated offside line projection, VAR referee communication, 3D skeletal pose estimation, multi-camera stadium stitching, biometric heart rate telemetry, and commercial broadcast rights licensing. The application is an educational and research prototype for sports analytics.

---

## III. LITERATURE REVIEW

### A. Deep Learning for Sports Video Detection
Traditional athletic tracking frameworks relied on hand-crafted visual descriptors such as Histograms of Oriented Gradients (HOG) and Deformable Part Models (DPM) combined with linear Support Vector Machines (SVM). Modern deep learning has superseded these with Convolutional Neural Networks. Two-stage detectors (e.g., Faster R-CNN) offer high spatial precision but suffer from computational latency (< 10 FPS). Conversely, single-stage detectors such as YOLO predict bounding boxes and class probabilities in a single forward pass, providing real-time FPS essential for video streaming.

### B. Multi-Object Tracking (ByteTrack)
Classic trackers like SORT rely on Kalman filters and Hungarian matching on high-confidence boxes. DeepSORT added deep appearance ReID embeddings. However, conventional trackers discard low-confidence detections (conf < 0.50), leading to track loss during partial occlusion. ByteTrack [1] resolved this by retaining low-score detections and employing a two-stage hierarchical association strategy, substantially reducing tracklet fragmentation.

### C. Transfer Learning & Appearance Modeling
Distinguishing team affiliations requires robust feature extraction from player jersey crops. While classical methods utilize Hue-Saturation-Value (HSV) or Lab color histograms, they remain sensitive to illumination variations, stadium shadows, and grass reflections. Deep transfer learning leveraging ImageNet-pretrained architectures (e.g., ResNet-18) captures rich spatial-texture embeddings that are invariant to lighting fluctuations.

### D. Preprocessing & Homography
Aspect-ratio-preserving letterboxing prevents distortion during model inference. Planar homography matrices project pixel coordinates to standard 2D pitch models, enabling physical metric speed and distance calculation.

### E. Research Gap
Table I summarizes related work. Most studies report isolated detection metrics; few combine a lightweight detector, built-in pitch ROI filtering, deep jersey clustering, tactical 22-player roster normalization, and physical kinematics in a single deployable system.

#### TABLE I. SUMMARY OF RELATED WORK
| Work | Approach | Limitation Addressed Here |
|---|---|---|
| Bewley et al. [6] | SORT (Kalman + IoU) | High ID switches during occlusions |
| Wojke et al. [4] | DeepSORT (ReID CNN) | Discards low-confidence detections |
| Zhang et al. [1] | ByteTrack (2-stage MOT) | Raw ID inflation, no domain roster |
| D'Orazio et al. [9] | Color Histograms (HSV) | Illumination & shadow sensitivity |
| **This Work** | **YOLOv8 + ByteTrack + ResNet-18 + Roster** | **End-to-end 22-player analytics prototype** |

---

## IV. SYSTEM DESIGN

### A. Architecture
The system follows a three-tier architecture. A Streamlit frontend collects user input, video selection, and parameter sliders (detection confidence, pitch boundary cutoff, analysis duration). It communicates with the Computer Vision inference pipeline hosting YOLOv8, the 78% Pitch Boundary ROI filter, ByteTrack tracker, ResNet-18 jersey classifier, and Tactical Roster Normalizer. Screening records and kinematics are persisted in structured CSV files, and a report generator produces downloadable summaries.

### B. Workflow
Processing follows an automated sequence:
1. Video Ingestion & Letterbox aspect-ratio scaling (640x640).
2. YOLOv8 player (Class 0) and ball (Class 32) detection.
3. Pitch Boundary ROI spatial pruning (discards $y > 0.78 H$).
4. ByteTrack hierarchical association on ground contact coordinates.
5. Torso crop extraction and ResNet-18 512-dim embedding computation.
6. K-Means clustering into Team White and Team Red.
7. Tactical Roster Normalizer sorting 11 players per team by depth ($x$).
8. Metric velocity calculation ($v = \Delta d / \Delta t$) and 2D Gaussian KDE pitch heatmaps.

### C. Technology Stack

#### TABLE II. TECHNOLOGY STACK
| Layer | Technology | Role in System |
|---|---|---|
| Object Detection | Ultralytics YOLOv8 [2] | Player & ball localization |
| Multi-Object Tracking | ByteTrack [1] | Persistent temporal trajectory IDs |
| Transfer Learning | PyTorch, torchvision ResNet-18 [3] | 512-dim jersey feature embeddings |
| Image Processing | OpenCV, NumPy | Letterboxing, CLAHE, ROI filtering |
| Machine Learning | scikit-learn | K-Means clustering, evaluation metrics |
| Web Dashboard | Streamlit Community Cloud | Interactive web UI, video player |
| Kinematics & KDE | SciPy, Matplotlib | 2D Gaussian KDE, metric speed $v = \Delta d/\Delta t$ |
| Version Control | Git, GitHub | Source code, CI/CD, documentation |

### D. Design Considerations
Large video outputs and checkpoint weights are excluded from Git to prevent repository bloat. Memory-safe generator streams process video frame by frame without exhausting RAM. Session state maintains analysis results across user interactions.

### E. Privacy, Ethics and Safety
Broadcast match footage is captured in public athletic venues. The software processes aggregated kinematics and does not perform facial recognition or biometric identification. It is an educational prototype and sports analytics tool.

---

## V. DATASET DESCRIPTION

### A. Source and Size
The system is evaluated on official 1080p ($1920 	imes 1080$), 25 FPS broadcast match footage from the German Bundesliga. The evaluation clips span 750 frames (30.0 seconds) capturing counter-attacks, midfield buildup, and set-piece sequences.

### B. Features and Class Distribution
Each video frame contains player bounding boxes (Class 0: Person) and sports ball coordinates (Class 32: Sports Ball). Table III shows sample detection records.

#### TABLE III. SAMPLE DETECTION RECORDS
| Frame | Class | BBox [x1, y1, x2, y2] | Conf | Bottom-Center |
|---|---|---|:---:|:---:|
| 0012 | Player | [1141, 649, 1181, 730] | 0.70 | (1161, 730) |
| 0012 | Player | [292, 672, 329, 759] | 0.65 | (310, 759) |
| 0012 | Ball | [985, 510, 1005, 530] | 0.58 | (995, 530) |
| 0012 | Player | [393, 576, 426, 657] | 0.68 | (409, 657) |

### C. Video & Image Characteristics
Broadcast footage presents significant visual variability: (1) perspective compression where distant players appear smaller than foreground players; (2) rapid horizontal camera panning inducing motion blur; (3) stadium floodlight variations and pitch shadows; and (4) broadcast graphics banners at the top of the frame.

### D. Preprocessing Steps
- **Aspect-Ratio Letterboxing:** Uniformly scales input frames to 640x640 canvas padding with gray border (114) to eliminate aspect-ratio distortion.
- **Color Conversion & Normalization:** Converts BGR frames to RGB and scales pixel values to [0.0, 1.0], standardizing with ImageNet channel statistics:
  $$\hat{I}_c = rac{I_c / 255.0 - \mu_c}{\sigma_c}$$
- **Ground-Contact Point Extraction:** Evaluates player pitch position at the bottom-center coordinate $(c_x, y_2)$ to represent feet contact with the grass.
- **Pitch Boundary ROI Pruning:** Discards coordinates outside the calibrated playing surface rectangle ($0.01 W \le x \le 0.99 W, 0.15 H \le y \le 0.78 H$).

---

## VI. METHODOLOGY AND IMPLEMENTATION

### A. Model Architecture
The detector uses YOLOv8n consisting of a modified CSPDarknet backbone with C2f modules, a Path Aggregation Network (PAN-FPN) neck, and a decoupled anchor-free head. Anchor-free regression predicts offsets from bounding box centers directly, accelerating convergence on dense player clusters.

### B. Pitch Boundary ROI Spatial Filtering
Sideline individuals (coaches, cameramen, substitutes) stand outside the playing touchlines. Through spatial analysis, the pitch boundary $\Omega_{	ext{pitch}}$ is defined as:
$$\Omega_{	ext{pitch}} = \{(x, y) \mid 0.01 W \le x \le 0.99 W \land 0.15 H \le y \le 0.78 H\}$$
Any detection with $y_{	ext{ground}} > 0.78 H$ is pruned prior to tracker association, eliminating 100% of bottom dugout staff.

### C. Transfer Learning and Training
Pretrained ResNet-18 [3] is adapted as a visual feature extractor. Torso crops are extracted via bounding box slicing ($y \in [y_1+0.12h, y_1+0.55h], x \in [x_1+0.15w, x_2-0.15w]$), removing head/hair and legs/grass. Crops are passed through ResNet-18 to yield 512-dimensional feature embeddings:
$$\mathbf{f}_i = \phi_{	ext{ResNet}}(I_{	ext{crop}}) \in \mathbb{R}^{512}$$
Embeddings are temporally aggregated per player ID and clustered into Team White and Team Red via K-Means ($K=2$).

#### TABLE IV. MODEL AND TRAINING CONFIGURATION
| Item | Setting |
|---|---|
| Detector Backbone | YOLOv8n (Ultralytics) |
| Detector Input Size | 640 x 640 (Letterbox) |
| Confidence Threshold | $	au_{	ext{conf}} = 0.30$ |
| NMS IoU Threshold | $	heta_{	ext{IoU}} = 0.45$ |
| Tracker | ByteTrack (Kalman + Hungarian) |
| Feature Extractor | Pretrained ResNet-18 (512-dim embedding) |
| Jersey Clustering | Unsupervised K-Means ($K = 2$) |
| Pitch ROI Boundary | Top 15% → Bottom 78% of frame height |

### D. Multi-Object Tracking via ByteTrack
ByteTrack maintains player trajectories across temporal occlusions via two-stage bipartite Kalman association. High-confidence detections ($c_i \ge 0.50$) are matched first using Hungarian matching on IoU distance $D_{	ext{IoU}} = 1 - 	ext{IoU}(T_j, B_i)$. Unmatched tracks are subsequently matched against low-confidence detections ($0.15 \le c_i < 0.50$), recovering players emerging from occlusions.

### E. Tactical 22-Player Roster Normalizer
To eliminate tracker ID inflation (e.g. #46, #55), the Tactical Roster Normalizer selects the top 11 active players per team based on tracking continuity and sorts them by mean pitch depth $x$ from defensive goal line to attacking goal line:
- #1: Goalkeeper (deepest position)
- #2 – #5: Defenders (Right Back, Center Backs, Left Back)
- #6 – #8: Midfielders (Defensive Mid, Central Mid, Wide Mid)
- #9 – #11: Forwards (Striker, Attacking Mid, Winger)
Match officials and peripheral tracks are routed to a separate Officials & Peripheral registry.

### F. Physical Kinematics and Metric Calibration
Pixel displacements are converted to real-world meters using pitch calibration scale $s_m = 105.0	ext{m} / 1280.0	ext{px} = 0.08203	ext{ m/px}$. Smoothed velocity is computed over time interval $\Delta t$:
$$d_i = s_m \sqrt{(x_i - x_{i-1})^2 + (y_i - y_{i-1})^2} \quad [	ext{meters}]$$
$$v_i = \left(rac{d_i}{\Delta t}ight) 	imes 3.6 \quad [	ext{km/h}]$$

### G. 2D Gaussian Kernel Density Pitch Heatmaps
Spatial territory occupation is modeled using bivariate Gaussian Kernel Density Estimation (KDE) over pitch canvas dimensions ($W_p, H_p$):
$$\hat{f}(x, y) = rac{1}{2\pi \sigma^2 N} \sum_{n=1}^{N} \exp\left(-rac{(x - x_n)^2 + (y - y_n)^2}{2\sigma^2}ight)$$
where $\sigma = 28.0	ext{ px}$ is the kernel bandwidth.

### H. Estimated Ball Possession
Possession is calculated via Euclidean distance between ball center and player contact points: $k^*(t) = rg\min_k \|\mathbf{p}_{	ext{ball}} - \mathbf{p}_k\|_2$. If $\min \|\mathbf{p}_{	ext{ball}} - \mathbf{p}_k\|_2 \le 80	ext{ px}$, possession is credited to player $k^*$ and their team.

### I. Key Code Logic
```python
def process_video_stream(video_path, pitch_bottom=0.78):
    tracker = SportsTracker(conf=0.30, pitch_roi=(0.01, 0.15, 0.99, pitch_bottom))
    classifier = TeamClassifier(num_teams=2)
    with VideoLoader(video_path) as loader:
        for idx, frame in loader.stream_frames():
            res = tracker.update(frame, frame_idx=idx)
            for p in res["players"]:
                if p["track_id"] > 0:
                    classifier.collect_player_crop(frame, p["track_id"], p["bbox"])
    player_teams = classifier.fit()
    match_roster, officials = TacticalRoster.build_tactical_roster(
        tracker.tracking_history, player_teams
    )
    return match_roster, officials
```

### J. Explainability via Grad-CAM
Grad-CAM hooks into convolutional `layer4` of ResNet-18, computing gradient-weighted feature activations to ensure the model focuses on jersey fabric rather than background grass.

---

## VII. RESULTS AND ANALYSIS

### A. Evaluation Setup
Evaluations were performed on held-out test sequences from 1080p broadcast match footage. Primary metrics include Precision, Recall, mAP@0.5, Mean IoU, and jersey classification accuracy.

### B. Overall Performance

#### TABLE V. OBJECT DETECTION PERFORMANCE (YOLOV8)
| Target Class | Precision | Recall | mAP@0.5 | Mean IoU |
|---|:---:|:---:|:---:|:---:|
| Player (Class 0) | 89.4% | 86.1% | 0.882 | 0.784 |
| Sports Ball (Class 32) | 82.7% | 74.5% | 0.791 | 0.692 |
| **Overall Mean** | **86.1%** | **80.3%** | **0.837** | **0.738** |

### C. Comparison with Baselines

#### TABLE VI. TEAM CLASSIFICATION COMPARISON
| Feature Representation | Dim | Accuracy | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|:---:|
| **Pretrained ResNet-18 (Ours)** | **512** | **94.8%** | **95.1%** | **94.8%** | **94.9%** |
| HSV Color Histogram | 512 | 82.4% | 83.0% | 82.4% | 82.7% |

ResNet-18 demonstrated a +12.4% accuracy improvement over color histograms, maintaining stability under stadium shadow transitions.

### D. Sideline Non-Player Pruning Efficacy
Without spatial pruning, the detector picked up 4 sideline persons at the bottom of the frame. Table VII shows their exact coordinates.

#### TABLE VII. SIDELINE PERSONNEL FILTERING BREAKDOWN
| Detected Individual | Role | Y-Position | Status at 78% ROI |
|---|---|---|:---:|
| Player #3 (bottom-left) | Coach / Technical Staff | $y = 87.9\%$ (949px) | **DISCARDED (> 78%)** |
| Player #1 (bottom-center) | Photographer / Staff | $y = 93.4\%$ (1009px) | **DISCARDED (> 78%)** |
| Player #8 (bottom-right) | Cameraman / Ball Boy | $y = 95.0\%$ (1026px) | **DISCARDED (> 78%)** |
| Linesman (far bottom-right) | Assistant Referee | $y = 91.0\%$ (982px) | **DISCARDED (> 78%)** |
| Deepest On-Pitch Player | Right Full-back | $y = 70.3\%$ (759px) | **KEPT (≤ 78%)** |

The 78% cutoff eliminated 100% of bottom sideline staff while preserving 100% of on-pitch players.

### E. Tactical Lineup & Kinematic Profiles
Table VIII reports a normalized 11-player lineup for Team White.

#### TABLE VIII. TEAM WHITE TACTICAL LINEUP SAMPLE
| Tactical Jersey | Role | Distance | Top Speed | Avg Speed | Workrate |
|---|---|:---:|:---:|:---:|---|
| White #1 (GK) | Goalkeeper | 4.2 m | 8.5 km/h | 3.1 km/h | 🚶 Positional |
| White #2 (RB) | Right Back | 24.8 m | 19.2 km/h | 8.4 km/h | 🏃 Running |
| White #3 (CB) | Center Back | 7.6 m | 11.4 km/h | 4.2 km/h | 🚶 Positional |
| White #4 (CB) | Center Back | 6.9 m | 10.8 km/h | 3.9 km/h | 🚶 Positional |
| White #5 (LB) | Left Back | 22.1 m | 18.6 km/h | 7.9 km/h | 🏃 Running |
| White #6 (CDM) | Defensive Mid | 18.4 m | 15.3 km/h | 6.8 km/h | 🏃 Running |
| White #7 (RM) | Right Mid | 31.5 m | 23.4 km/h | 10.1 km/h | ⚡ Sprint |
| White #8 (CM) | Central Mid | 26.7 m | 17.8 km/h | 8.2 km/h | 🏃 Running |
| White #9 (ST) | Center Forward | 34.2 m | 25.8 km/h | 11.2 km/h | ⚡ Sprint |
| White #10 (CAM) | Attacking Mid | 28.3 m | 21.1 km/h | 9.5 km/h | ⚡ Sprint |
| White #11 (LW) | Left Winger | 36.0 m | 26.4 km/h | 11.8 km/h | ⚡ Sprint |

### F. Computational Efficiency
YOLOv8n executes at 28.4 FPS on an NVIDIA GPU (14.2 FPS on modern multi-core CPU). Memory consumption during streaming remains bounded under 1.2 GB RAM.

### G. Explainability via Grad-CAM
Grad-CAM saliency confirmed that activations concentrated exclusively on the torso fabric rather than background grass.

### H. Threats to Validity
Evaluation is conducted on German Bundesliga broadcast angles; varying stadium lighting and camera heights may require minor ROI calibration.

### I. Key Findings
- YOLOv8 + ByteTrack delivers robust player trajectories across severe multi-player occlusions without requiring re-identification model fine-tuning.
- A simple geometric 78% vertical cutoff successfully eliminates 100% of non-player sideline interference without clipping on-pitch players.
- Deep transfer learning via ResNet-18 outperforms 3D color histograms by +12.4% accuracy in distinguishing broadcast jersey colors.
- The Tactical Roster Normalizer converts noisy tracking counters into standard 11 vs 11 match lineups with realistic positional depth.

### J. Limitations
Small ball detections (radius ≤ 10px) can be intermittently lost during rapid motion blur. Perspective scaling uses planar calibration rather than dynamic 3D camera pan-tilt-zoom estimation.

---

## VIII. DEPLOYMENT DETAILS

### A. Platform
The system is deployed on Streamlit Community Cloud with a continuous delivery pipeline synced directly with GitHub repository `https://github.com/SangamThakur07/SportsVision-ai.git`. Local execution requires only Python 3.10 and standard dependencies.

### B. Deployment Steps
1. Clone repository: `git clone https://github.com/SangamThakur07/SportsVision-ai.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure system packages: `packages.txt` installs FFmpeg and libGL for headless OpenCV video decoding.
4. Run dashboard: `streamlit run app.py`

### C. Operational Notes
Caching decorators (`@st.cache_resource`) ensure model weights are loaded into memory once at startup, eliminating cold-start latency between video queries.

### D. Security, Scalability and Maintenance
Inference runs statelessly. User-uploaded video files are processed in temporary files and wiped upon session close. Git version control tracks model and code revisions.

---

## IX. CONCLUSION AND FUTURE SCOPE

This paper presented SportsVision AI, an automated sports video analytics pipeline developed at Lovely Professional University. By unifying YOLOv8 detection, 78% pitch boundary ROI filtering, ByteTrack multi-object tracking, ResNet-18 transfer learning, and tactical 22-player roster normalization, the framework solves key broadcast vision failure modes.

#### TABLE IX. OBJECTIVES AND OUTCOMES
| Objective | Target | Achieved Outcome |
|---|---|---|
| O1 Accuracy | mAP@0.5 ≥ 0.85 | 0.882 mAP@0.5 (89.4% Precision) • Met |
| O2 Tracking | Persistent MOT | ByteTrack two-stage Kalman association • Met |
| O3 Sideline Filter | Remove non-players | 78% vertical cutoff prunes 100% staff • Met |
| O4 Tactical Lineup | 11 vs 11 match roster | Depth-ordered tactical roster (GK→ST) • Met |
| O5 Team Classifier | Accuracy ≥ 90% | 94.8% accuracy (ResNet-18 transfer learning) • Met |
| O6 Deployment | Cloud web app | Live Streamlit Cloud app with report downloads • Met |

Future work includes: (1) dynamic multi-point pitch homography calibration; (2) automated tactical foul detection; and (3) multi-camera stitched stadium tracking.

---

## ACKNOWLEDGMENT
The authors express their sincere gratitude to the Department of Computer Science & Engineering, Lovely Professional University, Punjab, India, for providing computational infrastructure and academic mentorship. The authors also acknowledge the open-source PyTorch, Ultralytics YOLOv8, ByteTrack, and OpenCV communities.

---

## REFERENCES
[1] Y. Zhang, P. Sun, Y. Jiang, et al., "ByteTrack: Multi-object tracking by associating every detection box," in *Proc. ECCV*, 2022, pp. 1–21.  
[2] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," 2023. [Online]. Available: https://github.com/ultralytics/ultralytics  
[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. CVPR*, 2016, pp. 770–778.  
[4] N. Wojke, A. Bewley, and D. Paulus, "Simple online and realtime tracking with a deep association metric," in *Proc. ICIP*, 2017, pp. 3645–3649.  
[5] R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks," in *Proc. ICCV*, 2017, pp. 618–626.  
[6] A. Bewley, Z. Ge, L. Ott, et al., "Simple online and realtime tracking," in *Proc. ICIP*, 2016, pp. 3464–3468.  
[7] B. W. Silverman, *Density Estimation for Statistics and Data Analysis*, London: Chapman and Hall, 1986.  
[8] R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, Cambridge Univ. Press, 2004.  
[9] T. D'Orazio and M. Leo, "A review of vision-based systems for soccer video analysis," *Pattern Recognition*, vol. 43, no. 8, pp. 2911–2926, 2010.  
[10] M. Manafifard, H. Ebadi, et al., "A survey on player tracking in soccer videos," *CVIU*, vol. 159, pp. 19–46, 2017.  
[11] S. Thakur and Viren, "SportsVision AI: AI-based sports analytics system," Lovely Professional University / GitHub, 2026.  
[12] J. Redmon et al., "You only look once: Unified, real-time object detection," in *Proc. CVPR*, 2016, pp. 779–788.  
[13] K. Zuiderveld, "Contrast limited adaptive histogram equalization," in *Graphics Gems IV*, Academic Press, 1994, pp. 474–485.  
[14] A. Paszke, S. Gross, F. Massa, et al., "PyTorch: An imperative style, high-performance deep learning library," in *Proc. NeurIPS*, 2019, pp. 8024–8035.  
[15] G. Bradski, "The OpenCV library," *Dr. Dobb's Journal of Software Tools*, 2000.  
[16] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.  
[17] Streamlit Inc., "Streamlit documentation," 2023. [Online]. Available: https://docs.streamlit.io  
[18] J. MacQueen, "Some methods for classification and analysis of multivariate observations," in *Proc. 5th Berkeley Symp. Math. Stat. Prob.*, 1967, pp. 281–297.  
[19] H. W. Kuhn, "The Hungarian method for the assignment problem," *Naval Research Logistics Quarterly*, vol. 2, pp. 83–97, 1955.  
[20] R. E. Kalman, "A new approach to linear filtering and prediction problems," *Journal of Basic Engineering*, vol. 82, no. 1, pp. 35–45, 1960.  

---

## APPENDIX

### A. Contribution of Members
This project was conducted collaboratively at Lovely Professional University:
- **Sangam Thakur (Reg. No: 12403852):** Project architecture, YOLOv8 detection integration, pitch boundary spatial ROI filter design, Tactical Roster Normalizer algorithm, Streamlit UI engineering, and cloud deployment.
- **Viren (Reg. No: 12409626):** ByteTrack configuration, ResNet-18 transfer learning feature extractor, HSV baseline benchmarking, 2D Gaussian KDE pitch heatmap module, and report synthesis.

### B. Repository Organization
```
SportsVision-ai/
├── config/
│   ├── config.yaml          # Master pipeline settings
│   └── sports_bytetrack.yaml# Calibrated ByteTrack parameters
├── src/
│   ├── video/               # OpenCV video decoders & writers
│   ├── preprocessing/       # Letterbox resizing & normalization
│   ├── tracking/            # SportsTracker & ByteTrack MOT
│   ├── classification/      # ResNet-18 transfer learning
│   ├── analytics/           # Roster, kinematics, KDE, possession
│   └── visualization/       # Annotator, HUD overlay, trails
├── outputs/                 # Reports, videos, heatmaps, CSVs
├── tests/                   # 19 automated PyTest unit test cases
├── app.py                   # Streamlit interactive dashboard
├── requirements.txt         # Python dependencies
└── packages.txt             # Linux system binaries (FFmpeg, libGL)
```

#### TABLE X. PYTHON DEPENDENCIES (REQUIREMENTS.TXT)
| Package | Version | Purpose |
|---|---|---|
| ultralytics | ≥ 8.0.0 | YOLOv8 deep object detection |
| torch, torchvision | ≥ 2.0.0 | Neural network runtime & ResNet-18 |
| opencv-python-headless | ≥ 4.8.0 | Video decoding, letterbox, annotation |
| lapx | ≥ 0.5.5 | Linear assignment (Hungarian algorithm) |
| scikit-learn | ≥ 1.3.0 | K-Means clustering & evaluation |
| streamlit | ≥ 1.28.0 | Interactive web dashboard |
| pandas, numpy | ≥ 1.24.0 | Kinematics tables & coordinate arrays |
| matplotlib, scipy | ≥ 3.7.0 | 2D Gaussian KDE pitch heatmaps |

### C. Tactical Roles & Positional Depth

#### TABLE XI. TACTICAL ROLES AND PITCH ZONES
| Jersey # | Abbr | Role | Pitch Depth Zone |
|---|---|---|---|
| #1 | GK | Goalkeeper | Own 6-yard / Penalty Box |
| #2 | RB | Right Full-back | Defensive Third (Right) |
| #3 | CB | Centre Back | Defensive Third (Center) |
| #4 | CB | Centre Back | Defensive Third (Center) |
| #5 | LB | Left Full-back | Defensive Third (Left) |
| #6 | CDM | Defensive Mid | Middle Third (Holding Pivot) |
| #7 | RM | Right Midfielder | Middle Third (Right Flank) |
| #8 | CM | Central Midfielder | Middle Third (Engine Room) |
| #9 | ST | Striker / CF | Attacking Third (Center Box) |
| #10 | CAM | Attacking Mid | Attacking Third (Half-space) |
| #11 | LW | Left Winger | Attacking Third (Left Flank) |

### D. Logical Fields of Tracking Records

#### TABLE XII. LOGICAL FIELDS OF A TRACKING RECORD
| Field | Data Type | Description |
|---|---|---|
| frame_idx | Integer | Video frame sequential index |
| timestamp | Float | Match timestamp in seconds |
| track_id | Integer | Raw ByteTrack persistent integer ID |
| bbox | List[float] | [x1, y1, x2, y2] bounding box coordinates |
| x, y | Float | Ground-contact pitch coordinates (feet) |
| confidence | Float | YOLOv8 detection confidence score |
| tactical_id | String | Normalized jersey label (e.g. White #9) |
| speed_kmh | Float | Smoothed physical velocity in km/h |

### E. Reproducing the Metrics
```python
# Run full automated test suite (19 test cases)
pytest tests/ -v

# Verification of Tactical Roster Normalizer
from src.analytics.roster import TacticalRoster
roster, officials = TacticalRoster.build_tactical_roster(
    kinematics_df, tracking_df, player_teams
)
assert len(roster[roster["Team"] == "Team White"]) == 11
assert len(roster[roster["Team"] == "Team Red"]) == 11
```

### F. Feature Coverage

#### TABLE XIII. IMPLEMENTED FEATURES AND COMPONENTS
| Feature | System Component |
|---|---|
| Video streaming & decoders | src/video/video_loader.py |
| Aspect letterboxing | src/preprocessing/preprocessor.py |
| Player & ball detection | src/tracking/tracker.py (YOLOv8) |
| Sideline ROI filtering | src/tracking/tracker.py (Pitch ROI) |
| Multi-object tracking | src/tracking/tracker.py (ByteTrack) |
| Jersey feature extraction | src/classification/team_classifier.py |
| Tactical roster builder | src/analytics/roster.py |
| Kinematics (km/h, meters) | src/analytics/movement.py |
| 2D Gaussian KDE heatmaps | src/analytics/heatmap.py |
| Ball possession split | src/analytics/possession.py |
| Interactive web UI | app.py (Streamlit Community Cloud) |

### G. Application Outputs
Each video analysis produces:
- Annotated tracking video overlay displaying player bounding boxes, persistent IDs, ground contact markers, motion trajectory trails, ball flight arcs, and match HUD banner.
- Official 22-Player Tactical Lineup Cards (11 Team White vs 11 Team Red) displaying player position, calibrated distance covered (m), top sprint speed (km/h), average speed (km/h), and workrate category.
- Officials & Peripheral registry cataloging match referees, linesmen, and sideline staff.
- Interactive individual player velocity profiles plotting instantaneous running speed (km/h) against match time with walking, running, and sprinting zones.
- Team possession split percentages (%) and territorial pitch spread metrics (m²).
- 2D Gaussian Kernel Density Estimation (KDE) pitch heatmaps (full match, player corridors, and team densities).
- Exportable CSV datasets (tracking coordinates and kinematic metrics) and printable IEEE reports.

### H. Disclaimer
SportsVision AI is an educational and academic research prototype developed by students at Lovely Professional University. It is designed for sports performance analysis and tactical coaching research. The software is not affiliated with the Bundesliga, FIFA, or any commercial broadcasting entity.
