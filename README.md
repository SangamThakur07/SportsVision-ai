# SportsVision AI: AI-Based Sports Analytics System
### University Computer Vision & Deep Learning Course Project

A high-performance, modular computer vision system that analyzes football match videos to detect and track players and the ball, extract spatial movement metrics, compute 2D pitch heatmaps, and estimate ball possession.

---

## 1. System Architecture & Pipeline

```
Football Video File
        │
        ▼
[1] OpenCV Frame Stream (Memory-efficient frame generator)
        │
        ▼
[2] Image Preprocessing (Letterbox resizing, color conversion BGR->RGB, normalization)
        │
        ▼
[3] YOLOv8 Object Detection (Classes: 0: Player, 32: Ball, bounding box [x1, y1, x2, y2], confidence, centroids)
        │
        ▼
[4] Multi-Object Tracking (ByteTrack associating persistent player IDs across frames)
        │
        ▼
[5] Jersey Classification (Transfer Learning with Pretrained ResNet on upper-torso crops)
        │
        ▼
[6] Kinematics & Spatial Analytics
    ├── Pixel displacement & smoothed velocity (v = Δd / Δt)
    ├── 2D Gaussian Kernel Density Estimation (KDE) heatmaps
    └── Nearest-neighbor proximity-based estimated ball possession
        │
        ▼
[7] Professional Streamlit Dashboard & Video Visualizer
```

---

## 2. University Syllabus & Viva Examination Guide

This project directly maps to the core units of the Computer Vision and Deep Learning curriculum:

### Unit I: Image Processing Fundamentals
- **Color Spaces**: OpenCV natively decodes video in **BGR** format. Convolutional backbones (YOLO, ResNet) expect **RGB** normalized inputs. The preprocessor handles BGR $\to$ RGB conversion. For jersey analysis, **HSV** (Hue, Saturation, Value) separates chromatic color from lighting variations.
- **Aspect-Ratio Preserving Resize**: Direct resizing distorts aspect ratios, distorting player shapes. Letterboxing scales the image uniformly and pads the remaining dimensions with neutral gray (114).
- **Pixel Normalization**: Raw pixel values in $[0, 255]$ are scaled to $[0.0, 1.0]$ and standardized using ImageNet dataset channel means and standard deviations:
  $$\hat{I}_{c} = \frac{I_{c} / 255 - \mu_c}{\sigma_c}$$
  This prevents internal covariate shift and stabilizes neural network activations.

### Unit II: CNNs & Deep Architectures
- **Convolution Operations**: Sliding learnable kernel matrices across feature maps to detect spatial hierarchies (edges $\to$ textures $\to$ jersey shapes).
- **Residual Networks (ResNet)**: Deep CNNs suffer from the vanishing gradient problem. ResNet introduces identity shortcut connections:
  $$y = \mathcal{F}(x, \{W_i\}) + x$$
  allowing gradients to flow directly through the skip connections during backpropagation.

### Unit III: Transfer Learning & Fine-Tuning
- **Transfer Learning**: Instead of training millions of parameters from scratch on limited sports data, we transfer visual representations learned by ResNet on 1.4M ImageNet images.
- **Feature Extraction vs. Fine-Tuning**: In feature extraction, convolutional layers are frozen ($\nabla W = 0$), and only the final classification head is updated. In fine-tuning, the upper residual stages are unfrozen with a small learning rate.

### Unit IV: Object Detection & YOLO
- **Single-Stage Detection**: YOLO formulates detection as a direct regression problem, predicting bounding box coordinates and class probabilities in a single forward pass, achieving real-time FPS.
- **Bounding Boxes & Coordinates**: Represented as $[x_1, y_1, x_2, y_2]$ (top-left, bottom-right). Centroids are calculated as:
  $$c_x = \frac{x_1 + x_2}{2}, \quad c_y = \frac{y_1 + y_2}{2}$$
  Ground-contact pitch coordinates use the bottom center: $(c_x, y_2)$ representing player feet.
- **Intersection over Union (IoU)**:
  $$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{|B_1 \cap B_2|}{|B_1 \cup B_2|}$$
- **Non-Maximum Suppression (NMS)**: Suppresses redundant bounding boxes that overlap above an IoU threshold ($\approx 0.45$) while retaining the box with the highest confidence score.
- **Mean Average Precision (mAP)**: Evaluates precision-recall trade-offs across different IoU thresholds ($0.5$ and $0.5:0.95$).

### Unit V: Multi-Object Tracking (ByteTrack)
- **Tracking vs. Detection**: Detectors treat frames independently. Trackers maintain persistent identities across temporal occlusions and motion blur.
- **ByteTrack Innovation**: Traditional trackers discard low-confidence detections ($conf < 0.5$). ByteTrack associates both high-confidence and low-confidence detections using Kalman filtering and Hungarian bipartite matching, preserving tracks when players are partially occluded.

### Spatial Kinematics & Analytics
- **Pixel Displacement**:
  $$\Delta d = \sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2} \quad \text{[pixels]}$$
- **Estimated Speed**:
  $$v = \frac{\Delta d}{\Delta t} = \frac{\Delta d \cdot \text{FPS}}{\Delta \text{frames}} \quad \text{[pixels/second]}$$
  *Note: Pixel velocity differs from physical meters/sec unless calibrated via homography.*
- **Estimated Ball Possession**:
  Measures the Euclidean distance between the ball center $(x_b, y_b)$ and each player's feet $(x_p, y_p)$. The closest player within threshold $T_{possession}$ is awarded instantaneous frame possession. Possession percentages are aggregated across match frames.

---

## 3. Installation & Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/sportsvision-ai.git
cd sportsvision-ai

# Install dependencies
pip install -r requirements.txt

# Run Interactive Streamlit Analytics Dashboard
streamlit run app.py

# Or Run Standalone CLI Pipeline (Tracking + Analytics)
python run_pipeline.py --input data/sample/synthetic_match.mp4 --output outputs/videos/output_analytics.mp4 --mode tracking

# Run Full Test Suite (14 Tests)
pytest tests/ -v
```

---

## 4. Streamlit Dashboard Features

The web interface is divided into 8 distinct analytical views:
1. **Home**: Architecture breakdown, tech stack summary, and preview.
2. **Video Analysis**: Video uploader with live processing progress bar and FPS counter.
3. **Tracking & Detection**: Frame inspection displaying persistent IDs, motion trails, and ball flight paths.
4. **Player Analytics**: Interactive player kinematics table with speed, distance covered, and tracking duration.
5. **Team & Possession**: Possession share breakdown (cards and bar charts) and team centroid information.
6. **Movement Heatmaps**: 2D Gaussian density estimation overlaid on a calibrated football pitch.
7. **Model Evaluation**: Precision, Recall, mAP, IoU, and ResNet-18 vs HSV Color Histogram benchmark.
8. **Viva & Methodology**: Academic viva revision guide explaining core Computer Vision concepts.

---

## 5. Configuration

All parameters can be tuned in `config/config.yaml`:
- Detection confidence & IoU thresholds
- Target classes (`player`, `ball`)
- Visual styles (colors, font size, HUD)
- Hardware device (`cuda` or `cpu`)
