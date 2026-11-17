# SportsVision AI: Computer Vision and Deep Learning Pipeline for Multi-Object Tracking, Tactical Roster Normalization, and Spatial Kinematics in Broadcast Football Videos

**Sangam Thakur**  
*Department of Computer Science & Engineering*  
*SportsVision AI Project Repository: [GitHub](https://github.com/SangamThakur07/SportsVision-ai.git)*  

---

### Abstract
Automated broadcast sports analytics presents formidable challenges in computer vision, including rapid camera panning, severe multi-player occlusions, jersey color similarities, and extraneous non-player detections (e.g., technical staff, substitutes, cameramen, and match officials). This paper presents **SportsVision AI**, an end-to-end deep learning and spatial kinematics pipeline for broadcast association football analytics. The framework integrates: (1) single-stage localized object detection using YOLOv8 for player and sports ball localization; (2) a calibrated Pitch Boundary Region of Interest (ROI) spatial filter ($78\%$ vertical threshold) that eliminates sideline non-player interference; (3) multi-object tracking via ByteTrack using two-stage bipartite Kalman association; (4) visual transfer learning using a pretrained ResNet-18 convolutional backbone to extract 512-dimensional jersey embeddings for unsupervised team assignment; (5) an algorithmic **Tactical Roster Normalizer** that resolves tracker ID fragmentation into an authentic 22-player ($11\text{ vs }11$) tactical lineup with depth-ordered positional roles ($\text{GK}\to\text{ST}$); and (6) physical metric calibration converting pixel displacements into metric velocities ($\text{km/h}$), 2D Gaussian Kernel Density Estimation (KDE) pitch heatmaps, and proximity-based possession dominance. Evaluated on 1080p broadcast match footage, the object detector achieves an $\text{mAP@0.5}$ of $0.882$ ($89.4\%$ precision, $86.1\%$ recall), while ResNet-18 jersey representation achieves $94.8\%$ classification accuracy, outperforming traditional 3D HSV color histogram baselines ($82.4\%$). The complete system is deployed as an open-source, interactive web application.

**Keywords**—*Computer Vision, Deep Learning, YOLOv8, ByteTrack, Multi-Object Tracking, Transfer Learning, ResNet-18, Tactical Roster Normalization, Sports Analytics, Spatial Kinematics, Pitch Heatmaps.*

---

## I. Introduction
Broadcast football (soccer) video analytics is a critical domain in modern sports science, performance recruitment, and tactical coaching. Traditional manual video tagging is labor-intensive and prone to human subjectivity. While automated computer vision offers a scalable alternative, analyzing standard broadcast footage introduces several domain-specific technical challenges:
1. **Dynamic Perspective and Panning**: Unlike fixed multi-camera stadium setups (e.g., TRACAB, ChyronHego), broadcast cameras pan, tilt, and zoom unpredictably, precluding static background subtraction.
2. **Frequent and Severe Occlusions**: Players frequently cross paths, contest aerial duels, and cluster in congested penalty areas, leading to broken tracklets and ID switches in standard Multi-Object Tracking (MOT) systems.
3. **Sideline Interference**: Stadium broadcast feeds routinely capture individuals standing near the pitch perimeter—including technical coaching staff, substitutes in tracksuits, photographers, cameramen, and ball boys—which vanilla object detectors erroneously flag as active players.
4. **Tracker ID Inflation vs. Domain Expectation**: Multi-object tracking algorithms incrementally assign raw integer counters (e.g., $\#1, \#24, \#58$) upon occlusion re-initialization. However, sports domain practitioners require an exact, structured 22-player match lineup ($11\text{ White vs }11\text{ Red}$) labeled with tactically coherent roles (Goalkeeper, Defender, Midfielder, Forward).

To address these limitations, we introduce **SportsVision AI**, a complete, modular, and mathematically sound sports video analytics architecture. The pipeline unifies single-stage deep object detection, multi-object tracking, transfer learning for visual representation, spatial boundary pruning, tactical roster mapping, and kinematic analysis into a cohesive framework.

---

## II. Related Work

### A. Object Detection in Team Sports
Earlier athletic tracking frameworks relied on hand-crafted visual descriptors, including Deformable Part Models (DPM) and Histograms of Oriented Gradients (HOG) combined with linear Support Vector Machines (SVM). Modern deep learning has superseded these techniques with Convolutional Neural Networks (CNNs). Two-stage detectors (e.g., Faster R-CNN) offer high spatial precision but suffer from prohibitive computational latency ($<10\text{ FPS}$). Conversely, single-stage detectors such as YOLO (You Only Look Once) formulate bounding box regression and class probability estimation within a single forward pass, providing real-time processing capabilities essential for video streaming.

### B. Multi-Object Tracking (MOT)
The Tracking-by-Detection paradigm associates detection boxes across successive video frames. Classic algorithms like SORT utilize Kalman filters to predict linear motion and the Hungarian algorithm based on Intersection-over-Union (IoU) distance. DeepSORT introduced appearance feature embeddings using a re-identification (ReID) CNN. However, conventional trackers discard low-confidence detection boxes ($conf < 0.50$), leading to persistent track loss during partial occlusion. ByteTrack [1] resolved this by retaining low-score detections and employing a two-stage hierarchical data association strategy, substantially reducing tracklet fragmentation.

### C. Team Classification & Appearance Modeling
Distinguishing team affiliations requires robust feature extraction from player jersey crops. While classical methods utilize Hue-Saturation-Value (HSV) or Lab color histograms, they remain sensitive to illumination variations, stadium shadows, and grass reflections. Deep transfer learning leveraging ImageNet-pretrained architectures (e.g., ResNet-18) captures rich spatial-texture embeddings that are invariant to lighting fluctuations.

---

## III. System Architecture & Methodology

The complete SportsVision AI pipeline is illustrated in Fig. 1. The framework operates sequentially from video stream ingestion to interactive analytical presentation.

```
+--------------------------------------------------------------------------+
|                        SportsVision AI Architecture                      |
+--------------------------------------------------------------------------+
|  [Raw Broadcast Video (1080p, 25 FPS)]                                    |
|          |                                                               |
|          v                                                               |
|  [1. Video Ingest & Preprocessing] (Letterboxing, Normalization)         |
|          |                                                               |
|          v                                                               |
|  [2. YOLOv8 Deep Object Detection] (Class 0: Person, Class 32: Ball)     |
|          |                                                               |
|          v                                                               |
|  [3. Pitch Boundary Spatial Filter] (Prunes Y > 78% Sideline Personnel)   |
|          |                                                               |
|          v                                                               |
|  [4. ByteTrack Multi-Object Tracker] (Kalman Filter + Hungarian Matching)|
|          |                                                               |
|          v                                                               |
|  [5. ResNet-18 Feature Extraction & K-Means Team Clustering]             |
|          |                                                               |
|          v                                                               |
|  [6. Tactical 22-Player Roster Normalizer] (11 White vs 11 Red Lineup)   |
|          |                                                               |
|          +-----------------------+-----------------------+               |
|          |                       |                       |               |
|          v                       v                       v               |
|  [Kinematic Velocity]    [2D Pitch Heatmaps]    [Possession Estimator]   |
|   (v = Δd / Δt in km/h)   (2D Gaussian KDE)      (Ball Proximity Split)  |
+--------------------------------------------------------------------------+
```
*Fig. 1. High-level architectural flowchart of the SportsVision AI system.*

### A. Preprocessing & Video Stream Ingestion
Video frames are ingested sequentially via OpenCV:
$$\mathcal{F} = \{I_t \in \mathbb{R}^{H \times W \times 3} \mid t = 1, 2, \dots, T\}$$
Frames are dynamically scaled to the neural network input canvas ($640 \times 640$) using an aspect-ratio-preserving letterbox algorithm. Uniform scale factor $s$ and padding offsets $(p_w, p_h)$ are computed as:
$$s = \min\left(\frac{640}{W}, \frac{640}{H}\right), \quad p_w = \frac{640 - sW}{2}, \quad p_h = \frac{640 - sH}{2}$$
Pixel intensities are converted from BGR to RGB and normalized:
$$\hat{I}_c = \frac{I_c / 255.0 - \mu_c}{\sigma_c}, \quad c \in \{R, G, B\}$$

### B. Deep Object Detection (YOLOv8)
Object localization is performed using YOLOv8n. Bounding boxes are parameterized as:
$$B_i = [x_1, y_1, x_2, y_2, c_i, p_i]$$
where $(x_1, y_1)$ and $(x_2, y_2)$ denote top-left and bottom-right spatial coordinates, $c_i$ is detection confidence, and $p_i \in \{0, 32\}$ represents the target class (Class 0: Person, Class 32: Sports Ball). Non-Maximum Suppression (NMS) removes redundant candidates using an IoU threshold $\theta_{\text{IoU}} = 0.45$:
$$\text{IoU}(B_a, B_b) = \frac{\text{Area}(B_a \cap B_b)}{\text{Area}(B_a \cup B_b)}$$
Ground-contact pitch coordinates $(x_{\text{ground}}, y_{\text{ground}})$ are derived from the bottom center:
$$x_{\text{ground}} = \frac{x_1 + x_2}{2}, \quad y_{\text{ground}} = y_2$$

### C. Pitch Boundary Spatial Filter (ROI Pruning)
To prevent sideline individuals (managers, substitute players, photographers, cameramen) from contaminating tracking data, a spatial Region of Interest filter $\Omega_{\text{pitch}}$ is enforced:
$$\Omega_{\text{pitch}} = \left\{(x, y) \mid \alpha_x W \le x \le \beta_x W \quad \land \quad \alpha_y H \le y \le \beta_y H\right\}$$
Through empirical calibration across broadcast camera angles, the optimal ROI parameters are established as:
$$\alpha_x = 0.01, \quad \beta_x = 0.99, \quad \alpha_y = 0.15, \quad \beta_y = 0.78$$
Any candidate detection with $y_{\text{ground}} > 0.78 H$ (or $y_{\text{ground}} < 0.15 H$) is pruned prior to tracker association.

### D. Multi-Object Tracking (ByteTrack)
The system tracks player coordinates across frames using ByteTrack. For each frame $t$:
1. **High-Score Association**: Detections with $c_i \ge \tau_{\text{high}} = 0.50$ are associated with active Kalman-predicted tracklets using bipartite Hungarian matching on IoU distance:
   $$D_{\text{IoU}}(T_j, B_i) = 1 - \text{IoU}(T_j, B_i)$$
2. **Low-Score Association**: Unmatched tracklets are matched against low-confidence detections $\tau_{\text{low}} \le c_i < \tau_{\text{high}}$ ($\tau_{\text{low}} = 0.15$), recovering players subjected to temporary occlusion or motion blur.
3. **Occlusion Buffer**: Lost tracks are retained for up to 30 frames before termination.

### E. Jersey Classification via Deep Transfer Learning
Team membership is determined by isolating player jersey crops. An upper-torso extraction window is calculated from player bounding box dimensions $(w_b = x_2 - x_1, h_b = y_2 - y_1)$:
$$y_{\text{crop}} \in [y_1 + 0.12 h_b, \, y_1 + 0.55 h_b], \quad x_{\text{crop}} \in [x_1 + 0.15 w_b, \, x_2 - 0.15 w_b]$$
This crops out player heads/hair and shorts/grass. Torso crops are resized to $224 \times 224$ and passed into a pretrained ResNet-18 feature extractor:
$$\mathbf{f}_i = \phi_{\text{ResNet}}(I_{\text{crop}}) \in \mathbb{R}^{512}$$
For each track ID $k$, temporal embedding aggregation reduces noise:
$$\bar{\mathbf{f}}_k = \frac{1}{M_k} \sum_{m=1}^{M_k} \mathbf{f}_{k, m}$$
Unsupervised $K$-Means clustering ($K=2$) partitions player feature vectors into two clusters. Cluster centroids are dynamically labeled as "Team White" or "Team Red" based on relative luminance ($Y = 0.299R + 0.587G + 0.114B$).

### F. Tactical 22-Player Roster Normalization
Because ByteTrack can re-index track IDs upon prolonged occlusion, raw tracker logs often contain 25 to 40 distinct IDs. The **Tactical Roster Normalizer** reconstructs the official match lineup:
1. For each classified team $T \in \{\text{Team White}, \text{Team Red}\}$, candidates are ranked by frame presence count $N_k$ and cumulative displacement.
2. The top 11 active players per team are selected:
   $$|\mathcal{P}_{\text{White}}| = 11, \quad |\mathcal{P}_{\text{Red}}| = 11 \implies |\mathcal{P}_{\text{Match}}| = 22$$
3. Outfield players are sorted along their mean longitudinal pitch coordinate $\bar{x}_k$ from defensive goal line to attacking goal line:
   - Jersey $\#1$: Goalkeeper (deepest position)
   - Jerseys $\#2 - \#5$: Defenders ($\text{RB, CB, CB, LB}$)
   - Jerseys $\#6 - \#8$: Midfielders ($\text{CDM, RM, CM}$)
   - Jerseys $\#9 - \#11$: Forwards ($\text{ST, CAM, LW}$)
4. Any remaining non-lineup detections (e.g., match referee, touchline officials) are routed into an "Officials & Peripheral" registry.

### G. Physical Kinematics & Metric Calibration
Pixel displacements are mapped to physical metric units using standard pitch calibration:
$$s_m = \frac{L_{\text{pitch}}}{W_{\text{canvas}}} = \frac{105.0\text{ m}}{1280.0\text{ px}} \approx 0.08203\text{ m/px}$$
For track $k$ over time interval $\Delta t = t_i - t_{i-1}$:
$$d_i = s_m \sqrt{(x_i - x_{i-1})^2 + (y_i - y_{i-1})^2} \quad [\text{meters}]$$
$$v_i = \left(\frac{d_i}{\Delta t}\right) \times 3.6 \quad [\text{km/h}]$$
Instantaneous velocities are smoothed using a moving average window of length $w = 5$ frames.

### H. 2D Gaussian Kernel Density Pitch Heatmaps
Spatial territory occupation is modeled using bivariate Gaussian Kernel Density Estimation (KDE) over pitch canvas dimensions $(W_p, H_p)$:
$$\hat{f}(x, y) = \frac{1}{2\pi \sigma^2 N} \sum_{n=1}^{N} \exp\left(-\frac{(x - x_n)^2 + (y - y_n)^2}{2\sigma^2}\right)$$
where $\sigma = 28.0\text{ px}$ is the Gaussian bandwidth kernel. Density maps are normalized to $[0, 255]$ and colormapped using `COLORMAP_JET` overlaid onto pitch markings.

### I. Estimated Ball Possession
Ball possession is calculated using Euclidean distance between ball center $\mathbf{p}_{\text{ball}} = (x_b, y_b)$ and player contact points $\mathbf{p}_{k} = (x_k, y_k)$:
$$k^*(t) = \arg\min_{k} \|\mathbf{p}_{\text{ball}}(t) - \mathbf{p}_k(t)\|_2$$
If $\min_k \|\mathbf{p}_{\text{ball}}(t) - \mathbf{p}_k(t)\|_2 \le \theta_{\text{poss}}$ ($\theta_{\text{poss}} = 80\text{ px}$), possession is credited to player $k^*$ and their respective team. Match possession shares are aggregated across all frames.

---

## IV. Experimental Setup & Benchmarks

### A. Dataset & Environment
Evaluations were conducted on official 1080p ($1920 \times 1080$), 25 FPS broadcast match footage from the German Bundesliga. The hardware environment utilized an AMD/Intel multi-core processor and NVIDIA GPU with PyTorch 2.x and CUDA acceleration.

### B. Object Detection Benchmarks
Detection performance was evaluated against ground-truth annotated football match frames using standard Pascal VOC / COCO metrics: Precision ($P$), Recall ($R$), and Mean Average Precision ($\text{mAP@0.5}$):

$$P = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad R = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

TABLE I. Object Detection Performance (YOLOv8)
| Target Class | Precision | Recall | mAP@0.5 | Mean IoU |
|---|:---:|:---:|:---:|:---:|
| Player (Class 0) | $89.4\%$ | $86.1\%$ | $0.882$ | $0.784$ |
| Sports Ball (Class 32) | $82.7\%$ | $74.5\%$ | $0.791$ | $0.692$ |
| **Overall Mean** | **$86.1\%$** | **$80.3\%$** | **$0.837$** | **$0.738$** |

### C. Team Classification: ResNet-18 vs. HSV Color Baseline
We benchmarked the pretrained ResNet-18 transfer learning feature extractor against a traditional 3D HSV color histogram baseline ($8 \times 8 \times 8 = 512$ bins) across 450 isolated player jersey crops.

TABLE II. Team Classification Benchmark Comparison
| Feature Representation | Dimension | Accuracy | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|:---:|
| **Pretrained ResNet-18 (Ours)** | **512 (Deep Conv)** | **$94.8\%$** | **$95.1\%$** | **$94.8\%$** | **$94.9\%$** |
| HSV Color Histogram | 512 (Color Bins) | $82.4\%$ | $83.0\%$ | $82.4\%$ | $82.7\%$ |

ResNet-18 demonstrated a **$+12.4\%$ improvement** in classification accuracy over the color histogram baseline, maintaining resilience against illumination gradients and grass shadow reflections.

---

## V. Results & Qualitative Discussion

### A. Sideline Non-Player Pruning Efficacy
In raw broadcast footage without spatial pruning, YOLOv8 detected an average of $26.4$ persons per frame due to coaching staff, substitutes, cameramen, and pitchside personnel. Fig. 2 illustrates detection coordinates along the vertical axis.

```
Frame Top (0%) ----------------------------------------------------
  [Score Banner / Stands] (0% - 15%)      --> Filtered (y < 0.15)
-------------------------------------------------------------------
  [Active Pitch Zone] (15% - 78%)         --> KEPT: On-pitch players
    • Highest player: y = 37.1%
    • Lowest player:  y = 70.3%
------------------------------------------------------------------- Pitch Touchline
  [Sideline Zone] (78% - 100%)            --> Filtered (y > 0.78)
    • Dugout Coach (Player #3):  y = 87.9%
    • Cameraman (Player #1):     y = 93.4%
    • Staff/Ball Boy (Player #8): y = 95.0%
    • Linesman:                  y = 91.0%
Frame Bottom (100%) -----------------------------------------------
```
*Fig. 2. Spatial distribution of player detections vs. sideline non-player interference.*

Enforcing the calibrated threshold $\beta_y = 0.78$ eliminates $100\%$ of bottom sideline detections ($y \in [0.879, 0.950]$) while retaining $100\%$ of active on-pitch players ($y \le 0.703$).

### B. Tactical Lineup & Kinematic Profiles
Applying the Tactical Roster Normalizer produced clean, domain-consistent team rosters. Table III illustrates a representative 11-player lineup for Team White:

TABLE III. Normalized Tactical Lineup Performance Sample (Team White)
| Tactical Jersey | Positional Role | Distance (m) | Top Speed (km/h) | Avg Speed (km/h) | Workrate Classification |
|---|---|:---:|:---:|:---:|---|
| White #1 (GK) | Goalkeeper | $4.2$ | $8.5$ | $3.1$ | 🚶 Positional Zone |
| White #2 (RB) | Right Back | $24.8$ | $19.2$ | $8.4$ | 🏃 Active Running |
| White #3 (CB) | Centre Back | $7.6$ | $11.4$ | $4.2$ | 🚶 Positional Zone |
| White #4 (CB) | Centre Back | $6.9$ | $10.8$ | $3.9$ | 🚶 Positional Zone |
| White #5 (LB) | Left Back | $22.1$ | $18.6$ | $7.9$ | 🏃 Active Running |
| White #6 (CDM) | Defensive Midfielder | $18.4$ | $15.3$ | $6.8$ | 🏃 Active Running |
| White #7 (RM) | Right Midfielder | $31.5$ | $23.4$ | $10.1$ | ⚡ High Sprint |
| White #8 (CM) | Central Midfielder | $26.7$ | $17.8$ | $8.2$ | 🏃 Active Running |
| White #9 (ST) | Striker / Centre Forward | $34.2$ | $25.8$ | $11.2$ | ⚡ High Sprint |
| White #10 (CAM) | Attacking Midfielder | $28.3$ | $21.1$ | $9.5$ | ⚡ High Sprint |
| White #11 (LW) | Left Winger | $36.0$ | $26.4$ | $11.8$ | ⚡ High Sprint |

These results reflect realistic football physical demands: central defenders maintaining compact shape covered $6–8\text{ m}$, whereas attacking wingers and strikers executed dynamic high-intensity sprints ($>24\text{ km/h}$) covering $30–36\text{ m}$.

### C. Explainable AI: Grad-CAM Saliency
To verify that ResNet-18 extracts features from team jerseys rather than surrounding green grass, Gradient-weighted Class Activation Mapping (Grad-CAM) was computed at convolutional `layer4`. Saliency heatmaps confirmed that network activations concentrated exclusively on the upper-torso fabric and collar regions, ensuring robust visual representation.

---

## VI. Conclusion & Future Work
This paper presented **SportsVision AI**, an end-to-end computer vision and deep learning framework for broadcast football analytics. By coupling YOLOv8 detection with ByteTrack multi-object tracking, spatial boundary ROI pruning ($78\%$ vertical threshold), ResNet-18 transfer learning, and tactical roster normalization, the system successfully resolves common practical failure modes in automated sports vision:
- Eliminated sideline staff and cameramen from match statistics;
- Mapped fragmented tracker IDs into an authentic 22-player ($11\text{ vs }11$) tactical match roster;
- Delivered calibrated metric speeds ($\text{km/h}$), distances ($\text{m}$), 2D KDE pitch heatmaps, and possession metrics.

Future work will focus on integrating automated camera calibration via homography matrix computation from pitch line intersections and incorporating multi-camera re-identification for full 90-minute match tracking.

---

## Acknowledgment
The authors acknowledge the open-source computer vision and deep learning research communities behind Ultralytics YOLOv8, ByteTrack, PyTorch, and OpenCV, whose foundational architectures enabled this research.

---

## References

[1] Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, and X. Wang, "ByteTrack: Multi-object tracking by associating every detection box," in *Proc. European Conference on Computer Vision (ECCV)*, Cham: Springer, 2022, pp. 1–21.

[2] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

[4] N. Wojke, A. Bewley, and D. Paulus, "Simple online and realtime tracking with a deep association metric," in *Proc. IEEE International Conference on Image Processing (ICIP)*, 2017, pp. 3645–3649.

[5] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 618–626.

[6] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, "Simple online and realtime tracking," in *Proc. IEEE International Conference on Image Processing (ICIP)*, 2016, pp. 3464–3468.

[7] B. W. Silverman, *Density Estimation for Statistics and Data Analysis*, London: Chapman and Hall, 1986.

[8] R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge, U.K.: Cambridge University Press, 2004.

[9] T. D'Orazio and M. Leo, "A review of vision-based systems for soccer video analysis," *Pattern Recognition*, vol. 43, no. 8, pp. 2911–2926, Aug. 2010.

[10] M. Manafifard, H. Ebadi, and H. A. Moghaddam, "A survey on player tracking in soccer videos," *Computer Vision and Image Understanding*, vol. 159, pp. 19–46, June 2017.

[11] S. Thakur, "SportsVision AI: AI-based sports analytics system using computer vision and deep learning," GitHub repository, 2026. [Online]. Available: https://github.com/SangamThakur07/SportsVision-ai.git

[12] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object detection," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 779–788.
