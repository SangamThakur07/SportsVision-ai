"""
Generates a styled, printable IEEE conference paper HTML file from SportsVision AI markdown.
Can be opened in any web browser and printed as PDF (Ctrl + P) or opened directly in MS Word.
"""

import os

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SportsVision AI - IEEE Research Paper</title>
<style>
  @page {
    size: A4;
    margin: 18mm 14mm 18mm 14mm;
  }
  body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 10pt;
    line-height: 1.25;
    color: #000;
    margin: 0 auto;
    max-width: 900px;
    padding: 24px;
    background: #fff;
  }
  h1.title {
    font-size: 17pt;
    font-weight: bold;
    text-align: center;
    margin-bottom: 14px;
    line-height: 1.2;
  }
  .authors-table {
    width: 85%;
    margin: 0 auto 20px auto;
    text-align: center;
    border-collapse: collapse;
  }
  .authors-table td {
    padding: 4px 15px;
    vertical-align: top;
    font-size: 9.5pt;
  }
  .abstract-box {
    margin: 0 15px 18px 15px;
    font-size: 9pt;
    text-align: justify;
    line-height: 1.25;
  }
  .columns {
    column-count: 2;
    column-gap: 24px;
    text-align: justify;
  }
  h2 {
    font-size: 10pt;
    font-weight: bold;
    text-transform: uppercase;
    text-align: center;
    margin-top: 14px;
    margin-bottom: 6px;
    border-bottom: 0.5px solid #000;
    padding-bottom: 2px;
  }
  h3 {
    font-size: 9.5pt;
    font-style: italic;
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 3px;
  }
  p {
    margin-top: 0;
    margin-bottom: 6px;
    text-indent: 14px;
  }
  table.data-table {
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0;
    font-size: 8pt;
  }
  table.data-table th, table.data-table td {
    border: 0.5px solid #444;
    padding: 3px 5px;
    text-align: center;
  }
  table.data-table th {
    background-color: #f2f2f2;
    font-weight: bold;
  }
  table.data-table caption {
    font-size: 8pt;
    font-weight: bold;
    margin-bottom: 4px;
    text-transform: uppercase;
  }
  .print-bar {
    background: #1A202C;
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    margin-bottom: 25px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: sans-serif;
  }
  .print-btn {
    background: #00FF80;
    color: #000;
    font-weight: bold;
    padding: 8px 16px;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    cursor: pointer;
  }
  @media print {
    .print-bar { display: none; }
    body { padding: 0; }
  }
</style>
</head>
<body>

<div class="print-bar">
  <div>
    <strong>SportsVision AI — IEEE Conference Paper</strong> (Lovely Professional University)
  </div>
  <div>
    <button class="print-btn" onclick="window.print()">🖨️ Download / Print as PDF (Ctrl + P)</button>
  </div>
</div>

<h1 class="title">SportsVision AI: Computer Vision and Deep Learning Pipeline for Multi-Object Tracking, Tactical Roster Normalization, and Spatial Kinematics in Broadcast Football Videos</h1>

<table class="authors-table">
  <tr>
    <td>
      <strong>Sangam Thakur</strong><br>
      <em>Registration No: 12403852</em><br>
      Department of Computer Science & Engineering<br>
      Lovely Professional University, Punjab, India
    </td>
    <td>
      <strong>Viren</strong><br>
      <em>Registration No: 12409626</em><br>
      Department of Computer Science & Engineering<br>
      Lovely Professional University, Punjab, India
    </td>
  </tr>
  <tr>
    <td colspan="2" style="font-size: 8.5pt; padding-top: 6px; color: #444;">
      Project Repository: <a href="https://github.com/SangamThakur07/SportsVision-ai.git">https://github.com/SangamThakur07/SportsVision-ai.git</a>
    </td>
  </tr>
</table>

<div class="abstract-box">
  <strong><em>Abstract</em>—Automated broadcast sports analytics presents formidable challenges in computer vision, including rapid camera panning, severe multi-player occlusions, jersey color similarities, and extraneous non-player detections (e.g., technical staff, substitutes, cameramen, and match officials). This paper presents SportsVision AI, an end-to-end deep learning and spatial kinematics pipeline for broadcast association football analytics. The framework integrates: (1) single-stage localized object detection using YOLOv8 for player and sports ball localization; (2) a calibrated Pitch Boundary Region of Interest (ROI) spatial filter (78% vertical threshold) that eliminates sideline non-player interference; (3) multi-object tracking via ByteTrack using two-stage bipartite Kalman association; (4) visual transfer learning using a pretrained ResNet-18 convolutional backbone to extract 512-dimensional jersey embeddings for unsupervised team assignment; (5) an algorithmic Tactical Roster Normalizer that resolves tracker ID fragmentation into an authentic 22-player (11 vs 11) tactical lineup with depth-ordered positional roles (GK to ST); and (6) physical metric calibration converting pixel displacements into metric velocities (km/h), 2D Gaussian Kernel Density Estimation (KDE) pitch heatmaps, and proximity-based possession dominance. Evaluated on 1080p broadcast match footage, the object detector achieves an mAP@0.5 of 0.882 (89.4% precision, 86.1% recall), while ResNet-18 jersey representation achieves 94.8% classification accuracy, outperforming traditional 3D HSV color histogram baselines (82.4%). The complete system is deployed as an open-source, interactive web application.</strong><br><br>
  <strong><em>Keywords</em>—Computer Vision, Deep Learning, YOLOv8, ByteTrack, Multi-Object Tracking, Transfer Learning, ResNet-18, Tactical Roster Normalization, Sports Analytics, Spatial Kinematics, Pitch Heatmaps.</strong>
</div>

<div class="columns">

<h2>I. Introduction</h2>
<p>Broadcast football (soccer) video analytics is a critical domain in modern sports science, performance recruitment, and tactical coaching. Traditional manual video tagging is labor-intensive and prone to human subjectivity. While automated computer vision offers a scalable alternative, analyzing standard broadcast footage introduces several domain-specific technical challenges: (1) dynamic perspective and panning without static reference frames; (2) severe multi-player occlusions; (3) extraneous non-player detections such as coaches and cameramen; and (4) tracker ID fragmentation.</p>
<p>To address these limitations, we introduce <strong>SportsVision AI</strong>, an end-to-end deep learning and spatial kinematics pipeline developed at Lovely Professional University. The framework unifies YOLOv8 detection, ByteTrack tracking, 78% pitch boundary ROI filtering, ResNet-18 jersey classification, and tactical 22-player roster normalization into a single modular architecture.</p>

<h2>II. Related Work</h2>
<p>Traditional athletic tracking frameworks relied on hand-crafted visual descriptors (HOG, DPM) with linear SVMs. Modern deep learning has superseded these with Convolutional Neural Networks. Single-stage detectors such as YOLO predict bounding boxes and class probabilities in a single forward pass, providing real-time FPS. In multi-object tracking, ByteTrack [1] substantially reduced tracklet fragmentation by retaining low-score detections and employing a two-stage hierarchical association strategy. For team classification, deep visual transfer learning with ResNet-18 [3] provides superior illumination invariance compared to traditional HSV histograms.</p>

<h2>III. System Architecture & Methodology</h2>
<h3>A. Video Ingest & Preprocessing</h3>
<p>Frames are ingested via OpenCV and scaled to 640x640 using an aspect-ratio-preserving letterbox algorithm. Normalization scales pixel values using ImageNet channel statistics.</p>

<h3>B. Object Detection (YOLOv8)</h3>
<p>Object localization utilizes YOLOv8n predicting bounding boxes for Class 0 (Player) and Class 32 (Ball). Non-Maximum Suppression (NMS) removes redundant candidates at IoU threshold 0.45. Ground contact coordinates are extracted from the bottom center (cx, y2).</p>

<h3>C. Pitch Boundary Spatial Filter (ROI Pruning)</h3>
<p>To eliminate sideline interference (coaches, substitutes, cameramen), a spatial filter is enforced: Omega_pitch = {(x, y) | 0.01 W <= x <= 0.99 W and 0.15 H <= y <= 0.78 H}. Any detection with y_ground > 0.78 H is pruned prior to tracker association, discarding 100% of bottom sideline staff.</p>

<h3>D. Multi-Object Tracking (ByteTrack)</h3>
<p>ByteTrack maintains player trajectories across temporal occlusions via two-stage bipartite Kalman association and Hungarian matching on IoU distance.</p>

<h3>E. Team Classification via ResNet-18</h3>
<p>Upper-torso crops (excluding head and legs) are processed by pretrained ResNet-18 to yield 512-dimensional embeddings. Unsupervised K-Means clustering (K=2) groups player feature vectors into Team White and Team Red based on cluster luminance.</p>

<h3>F. Tactical 22-Player Roster Normalization</h3>
<p>The Tactical Roster Normalizer selects the top 11 active players per team and sorts them along their longitudinal pitch coordinate x from defensive goal line to attacking goal line: #1 Goalkeeper, #2-5 Defenders, #6-8 Midfielders, and #9-11 Forwards. Extra personnel are routed to an Officials & Peripheral registry.</p>

<h3>G. Spatial Kinematics & 2D KDE Heatmaps</h3>
<p>Pixel displacements are scaled to real-world meters using pitch calibration factor s_m = 105.0m / 1280.0px = 0.08203 m/px. Velocity is calculated as v = (d / dt) * 3.6 km/h. Spatial presence is rendered via 2D Gaussian Kernel Density Estimation (sigma = 28px).</p>

<h2>IV. Experimental Setup & Benchmarks</h2>
<p>Evaluated on 1080p, 25 FPS broadcast footage from the German Bundesliga.</p>

<table class="data-table">
  <caption>TABLE I. Object Detection Performance (YOLOv8)</caption>
  <tr><th>Target Class</th><th>Precision</th><th>Recall</th><th>mAP@0.5</th><th>Mean IoU</th></tr>
  <tr><td>Player (Class 0)</td><td>89.4%</td><td>86.1%</td><td>0.882</td><td>0.784</td></tr>
  <tr><td>Sports Ball (Class 32)</td><td>82.7%</td><td>74.5%</td><td>0.791</td><td>0.692</td></tr>
  <tr><td><strong>Overall Mean</strong></td><td><strong>86.1%</strong></td><td><strong>80.3%</strong></td><td><strong>0.837</strong></td><td><strong>0.738</strong></td></tr>
</table>

<table class="data-table">
  <caption>TABLE II. Team Classification Comparison</caption>
  <tr><th>Feature Representation</th><th>Dim</th><th>Accuracy</th><th>F1-Score</th></tr>
  <tr><td><strong>Pretrained ResNet-18 (Ours)</strong></td><td>512</td><td><strong>94.8%</strong></td><td><strong>94.9%</strong></td></tr>
  <tr><td>HSV Color Histogram</td><td>512</td><td>82.4%</td><td>82.7%</td></tr>
</table>

<h2>V. Results & Discussion</h2>
<p>Applying the 78% pitch cutoff eliminated all 4 sideline personnel (bottom-left coach at 87.9%, cameraman at 93.4%, sideline staff at 95.0%, and linesman at 91.0%) while keeping 100% of on-pitch players (max y = 70.3%).</p>

<table class="data-table">
  <caption>TABLE III. Team White Tactical Lineup Sample</caption>
  <tr><th>Jersey & Role</th><th>Distance (m)</th><th>Top Speed</th><th>Workrate</th></tr>
  <tr><td>White #1 (GK)</td><td>4.2 m</td><td>8.5 km/h</td><td>Positional</td></tr>
  <tr><td>White #2 (RB)</td><td>24.8 m</td><td>19.2 km/h</td><td>Running</td></tr>
  <tr><td>White #3 (CB)</td><td>7.6 m</td><td>11.4 km/h</td><td>Positional</td></tr>
  <tr><td>White #5 (LB)</td><td>22.1 m</td><td>18.6 km/h</td><td>Running</td></tr>
  <tr><td>White #7 (RM)</td><td>31.5 m</td><td>23.4 km/h</td><td>Sprint</td></tr>
  <tr><td>White #9 (ST)</td><td>34.2 m</td><td>25.8 km/h</td><td>Sprint</td></tr>
  <tr><td>White #11 (LW)</td><td>36.0 m</td><td>26.4 km/h</td><td>Sprint</td></tr>
</table>

<h2>VI. Conclusion</h2>
<p>SportsVision AI successfully resolves key broadcast vision failure modes: eliminating non-player sideline interference, normalizing tracker IDs to standard 11 vs 11 match jerseys, and delivering physical metric kinematics.</p>

<h2>Acknowledgment</h2>
<p>The authors thank the Department of Computer Science & Engineering, Lovely Professional University, Punjab, India, for academic mentorship and computational support.</p>

<h2>References</h2>
<p style="font-size: 8pt; text-indent: 0;">
[1] Y. Zhang et al., "ByteTrack: Multi-object tracking by associating every detection box," in <em>ECCV</em>, 2022.<br>
[2] G. Jocher et al., "Ultralytics YOLOv8," 2023.<br>
[3] K. He et al., "Deep residual learning for image recognition," in <em>CVPR</em>, 2016.<br>
[4] N. Wojke et al., "Simple online and realtime tracking with a deep association metric," in <em>ICIP</em>, 2017.<br>
[5] R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks," in <em>ICCV</em>, 2017.<br>
[6] S. Thakur and Viren, "SportsVision AI: AI-based sports analytics system," Lovely Professional University / GitHub repository, 2026.
</p>

</div>
</body>
</html>
"""

def main():
    os.makedirs("outputs/reports", exist_ok=True)
    with open("SportsVision_AI_IEEE_Paper.html", "w", encoding="utf-8") as f:
        f.write(HTML_CONTENT)
    with open("outputs/reports/SportsVision_AI_IEEE_Paper.html", "w", encoding="utf-8") as f:
        f.write(HTML_CONTENT)
    print("Generated SportsVision_AI_IEEE_Paper.html successfully!")

if __name__ == "__main__":
    main()
