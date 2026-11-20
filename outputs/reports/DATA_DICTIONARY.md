# DATA DICTIONARY
## SportsVision AI — Broadcast Football Tracking & Evaluation Benchmark

---

**Version:** 1.0.0  
**Institution:** Lovely Professional University (LPU), Punjab, India  
**Department:** Computer Science & Engineering  
**Authors:** Sangam Thakur (Reg. No: 12403852) · Viren (Reg. No: 12409626)  
**Repository:** https://github.com/SangamThakur07/SportsVision-ai.git  
**Dataset Files:** `sample_training_dataset.csv` · `sample_training_dataset.json`  
**Total Records:** 50 representative records across 4 video frames  

---

## 1. Dataset Overview

This dataset contains representative annotation records extracted from a 1080p Bundesliga broadcast
football match video, processed at 25 FPS through the SportsVision AI pipeline. Each record represents
one detected and tracked object (player, ball, referee, or sideline person) in one video frame.

The 50 records span:
- **22 players** (11 Team White + 11 Team Red) across multiple frames
- **1 Match Referee** (pitch official)
- **1 Sports Ball** (including an occlusion edge-case)
- **4 Sideline persons** (coach, cameraman, ball boy, linesman) demonstrating the 78% ROI filter

---

## 2. Coordinate Systems

### 2.1 Image Pixel Space
- **Origin:** Top-left corner `(0, 0)`
- **Canvas size:** `1280 × 720` pixels (HD broadcast frame)
- **X-axis:** Left → Right (0 to 1280 px)
- **Y-axis:** Top → Bottom (0 to 720 px)

### 2.2 YOLO Normalized Space
- **Range:** `[0.0, 1.0]` for both axes
- **Format:** `[x_center, y_center, width, height]` as fractions of canvas dimensions
- **Formula:** `x_norm = x_px / 1280.0`, `y_norm = y_px / 720.0`

### 2.3 FIFA Pitch Metric Space (Calibrated)
- **Standard pitch:** `105.0 m (length) × 68.0 m (width)`
- **Scale X:** `105.0 / 1280.0 ≈ 0.08203 m/px`
- **Scale Y:** `68.0 / 720.0 ≈ 0.09444 m/px`
- **Formula:** `pitch_x_m = ground_x_px × 0.08203`, `pitch_y_m = ground_y_px × 0.09444`

### 2.4 Pitch ROI Filter
- **Threshold:** `y_norm > 0.78` (ground contact Y > 561.6 px) → sideline / outside pitch
- **Logic:** Any detection whose **ground contact Y coordinate** exceeds 78% of the frame height
  is classified as outside the active pitch zone and excluded from tactical analysis.

---

## 3. Field Descriptions

| # | Field Name | Data Type | Nullable | Range / Values | Description | Formula / Notes | Example |
|---|-----------|-----------|----------|----------------|-------------|-----------------|---------|
| 1 | `record_id` | Integer | No | 1 – 50 | Unique sequential primary key for each record in the dataset | Auto-incremented from 1 | `1` |
| 2 | `frame_idx` | Integer | No | 0, 25, 50, 75, ... | Zero-based video frame index. At 25 FPS: frame 0 = 0.0s, frame 25 = 1.0s, frame 50 = 2.0s | `frame_idx = timestamp_sec × fps` | `25` |
| 3 | `timestamp_sec` | Float | No | 0.00 – ∞ | Time in seconds from video start corresponding to this frame | `timestamp_sec = frame_idx / fps` | `1.00` |
| 4 | `class_id` | Integer | No | `0`, `32` | COCO dataset class ID. `0` = Person (Player/Referee/Sideline). `32` = Sports Ball | YOLOv8 COCO-pretrained class label | `0` |
| 5 | `class_name` | String | No | `Player`, `Referee`, `Person (Sideline)`, `Sports Ball`, `Sports Ball (Occluded)` | Human-readable class label including context-aware sub-classification | Derived from class_id + ROI filter + confidence threshold | `"Player"` |
| 6 | `track_id` | Integer | No | 1 – 104, `-1` for ball | Persistent ByteTrack multi-object tracking ID assigned across all frames. `-1` indicates the sports ball (not tracked via ByteTrack). Sideline persons use IDs 101–104. | ByteTrack MOT algorithm output | `27` |
| 7 | `tactical_jersey` | String | No | `White #1 (GK)` – `White #11 (LW)`, `Red #1 (GK)` – `Red #11 (LW)`, `Referee #1 (REF)`, `Ball`, `Sideline #1` – `#4` | Normalized tactical jersey identifier. Maps raw track_id to a stable semantic label after team assignment and jersey calibration. | Assigned by `src/analytics/roster.py` | `"White #9 (ST)"` |
| 8 | `tactical_role` | String | No | Goalkeeper, Right Back, Center Back, Left Back, Defensive Midfielder, Right Midfielder, Central Midfielder, Left Winger, Attacking Midfielder, Center Forward, Match Referee, Match Ball, Dugout Coach, Cameraman / Photographer, Ball Boy / Pitch Staff, Assistant Referee (Linesman) | Tactical / functional position on the pitch or sideline role | Derived from jersey number and team formation (4-3-3 default) | `"Center Forward"` |
| 9 | `team` | String | No | `Team White`, `Team Red`, `Match Officials`, `Neutral`, `Staff`, `Media`, `Officials` | Team or category assignment for the detected object | Assigned via K-Means color clustering on jersey HSV histogram | `"Team White"` |
| 10 | `bbox_x1` | Float | No | 0.0 – 1280.0 | Left edge X-coordinate of the bounding box in pixel space | Raw YOLOv8 output converted from normalized to pixel | `124.0` |
| 11 | `bbox_y1` | Float | No | 0.0 – 720.0 | Top edge Y-coordinate of the bounding box in pixel space | Raw YOLOv8 output converted from normalized to pixel | `285.0` |
| 12 | `bbox_x2` | Float | No | 0.0 – 1280.0 | Right edge X-coordinate of the bounding box in pixel space | `bbox_x2 = bbox_x1 + bbox_width` | `156.0` |
| 13 | `bbox_y2` | Float | No | 0.0 – 720.0 | Bottom edge Y-coordinate of the bounding box in pixel space | `bbox_y2 = bbox_y1 + bbox_height` | `360.0` |
| 14 | `bbox_norm_cx` | Float | No | 0.0 – 1.0 | Normalized X center of bounding box (YOLO format) | `bbox_norm_cx = (bbox_x1 + bbox_x2) / (2 × 1280)` | `0.1094` |
| 15 | `bbox_norm_cy` | Float | No | 0.0 – 1.0 | Normalized Y center of bounding box (YOLO format) | `bbox_norm_cy = (bbox_y1 + bbox_y2) / (2 × 720)` | `0.4479` |
| 16 | `bbox_norm_w` | Float | No | 0.0 – 1.0 | Normalized bounding box width (YOLO format) | `bbox_norm_w = (bbox_x2 - bbox_x1) / 1280` | `0.0250` |
| 17 | `bbox_norm_h` | Float | No | 0.0 – 1.0 | Normalized bounding box height (YOLO format) | `bbox_norm_h = (bbox_y2 - bbox_y1) / 720` | `0.1042` |
| 18 | `ground_x_px` | Float | No | 0.0 – 1280.0 | X-coordinate of the player's ground contact point (foot position) in pixel space | `ground_x_px = (bbox_x1 + bbox_x2) / 2` (horizontal center) | `140.0` |
| 19 | `ground_y_px` | Float | No | 0.0 – 720.0 | Y-coordinate of the player's ground contact point (bottom of bounding box) in pixel space | `ground_y_px = bbox_y2` (bottom edge = foot level) | `360.0` |
| 20 | `pitch_x_m` | Float | No | 0.0 – 105.0 | Calibrated pitch X position in meters from the left goal line | `pitch_x_m = ground_x_px × (105.0 / 1280.0)` | `11.48` |
| 21 | `pitch_y_m` | Float | No | 0.0 – 68.0 | Calibrated pitch Y position in meters from the top touchline | `pitch_y_m = ground_y_px × (68.0 / 720.0)` | `34.00` |
| 22 | `speed_kmh` | Float | No | 0.0 – 45.0 | Instantaneous speed of the detected object in km/h, computed between consecutive frames | `speed_kmh = (Δd_px × scale_m/px) / Δt × 3.6` where `Δt = Δframes / fps` | `8.4` |
| 23 | `distance_m` | Float | No | 0.0 – 15.0 | Distance travelled (meters) in the current frame segment since the previous tracked position | `distance_m = sqrt((Δpitch_x)² + (Δpitch_y)²)` | `2.5` |
| 24 | `workrate_category` | String | No | `⚡ High Sprint`, `🏃 Active Running`, `🚶 Positional Zone` | Categorical work-rate classification based on current speed thresholds | High Sprint: speed > 20 km/h · Active Running: 10–20 km/h · Positional Zone: < 10 km/h | `"⚡ High Sprint"` |
| 25 | `detection_confidence` | Float | No | 0.0 – 1.0 | YOLOv8 object detection confidence score for this bounding box | Raw model output score (sigmoid of final logit). Threshold for display: 0.40 | `0.84` |
| 26 | `is_on_pitch` | Boolean | No | `True`, `False` | Whether the detection's ground contact point lies within the pitch boundary polygon (homography-mapped) | Computed via point-in-polygon test on calibrated pitch corners | `True` |
| 27 | `pitch_roi_valid` | Boolean | No | `True`, `False` | Whether the detection passes the vertical ROI cutoff filter (ground_y_px / 720 ≤ 0.78). Sideline persons have `False`. | `pitch_roi_valid = (ground_y_px / 720.0) <= 0.78` | `False` (for sideline) |

---

## 4. Workrate Speed Thresholds

| Category | Emoji | Speed Range | Tactical Interpretation |
|----------|-------|-------------|------------------------|
| High Sprint | ⚡ | > 20 km/h | Attacking run, counter-press, tracking run |
| Active Running | 🏃 | 10 – 20 km/h | Transitional movement, pressing, repositioning |
| Positional Zone | 🚶 | 0 – 10 km/h | Ball-watching, set-piece positioning, holding shape |

---

## 5. Class ID Reference (COCO)

| class_id | class_name | Description |
|----------|-----------|-------------|
| `0` | Person | All humans detected (players, referee, sideline persons) |
| `32` | Sports Ball | Football / soccer ball |

---

## 6. Sample Records by Frame

| Frame | Timestamp | Objects Included |
|-------|-----------|-----------------|
| 0 | 0.00s | 11 White + 11 Red players, Referee, Ball, 4 Sideline (ROI filtered) |
| 25 | 1.00s | 12 key sprinters + Ball (pass delivered forward) |
| 50 | 2.00s | Corner kick scenario — Ball + 3 key players + Referee |
| 75 | 3.00s | Ball (occluded, low confidence) + 3 players — testing edge cases |

---

## 7. Pipeline Architecture (Data Generation)

```
1080p Broadcast MP4
     │
     ▼
YOLOv8x Detection (class_id 0, 32)
     │
     ▼
ByteTrack MOT  →  track_id assignment
     │
     ▼
Pitch ROI Filter (78% Y threshold)  →  pitch_roi_valid
     │
     ▼
K-Means Team Assignment (jersey HSV)  →  team
     │
     ▼
Roster Mapping (src/analytics/roster.py)  →  tactical_jersey, tactical_role
     │
     ▼
Homography Calibration  →  pitch_x_m, pitch_y_m
     │
     ▼
Kinematic Computation  →  speed_kmh, distance_m, workrate_category
     │
     ▼
CSV / JSON Export  →  sample_training_dataset.csv / .json
```

---

## 8. Notes & Limitations

1. **Synthetic representative data:** This sample dataset contains realistic but synthetically-generated coordinates based on typical Bundesliga broadcast camera perspectives. Actual production data is extracted from real video using the full pipeline.
2. **Fixed camera assumption:** The pitch-to-pixel calibration assumes a fixed broadcast camera. Pan/zoom operations require re-calibration per frame.
3. **Ball track_id = -1:** The sports ball is detected by YOLO but not assigned a ByteTrack ID due to its small size and fast motion. A dedicated ball-tracking module is under development.
4. **Sideline records:** Records 38–41 (Sideline #1–#4) have `pitch_roi_valid = False` and `is_on_pitch = False`. They are included to demonstrate correct filtering behavior.
5. **Speed computation:** Speed values for Frame 0 records represent average speed from a prior assumed position; they are initialized from tactical context rather than computed from frame delta.

---

*Last Updated: November 2026 | SportsVision AI v1.0.0 | Lovely Professional University*
