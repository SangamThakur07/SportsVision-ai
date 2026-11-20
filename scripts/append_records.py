"""
Appends 9 additional records to reach 50 total in the sample dataset.
Covers: Frame 50 (t=2.0s) corner kick scenario + ball occlusion edge case
"""
import os, json, pandas as pd

csv_path = "d:/sports analytics/data/sample_training_dataset/sample_training_dataset.csv"
json_path = "d:/sports analytics/data/sample_training_dataset/sample_training_dataset.json"

df = pd.read_csv(csv_path)
start_id = len(df) + 1
scale_m = 105.0 / 1280.0
pitch_y_scale = 68.0 / 720.0

extra = [
    # Frame 50 (t=2.0s) - Corner kick scenario (ball near corner flag)
    {"record_id": start_id,     "frame_idx": 50, "timestamp_sec": 2.00, "class_id": 32, "class_name": "Sports Ball",         "track_id": -1,  "tactical_jersey": "Ball",             "tactical_role": "Match Ball",             "team": "Neutral",      "bbox_x1": 15.0,  "bbox_y1": 118.0, "bbox_x2": 31.0,  "bbox_y2": 134.0, "bbox_norm_cx": 0.0180, "bbox_norm_cy": 0.1764, "bbox_norm_w": 0.0125, "bbox_norm_h": 0.0222, "ground_x_px": 23.0,   "ground_y_px": 134.0, "pitch_x_m": 1.89,   "pitch_y_m": 12.65, "speed_kmh": 0.0,  "distance_m": 0.0, "workrate_category": "🚶 Positional Zone", "detection_confidence": 0.69, "is_on_pitch": True,  "pitch_roi_valid": True},
    {"record_id": start_id + 1, "frame_idx": 50, "timestamp_sec": 2.00, "class_id": 0,  "class_name": "Player",               "track_id": 7,   "tactical_jersey": "White #7 (RM)",    "tactical_role": "Right Midfielder",       "team": "Team White",   "bbox_x1": 20.0,  "bbox_y1": 505.0, "bbox_x2": 52.0,  "bbox_y2": 580.0, "bbox_norm_cx": 0.0281, "bbox_norm_cy": 0.7674, "bbox_norm_w": 0.0250, "bbox_norm_h": 0.1042, "ground_x_px": 36.0,   "ground_y_px": 580.0, "pitch_x_m": 2.95,   "pitch_y_m": 54.78, "speed_kmh": 29.1, "distance_m": 7.9, "workrate_category": "⚡ High Sprint",     "detection_confidence": 0.84, "is_on_pitch": True,  "pitch_roi_valid": True},
    {"record_id": start_id + 2, "frame_idx": 50, "timestamp_sec": 2.00, "class_id": 0,  "class_name": "Player",               "track_id": 2,   "tactical_jersey": "Red #2 (RB)",      "tactical_role": "Right Back",             "team": "Team Red",     "bbox_x1": 35.0,  "bbox_y1": 65.0,  "bbox_x2": 67.0,  "bbox_y2": 140.0, "bbox_norm_cx": 0.0398, "bbox_norm_cy": 0.1424, "bbox_norm_w": 0.0250, "bbox_norm_h": 0.1042, "ground_x_px": 51.0,   "ground_y_px": 140.0, "pitch_x_m": 4.18,   "pitch_y_m": 13.22, "speed_kmh": 25.6, "distance_m": 6.8, "workrate_category": "⚡ High Sprint",     "detection_confidence": 0.78, "is_on_pitch": True,  "pitch_roi_valid": True},
    {"record_id": start_id + 3, "frame_idx": 50, "timestamp_sec": 2.00, "class_id": 0,  "class_name": "Player",               "track_id": 19,  "tactical_jersey": "White #11 (LW)",   "tactical_role": "Left Winger",            "team": "Team White",   "bbox_x1": 42.0,  "bbox_y1": 51.0,  "bbox_x2": 74.0,  "bbox_y2": 126.0, "bbox_norm_cx": 0.0453, "bbox_norm_cy": 0.1229, "bbox_norm_w": 0.0250, "bbox_norm_h": 0.1042, "ground_x_px": 58.0,   "ground_y_px": 126.0, "pitch_x_m": 4.76,   "pitch_y_m": 11.90, "speed_kmh": 32.1, "distance_m": 8.6, "workrate_category": "⚡ High Sprint",     "detection_confidence": 0.91, "is_on_pitch": True,  "pitch_roi_valid": True},
    {"record_id": start_id + 4, "frame_idx": 50, "timestamp_sec": 2.00, "class_id": 0,  "class_name": "Referee",              "track_id": 50,  "tactical_jersey": "Referee #1 (REF)", "tactical_role": "Match Referee",          "team": "Match Officials","bbox_x1": 142.0,"bbox_y1": 302.0, "bbox_x2": 167.0, "bbox_y2": 367.0, "bbox_norm_cx": 0.1207, "bbox_norm_cy": 0.4785, "bbox_norm_w": 0.0195, "bbox_norm_h": 0.0903, "ground_x_px": 154.5,  "ground_y_px": 367.0, "pitch_x_m": 12.67,  "pitch_y_m": 34.65, "speed_kmh": 14.2, "distance_m": 2.1, "workrate_category": "🏃 Active Running",  "detection_confidence": 0.86, "is_on_pitch": True,  "pitch_roi_valid": True},
    # Edge case: low-confidence ball occlusion at Frame 75 (t=3.0s)
    {"record_id": start_id + 5, "frame_idx": 75, "timestamp_sec": 3.00, "class_id": 32, "class_name": "Sports Ball (Occluded)","track_id": -1, "tactical_jersey": "Ball",             "tactical_role": "Match Ball",             "team": "Neutral",      "bbox_x1": 598.0, "bbox_y1": 348.0, "bbox_x2": 614.0, "bbox_y2": 364.0, "bbox_norm_cx": 0.4773, "bbox_norm_cy": 0.4944, "bbox_norm_w": 0.0125, "bbox_norm_h": 0.0222, "ground_x_px": 606.0,  "ground_y_px": 364.0, "pitch_x_m": 49.72,  "pitch_y_m": 34.38, "speed_kmh": 0.0,  "distance_m": 0.0, "workrate_category": "🚶 Positional Zone", "detection_confidence": 0.41, "is_on_pitch": True,  "pitch_roi_valid": True},
    # Additional unique player frames at Frame 75 to cover more scenario coverage
    {"record_id": start_id + 6, "frame_idx": 75, "timestamp_sec": 3.00, "class_id": 0,  "class_name": "Player",               "track_id": 5,   "tactical_jersey": "White #9 (ST)",    "tactical_role": "Center Forward",         "team": "Team White",   "bbox_x1": 892.0, "bbox_y1": 303.0, "bbox_x2": 924.0, "bbox_y2": 378.0, "bbox_norm_cx": 0.7094, "bbox_norm_cy": 0.4743, "bbox_norm_w": 0.0250, "bbox_norm_h": 0.1042, "ground_x_px": 908.0,  "ground_y_px": 378.0, "pitch_x_m": 74.47,  "pitch_y_m": 35.70, "speed_kmh": 11.2, "distance_m": 3.1, "workrate_category": "🏃 Active Running",  "detection_confidence": 0.88, "is_on_pitch": True,  "pitch_roi_valid": True},
    {"record_id": start_id + 7, "frame_idx": 75, "timestamp_sec": 3.00, "class_id": 0,  "class_name": "Player",               "track_id": 18,  "tactical_jersey": "Red #9 (ST)",      "tactical_role": "Center Forward",         "team": "Team Red",     "bbox_x1": 416.0, "bbox_y1": 273.0, "bbox_x2": 448.0, "bbox_y2": 348.0, "bbox_norm_cx": 0.3313, "bbox_norm_cy": 0.4354, "bbox_norm_w": 0.0250, "bbox_norm_h": 0.1042, "ground_x_px": 432.0,  "ground_y_px": 348.0, "pitch_x_m": 35.44,  "pitch_y_m": 32.87, "speed_kmh": 10.9, "distance_m": 2.9, "workrate_category": "🏃 Active Running",  "detection_confidence": 0.82, "is_on_pitch": True,  "pitch_roi_valid": True},
    {"record_id": start_id + 8, "frame_idx": 75, "timestamp_sec": 3.00, "class_id": 0,  "class_name": "Player",               "track_id": 23,  "tactical_jersey": "White #10 (CAM)",  "tactical_role": "Attacking Midfielder",   "team": "Team White",   "bbox_x1": 751.0, "bbox_y1": 273.0, "bbox_x2": 783.0, "bbox_y2": 348.0, "bbox_norm_cx": 0.5992, "bbox_norm_cy": 0.4354, "bbox_norm_w": 0.0250, "bbox_norm_h": 0.1042, "ground_x_px": 767.0,  "ground_y_px": 348.0, "pitch_x_m": 62.93,  "pitch_y_m": 32.87, "speed_kmh": 21.1, "distance_m": 5.6, "workrate_category": "⚡ High Sprint",     "detection_confidence": 0.90, "is_on_pitch": True,  "pitch_roi_valid": True},
]

df_extra = pd.DataFrame(extra)
df_final = pd.concat([df, df_extra], ignore_index=True)
print(f"Final record count: {len(df_final)}")

# Save CSV
df_final.to_csv(csv_path, index=False)
df_final.to_csv("d:/sports analytics/outputs/reports/sample_training_dataset.csv", index=False)
print("CSV saved.")

# Update JSON
with open(json_path, "r", encoding="utf-8") as f:
    jdata = json.load(f)

jdata["records"] = df_final.to_dict(orient="records")
jdata["metadata"]["total_records"] = len(df_final)

for jpath in [json_path, "d:/sports analytics/outputs/reports/sample_training_dataset.json"]:
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(jdata, f, indent=2)
print("JSON saved.")
print("Done! Total records:", len(df_final))
