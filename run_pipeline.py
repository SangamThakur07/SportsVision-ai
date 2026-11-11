"""
SportsVision AI: Standalone Pipeline Execution Script
Supports:
- Phase 1: Video Preprocessing & YOLO Object Detection
- Phase 2: ByteTrack Multi-Object Tracking & Ball Trajectory
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, List
import yaml
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.video.video_loader import VideoLoader
from src.video.video_writer import VideoWriter
from src.preprocessing.preprocessor import Preprocessor
from src.detection.detector import YOLODetector
from src.tracking.tracker import SportsTracker
from src.classification.team_classifier import TeamClassifier
from src.visualization.annotator import Annotator
from src.analytics.movement import MovementAnalyzer
from src.analytics.heatmap import HeatmapGenerator
from src.analytics.possession import PossessionEstimator
from src.analytics.team_analytics import TeamAnalytics


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(
    input_video: str,
    output_video: str,
    mode: str = "tracking",
    config_path: str = "config/config.yaml",
    conf_threshold: float = None,
    device: str = None,
    max_frames: int = None,
    save_csv: str = None
):
    """
    Execute SportsVision AI Pipeline (Detection or Tracking).
    """
    phase_title = "Phase 2 Object Tracking (ByteTrack)" if mode == "tracking" else "Phase 1 Object Detection"
    print("=" * 60)
    print(f" SportsVision AI: {phase_title}")
    print("=" * 60)

    # 1. Load Configuration
    config = load_config(config_path)
    det_cfg = config.get("detection", {})
    vis_cfg = config.get("visualization", {})
    prep_cfg = config.get("preprocessing", {})
    track_cfg = config.get("tracking", {})

    confidence = conf_threshold if conf_threshold is not None else det_cfg.get("confidence_threshold", 0.30)
    dev = device if device is not None else config.get("device", "cuda")
    model_path = det_cfg.get("model_path", "models/weights/yolov8n.pt")

    # 2. Initialize Video Loader
    print(f"\n[1/5] Opening video: {input_video}")
    loader = VideoLoader(input_video)
    info = loader.get_info()

    print("-" * 40)
    print(f" Resolution:   {info['resolution']}")
    print(f" FPS:          {info['fps']}")
    print(f" Total Frames: {info['total_frames']}")
    print(f" Duration:     {info['duration_formatted']} ({info['duration_sec']}s)")
    print("-" * 40)

    # 3. Initialize Preprocessor
    print("\n[2/5] Initializing Computer Vision Preprocessor...")
    target_size = tuple(prep_cfg.get("target_size")) if prep_cfg.get("target_size") else None
    preprocessor = Preprocessor(
        target_size=target_size,
        color_space=prep_cfg.get("color_space", "RGB"),
        normalize=prep_cfg.get("normalize", False)
    )

    # 4. Initialize Model (Detector or Tracker)
    detector = None
    tracker = None
    possession_estimator = None

    if mode == "tracking":
        print(f"\n[3/5] Initializing ByteTrack Multi-Object Tracker ({model_path})...")
        tracker = SportsTracker(
            model_path=model_path,
            confidence_threshold=confidence,
            iou_threshold=det_cfg.get("iou_threshold", 0.45),
            max_trail_len=track_cfg.get("track_buffer", 30),
            device=dev
        )
        possession_estimator = PossessionEstimator(proximity_threshold_px=80.0)
        team_classifier = TeamClassifier(num_teams=2, device="cpu")
        frame_track_cache = []
        print(f" Inferred on device: {tracker.device}")
    else:
        print(f"\n[3/5] Loading YOLO Object Detector ({model_path})...")
        detector = YOLODetector(
            model_path=model_path,
            confidence_threshold=confidence,
            iou_threshold=det_cfg.get("iou_threshold", 0.45),
            target_classes=det_cfg.get("target_classes"),
            device=dev
        )
        print(f" Inferred on device: {detector.device}")

    # 5. Initialize Visual Annotator
    print("\n[4/5] Setting up Annotator and Output Video Writer...")
    annotator = Annotator(
        player_color=tuple(vis_cfg.get("player_box_color", [0, 255, 128])),
        ball_color=tuple(vis_cfg.get("ball_box_color", [0, 140, 255])),
        referee_color=tuple(vis_cfg.get("referee_box_color", [0, 255, 255])),
        box_thickness=vis_cfg.get("box_thickness", 2),
        font_scale=vis_cfg.get("font_scale", 0.55),
        text_thickness=vis_cfg.get("text_thickness", 2),
        show_confidence=vis_cfg.get("show_confidence", True),
        show_center_point=vis_cfg.get("show_center_point", True),
        show_hud=vis_cfg.get("show_hud", True),
        show_trajectories=True
    )

    output_dir = os.path.dirname(os.path.abspath(output_video))
    os.makedirs(output_dir, exist_ok=True)

    out_width = target_size[1] if target_size else info["width"]
    out_height = target_size[0] if target_size else info["height"]

    writer = VideoWriter(
        output_path=output_video,
        fps=info["fps"],
        frame_size=(out_width, out_height)
    )

    # 6. Stream Frames and Process
    total_to_process = min(info["total_frames"], max_frames) if max_frames else info["total_frames"]
    print(f"\n[5/5] Processing {total_to_process} frames (Mode: {mode.upper()})...")

    start_time = time.time()
    frames_processed = 0
    unique_track_ids = set()
    ball_found_count = 0

    try:
        for frame_idx, frame in loader.stream_frames(max_frames=max_frames):
            if target_size:
                frame = preprocessor.resize(frame, target_size)

            if mode == "tracking":
                tracking_result = tracker.update(frame, frame_idx=frame_idx, fps=info["fps"])
                for p in tracking_result["players"]:
                    if p["track_id"] > 0:
                        unique_track_ids.add(p["track_id"])
                        team_classifier.collect_player_crop(frame, p["track_id"], p["bbox"])
                if tracking_result["ball"]["is_detected"]:
                    ball_found_count += 1

                frame_track_cache.append((frame_idx, tracking_result["ball"], tracking_result["players"]))

                annotated_frame = annotator.annotate_tracking(
                    frame,
                    tracking_result,
                    fps=info["fps"]
                )
            else:
                detections = detector.detect_frame(frame, frame_idx=frame_idx)
                for d in detections:
                    if "ball" in d["class_name"].lower():
                        ball_found_count += 1

                annotated_frame = annotator.annotate_frame(
                    frame,
                    detections,
                    frame_idx=frame_idx,
                    fps=info["fps"]
                )

            writer.write_frame(annotated_frame)
            frames_processed += 1

            if frames_processed % 25 == 0 or frames_processed == total_to_process:
                elapsed = time.time() - start_time
                fps_proc = frames_processed / elapsed if elapsed > 0 else 0
                pct = (frames_processed / total_to_process * 100) if total_to_process > 0 else 0
                track_info = f" | Unique IDs: {len(unique_track_ids)}" if mode == "tracking" else ""
                print(f" Progress: {frames_processed}/{total_to_process} frames ({pct:.1f}%) - Speed: {fps_proc:.2f} FPS{track_info}")

    finally:
        loader.release()
        writer.release()

    total_time = time.time() - start_time
    avg_fps = frames_processed / total_time if total_time > 0 else 0

    # Analytics Post-Processing
    reports_dir = config.get("paths", {}).get("output_report_dir", "outputs/reports")
    heatmaps_dir = config.get("paths", {}).get("output_heatmap_dir", "outputs/heatmaps")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(heatmaps_dir, exist_ok=True)

    csv_saved_path = None
    kinematics_saved_path = None
    heatmap_saved_paths = []
    possession_res = None

    if mode == "tracking" and tracker is not None and len(tracker.tracking_history) > 0:
        # 1. Export Raw Tracking CSV
        csv_path = save_csv or os.path.join(reports_dir, "tracking_data.csv")
        df = pd.DataFrame(tracker.tracking_history)
        df.to_csv(csv_path, index=False)
        csv_saved_path = csv_path

        # 2. Player Kinematics Analytics
        movement_analyzer = MovementAnalyzer(fps=info["fps"])
        player_metrics = movement_analyzer.analyze_all_players(tracker.tracking_history)
        kinematics_df = movement_analyzer.to_dataframe(player_metrics)
        kin_path = os.path.join(reports_dir, "player_kinematics.csv")
        kinematics_df.to_csv(kin_path, index=False)
        kinematics_saved_path = kin_path

        # 3. Team Classification & Ball Possession
        if possession_estimator is not None:
            try:
                player_teams = team_classifier.fit()
            except Exception:
                player_teams = {tid: ("Team White" if tid % 2 == 1 else "Team Red") for tid in unique_track_ids}

            possession_estimator.reset()
            for f_idx, b_data, p_data in frame_track_cache:
                possession_estimator.update_frame(
                    frame_idx=f_idx,
                    timestamp=f_idx / info["fps"],
                    ball_data=b_data,
                    players=p_data,
                    player_teams=player_teams
                )
            possession_res = possession_estimator.compute_summary()

        # 4. Movement Heatmaps Generation
        heatmap_gen = HeatmapGenerator(pitch_width=1280, pitch_height=720, sigma=28.0)
        frame_sz = (info["width"], info["height"])

        # Full Match Overall Activity Heatmap
        all_positions = [(r["x"], r["y"]) for r in tracker.tracking_history if r.get("track_id", -1) > 0]
        if all_positions:
            match_hm_path = os.path.join(heatmaps_dir, "match_activity_heatmap.png")
            heatmap_gen.save_heatmap(all_positions, match_hm_path, title="Full Match Spatial Activity Heatmap", frame_size=frame_sz)
            heatmap_saved_paths.append(match_hm_path)

        # Individual Heatmap for Most Active Tracked Player
        if player_metrics:
            top_player_id = max(player_metrics.keys(), key=lambda pid: player_metrics[pid]["frame_count"])
            top_positions = player_metrics[top_player_id]["positions"]
            top_hm_path = os.path.join(heatmaps_dir, f"player_{top_player_id}_heatmap.png")
            heatmap_gen.save_heatmap(top_positions, top_hm_path, title=f"Player #{top_player_id} Movement Heatmap", frame_size=frame_sz)
            heatmap_saved_paths.append(top_hm_path)

    print("\n" + "=" * 60)
    print(f" {phase_title} Completed Successfully!")
    print("=" * 60)
    print(f" Frames Processed:      {frames_processed}")
    print(f" Total Elapsed Time:    {total_time:.2f} seconds")
    print(f" Processing Speed:      {avg_fps:.2f} FPS")
    if mode == "tracking":
        print(f" Unique Players Tracked: {len(unique_track_ids)}")
        print(f" Track IDs Assigned:    {sorted(list(unique_track_ids))}")
        ball_pct = (ball_found_count / frames_processed * 100) if frames_processed > 0 else 0
        print(f" Ball Track Visibility: {ball_found_count}/{frames_processed} frames ({ball_pct:.1f}%)")
        if csv_saved_path:
            print(f" Tracking Data Export:  {csv_saved_path} ({len(tracker.tracking_history)} rows)")
        if kinematics_saved_path:
            print(f" Player Kinematics:     {kinematics_saved_path} ({len(player_metrics)} players)")
        if possession_res:
            print(f" Estimated Possession:  {possession_res['team_a_name']} {possession_res['team_a_pct']}% | {possession_res['team_b_name']} {possession_res['team_b_pct']}%")
        if heatmap_saved_paths:
            print(f" Heatmaps Generated:    {len(heatmap_saved_paths)} images in {heatmaps_dir}/")
    print(f" Output Video Saved:    {output_video}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SportsVision AI Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input football video (.mp4/.avi)")
    parser.add_argument("--output", "-o", type=str, default="outputs/videos/output_tracking.mp4", help="Path for output video")
    parser.add_argument("--mode", "-m", type=str, choices=["detection", "tracking"], default="tracking", help="Pipeline mode")
    parser.add_argument("--config", "-c", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--csv", type=str, default=None, help="Path to save tracking CSV data")

    args = parser.parse_args()
    run_pipeline(
        input_video=args.input,
        output_video=args.output,
        mode=args.mode,
        config_path=args.config,
        conf_threshold=args.conf,
        device=args.device,
        max_frames=args.max_frames,
        save_csv=args.csv
    )
