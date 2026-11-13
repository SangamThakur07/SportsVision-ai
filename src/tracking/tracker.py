"""
Object Tracking Module
Implements multi-object tracking for players using ByteTrack and dedicated ball trajectory tracking.
Maintains persistent player IDs across frames, trajectory history trails, and occlusion recovery.
Directly aligns with Computer Vision Syllabus Unit V.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
from collections import deque
import numpy as np
import torch
from ultralytics import YOLO


class TrackedPlayer:
    """Represents a tracked player entity in the current frame."""

    def __init__(
        self,
        track_id: int,
        bbox: List[float],
        confidence: float,
        frame_idx: int,
        timestamp: float,
        team: Optional[str] = None
    ):
        self.track_id = track_id
        self.bbox = [round(float(v), 2) for v in bbox]
        self.confidence = float(confidence)
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.team = team

        x1, y1, x2, y2 = self.bbox
        self.center = (round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2))
        self.bottom_center = (self.center[0], y2)  # Ground-contact position

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "center": self.center,
            "bottom_center": self.bottom_center,
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "team": self.team
        }


class BallTrack:
    """Manages ball detection, trajectory trail, and occlusion status."""

    def __init__(self, max_trajectory_len: int = 40):
        self.max_len = max_trajectory_len
        self.trajectory: deque = deque(maxlen=max_trajectory_len)
        self.is_detected: bool = False
        self.current_bbox: Optional[List[float]] = None
        self.current_center: Optional[Tuple[float, float]] = None
        self.confidence: float = 0.0
        self.frames_lost: int = 0
        self.status: str = "SEARCHING"

    def update(
        self,
        detected: bool,
        bbox: Optional[List[float]] = None,
        confidence: float = 0.0
    ):
        """Update ball state for the current frame."""
        self.is_detected = detected

        if detected and bbox is not None:
            self.current_bbox = [round(float(v), 2) for v in bbox]
            self.confidence = float(confidence)
            x1, y1, x2, y2 = self.current_bbox
            cx = round((x1 + x2) / 2.0, 2)
            cy = round((y1 + y2) / 2.0, 2)
            self.current_center = (cx, cy)
            self.trajectory.append(self.current_center)
            self.frames_lost = 0
            self.status = "DETECTED"
        else:
            self.current_bbox = None
            self.current_center = None
            self.confidence = 0.0
            self.frames_lost += 1
            if self.frames_lost > 30:
                self.status = "LOST"
                # Clear very old trajectory if lost for too long to avoid stale trails
                if len(self.trajectory) > 0 and self.frames_lost > 60:
                    self.trajectory.popleft()
            else:
                self.status = "OCCLUDED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_detected": self.is_detected,
            "bbox": self.current_bbox,
            "center": self.current_center,
            "confidence": self.confidence,
            "frames_lost": self.frames_lost,
            "status": self.status,
            "trajectory": list(self.trajectory)
        }


class SportsTracker:
    """
    Core tracker coordinating ByteTrack player tracking and ball trajectory estimation.
    """

    def __init__(
        self,
        model_path: str = "models/weights/yolov8n.pt",
        confidence_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        max_trail_len: int = 30,
        device: Optional[str] = None,
        pitch_roi: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        Args:
            pitch_roi: Optional (x1_pct, y1_pct, x2_pct, y2_pct) in [0.0–1.0] defining
                       the pitch boundary as a fraction of frame size. Detections whose
                       bottom-center falls outside this region are discarded (coaches,
                       cameramen, ball boys at the frame edges).
                       Default: (0.02, 0.05, 0.98, 0.88) — trims ~12% from the bottom
                       (dugout/sideline row) and 2% from each horizontal edge.
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_trail_len = max_trail_len

        # Pitch boundary filter (fractions of frame width/height)
        self.pitch_roi = pitch_roi if pitch_roi is not None else (0.02, 0.05, 0.98, 0.88)

        # Device selection
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

        # Load YOLO model
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Player trajectory histories: {track_id: deque([(x, y), ...])}
        self.player_trajectories: Dict[int, deque] = {}

        # Ball tracker
        self.ball_tracker = BallTrack(max_trajectory_len=max_trail_len)

        # Complete tracking log for analytics: list of dicts
        self.tracking_history: List[Dict[str, Any]] = []

    def update(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float = 25.0
    ) -> Dict[str, Any]:
        """
        Track players and ball on the current frame.

        Args:
            frame: Input BGR image frame.
            frame_idx: Frame sequence index.
            fps: Video frames per second.

        Returns:
            Dictionary with 'players', 'ball', 'frame_idx', and 'timestamp'.
        """
        timestamp = round(frame_idx / fps, 3) if fps > 0 else 0.0

        # Run YOLO with ByteTrack (persist=True maintains IDs across frames)
        # Class 0: person (player), Class 32: sports ball
        # Use optimized sports tracker config if available to reduce ID switching across occlusions
        tracker_cfg = "config/sports_bytetrack.yaml" if os.path.exists("config/sports_bytetrack.yaml") else "bytetrack.yaml"

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=tracker_cfg,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[0, 32],
            device=self.device,
            verbose=False
        )

        tracked_players: List[TrackedPlayer] = []
        ball_detected = False
        ball_bbox = None
        ball_conf = 0.0

        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

                for i in range(len(xyxy)):
                    cls_id = int(classes[i])
                    conf = float(confs[i])
                    bbox = [float(v) for v in xyxy[i]]

                    if cls_id == 0:  # Player (person)
                        # If tracker assigned an ID, use it; otherwise assign negative temporary ID
                        t_id = int(track_ids[i]) if (track_ids is not None and i < len(track_ids)) else -(i + 1)
                        player = TrackedPlayer(
                            track_id=t_id,
                            bbox=bbox,
                            confidence=conf,
                            frame_idx=frame_idx,
                            timestamp=timestamp
                        )

                        # ── Pitch boundary filter ───────────────────────────────────────────────
                        # Discard detections of coaches, cameramen, and ball boys whose
                        # feet (bottom_center) land outside the visible pitch rectangle.
                        # roi = (x1%, y1%, x2%, y2%) as fractions of frame dimensions.
                        h_frame, w_frame = frame.shape[:2]
                        roi_x1 = self.pitch_roi[0] * w_frame
                        roi_y1 = self.pitch_roi[1] * h_frame
                        roi_x2 = self.pitch_roi[2] * w_frame
                        roi_y2 = self.pitch_roi[3] * h_frame
                        bx, by = player.bottom_center
                        if not (roi_x1 <= bx <= roi_x2 and roi_y1 <= by <= roi_y2):
                            continue   # Outside pitch → skip (coach / cameraman)
                        # ────────────────────────────────────────────────────────────────────────

                        tracked_players.append(player)

                        # Update player trajectory trail (feet position on pitch)
                        if t_id not in self.player_trajectories:
                            self.player_trajectories[t_id] = deque(maxlen=self.max_trail_len)
                        self.player_trajectories[t_id].append(player.bottom_center)

                        # Log to history for Phase 3 analytics
                        self.tracking_history.append({
                            "frame_idx": frame_idx,
                            "timestamp": timestamp,
                            "track_id": t_id,
                            "x": player.bottom_center[0],
                            "y": player.bottom_center[1],
                            "bbox": player.bbox,
                            "confidence": player.confidence
                        })

                    elif cls_id == 32:  # Sports Ball
                        # Take the highest confidence ball detection
                        if not ball_detected or conf > ball_conf:
                            ball_detected = True
                            ball_bbox = bbox
                            ball_conf = conf

        # Update ball state
        self.ball_tracker.update(ball_detected, ball_bbox, ball_conf)

        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "players": [p.to_dict() for p in tracked_players],
            "player_trajectories": {
                p.track_id: list(self.player_trajectories[p.track_id])
                for p in tracked_players
            },
            "ball": self.ball_tracker.to_dict()
        }

    def reset(self):
        """Reset all tracking states."""
        self.player_trajectories.clear()
        self.ball_tracker = BallTrack(max_trajectory_len=self.max_trail_len)
        self.tracking_history.clear()
