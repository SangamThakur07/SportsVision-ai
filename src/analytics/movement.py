"""
Player Movement and Kinematics Analytics Module
Calculates pixel displacement, smoothed velocity, total distance covered,
and tracking duration for each tracked player.
Directly aligns with Computer Vision Syllabus Unit I & V.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


class MovementAnalyzer:
    """
    Computes kinematics metrics (displacement, smoothed velocity, distance covered)
    from multi-object tracking records.
    """

    def __init__(
        self,
        fps: float = 25.0,
        smoothing_window: int = 5,
        max_displacement_threshold: float = 250.0  # Max plausible pixel jump per frame
    ):
        """
        Args:
            fps: Video frames per second.
            smoothing_window: Moving average window size for velocity smoothing.
            max_displacement_threshold: Discard erratic coordinate jumps caused by ID swaps.
        """
        self.fps = fps if fps > 0 else 25.0
        self.smoothing_window = max(1, smoothing_window)
        self.max_displacement_threshold = max_displacement_threshold
        self.records: List[Dict[str, Any]] = []

    def update_position(
        self,
        track_id: int,
        x: float,
        y: float,
        timestamp: Optional[float] = None,
        frame_idx: Optional[int] = None
    ):
        """Append an instantaneous player tracking coordinate."""
        if track_id <= 0:
            return
        t = timestamp if timestamp is not None else (len(self.records) / self.fps)
        f = frame_idx if frame_idx is not None else int(round(t * self.fps))
        self.records.append({
            "track_id": track_id,
            "x": float(x),
            "y": float(y),
            "timestamp": float(t),
            "frame_idx": f
        })

    def calculate_player_metrics(
        self,
        player_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute kinematic statistics for a single player from ordered tracking records.

        Args:
            player_records: List of records containing 'frame_idx', 'x', 'y', and 'timestamp'.

        Returns:
            Dictionary with distance, speed, and trajectory metrics.
        """
        if not player_records or len(player_records) < 2:
            single = player_records[0] if player_records else {}
            return {
                "track_id": single.get("track_id", -1),
                "total_distance_px": 0.0,
                "avg_speed_px_s": 0.0,
                "max_speed_px_s": 0.0,
                "duration_sec": 0.0,
                "frame_count": len(player_records),
                "positions": [(single.get("x", 0.0), single.get("y", 0.0))] if single else [],
                "speeds": [0.0] if single else []
            }

        # Sort records by frame_idx
        sorted_records = sorted(player_records, key=lambda r: r["frame_idx"])
        track_id = sorted_records[0]["track_id"]

        positions = [(r["x"], r["y"]) for r in sorted_records]
        frame_indices = [r["frame_idx"] for r in sorted_records]
        timestamps = [r.get("timestamp", r["frame_idx"] / self.fps) for r in sorted_records]

        displacements: List[float] = []
        raw_speeds: List[float] = []

        for i in range(1, len(positions)):
            x1, y1 = positions[i - 1]
            x2, y2 = positions[i]

            dx = x2 - x1
            dy = y2 - y1
            dist = float(np.sqrt(dx * dx + dy * dy))

            # Filter unrealistic teleportation spikes (tracker ID reassignment)
            if dist > self.max_displacement_threshold:
                dist = 0.0

            dt = timestamps[i] - timestamps[i - 1]
            if dt <= 0:
                dt = 1.0 / self.fps

            inst_speed = dist / dt

            displacements.append(dist)
            raw_speeds.append(inst_speed)

        # Apply moving-average smoothing filter to reduce bounding box coordinate jitter
        smoothed_speeds = self._smooth_speeds(raw_speeds)

        total_distance = float(np.sum(displacements))
        duration = float(timestamps[-1] - timestamps[0])
        avg_speed = float(np.mean(smoothed_speeds)) if smoothed_speeds else 0.0
        max_speed = float(np.max(smoothed_speeds)) if smoothed_speeds else 0.0

        return {
            "track_id": track_id,
            "total_distance_px": round(total_distance, 2),
            "avg_speed_px_s": round(avg_speed, 2),
            "max_speed_px_s": round(max_speed, 2),
            "duration_sec": round(duration, 2),
            "frame_count": len(sorted_records),
            "positions": positions,
            "speeds": [round(s, 2) for s in smoothed_speeds],
            "timestamps": [round(t, 2) for t in timestamps]
        }

    def _smooth_speeds(self, speeds: List[float]) -> List[float]:
        """Apply sliding box filter smoothing to raw velocity series."""
        if not speeds:
            return []
        if len(speeds) < self.smoothing_window:
            return speeds

        smoothed: List[float] = []
        w = self.smoothing_window
        half_w = w // 2

        for i in range(len(speeds)):
            start_idx = max(0, i - half_w)
            end_idx = min(len(speeds), i + half_w + 1)
            window_slice = speeds[start_idx:end_idx]
            smoothed.append(float(np.mean(window_slice)))

        return smoothed

    def analyze_all_players(
        self,
        tracking_history: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Analyze movement metrics for all tracked players in the tracking history.

        Args:
            tracking_history: List of player tracking dictionaries.

        Returns:
            Dictionary mapping track_id -> player metrics dictionary.
        """
        # Group records by track_id
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for record in tracking_history:
            t_id = record.get("track_id", -1)
            if t_id <= 0:
                continue  # Skip unassigned or negative IDs
            if t_id not in grouped:
                grouped[t_id] = []
            grouped[t_id].append(record)

        results: Dict[int, Dict[str, Any]] = {}
        for t_id, records in grouped.items():
            results[t_id] = self.calculate_player_metrics(records)

        return results

    def to_dataframe(self, analyzed_metrics: Optional[Dict[int, Dict[str, Any]]] = None) -> pd.DataFrame:
        """Convert player kinematics summary into a clean Pandas DataFrame."""
        if analyzed_metrics is None:
            analyzed_metrics = self.analyze_all_players(self.records)

        rows = []
        for t_id, m in sorted(analyzed_metrics.items()):
            rows.append({
                "Player ID": f"#{t_id}",
                "Tracking Duration (s)": m["duration_sec"],
                "Frames Tracked": m["frame_count"],
                "Total Distance (px)": m["total_distance_px"],
                "Average Speed (px/s)": m["avg_speed_px_s"],
                "Max Speed (px/s)": m["max_speed_px_s"]
            })
        return pd.DataFrame(rows)
