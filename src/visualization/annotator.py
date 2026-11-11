"""
Visual Annotation Module
Renders professional bounding boxes, persistent player ID badges, motion trajectories,
ball flight trails, and match summary HUD onto video frames.
Supports both Phase 1 (Detection) and Phase 2 (Tracking).
"""

from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np


class Annotator:
    """
    Renders visual detections, tracking IDs, motion trails, and analytics HUD onto frames.
    """

    # High-contrast color palette for distinct player IDs
    PALETTE = [
        (0, 255, 128),   # Emerald Green
        (255, 128, 0),   # Sky Blue
        (0, 215, 255),   # Gold / Amber
        (255, 0, 128),   # Magenta
        (0, 255, 255),   # Yellow
        (255, 255, 0),   # Cyan
        (180, 105, 255), # Soft Pink
        (50, 205, 50),   # Lime
        (238, 130, 238), # Violet
        (0, 165, 255),   # Deep Orange
        (220, 20, 60),   # Crimson
        (127, 255, 0)    # Chartreuse
    ]

    def __init__(
        self,
        player_color: Tuple[int, int, int] = (0, 255, 128),     # BGR: Emerald Green
        ball_color: Tuple[int, int, int] = (0, 140, 255),       # BGR: Vivid Orange
        referee_color: Tuple[int, int, int] = (0, 255, 255),    # BGR: Yellow
        box_thickness: int = 2,
        font_scale: float = 0.55,
        text_thickness: int = 2,
        show_confidence: bool = True,
        show_center_point: bool = True,
        show_hud: bool = True,
        show_trajectories: bool = True
    ):
        self.player_color = player_color
        self.ball_color = ball_color
        self.referee_color = referee_color
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.text_thickness = text_thickness
        self.show_confidence = show_confidence
        self.show_center_point = show_center_point
        self.show_hud = show_hud
        self.show_trajectories = show_trajectories
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Return consistent distinct BGR color for a given track ID."""
        if track_id < 0:
            return (180, 180, 180)
        return self.PALETTE[track_id % len(self.PALETTE)]

    def get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Return BGR color tuple based on detection class."""
        name = class_name.lower()
        if "player" in name:
            return self.player_color
        elif "ball" in name:
            return self.ball_color
        elif "referee" in name:
            return self.referee_color
        return (200, 200, 200)

    def draw_badge(
        self,
        frame: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        bg_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int] = (0, 0, 0)
    ):
        """Draw a filled text badge above or inside a bounding box."""
        x, y = pos
        (w, h), _ = cv2.getTextSize(text, self.font, self.font_scale, self.text_thickness)
        pad = 4

        # Background rectangle
        cv2.rectangle(
            frame,
            (x, max(0, y - h - 2 * pad)),
            (x + w + 2 * pad, max(h + 2 * pad, y)),
            bg_color,
            -1
        )
        # Foreground text
        cv2.putText(
            frame,
            text,
            (x + pad, max(h + pad, y - pad)),
            self.font,
            self.font_scale,
            text_color,
            self.text_thickness,
            lineType=cv2.LINE_AA
        )

    def draw_detection(self, frame: np.ndarray, det: Dict[str, Any]):
        """Draw a single detection on the frame (Phase 1)."""
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_name = det["class_name"]
        conf = det["confidence"]
        color = self.get_color(cls_name)

        if "ball" in cls_name.lower():
            cx, cy = map(int, det["center"])
            radius = max(8, int((x2 - x1) / 2) + 4)
            cv2.circle(frame, (cx, cy), radius, color, self.box_thickness + 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)
            label = f"Ball {conf:.2f}" if self.show_confidence else "Ball"
            self.draw_badge(frame, label, (x1, y1 - 4), color, (255, 255, 255))
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness, cv2.LINE_AA)
            if self.show_center_point:
                bx, by = map(int, det["bottom_center"])
                cv2.circle(frame, (bx, by), 4, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (bx, by), 6, (0, 0, 0), 1, cv2.LINE_AA)
            label = f"{cls_name.capitalize()}"
            if self.show_confidence:
                label += f" {conf:.2f}"
            self.draw_badge(frame, label, (x1, y1 - 2), color, (0, 0, 0))

    def draw_player_track(
        self,
        frame: np.ndarray,
        player: Dict[str, Any],
        trajectory: Optional[List[Tuple[float, float]]] = None
    ):
        """Draw tracked player with persistent ID badge and pitch motion trail."""
        track_id = player.get("track_id", -1)
        x1, y1, x2, y2 = map(int, player["bbox"])
        conf = player.get("confidence", 0.0)
        color = self.get_track_color(track_id)

        # Draw motion trajectory trail (feet coordinates)
        if self.show_trajectories and trajectory and len(trajectory) > 1:
            pts = [tuple(map(int, pt)) for pt in trajectory]
            num_pts = len(pts)
            for i in range(1, num_pts):
                # Fade alpha: older points are thinner
                alpha = i / num_pts
                thickness = max(1, int(3 * alpha))
                cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness, cv2.LINE_AA)

        # Ground-contact marker (player feet)
        if self.show_center_point:
            bx, by = map(int, player["bottom_center"])
            cv2.circle(frame, (bx, by), 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (bx, by), 7, (0, 0, 0), 1, cv2.LINE_AA)

        # ID Badge
        id_str = f"#{track_id}" if track_id > 0 else "Player"
        if player.get("team"):
            label = f"{player['team']} {id_str}"
        else:
            label = f"Player {id_str}"
        if self.show_confidence:
            label += f" {conf:.2f}"

        self.draw_badge(frame, label, (x1, y1 - 2), color, (0, 0, 0))

    def draw_ball_track(self, frame: np.ndarray, ball_data: Dict[str, Any]):
        """Draw ball detection and trajectory trail."""
        trajectory = ball_data.get("trajectory", [])
        is_detected = ball_data.get("is_detected", False)
        current_center = ball_data.get("center")
        color = self.ball_color

        # Draw ball trajectory trail
        if self.show_trajectories and len(trajectory) > 1:
            pts = [tuple(map(int, pt)) for pt in trajectory]
            num_pts = len(pts)
            for i in range(1, num_pts):
                alpha = i / num_pts
                thickness = max(1, int(3 * alpha))
                # Glowing orange-to-yellow trail
                cv2.line(frame, pts[i - 1], pts[i], (0, int(140 + 100 * alpha), 255), thickness, cv2.LINE_AA)
                if i % 3 == 0 or i == num_pts - 1:
                    cv2.circle(frame, pts[i], 2, (255, 255, 255), -1, cv2.LINE_AA)

        # Current ball highlight
        if is_detected and current_center is not None:
            cx, cy = map(int, current_center)
            bbox = ball_data.get("bbox")
            radius = max(8, int((bbox[2] - bbox[0]) / 2) + 4) if bbox else 10
            # Outer ring + center dot
            cv2.circle(frame, (cx, cy), radius, color, self.box_thickness + 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)
            conf = ball_data.get("confidence", 0.0)
            label = f"Ball {conf:.2f}" if self.show_confidence else "Ball"
            x1 = bbox[0] if bbox else cx - 10
            y1 = bbox[1] if bbox else cy - 10
            self.draw_badge(frame, label, (int(x1), int(y1) - 4), color, (255, 255, 255))

    def draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        num_players: int,
        ball_status: str,
        phase_label: str = "Tracking"
    ):
        """Draw sleek match information HUD banner at the top."""
        h, w = frame.shape[:2]
        hud_height = 42
        hud_overlay = frame.copy()

        cv2.rectangle(hud_overlay, (0, 0), (w, hud_height), (15, 15, 15), -1)
        cv2.addWeighted(hud_overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, hud_height), (w, hud_height), (0, 255, 128), 2)

        time_sec = frame_idx / fps if fps > 0 else 0.0
        time_str = f"Time: {int(time_sec // 60):02d}:{int(time_sec % 60):02d}.{int((time_sec % 1) * 10)}"

        cv2.putText(frame, f"SportsVision AI [{phase_label}]", (16, 26),
                    self.font, 0.6, (0, 255, 128), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {frame_idx:04d} | {time_str}", (340, 26),
                    self.font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Tracked Players: {num_players}", (w - 420, 26),
                    self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        b_color = (0, 140, 255) if "DETECTED" in ball_status or "FOUND" in ball_status else (
            (0, 215, 255) if "OCCLUDED" in ball_status else (120, 120, 120)
        )
        cv2.putText(frame, f"Ball: {ball_status}", (w - 200, 26),
                    self.font, 0.5, b_color, 2, cv2.LINE_AA)

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        frame_idx: int = 0,
        fps: float = 25.0
    ) -> np.ndarray:
        """Draw detections and HUD on frame (Phase 1)."""
        annotated = frame.copy()
        num_players = sum(1 for d in detections if "player" in d["class_name"].lower())
        ball_detected = any("ball" in d["class_name"].lower() for d in detections)
        for det in detections:
            self.draw_detection(annotated, det)
        if self.show_hud:
            ball_str = "FOUND" if ball_detected else "SEARCHING"
            self.draw_hud(annotated, frame_idx, fps, num_players, ball_str, "Detection")
        return annotated

    def annotate_tracking(
        self,
        frame: np.ndarray,
        tracking_result: Dict[str, Any],
        fps: float = 25.0
    ) -> np.ndarray:
        """
        Draw all tracked players with persistent IDs, motion trails,
        ball trajectory, and match status HUD (Phase 2).
        """
        annotated = frame.copy()
        players = tracking_result.get("players", [])
        trajectories = tracking_result.get("player_trajectories", {})
        ball_data = tracking_result.get("ball", {})
        frame_idx = tracking_result.get("frame_idx", 0)

        # 1. Draw player tracks and motion trails
        for player in players:
            t_id = player.get("track_id")
            traj = trajectories.get(t_id)
            self.draw_player_track(annotated, player, traj)

        # 2. Draw ball trajectory
        self.draw_ball_track(annotated, ball_data)

        # 3. Draw tracking HUD
        if self.show_hud:
            ball_status = ball_data.get("status", "SEARCHING")
            self.draw_hud(annotated, frame_idx, fps, len(players), ball_status, "ByteTrack MOT")

        return annotated
