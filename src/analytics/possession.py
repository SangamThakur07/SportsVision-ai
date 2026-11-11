"""
Estimated Ball Possession Analytics Module
Calculates spatial proximity between ball and players per frame to compute approximate
team and player possession statistics.
Directly aligns with Computer Vision Syllabus Unit I & V.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


class PossessionEstimator:
    """
    Computes estimated ball possession based on 2D Euclidean proximity between the ball
    and players across match video frames.
    """

    METHODOLOGY_NOTE = (
        "Estimated Ball Possession is calculated based on 2D camera-frame Euclidean proximity "
        "between the detected ball center and player foot contact points. "
        "It provides an automated approximation and does not account for 3D aerial flight, "
        "off-ball occlusion, or official FIFA match-tracking telemetry."
    )

    def __init__(
        self,
        proximity_threshold_px: float = 75.0,
        max_occlusion_hold_frames: int = 5,
        default_team_a_name: str = "Team Blue",
        default_team_b_name: str = "Team Red"
    ):
        """
        Args:
            proximity_threshold_px: Maximum distance (in pixels) for a player to be
                                     considered in control of the ball.
            max_occlusion_hold_frames: Hold last known possession if ball is lost for <= K frames.
        """
        self.proximity_threshold = proximity_threshold_px
        self.max_hold_frames = max_occlusion_hold_frames
        self.team_a_name = default_team_a_name
        self.team_b_name = default_team_b_name

        # State tracking
        self.frame_records: List[Dict[str, Any]] = []
        self.last_possessing_player: Optional[int] = None
        self.last_possessing_team: Optional[str] = None
        self.frames_since_ball_seen: int = 0

    def update_frame(
        self,
        frame_idx: int,
        timestamp: float,
        ball_data: Dict[str, Any],
        players: List[Dict[str, Any]],
        player_teams: Optional[Dict[int, str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate possession for a single frame.

        Args:
            frame_idx: Current frame index.
            timestamp: Elapsed time in seconds.
            ball_data: Ball tracker output dict (containing 'is_detected', 'center').
            players: List of tracked player dicts (containing 'track_id', 'bottom_center').
            player_teams: Optional mapping of track_id -> 'Team A' / 'Team B'.

        Returns:
            Frame possession record dictionary.
        """
        player_teams = player_teams or {}
        # Dynamically adopt discovered team names if present
        unique_teams = [t for t in set(player_teams.values()) if t]
        if len(unique_teams) >= 2:
            self.team_a_name = unique_teams[0]
            self.team_b_name = unique_teams[1]
        elif len(unique_teams) == 1:
            self.team_a_name = unique_teams[0]

        is_ball_detected = ball_data.get("is_detected", False)
        ball_center = ball_data.get("center")

        possessing_player = None
        possessing_team = None
        min_dist = float("inf")
        status = "NO_BALL"

        team_a_weight = 0.0
        team_b_weight = 0.0

        if is_ball_detected and ball_center is not None and players:
            self.frames_since_ball_seen = 0
            bx, by = ball_center

            for p in players:
                t_id = p.get("track_id")
                # Use ground-contact feet position if available, else centroid
                px, py = p.get("bottom_center", p.get("center", (0, 0)))

                dist = float(np.sqrt((px - bx) ** 2 + (py - by) ** 2))
                if dist < min_dist:
                    min_dist = dist
                    possessing_player = t_id

            if min_dist <= self.proximity_threshold:
                possessing_team = player_teams.get(possessing_player, self._heuristic_team(possessing_player))
                status = "CONTROLLED"
                self.last_possessing_player = possessing_player
                self.last_possessing_team = possessing_team
                if possessing_team == self.team_a_name:
                    team_a_weight = 1.0
                    team_b_weight = 0.0
                else:
                    team_a_weight = 0.0
                    team_b_weight = 1.0
            else:
                status = "LOOSE_BALL"
                possessing_player = None
                possessing_team = "In-Transit / Loose"
                team_a_weight = 0.5
                team_b_weight = 0.5
        else:
            self.frames_since_ball_seen += 1
            # Graceful hold for temporary ball occlusion
            if self.frames_since_ball_seen <= self.max_hold_frames and self.last_possessing_player is not None:
                possessing_player = self.last_possessing_player
                possessing_team = self.last_possessing_team
                status = "HELD_OVER_OCCLUSION"
                if possessing_team == self.team_a_name:
                    team_a_weight = 1.0
                    team_b_weight = 0.0
                else:
                    team_a_weight = 0.0
                    team_b_weight = 1.0
            else:
                # Proportional territorial dominance based on active player count
                team_a_players = [p for p in players if player_teams.get(p.get("track_id")) == self.team_a_name]
                team_b_players = [p for p in players if player_teams.get(p.get("track_id")) == self.team_b_name]
                cnt_a = len(team_a_players)
                cnt_b = len(team_b_players)
                if cnt_a + cnt_b > 0:
                    team_a_weight = cnt_a / (cnt_a + cnt_b)
                    team_b_weight = cnt_b / (cnt_a + cnt_b)
                    possessing_team = self.team_a_name if cnt_a >= cnt_b else self.team_b_name
                    status = "TERRITORIAL_DOMINANCE"
                else:
                    team_a_weight = 0.5
                    team_b_weight = 0.5
                    status = "SEARCHING"

        record = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "ball_detected": is_ball_detected,
            "possessing_player_id": possessing_player,
            "possessing_team": possessing_team,
            "team_a_weight": team_a_weight,
            "team_b_weight": team_b_weight,
            "min_distance_to_ball_px": round(min_dist, 2) if min_dist != float("inf") else None,
            "status": status
        }
        self.frame_records.append(record)
        return record

    def _heuristic_team(self, track_id: Optional[int]) -> str:
        """Fallback heuristic partition if teams are not yet classified."""
        if track_id is None:
            return "Unknown"
        # Deterministic partition for demonstration (odds: Team A, evens: Team B)
        return self.team_a_name if track_id % 2 == 1 else self.team_b_name

    def compute_summary(self) -> Dict[str, Any]:
        """
        Aggregate full match possession statistics.
        """
        if not self.frame_records:
            return {
                "team_a_pct": 50.0,
                "team_b_pct": 50.0,
                "team_a_name": self.team_a_name,
                "team_b_name": self.team_b_name,
                "team_a_frames": 0,
                "team_b_frames": 0,
                "loose_frames": 0,
                "total_frames": 0,
                "player_possession_seconds": {},
                "methodology": self.METHODOLOGY_NOTE
            }

        total_a_weight = sum(r.get("team_a_weight", 0.0) for r in self.frame_records)
        total_b_weight = sum(r.get("team_b_weight", 0.0) for r in self.frame_records)
        total_weight = total_a_weight + total_b_weight

        player_counts: Dict[int, int] = {}
        loose_count = 0

        for r in self.frame_records:
            p_id = r.get("possessing_player_id")
            if r.get("status") == "LOOSE_BALL":
                loose_count += 1
            if p_id is not None:
                player_counts[p_id] = player_counts.get(p_id, 0) + 1

        if total_weight > 0:
            team_a_pct = round((total_a_weight / total_weight) * 100.0, 1)
            team_b_pct = round(100.0 - team_a_pct, 1)
        else:
            team_a_pct = 50.0
            team_b_pct = 50.0

        # Convert frame counts to seconds
        total_frames = len(self.frame_records)
        fps = 25.0
        player_seconds = {
            f"Player #{pid}": round(cnt / fps, 2)
            for pid, cnt in sorted(player_counts.items(), key=lambda x: x[1], reverse=True)
        }

        return {
            "team_a_name": self.team_a_name,
            "team_b_name": self.team_b_name,
            "team_a_pct": team_a_pct,
            "team_b_pct": team_b_pct,
            "team_a_frames": int(round(total_a_weight)),
            "team_b_frames": int(round(total_b_weight)),
            "loose_frames": loose_count,
            "total_frames": total_frames,
            "player_possession_seconds": player_seconds,
            "methodology": self.METHODOLOGY_NOTE
        }

    def reset(self):
        """Reset internal accumulator state."""
        self.frame_records.clear()
        self.last_possessing_player = None
        self.last_possessing_team = None
        self.frames_since_ball_seen = 0
