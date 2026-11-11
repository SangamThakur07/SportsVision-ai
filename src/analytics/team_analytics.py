"""
Team Spatial Analytics Module
Computes team centroids, team dispersion / spacing, convex hull territory area,
and average player formations across match frames.
Directly aligns with Computer Vision Syllabus Unit I & V.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.spatial import ConvexHull


class TeamAnalytics:
    """
    Computes collective team positioning metrics (centroids, dispersion, territory area).
    """

    @staticmethod
    def calculate_centroid(positions: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """
        Calculate mathematical mean center of mass (centroid) for a team.
        """
        if not positions:
            return None
        arr = np.array(positions, dtype=np.float32)
        cx = float(np.mean(arr[:, 0]))
        cy = float(np.mean(arr[:, 1]))
        return (round(cx, 2), round(cy, 2))

    @staticmethod
    def calculate_team_spread(positions: List[Tuple[float, float]]) -> float:
        """
        Calculate mean distance of all team players from the team centroid (spacing metric).
        Higher value indicates stretched/wide formation; lower indicates compact defense.
        """
        if not positions or len(positions) < 2:
            return 0.0
        centroid = TeamAnalytics.calculate_centroid(positions)
        if centroid is None:
            return 0.0
        cx, cy = centroid
        dists = [np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in positions]
        return round(float(np.mean(dists)), 2)

    @staticmethod
    def calculate_territory_area(positions: List[Tuple[float, float]]) -> float:
        """
        Calculate convex hull area formed by player positions on the pitch (in square pixels).
        """
        if not positions or len(positions) < 3:
            return 0.0
        points = np.array(positions, dtype=np.float32)
        # Check if points are not collinear
        try:
            hull = ConvexHull(points)
            return round(float(hull.volume), 2)  # In 2D, hull.volume is area
        except Exception:
            return 0.0

    @staticmethod
    def compute_average_positions(
        tracking_history: List[Dict[str, Any]],
        player_teams: Optional[Dict[int, str]] = None
    ) -> Dict[str, Dict[int, Tuple[float, float]]]:
        """
        Compute average pitch position for each player, grouped by team.

        Returns:
            Dictionary: {'Team A': {track_id: (mean_x, mean_y)}, 'Team B': {...}}
        """
        player_teams = player_teams or {}
        player_coords: Dict[int, List[Tuple[float, float]]] = {}

        for rec in tracking_history:
            t_id = rec.get("track_id")
            if t_id is not None and t_id > 0:
                if t_id not in player_coords:
                    player_coords[t_id] = []
                player_coords[t_id].append((rec["x"], rec["y"]))

        teams_avg: Dict[str, Dict[int, Tuple[float, float]]] = {
            "Team A": {},
            "Team B": {}
        }

        for t_id, coords in player_coords.items():
            if not coords:
                continue
            mean_x = float(np.mean([c[0] for c in coords]))
            mean_y = float(np.mean([c[1] for c in coords]))
            team = player_teams.get(t_id, "Team A" if t_id % 2 == 1 else "Team B")
            if team not in teams_avg:
                teams_avg[team] = {}
            teams_avg[team][t_id] = (round(mean_x, 2), round(mean_y, 2))

        return teams_avg
