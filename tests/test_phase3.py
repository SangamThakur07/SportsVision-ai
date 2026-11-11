"""
SportsVision AI - Phase 3 Analytics Verification Tests
Tests MovementAnalyzer, HeatmapGenerator, PossessionEstimator, and TeamAnalytics modules.
"""

import os
import sys
import pytest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics.movement import MovementAnalyzer
from src.analytics.heatmap import HeatmapGenerator
from src.analytics.possession import PossessionEstimator
from src.analytics.team_analytics import TeamAnalytics


def test_movement_analyzer_metrics():
    """Verify player displacement, smoothed speed, and distance calculations."""
    analyzer = MovementAnalyzer(fps=25.0, smoothing_window=3)

    # Simulated linear player run: 10 pixels per frame across 10 frames
    records = [
        {"track_id": 7, "frame_idx": i, "x": 100.0 + i * 10.0, "y": 200.0, "timestamp": i * 0.04}
        for i in range(10)
    ]

    metrics = analyzer.calculate_player_metrics(records)
    assert metrics["track_id"] == 7
    assert metrics["frame_count"] == 10
    # Total distance: 9 steps of 10 pixels = 90.0 pixels
    assert abs(metrics["total_distance_px"] - 90.0) < 1.0
    # Speed: 10 pixels / 0.04 sec = 250 px/s
    assert abs(metrics["avg_speed_px_s"] - 250.0) < 10.0
    assert metrics["duration_sec"] > 0


def test_movement_analyzer_dataframe():
    """Verify multi-player kinematics DataFrame export."""
    analyzer = MovementAnalyzer(fps=25.0)
    history = [
        {"track_id": 1, "frame_idx": 0, "x": 10.0, "y": 20.0, "timestamp": 0.0},
        {"track_id": 1, "frame_idx": 1, "x": 15.0, "y": 20.0, "timestamp": 0.04},
        {"track_id": 2, "frame_idx": 0, "x": 100.0, "y": 200.0, "timestamp": 0.0},
        {"track_id": 2, "frame_idx": 1, "x": 102.0, "y": 200.0, "timestamp": 0.04}
    ]
    analyzed = analyzer.analyze_all_players(history)
    assert 1 in analyzed
    assert 2 in analyzed

    df = analyzer.to_dataframe(analyzed)
    assert len(df) == 2
    assert "Player ID" in df.columns
    assert "Total Distance (px)" in df.columns


def test_heatmap_generation(tmp_path):
    """Verify 2D Gaussian density calculation and heatmap rendering."""
    generator = HeatmapGenerator(pitch_width=640, pitch_height=360, sigma=15.0)

    # Verify pitch template
    pitch = generator.draw_pitch_template()
    assert pitch.shape == (360, 640, 3)

    # Simulated positions centered around penalty box
    positions = [(150.0 + np.random.randn() * 10, 180.0 + np.random.randn() * 10) for _ in range(30)]

    density = generator.generate_density_map(positions)
    assert density.shape == (360, 640)
    assert np.max(density) == 1.0

    blended = generator.render_heatmap(positions, title="Player #7 Heatmap")
    assert blended.shape == (360, 640, 3)

    out_file = str(tmp_path / "test_heatmap.png")
    generator.save_heatmap(positions, out_file, title="Test Heatmap")
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000


def test_possession_estimation():
    """Verify proximity-based estimated ball possession calculations."""
    estimator = PossessionEstimator(proximity_threshold_px=50.0)

    # Frame 1: Ball close to Player #7 (dist = 20 px)
    ball_1 = {"is_detected": True, "center": (100.0, 100.0)}
    players_1 = [{"track_id": 7, "bottom_center": (100.0, 120.0), "center": (100.0, 110.0)}]
    rec_1 = estimator.update_frame(0, 0.0, ball_1, players_1, {7: "Team Blue"})
    assert rec_1["possessing_player_id"] == 7
    assert rec_1["possessing_team"] == "Team Blue"
    assert rec_1["status"] == "CONTROLLED"

    # Frame 2: Ball far from Player #8 (dist = 200 px) -> Loose Ball
    ball_2 = {"is_detected": True, "center": (500.0, 500.0)}
    players_2 = [{"track_id": 8, "bottom_center": (100.0, 100.0), "center": (100.0, 90.0)}]
    rec_2 = estimator.update_frame(1, 0.04, ball_2, players_2, {8: "Team Red"})
    assert rec_2["possessing_player_id"] is None
    assert rec_2["status"] == "LOOSE_BALL"

    summary = estimator.compute_summary()
    assert summary["team_a_pct"] + summary["team_b_pct"] == 100.0
    assert "methodology" in summary


def test_team_spatial_analytics():
    """Verify team centroid, spread, and convex hull territory calculations."""
    positions = [(100.0, 200.0), (200.0, 200.0), (150.0, 300.0), (150.0, 100.0)]

    centroid = TeamAnalytics.calculate_centroid(positions)
    assert centroid == (150.0, 200.0)

    spread = TeamAnalytics.calculate_team_spread(positions)
    assert spread > 0.0

    territory = TeamAnalytics.calculate_territory_area(positions)
    # Area of diamond with diagonals 100 and 200 is 0.5 * 100 * 200 = 10000
    assert abs(territory - 10000.0) < 100.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
