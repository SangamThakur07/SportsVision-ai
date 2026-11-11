"""
SportsVision AI - Phase 2 Tracking Verification Tests
Tests ByteTrack player tracker, BallTrack trajectory engine, and tracking visualizer.
"""

import os
import sys
import pytest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracking.tracker import SportsTracker, TrackedPlayer, BallTrack
from src.visualization.annotator import Annotator


def test_tracked_player_dataclass():
    """Verify TrackedPlayer initialization and coordinate calculations."""
    player = TrackedPlayer(
        track_id=7,
        bbox=[100.0, 200.0, 160.0, 320.0],
        confidence=0.89,
        frame_idx=10,
        timestamp=0.4,
        team="Team A"
    )
    p_dict = player.to_dict()
    assert p_dict["track_id"] == 7
    assert p_dict["center"] == (130.0, 260.0)
    assert p_dict["bottom_center"] == (130.0, 320.0)
    assert p_dict["team"] == "Team A"


def test_ball_track_lifecycle():
    """Verify BallTrack state transitions: DETECTED -> OCCLUDED -> LOST."""
    ball = BallTrack(max_trajectory_len=10)
    assert ball.status == "SEARCHING"
    assert len(ball.trajectory) == 0

    # 1. Detection
    ball.update(detected=True, bbox=[50.0, 50.0, 70.0, 70.0], confidence=0.75)
    assert ball.is_detected is True
    assert ball.status == "DETECTED"
    assert ball.current_center == (60.0, 60.0)
    assert len(ball.trajectory) == 1

    # 2. Temporary occlusion (under 30 frames)
    for _ in range(5):
        ball.update(detected=False)
    assert ball.is_detected is False
    assert ball.status == "OCCLUDED"
    assert ball.frames_lost == 5
    assert len(ball.trajectory) == 1  # Keeps recent trail

    # 3. Prolonged occlusion (> 30 frames)
    for _ in range(30):
        ball.update(detected=False)
    assert ball.status == "LOST"


def test_annotator_tracking_rendering():
    """Verify Annotator renders tracking IDs and trajectory polylines without crashing."""
    annotator = Annotator(show_hud=True, show_trajectories=True)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    tracking_result = {
        "frame_idx": 15,
        "timestamp": 0.6,
        "players": [
            {
                "track_id": 9,
                "bbox": [200, 250, 260, 370],
                "confidence": 0.85,
                "center": (230, 310),
                "bottom_center": (230, 370),
                "team": None
            }
        ],
        "player_trajectories": {
            9: [(210, 370), (215, 370), (220, 370), (230, 370)]
        },
        "ball": {
            "is_detected": True,
            "bbox": [400, 350, 420, 370],
            "center": (410, 360),
            "confidence": 0.80,
            "frames_lost": 0,
            "status": "DETECTED",
            "trajectory": [(380, 350), (395, 355), (410, 360)]
        }
    }

    annotated = annotator.annotate_tracking(frame, tracking_result, fps=25.0)
    assert annotated is not None
    assert annotated.shape == (720, 1280, 3)
    assert np.any(annotated > 0)


def test_sports_tracker_execution():
    """Verify SportsTracker updates state and returns structured tracking dictionary."""
    tracker = SportsTracker(device="cpu", confidence_threshold=0.25)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    result = tracker.update(dummy_frame, frame_idx=0, fps=25.0)
    assert "frame_idx" in result
    assert "timestamp" in result
    assert "players" in result
    assert "player_trajectories" in result
    assert "ball" in result


if __name__ == "__main__":
    pytest.main(["-v", __file__])
