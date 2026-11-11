"""
SportsVision AI - Advanced Features Verification Tests
Tests PitchHomography and GradCAM explainability modules.
"""

import os
import sys
import pytest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics.homography import PitchHomography
from src.evaluation.gradcam import GradCAM


def test_pitch_homography_projection():
    """Verify camera to metric pitch coordinate projection."""
    homography = PitchHomography(frame_width=1280, frame_height=720)

    # Top-Left corner (60, 40) should project to near (0, 0) meters
    tx, ty = homography.transform_point(60.0, 40.0)
    assert abs(tx - 0.0) < 1.0
    assert abs(ty - 0.0) < 1.0

    # Bottom-Right corner (1220, 680) should project to near (105, 68) meters
    tx_br, ty_br = homography.transform_point(1220.0, 680.0)
    assert abs(tx_br - 105.0) < 1.0
    assert abs(ty_br - 68.0) < 1.0


def test_physical_speed_calculation():
    """Verify physical distance in meters and speed in km/h."""
    homography = PitchHomography(frame_width=1280, frame_height=720)

    pt1 = (60.0, 40.0)
    pt2 = (1220.0, 40.0)  # Length of pitch = 105m

    dist_m = homography.calculate_physical_distance(pt1, pt2)
    assert abs(dist_m - 105.0) < 1.0

    # If covered in 10.5 seconds, speed = 10 m/s = 36.0 km/h
    speed_kmh = homography.calculate_speed_kmh(pt1, pt2, dt_seconds=10.5)
    assert abs(speed_kmh - 36.0) < 1.0


def test_gradcam_execution(tmp_path):
    """Verify Grad-CAM saliency map generation and explanation saving."""
    gradcam = GradCAM(device="cpu")

    # Synthetic player crop (224x224x3)
    dummy_crop = np.full((224, 224, 3), (40, 40, 220), dtype=np.uint8)  # Red jersey

    heatmap, overlay = gradcam.generate_heatmap(dummy_crop)
    assert heatmap.shape == (224, 224)
    assert overlay.shape == (224, 224, 3)
    assert np.max(heatmap) <= 1.0
    assert np.min(heatmap) >= 0.0

    out_file = str(tmp_path / "gradcam_test.png")
    gradcam.save_explanation(dummy_crop, out_file)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000


if __name__ == "__main__":
    pytest.main(["-v", __file__])
