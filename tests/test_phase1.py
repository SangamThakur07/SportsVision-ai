"""
SportsVision AI - Phase 1 Verification Tests
Tests configuration, preprocessing, video streaming, annotation, and detection modules.
"""

import os
import sys
import pytest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.preprocessor import Preprocessor
from src.visualization.annotator import Annotator
from run_pipeline import load_config


def test_config_loading():
    """Verify master configuration loads correctly."""
    config = load_config("config/config.yaml")
    assert "detection" in config
    assert "visualization" in config
    assert "preprocessing" in config
    assert config["detection"]["confidence_threshold"] > 0
    assert 0 in config["detection"]["target_classes"]


def test_preprocessor_resizing():
    """Verify aspect-ratio and letterbox resizing."""
    prep = Preprocessor(target_size=(360, 640), color_space="RGB", normalize=False)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    resized = prep.resize(dummy_frame, (360, 640))
    assert resized.shape == (360, 640, 3)

    letterboxed = prep.resize(dummy_frame, (360, 640), preserve_aspect_ratio=True)
    assert letterboxed.shape == (360, 640, 3)


def test_preprocessor_color_and_norm():
    """Verify BGR to RGB and normalization scaling."""
    prep = Preprocessor(color_space="RGB", normalize=True)
    bgr_img = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)  # Pure Blue in BGR

    rgb_img = prep.convert_color(bgr_img, "RGB")
    # In RGB, blue is channel 2
    assert rgb_img[0, 0, 2] == 255
    assert rgb_img[0, 0, 0] == 0

    norm_img = prep.normalize_tensor(bgr_img)
    assert norm_img.dtype == np.float32


def test_preprocessor_crop_extraction():
    """Verify safe bounding box crop extraction."""
    prep = Preprocessor()
    frame = np.ones((400, 600, 3), dtype=np.uint8) * 128
    bbox = [50, 60, 150, 200]

    crop = prep.extract_crop(frame, bbox, target_size=(224, 224))
    assert crop is not None
    assert crop.shape == (224, 224, 3)

    # Test out-of-bound bbox handling
    oob_bbox = [-20, -10, 700, 500]
    crop_oob = prep.extract_crop(frame, oob_bbox, target_size=(224, 224))
    assert crop_oob is not None
    assert crop_oob.shape == (224, 224, 3)


def test_annotator_rendering():
    """Verify visual annotation and HUD drawing."""
    annotator = Annotator(show_hud=True)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    detections = [
        {
            "frame_idx": 1,
            "class_id": 0,
            "class_name": "player",
            "confidence": 0.88,
            "bbox": [100, 100, 200, 300],
            "center": (150, 200),
            "bottom_center": (150, 300)
        },
        {
            "frame_idx": 1,
            "class_id": 32,
            "class_name": "ball",
            "confidence": 0.75,
            "bbox": [500, 400, 530, 430],
            "center": (515, 415),
            "bottom_center": (515, 430)
        }
    ]

    annotated = annotator.annotate_frame(frame, detections, frame_idx=1, fps=25.0)
    assert annotated is not None
    assert annotated.shape == (720, 1280, 3)
    # Check that pixels were actually modified (not all zero anymore)
    assert np.any(annotated > 0)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
