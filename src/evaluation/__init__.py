"""
Model Evaluation Package for SportsVision AI.
"""

from src.evaluation.metrics import (
    calculate_iou,
    evaluate_detection_batch,
    evaluate_classification,
    benchmark_resnet_vs_hsv
)

__all__ = [
    "calculate_iou",
    "evaluate_detection_batch",
    "evaluate_classification",
    "benchmark_resnet_vs_hsv"
]
