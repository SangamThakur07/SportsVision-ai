"""
Model Evaluation and Metrics Module
Implements Intersection over Union (IoU), detection evaluation, classification metrics
(Accuracy, Precision, Recall, F1, Confusion Matrix), and benchmark comparisons.
Directly aligns with Computer Vision Syllabus Unit IV & Evaluation Requirements.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans

from src.classification.team_classifier import ResNetFeatureExtractor, ColorHistogramBaseline


def calculate_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].

    Formula:
        IoU = Area(Overlap) / Area(Union)
    """
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    # Intersection rectangle coordinates
    x1_inter = max(x1_a, x1_b)
    y1_inter = max(y1_a, y1_b)
    x2_inter = min(x2_a, x2_b)
    y2_inter = min(y2_a, y2_b)

    inter_w = max(0.0, x2_inter - x1_inter)
    inter_h = max(0.0, y2_inter - y1_inter)
    area_inter = inter_w * inter_h

    # Union area
    area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
    area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)
    area_union = area_a + area_b - area_inter

    if area_union <= 0.0:
        return 0.0

    return float(area_inter / area_union)


def evaluate_detection_batch(
    pred_boxes: List[List[float]],
    gt_boxes: List[List[float]],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate object detection predictions against ground truth bounding boxes.
    """
    if not gt_boxes:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "mean_iou": 0.0,
            "true_positives": 0,
            "false_positives": len(pred_boxes),
            "false_negatives": 0,
            "total_ground_truth": 0
        }

    matched_gt = set()
    true_positives = 0
    ious = []

    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = calculate_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            true_positives += 1
            matched_gt.add(best_gt_idx)
            ious.append(best_iou)

    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - len(matched_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mean_iou": round(mean_iou, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_ground_truth": len(gt_boxes)
    }


def evaluate_classification(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute actual classification evaluation metrics:
    Accuracy, Macro Precision, Macro Recall, Macro F1, and Confusion Matrix.
    """
    unique_labels = labels or sorted(list(set(y_true) | set(y_pred)))
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=unique_labels, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=unique_labels, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=unique_labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "labels": unique_labels,
        "confusion_matrix": cm.tolist()
    }


def benchmark_resnet_vs_hsv(
    crops: List[np.ndarray],
    ground_truth_labels: List[str]
) -> pd.DataFrame:
    """
    Benchmark Transfer Learning (ResNet-18) against HSV Color Histogram baseline.
    """
    if not crops or len(crops) != len(ground_truth_labels):
        raise ValueError("Crops and labels must be non-empty and of matching length.")

    unique_classes = sorted(list(set(ground_truth_labels)))
    k = len(unique_classes)

    # 1. Evaluate ResNet-18 Transfer Learning
    extractor = ResNetFeatureExtractor()
    resnet_feats = extractor.extract(crops)
    kmeans_resnet = KMeans(n_clusters=k, random_state=42, n_init=10)
    resnet_clusters = kmeans_resnet.fit_predict(resnet_feats)

    # Map clusters to true labels via majority vote
    mapped_resnet_pred = _map_clusters_to_truth(resnet_clusters, ground_truth_labels, k)
    metrics_resnet = evaluate_classification(ground_truth_labels, mapped_resnet_pred)

    # 2. Evaluate HSV Histogram Baseline
    hsv_feats = np.array([ColorHistogramBaseline.extract_histogram(c) for c in crops])
    kmeans_hsv = KMeans(n_clusters=k, random_state=42, n_init=10)
    hsv_clusters = kmeans_hsv.fit_predict(hsv_feats)

    mapped_hsv_pred = _map_clusters_to_truth(hsv_clusters, ground_truth_labels, k)
    metrics_hsv = evaluate_classification(ground_truth_labels, mapped_hsv_pred)

    comparison = [
        {
            "Model / Method": "Pretrained ResNet-18 (Transfer Learning)",
            "Feature Representation": "Deep 512-dim Conv Embeddings",
            "Accuracy": f"{metrics_resnet['accuracy'] * 100:.1f}%",
            "Precision": f"{metrics_resnet['precision'] * 100:.1f}%",
            "Recall": f"{metrics_resnet['recall'] * 100:.1f}%",
            "F1-Score": f"{metrics_resnet['f1_score'] * 100:.1f}%"
        },
        {
            "Model / Method": "HSV Color Histogram Baseline",
            "Feature Representation": "8x8x8 3D Color Bins",
            "Accuracy": f"{metrics_hsv['accuracy'] * 100:.1f}%",
            "Precision": f"{metrics_hsv['precision'] * 100:.1f}%",
            "Recall": f"{metrics_hsv['recall'] * 100:.1f}%",
            "F1-Score": f"{metrics_hsv['f1_score'] * 100:.1f}%"
        }
    ]

    return pd.DataFrame(comparison)


def _map_clusters_to_truth(cluster_labels: np.ndarray, true_labels: List[str], k: int) -> List[str]:
    """Map unsupervised cluster indices to ground truth class labels via majority voting."""
    cluster_mapping = {}
    for c in range(k):
        members = [true_labels[i] for i, cl in enumerate(cluster_labels) if cl == c]
        if members:
            # Majority class in this cluster
            maj = max(set(members), key=members.count)
            cluster_mapping[c] = maj
        else:
            cluster_mapping[c] = "Unknown"

    return [cluster_mapping[c] for c in cluster_labels]
