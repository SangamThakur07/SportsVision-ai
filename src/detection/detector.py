"""
YOLO Object Detection Module
Specialized for detecting football players and the ball using YOLOv8/YOLO11.
Extracts bounding boxes, confidence scores, centroids, and pitch ground-contact points.
Directly aligns with Computer Vision Syllabus Unit IV.
"""

from typing import List, Dict, Any, Optional
import os
import torch
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """
    Wrapper around YOLO for football player and ball detection.
    """

    # COCO Dataset standard class IDs:
    # 0 = person, 32 = sports ball
    DEFAULT_CLASS_MAPPING = {
        0: "player",
        32: "ball"
    }

    def __init__(
        self,
        model_path: str = "models/weights/yolov8n.pt",
        confidence_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        target_classes: Optional[Dict[int, str]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the YOLO detector.

        Args:
            model_path: Path or name of YOLO weights (e.g. 'yolov8n.pt').
            confidence_threshold: Minimum confidence score for detection.
            iou_threshold: IoU threshold for Non-Maximum Suppression (NMS).
            target_classes: Dict mapping class ID to custom name.
            device: 'cuda', 'cpu', or None for auto-detection.
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or self.DEFAULT_CLASS_MAPPING

        # Hardware selection: prioritize GPU if available
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

        # Ensure directory for weights exists
        weights_dir = os.path.dirname(os.path.abspath(model_path))
        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)

        # Initialize model (Ultralytics auto-downloads weights if not found locally)
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect_frame(self, frame: np.ndarray, frame_idx: int = 0) -> List[Dict[str, Any]]:
        """
        Run object detection on a single frame.

        Args:
            frame: Input BGR frame (numpy array).
            frame_idx: Frame index in the video sequence.

        Returns:
            List of detection dictionaries containing:
            - frame_idx: int
            - class_id: int
            - class_name: str ('player' or 'ball')
            - confidence: float
            - bbox: [x1, y1, x2, y2]
            - center: (cx, cy)
            - bottom_center: (cx, y2)  [player ground-contact feet position]
        """
        target_class_ids = list(self.target_classes.keys())

        # Perform YOLO inference with NMS
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=target_class_ids,
            device=self.device,
            verbose=False
        )

        detections: List[Dict[str, Any]] = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # Extract coordinates, confidences, and class IDs
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            cls_id = int(classes[i])
            conf = float(confs[i])
            x1, y1, x2, y2 = [round(float(v), 2) for v in xyxy[i]]

            # Mathematical center of the bounding box
            cx = round((x1 + x2) / 2.0, 2)
            cy = round((y1 + y2) / 2.0, 2)

            # Bottom center (crucial for pitch coordinate projection)
            bottom_x = cx
            bottom_y = y2

            class_name = self.target_classes.get(cls_id, f"class_{cls_id}")

            detections.append({
                "frame_idx": frame_idx,
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "bottom_center": (bottom_x, bottom_y)
            })

        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        start_frame_idx: int = 0
    ) -> List[List[Dict[str, Any]]]:
        """
        Run batch detection on a list of frames.
        """
        target_class_ids = list(self.target_classes.keys())
        results = self.model.predict(
            source=frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=target_class_ids,
            device=self.device,
            verbose=False
        )

        batch_detections: List[List[Dict[str, Any]]] = []
        for offset, res in enumerate(results):
            f_idx = start_frame_idx + offset
            frame_dets: List[Dict[str, Any]] = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy().astype(int)

                for i in range(len(xyxy)):
                    cls_id = int(classes[i])
                    conf = float(confs[i])
                    x1, y1, x2, y2 = [round(float(v), 2) for v in xyxy[i]]
                    cx = round((x1 + x2) / 2.0, 2)
                    cy = round((y1 + y2) / 2.0, 2)

                    frame_dets.append({
                        "frame_idx": f_idx,
                        "class_id": cls_id,
                        "class_name": self.target_classes.get(cls_id, f"class_{cls_id}"),
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "center": (cx, cy),
                        "bottom_center": (cx, y2)
                    })
            batch_detections.append(frame_dets)

        return batch_detections
