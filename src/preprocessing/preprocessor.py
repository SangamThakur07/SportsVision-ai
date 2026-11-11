"""
Image Preprocessing Module
Implements fundamental Computer Vision preprocessing operations:
resizing, color conversions, letterboxing, and normalization.
Directly aligns with Computer Vision Syllabus Unit I.
"""

from typing import Tuple, Optional, Union, List
import cv2
import numpy as np


class Preprocessor:
    """
    Modular preprocessing utility for video frames and player image crops.
    """

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        color_space: str = "RGB",
        normalize: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ):
        """
        Args:
            target_size: Optional (height, width) to resize frames.
            color_space: Target color format: 'RGB', 'BGR', or 'HSV'.
            normalize: Whether to scale pixel intensities to [0.0, 1.0].
            mean: Optional list of channel means for z-score standardization.
            std: Optional list of channel stds for z-score standardization.
        """
        self.target_size = target_size
        self.color_space = color_space.upper()
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32) if mean else np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array(std, dtype=np.float32) if std else np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def convert_color(self, image: np.ndarray, target_color: Optional[str] = None) -> np.ndarray:
        """
        Convert image from OpenCV default BGR to specified color space.
        """
        fmt = (target_color or self.color_space).upper()
        if fmt == "RGB":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif fmt == "HSV":
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif fmt == "GRAY":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif fmt == "BGR":
            return image
        else:
            raise ValueError(f"Unsupported color space: {fmt}")

    def resize(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        preserve_aspect_ratio: bool = False
    ) -> np.ndarray:
        """
        Resize image to target_size (height, width).
        """
        size = target_size or self.target_size
        if size is None:
            return image

        target_h, target_w = size
        h, w = image.shape[:2]

        if (h, w) == (target_h, target_w):
            return image

        if not preserve_aspect_ratio:
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Aspect-ratio preserving letterbox
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad with gray background (114, 114, 114) standard for YOLO
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas

    def normalize_tensor(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize uint8 image [0, 255] to float32 [0.0, 1.0],
        and optionally apply channel standardization.
        """
        img_float = image.astype(np.float32) / 255.0
        if self.mean is not None and self.std is not None and img_float.ndim == 3:
            img_float = (img_float - self.mean) / self.std
        return img_float

    def extract_crop(
        self,
        frame: np.ndarray,
        bbox: Union[List[int], Tuple[int, int, int, int]],
        target_size: Optional[Tuple[int, int]] = (224, 224)
    ) -> Optional[np.ndarray]:
        """
        Safely extract a bounding box crop [x1, y1, x2, y2] from frame,
        handling boundaries and resizing if requested.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        # Clip to image boundaries
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return None

        if target_size is not None:
            crop = cv2.resize(crop, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

        return crop

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline execution on a single frame.
        """
        processed = self.resize(frame)
        processed = self.convert_color(processed)
        if self.normalize:
            processed = self.normalize_tensor(processed)
        return processed
