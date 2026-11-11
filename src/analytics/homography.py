"""
Perspective Transformation & Pitch Homography Module
Maps camera-view image coordinates to calibrated 2D pitch coordinates in real-world meters.
Enables metric distance (meters) and physical speed (km/h) estimation.
Directly aligns with Computer Vision Syllabus Unit I (Geometric Transformations & Homography).
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2


class PitchHomography:
    """
    Computes and applies perspective transformation homography matrix H
    mapping 2D camera image coordinates to 2D pitch coordinates in meters.
    """

    # Standard FIFA football pitch dimensions in meters (105m x 68m)
    PITCH_LENGTH_METERS = 105.0
    PITCH_WIDTH_METERS = 68.0

    def __init__(
        self,
        src_points: Optional[List[Tuple[float, float]]] = None,
        dst_points: Optional[List[Tuple[float, float]]] = None,
        frame_width: int = 1280,
        frame_height: int = 720
    ):
        """
        Args:
            src_points: 4 camera frame coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            dst_points: 4 corresponding physical pitch coordinates in meters.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Default standard 4-point broadcast calibration if not provided
        if src_points is None:
            # 4 pitch touchline corners in camera frame
            self.src_points = np.array([
                [60.0, 40.0],                        # Top-Left
                [frame_width - 60.0, 40.0],          # Top-Right
                [frame_width - 60.0, frame_height - 40.0],  # Bottom-Right
                [60.0, frame_height - 40.0]           # Bottom-Left
            ], dtype=np.float32)
        else:
            self.src_points = np.array(src_points, dtype=np.float32)

        if dst_points is None:
            # Pitch rectangle in meters [0..105, 0..68]
            self.dst_points = np.array([
                [0.0, 0.0],
                [self.PITCH_LENGTH_METERS, 0.0],
                [self.PITCH_LENGTH_METERS, self.PITCH_WIDTH_METERS],
                [0.0, self.PITCH_WIDTH_METERS]
            ], dtype=np.float32)
        else:
            self.dst_points = np.array(dst_points, dtype=np.float32)

        # Compute 3x3 Homography Matrix H
        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.H_inv = np.linalg.inv(self.H)

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Map a single (x, y) pixel coordinate to real-world (pitch_x_m, pitch_y_m) in meters.
        """
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        tx = float(transformed[0, 0, 0])
        ty = float(transformed[0, 0, 1])
        return (round(tx, 2), round(ty, 2))

    def transform_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Map a list of (x, y) pixel coordinates to metric pitch coordinates.
        """
        if not points:
            return []
        arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(arr, self.H)
        return [(round(float(p[0, 0]), 2), round(float(p[0, 1]), 2)) for p in transformed]

    def calculate_physical_distance(
        self,
        pt1: Tuple[float, float],
        pt2: Tuple[float, float]
    ) -> float:
        """
        Calculate Euclidean distance in real-world meters between two camera coordinates.
        """
        m1 = self.transform_point(*pt1)
        m2 = self.transform_point(*pt2)
        dx = m2[0] - m1[0]
        dy = m2[1] - m1[1]
        return float(np.sqrt(dx * dx + dy * dy))

    def calculate_speed_kmh(
        self,
        pt1: Tuple[float, float],
        pt2: Tuple[float, float],
        dt_seconds: float
    ) -> float:
        """
        Calculate physical speed in kilometers per hour (km/h).
        """
        if dt_seconds <= 0:
            return 0.0
        dist_m = self.calculate_physical_distance(pt1, pt2)
        speed_mps = dist_m / dt_seconds
        speed_kmh = speed_mps * 3.6  # 1 m/s = 3.6 km/h
        return round(float(speed_kmh), 2)
