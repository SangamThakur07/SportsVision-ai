"""
Video Loader Module
Handles video ingest, metadata extraction, and memory-safe frame streaming using OpenCV.
"""

import os
from typing import Generator, Tuple, Dict, Any, Optional
import cv2
import numpy as np


class VideoLoader:
    """
    Memory-efficient video loader providing frame-by-frame streaming
    and comprehensive metadata extraction.
    """

    def __init__(self, video_path: str):
        """
        Initialize the video loader with a given video path.

        Args:
            video_path: Path to the input video file.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video stream: {video_path}")

        # Extract video metadata
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 25.0  # Fallback standard FPS
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.total_frames > 0 else 0.0

    def get_info(self) -> Dict[str, Any]:
        """
        Return metadata summary dictionary for the video.
        """
        return {
            "file_path": self.video_path,
            "filename": os.path.basename(self.video_path),
            "width": self.width,
            "height": self.height,
            "resolution": f"{self.width}x{self.height}",
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration, 2),
            "duration_formatted": f"{int(self.duration // 60):02d}:{int(self.duration % 60):02d}"
        }

    def stream_frames(
        self,
        start_frame: int = 0,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Memory-safe frame-by-frame generator.
        Yields (frame_idx, frame_bgr) without loading all frames into RAM.

        Args:
            start_frame: Frame index to begin streaming from.
            max_frames: Optional upper limit on frames to yield.
        """
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        frames_yielded = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break

            yield frame_idx, frame
            frame_idx += 1
            frames_yielded += 1

            if max_frames is not None and frames_yielded >= max_frames:
                break

    def reset(self):
        """Reset video capture to frame 0."""
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        """Release underlying OpenCV VideoCapture resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
