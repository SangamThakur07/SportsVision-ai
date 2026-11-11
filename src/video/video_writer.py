"""
Video Writer Module
Handles robust encoding and saving of processed video frames using OpenCV.
"""

import os
from typing import Tuple, Optional
import cv2
import numpy as np


class VideoWriter:
    """
    OpenCV VideoWriter wrapper with automatic codec fallback and directory creation.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],  # (width, height)
        codec: str = "mp4v"
    ):
        """
        Initialize video writer.

        Args:
            output_path: Target video file path (.mp4 / .avi).
            fps: Video frames per second.
            frame_size: Tuple of (width, height) in pixels.
            codec: FourCC codec string (e.g., 'mp4v', 'avc1', 'XVID').
        """
        self.output_path = output_path
        self.fps = fps if fps > 0 else 25.0
        self.frame_size = frame_size  # (width, height)
        self.codec = codec

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            self.frame_size,
            isColor=True
        )

        if not self.writer.isOpened():
            # Codec fallback attempt: try XVID or fallback to avc1
            alt_codec = "XVID" if output_path.endswith(".avi") else "avc1"
            fourcc_alt = cv2.VideoWriter_fourcc(*alt_codec)
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc_alt,
                self.fps,
                self.frame_size,
                isColor=True
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not initialize OpenCV VideoWriter for: {output_path}")

        self.frames_written = 0

    def write_frame(self, frame: np.ndarray):
        """
        Write a single BGR frame to the output video.
        Resizes automatically if frame dimensions differ from initialized size.
        """
        if frame is None:
            return

        h, w = frame.shape[:2]
        if (w, h) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)

        self.writer.write(frame)
        self.frames_written += 1

    def release(self):
        """Finalize video stream and release file handles."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        # Re-encode to browser-compatible H.264 (yuv420p) if imageio_ffmpeg is available
        if self.output_path.endswith(".mp4") and os.path.exists(self.output_path):
            try:
                import imageio_ffmpeg
                import subprocess
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                tmp_path = self.output_path + ".tmp.mp4"
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                os.replace(self.output_path, tmp_path)
                cmd = [
                    ffmpeg_exe, "-y", "-i", tmp_path,
                    "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "22", "-preset", "veryfast",
                    self.output_path
                ]
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if res.returncode == 0 and os.path.exists(self.output_path):
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                else:
                    if os.path.exists(tmp_path):
                        os.replace(tmp_path, self.output_path)
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
