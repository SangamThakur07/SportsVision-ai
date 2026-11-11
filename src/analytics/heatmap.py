"""
2D Gaussian Kernel Density Heatmap Generator
Generates continuous movement heatmaps overlaid onto a standard football pitch canvas.
Directly aligns with Computer Vision Syllabus Unit I (Spatial Filtering & Distributions).
"""

from typing import List, Tuple, Optional, Union
import os
import cv2
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter


class HeatmapGenerator:
    """
    Renders 2D spatial density heatmaps over a football pitch diagram.
    """

    def __init__(
        self,
        pitch_width: int = 1280,
        pitch_height: int = 720,
        sigma: float = 28.0,
        colormap: str = "hot"
    ):
        """
        Args:
            pitch_width: Width of the output canvas in pixels.
            pitch_height: Height of the output canvas in pixels.
            sigma: Standard deviation for Gaussian kernel smoothing.
            colormap: Matplotlib colormap ('hot', 'inferno', 'jet').
        """
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.sigma = sigma
        self.colormap = colormap

    def draw_pitch_template(self) -> np.ndarray:
        """
        Draw a clean, professional football pitch template in BGR format.
        """
        w, h = self.pitch_width, self.pitch_height
        # Deep pitch green
        pitch = np.full((h, w, 3), (34, 110, 34), dtype=np.uint8)
        line_color = (235, 235, 235)
        thickness = 2

        pad_x = int(w * 0.05)
        pad_y = int(h * 0.05)
        pw = w - 2 * pad_x
        ph = h - 2 * pad_y

        # Touchlines & Goal lines
        cv2.rectangle(pitch, (pad_x, pad_y), (pad_x + pw, pad_y + ph), line_color, thickness)

        # Halfway line
        mid_x = pad_x + pw // 2
        cv2.line(pitch, (mid_x, pad_y), (mid_x, pad_y + ph), line_color, thickness)

        # Center circle
        circle_radius = int(ph * 0.18)
        cv2.circle(pitch, (mid_x, pad_y + ph // 2), circle_radius, line_color, thickness)
        cv2.circle(pitch, (mid_x, pad_y + ph // 2), 4, line_color, -1)

        # Left Penalty Box
        pen_w = int(pw * 0.16)
        pen_h = int(ph * 0.50)
        top_pen = pad_y + (ph - pen_h) // 2
        cv2.rectangle(pitch, (pad_x, top_pen), (pad_x + pen_w, top_pen + pen_h), line_color, thickness)

        # Left Goal Area
        goal_w = int(pw * 0.06)
        goal_h = int(ph * 0.25)
        top_goal = pad_y + (ph - goal_h) // 2
        cv2.rectangle(pitch, (pad_x, top_goal), (pad_x + goal_w, top_goal + goal_h), line_color, thickness)

        # Right Penalty Box
        cv2.rectangle(pitch, (pad_x + pw - pen_w, top_pen), (pad_x + pw, top_pen + pen_h), line_color, thickness)

        # Right Goal Area
        cv2.rectangle(pitch, (pad_x + pw - goal_w, top_goal), (pad_x + pw, top_goal + goal_h), line_color, thickness)

        return pitch

    def _project_positions_to_pitch(
        self,
        positions: List[Tuple[float, float]],
        frame_size: Optional[Tuple[int, int]] = None
    ) -> List[Tuple[float, float]]:
        """
        Scale and map arbitrary input coordinates (e.g. 1920x1080)
        cleanly inside the football pitch boundary canvas.
        """
        if not positions:
            return []

        w, h = self.pitch_width, self.pitch_height
        pad_x = int(w * 0.05)
        pad_y = int(h * 0.05)
        pw = w - 2 * pad_x
        ph = h - 2 * pad_y

        if frame_size is not None and frame_size[0] > 0 and frame_size[1] > 0:
            fw, fh = float(frame_size[0]), float(frame_size[1])
        else:
            max_x = max(p[0] for p in positions)
            max_y = max(p[1] for p in positions)
            fw = max(float(w), float(max_x))
            fh = max(float(h), float(max_y))

        projected = []
        for x, y in positions:
            nx = np.clip(float(x) / fw, 0.0, 1.0)
            ny = np.clip(float(y) / fh, 0.0, 1.0)
            px = pad_x + nx * pw
            py = pad_y + ny * ph
            projected.append((px, py))

        return projected

    def generate_density_map(
        self,
        positions: List[Tuple[float, float]],
        sigma: Optional[float] = None,
        frame_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Compute 2D Gaussian density distribution from a list of (x, y) coordinates.

        Returns:
            Normalized 2D density array of shape (pitch_height, pitch_width) in [0.0, 1.0].
        """
        w, h = self.pitch_width, self.pitch_height
        density = np.zeros((h, w), dtype=np.float32)

        if not positions:
            return density

        scaled_positions = self._project_positions_to_pitch(positions, frame_size=frame_size)

        # Accumulate 2D histogram
        for x, y in scaled_positions:
            ix = int(round(x))
            iy = int(round(y))
            if 0 <= ix < w and 0 <= iy < h:
                density[iy, ix] += 1.0

        # Apply 2D Gaussian filter
        kernel_sigma = sigma or self.sigma
        smoothed_density = gaussian_filter(density, sigma=kernel_sigma)

        # Normalize to [0.0, 1.0]
        max_val = np.max(smoothed_density)
        if max_val > 0:
            smoothed_density = smoothed_density / max_val

        return smoothed_density

    def render_heatmap(
        self,
        positions: List[Tuple[float, float]],
        title: Optional[str] = None,
        sigma: Optional[float] = None,
        alpha: float = 0.65,
        frame_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Render a blended heatmap over the football pitch canvas.

        Args:
            positions: List of (x, y) positions.
            title: Optional banner title text.
            sigma: Kernel size override.
            alpha: Maximum heatmap opacity overlay.
            frame_size: Source frame size (width, height) to scale coordinates correctly.

        Returns:
            BGR image array of shape (height, width, 3).
        """
        pitch = self.draw_pitch_template()
        if not positions:
            return pitch

        density = self.generate_density_map(positions, sigma=sigma, frame_size=frame_size)
        scaled_positions = self._project_positions_to_pitch(positions, frame_size=frame_size)

        # Get Matplotlib colormap
        cmap = getattr(cm, self.colormap, cm.hot)
        # Apply colormap: density is [0.0, 1.0] -> RGBA float [0.0, 1.0]
        colored_float = cmap(density)[:, :, :3]  # RGB
        # Convert RGB to BGR uint8
        heatmap_bgr = (colored_float[:, :, ::-1] * 255.0).astype(np.uint8)

        # Non-linear alpha mask: low densities are fully transparent
        # Threshold at 0.05 to avoid washing out the whole pitch
        alpha_mask = np.clip((density - 0.05) / 0.95, 0.0, 1.0) * alpha
        alpha_3d = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)

        # Alpha composite blend
        blended = (alpha_3d * heatmap_bgr + (1.0 - alpha_3d) * pitch).astype(np.uint8)

        # Draw trajectory point dots (subtle white dots)
        for x, y in scaled_positions[::max(1, len(scaled_positions) // 30)]:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < self.pitch_width and 0 <= iy < self.pitch_height:
                cv2.circle(blended, (ix, iy), 3, (255, 255, 255), -1, cv2.LINE_AA)

        # Draw Title Banner
        if title:
            cv2.rectangle(blended, (0, 0), (self.pitch_width, 42), (20, 20, 20), -1)
            cv2.line(blended, (0, 42), (self.pitch_width, 42), (0, 255, 128), 2)
            cv2.putText(
                blended,
                title,
                (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            count_str = f"Positions: {len(positions)}"
            cv2.putText(
                blended,
                count_str,
                (self.pitch_width - 220, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

        return blended

    def save_heatmap(
        self,
        positions: List[Tuple[float, float]],
        output_path: str,
        title: Optional[str] = None,
        frame_size: Optional[Tuple[int, int]] = None
    ) -> str:
        """
        Generate and save heatmap image to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img = self.render_heatmap(positions, title=title, frame_size=frame_size)
        cv2.imwrite(output_path, img)
        return output_path
