"""
Synthetic Match Clip Generator
Generates a realistic football video simulation with pitch lines, players, and a ball
for local testing and verification of the SportsVision AI pipeline.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.video.video_writer import VideoWriter


def generate_synthetic_match_clip(
    output_path: str = "data/sample/synthetic_match.mp4",
    num_frames: int = 150,
    fps: int = 25,
    width: int = 1280,
    height: int = 720
):
    """
    Generate an animated football video clip with:
    - Green grass pitch with standard markings (halfway line, center circle, penalty box)
    - Moving players (Team Blue & Team Red)
    - Rolling/bouncing soccer ball
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    writer = VideoWriter(
        output_path=output_path,
        fps=fps,
        frame_size=(width, height),
        codec="mp4v"
    )

    # Pitch color (lush green)
    pitch_bgr = (34, 139, 34)
    line_color = (240, 240, 240)

    # Initial player positions: 5 Team Blue (left), 5 Team Red (right), 1 Referee (yellow)
    np.random.seed(42)
    players = [
        # Team Blue: [x, y, dx, dy, color]
        {"x": 280, "y": 360, "dx": 2.2, "dy": 0.8, "color": (220, 50, 50), "label": "Blue"},
        {"x": 380, "y": 240, "dx": 1.5, "dy": 1.2, "color": (220, 50, 50), "label": "Blue"},
        {"x": 420, "y": 480, "dx": 1.8, "dy": -1.0, "color": (220, 50, 50), "label": "Blue"},
        {"x": 520, "y": 320, "dx": 2.5, "dy": 0.5, "color": (220, 50, 50), "label": "Blue"},
        {"x": 200, "y": 360, "dx": 0.5, "dy": 0.3, "color": (180, 20, 20), "label": "Blue GK"},

        # Team Red:
        {"x": 1000, "y": 360, "dx": -2.0, "dy": -0.6, "color": (50, 50, 220), "label": "Red"},
        {"x": 900, "y": 250, "dx": -1.6, "dy": 0.9, "color": (50, 50, 220), "label": "Red"},
        {"x": 860, "y": 470, "dx": -1.4, "dy": -1.1, "color": (50, 50, 220), "label": "Red"},
        {"x": 760, "y": 340, "dx": -2.2, "dy": 0.4, "color": (50, 50, 220), "label": "Red"},
        {"x": 1080, "y": 360, "dx": -0.4, "dy": -0.2, "color": (20, 20, 180), "label": "Red GK"},

        # Referee (Yellow)
        {"x": 640, "y": 200, "dx": 0.8, "dy": 1.5, "color": (0, 230, 255), "label": "Ref"}
    ]

    # Ball [x, y, dx, dy]
    ball = {"x": 580.0, "y": 350.0, "dx": 4.5, "dy": 1.8}

    print(f"Rendering {num_frames} synthetic football frames to {output_path}...")

    for f_idx in range(num_frames):
        # Create base pitch
        frame = np.full((height, width, 3), pitch_bgr, dtype=np.uint8)

        # Draw pitch markings
        # Outer touchlines
        cv2.rectangle(frame, (60, 40), (width - 60, height - 40), line_color, 3)
        # Halfway line
        cv2.line(frame, (width // 2, 40), (width // 2, height - 40), line_color, 3)
        # Center circle
        cv2.circle(frame, (width // 2, height // 2), 90, line_color, 3)
        cv2.circle(frame, (width // 2, height // 2), 4, line_color, -1)
        # Left penalty box
        cv2.rectangle(frame, (60, 180), (240, height - 180), line_color, 3)
        # Right penalty box
        cv2.rectangle(frame, (width - 240, 180), (width - 60, height - 180), line_color, 3)

        # Update and draw players
        for p in players:
            p["x"] += p["dx"] + np.random.uniform(-0.3, 0.3)
            p["y"] += p["dy"] + np.random.uniform(-0.3, 0.3)

            # Bounce off boundary
            if p["x"] < 80 or p["x"] > width - 80:
                p["dx"] *= -1
            if p["y"] < 60 or p["y"] > height - 60:
                p["dy"] *= -1

            px, py = int(p["x"]), int(p["y"])
            # Draw player as a stylized human figure (torso + head + limbs)
            # Torso / Jersey
            cv2.rectangle(frame, (px - 14, py - 32), (px + 14, py + 8), p["color"], -1)
            # Head
            cv2.circle(frame, (px, py - 42), 10, (190, 210, 240), -1)
            # Shorts
            cv2.rectangle(frame, (px - 12, py + 8), (px + 12, py + 22), (240, 240, 240), -1)
            # Legs
            cv2.line(frame, (px - 6, py + 22), (px - 8, py + 42), (190, 210, 240), 4)
            cv2.line(frame, (px + 6, py + 22), (px + 8, py + 42), (190, 210, 240), 4)
            # Shoes
            cv2.rectangle(frame, (px - 14, py + 42), (px - 4, py + 46), (20, 20, 20), -1)
            cv2.rectangle(frame, (px + 4, py + 42), (px + 14, py + 46), (20, 20, 20), -1)

        # Update and draw ball
        ball["x"] += ball["dx"]
        ball["y"] += ball["dy"]
        if ball["x"] < 80 or ball["x"] > width - 80:
            ball["dx"] *= -1
        if ball["y"] < 60 or ball["y"] > height - 60:
            ball["dy"] *= -1

        bx, by = int(ball["x"]), int(ball["y"])
        # Soccer ball (white with black pentagon spots)
        cv2.circle(frame, (bx, by), 10, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (bx, by), 10, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, (bx, by), 3, (0, 0, 0), -1, cv2.LINE_AA)

        writer.write_frame(frame)

    writer.release()
    print(f"Synthetic match video generated: {output_path} ({num_frames} frames, {width}x{height})")
    return output_path


if __name__ == "__main__":
    generate_synthetic_match_clip()
