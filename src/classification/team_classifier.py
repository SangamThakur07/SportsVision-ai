"""
Team Classification Module using Transfer Learning
Applies pretrained ResNet-18 as a visual feature extractor on upper-torso jersey crops
to classify players into Team A, Team B, and Referees.
Directly aligns with Computer Vision Syllabus Units II & III.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.cluster import KMeans


class JerseyCropExtractor:
    """Extracts upper-torso player crops isolating the team jersey."""

    @staticmethod
    def extract_jersey_crop(
        frame: np.ndarray,
        bbox: List[float],
        target_size: Tuple[int, int] = (224, 224)
    ) -> Optional[np.ndarray]:
        """
        Extract upper-torso region from player bounding box.
        Removes head/hair (top 15%) and legs/boots/grass (bottom 45%).
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(float, bbox)

        box_w = x2 - x1
        box_h = y2 - y1

        # Isolate jersey torso
        crop_y1 = int(max(0, y1 + 0.12 * box_h))
        crop_y2 = int(min(h, y1 + 0.55 * box_h))
        crop_x1 = int(max(0, x1 + 0.15 * box_w))
        crop_x2 = int(min(w, x2 - 0.15 * box_w))

        if crop_y2 <= crop_y1 + 5 or crop_x2 <= crop_x1 + 5:
            return None

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return None

        return cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)


class ResNetFeatureExtractor(nn.Module):
    """
    Pretrained ResNet-18 feature extractor.
    Uses ImageNet weights, removes final classification layer, and outputs 512-dim embedding.
    """

    def __init__(self, device: Optional[str] = None):
        super().__init__()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load pretrained ResNet-18 (ImageNet weights)
        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)

        # Freeze all backbone layers (Transfer Learning - Feature Extraction)
        for param in resnet.parameters():
            param.requires_grad = False

        # Replace 1000-class fc head with Identity layer
        resnet.fc = nn.Identity()
        self.backbone = resnet.to(self.device)
        self.backbone.eval()

        # ImageNet normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def extract(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract 512-dimensional feature embedding vectors for a batch of BGR crops.
        """
        if not crops:
            return np.empty((0, 512), dtype=np.float32)

        # Convert BGR crops to RGB normalized tensors
        rgb_tensors = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            rgb_tensors.append(tensor)

        batch = torch.stack(rgb_tensors).to(self.device)
        # Apply standard ImageNet normalization
        batch = (batch - self.mean) / self.std

        with torch.no_grad():
            embeddings = self.backbone(batch)  # Shape: (B, 512)

        return embeddings.cpu().numpy()


class ColorHistogramBaseline:
    """
    Simple Color Histogram Baseline for comparison against deep learning ResNet.
    Extracts 3D HSV color histograms for jersey crops.
    """

    @staticmethod
    def extract_histogram(crop: np.ndarray, bins: Tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
        """Compute normalized 3D HSV histogram."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()


class TeamClassifier:
    """
    High-level Team Classifier combining ResNet feature extraction,
    temporal embedding aggregation per player ID, and K-Means clustering.
    """

    def __init__(
        self,
        num_teams: int = 2,
        device: Optional[str] = None
    ):
        self.num_teams = num_teams
        self.crop_extractor = JerseyCropExtractor()
        self.feature_extractor = ResNetFeatureExtractor(device=device)
        self.kmeans: Optional[KMeans] = None
        self.cluster_to_team: Dict[int, str] = {}
        self.player_embeddings: Dict[int, List[np.ndarray]] = {}
        self.player_crops: Dict[int, List[np.ndarray]] = {}

    def collect_player_crop(
        self,
        frame: np.ndarray,
        track_id: int,
        bbox: List[float]
    ):
        """Extract and store jersey crop for a tracked player in the current frame."""
        if track_id <= 0:
            return
        crop = self.crop_extractor.extract_jersey_crop(frame, bbox)
        if crop is not None:
            if track_id not in self.player_crops:
                self.player_crops[track_id] = []
            # Keep up to 15 crops per player for memory efficiency
            if len(self.player_crops[track_id]) < 15:
                self.player_crops[track_id].append(crop)

    def fit(self) -> Dict[int, str]:
        """
        Extract ResNet embeddings across all collected player crops,
        cluster into teams using K-Means, and assign team labels to player IDs.

        Returns:
            Dictionary mapping track_id -> 'Team A' or 'Team B'.
        """
        player_mean_features: Dict[int, np.ndarray] = {}
        player_mean_colors: Dict[int, np.ndarray] = {}

        for track_id, crops in self.player_crops.items():
            if not crops:
                continue
            feats = self.feature_extractor.extract(crops)
            player_mean_features[track_id] = np.mean(feats, axis=0)

            # Average BGR color of jersey crops for dynamic team naming
            colors = [np.mean(c, axis=(0, 1)) for c in crops]
            player_mean_colors[track_id] = np.mean(colors, axis=0)

        if len(player_mean_features) < self.num_teams:
            # Fallback if too few players were observed
            return {pid: "Team A" if pid % 2 == 1 else "Team B" for pid in self.player_crops}

        # Cluster player feature vectors using K-Means
        track_ids = list(player_mean_features.keys())
        X = np.array([player_mean_features[tid] for tid in track_ids])

        self.kmeans = KMeans(n_clusters=self.num_teams, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X)

        # Inspect average dominant RGB color per cluster to assign intuitive names
        cluster_colors: Dict[int, List[np.ndarray]] = {c: [] for c in range(self.num_teams)}
        for tid, cluster_id in zip(track_ids, labels):
            cluster_colors[cluster_id].append(player_mean_colors[tid])

        self.cluster_to_team = {}
        if self.num_teams == 2:
            c0_colors = cluster_colors[0] if cluster_colors[0] else [np.array([128, 128, 128])]
            c1_colors = cluster_colors[1] if cluster_colors[1] else [np.array([128, 128, 128])]
            avg_bgr0 = np.mean(c0_colors, axis=0)
            avg_bgr1 = np.mean(c1_colors, axis=0)

            lum0 = 0.299 * avg_bgr0[2] + 0.587 * avg_bgr0[1] + 0.114 * avg_bgr0[0]
            lum1 = 0.299 * avg_bgr1[2] + 0.587 * avg_bgr1[1] + 0.114 * avg_bgr1[0]

            if lum0 > lum1:
                self.cluster_to_team[0] = "Team White"
                self.cluster_to_team[1] = "Team Red" if avg_bgr1[2] > avg_bgr1[0] else "Team Dark"
            else:
                self.cluster_to_team[1] = "Team White"
                self.cluster_to_team[0] = "Team Red" if avg_bgr0[2] > avg_bgr0[0] else "Team Dark"
        else:
            team_names = ["Team White", "Team Red", "Team Blue"]
            for c in range(self.num_teams):
                self.cluster_to_team[c] = team_names[c % len(team_names)]

        # Map each track_id to its assigned team
        player_team_assignments: Dict[int, str] = {}
        for tid, cluster_id in zip(track_ids, labels):
            player_team_assignments[tid] = self.cluster_to_team[cluster_id]

        return player_team_assignments
