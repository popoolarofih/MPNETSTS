from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image as PilImage
from image_model import ImageEmbedder

class VideoEmbedder:
    def __init__(self, device: str = "cpu") -> None:
        self.image_embedder = ImageEmbedder(device=device)
        self.device = device

    def encode_video(self, video_path: str, num_frames: int = 10) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError("Invalid video")
        step = max(1, total_frames // num_frames)
        count = 0
        while cap.isOpened() and len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(PilImage.fromarray(frame_rgb))
            count += 1
        cap.release()
        if not frames:
            raise ValueError("No frames extracted")
        embeddings = [self.image_embedder.encode_image(f) for f in frames]
        mean_emb = torch.mean(torch.stack(embeddings), dim=0)
        return F.normalize(mean_emb, p=2, dim=0)

    def perform_cosine_similarity_between_2_videos(self, videos: List[str]) -> float:
        if len(videos) != 2:
            raise ValueError("Exactly two video paths required")
        embeddings = [self.encode_video(path) for path in videos]
        return F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()

    def perform_cosine_similarity_and_return_highest(
        self, videos: List[str]
    ) -> Tuple[float, Tuple[int, int]]:
        if len(videos) < 2:
            raise ValueError("At least two videos required")
        embeddings = torch.stack([self.encode_video(v) for v in videos])
        similarities = F.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )
        similarities = torch.triu(similarities, diagonal=1)
        max_similarity, max_idx = torch.max(similarities.flatten(), dim=0)
        i, j = divmod(max_idx.item(), similarities.shape[1])
        return max_similarity.item(), (i, j)