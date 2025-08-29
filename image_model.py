from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image as PilImage

class ImageEmbedder:
    def __init__(self, model_name: str = "google/vit-base-patch16-224", device: str = "cpu") -> None:
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def encode_image(self, image: PilImage.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.pooler_output[0]
        return F.normalize(embedding, p=2, dim=0)

    def perform_cosine_similarity_between_2_images(self, images: List[PilImage.Image]) -> float:
        if len(images) != 2:
            raise ValueError("Exactly two images required")
        embeddings = [self.encode_image(img) for img in images]
        return F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()

    def perform_cosine_similarity_and_return_highest(
        self, images: List[PilImage.Image]
    ) -> Tuple[float, Tuple[int, int]]:
        if len(images) < 2:
            raise ValueError("At least two images required")
        embeddings = torch.stack([self.encode_image(img) for img in images])
        similarities = F.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )
        similarities = torch.triu(similarities, diagonal=1)
        max_similarity, max_idx = torch.max(similarities.flatten(), dim=0)
        i, j = divmod(max_idx.item(), similarities.shape[1])
        return max_similarity.item(), (i, j)