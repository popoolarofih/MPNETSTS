from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class AllMpnetBaseV2:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = "cpu") -> None:
        """Initialize the model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def perform_cosine_similarity_between_2_sentences(
        self, sentences: Optional[List[str]] = None
    ) -> float:
        """Calculate cosine similarity between two sentences.

        Args:
            sentences: List of two sentences. If None, raises an error.

        Returns:
            float: Cosine similarity score.

        Raises:
            ValueError: If the input is not a list of exactly two strings.
        """
        if sentences is None or not isinstance(sentences, list) or len(sentences) != 2:
            raise ValueError("Must provide exactly two sentences as a list of strings")
        if not all(isinstance(s, str) and s.strip() for s in sentences):
            raise ValueError("Sentences must be non-empty strings")

        embeddings = self._encode_sentences(sentences)
        return F.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()

    def perform_cosine_similarity_and_return_highest(
        self, sentences: List[str]
    ) -> Tuple[float, Tuple[str, str]]:
        """Find the pair of sentences with the highest cosine similarity.

        Args:
            sentences: List of sentences (at least two).

        Returns:
            Tuple[float, Tuple[str, str]]: Highest similarity score and the corresponding sentence pair.

        Raises:
            ValueError: If fewer than two valid sentences are provided.
        """
        if not isinstance(sentences, list) or len(sentences) < 2:
            raise ValueError("At least two sentences are required")
        if not all(isinstance(s, str) and s.strip() for s in sentences):
            raise ValueError("Sentences must be non-empty strings")

        embeddings = self._encode_sentences(sentences)
        # Compute all pairwise similarities efficiently
        similarities = F.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )
        # Mask diagonal (self-similarities) and lower triangle
        similarities = torch.triu(similarities, diagonal=1)
        max_similarity, max_idx = torch.max(similarities.flatten(), dim=0)
        i, j = divmod(max_idx.item(), similarities.shape[1])
        return max_similarity.item(), (sentences[i], sentences[j])

    def encode_sentence(self, sentence: str) -> torch.Tensor:
        """Encode a single sentence into a normalized embedding.

        Args:
            sentence: The input sentence.

        Returns:
            torch.Tensor: Normalized 768-dimensional embedding.

        Raises:
            ValueError: If the input is not a non-empty string.
        """
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError("Input must be a non-empty string")
        return self._encode_sentences([sentence])[0]

    def _encode_sentences(self, sentences: List[str]) -> torch.Tensor:
        """Encode a list of sentences into normalized embeddings.

        Args:
            sentences: List of sentences.

        Returns:
            torch.Tensor: Normalized embeddings of shape (num_sentences, 768).
        """
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        return F.normalize(embeddings, p=2, dim=1)

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform mean pooling on token embeddings, accounting for attention mask.

        Args:
            model_output: Model output containing token embeddings.
            attention_mask: Attention mask for the input tokens.

        Returns:
            torch.Tensor: Mean-pooled sentence embeddings.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )