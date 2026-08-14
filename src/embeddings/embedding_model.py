from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        """
        Convert a single text into a normalized embedding.
        """

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.astype("float32")

    def encode_batch(
        self,
        texts: list[str]
    ) -> np.ndarray:
        """
        Convert multiple texts into normalized embeddings.
        """

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings.astype("float32")

    @property
    def dimension(self) -> int:
        """
        Return the embedding dimension.
        """

        return self.model.get_embedding_dimension()