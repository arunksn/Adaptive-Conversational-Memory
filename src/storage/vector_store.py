from pathlib import Path
import json

import faiss
import numpy as np

from src.models.memory import Memory


class VectorStore:

    def __init__(
        self,
        dimension: int,
        storage_dir: str = "data/vector_store"
    ):
        self.dimension = dimension

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.index_path = (
            self.storage_dir / "memories.index"
        )

        self.metadata_path = (
            self.storage_dir / "memories.json"
        )

        # Inner Product is used because our
        # embeddings are normalized.
        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        # FAISS stores vectors, not our Memory objects.
        # This list maps FAISS positions → memory IDs.
        self.memory_ids: list[str] = []

        # Stores the actual memory metadata.
        self.memories: dict[str, dict] = {}

    def add(
        self,
        memory: Memory,
        embedding: np.ndarray
    ):

        vector = np.asarray(
            embedding,
            dtype="float32"
        ).reshape(1, -1)

        if vector.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.dimension}, "
                f"got {vector.shape[1]}"
            )

        self.index.add(vector)

        self.memory_ids.append(
            memory.memory_id
        )

        self.memories[memory.memory_id] = (
            memory.to_dict()
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ):

        if self.index.ntotal == 0:
            return []

        vector = np.asarray(
            query_embedding,
            dtype="float32"
        ).reshape(1, -1)

        scores, indices = self.index.search(
            vector,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            memory_id = self.memory_ids[index]

            results.append({
                "memory_id": memory_id,
                "score": float(score),
                "memory": self.memories[memory_id]
            })

        return results

    def save(self):

        faiss.write_index(
            self.index,
            str(self.index_path)
        )

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                {
                    "memory_ids": self.memory_ids,
                    "memories": self.memories
                },
                file,
                indent=2
            )

    def load(self):

        if not self.index_path.exists():
            return

        if not self.metadata_path.exists():
            return

        self.index = faiss.read_index(
            str(self.index_path)
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        self.memory_ids = data["memory_ids"]
        self.memories = data["memories"]

    def count(self) -> int:
        return self.index.ntotal