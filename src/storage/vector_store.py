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

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        # FAISS position → memory ID
        self.memory_ids: list[str] = []

        # memory ID → serialized memory
        self.memories: dict[str, dict] = {}

    # ADDED
 

    def add(
        self,
        memory: Memory,
        embedding: np.ndarray
    ):
        """
        Add a new memory and its embedding.
        """

        if memory.memory_id in self.memories:
            raise ValueError(
                f"Memory already exists: {memory.memory_id}"
            )

        vector = self._validate_embedding(embedding)

        self.index.add(vector)

        self.memory_ids.append(
            memory.memory_id
        )

        self.memories[memory.memory_id] = (
            memory.to_dict()
        )

    # SEARCH

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float | None = None
    ):
        """
        Search for the most similar memories.

        min_score can be used to reject memories
        that are not sufficiently similar.
        """

        if self.index.ntotal == 0:
            return []

        vector = self._validate_embedding(
            query_embedding
        )

        top_k = min(
            top_k,
            self.index.ntotal
        )

        scores, indices = self.index.search(
            vector,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            score = float(score)

            if (
                min_score is not None
                and score < min_score
            ):
                continue

            memory_id = self.memory_ids[index]

            results.append({
                "memory_id": memory_id,
                "score": score,
                "memory": self.memories[memory_id]
            })

        return results
    
    # UPDATE

    def update(
        self,
        memory: Memory,
        embedding: np.ndarray
    ):
        """
        Update an existing memory.

        FAISS IndexFlatIP does not directly provide
        an application-level update operation, so we
        rebuild the index after replacing the memory.
        """

        if memory.memory_id not in self.memories:
            raise ValueError(
                f"Memory not found: {memory.memory_id}"
            )

        vector = self._validate_embedding(
            embedding
        )

        old_index = self.memory_ids.index(
            memory.memory_id
        )

        vectors = self.index.reconstruct_n(
            0,
            self.index.ntotal
        )

        vectors[old_index] = vector[0]

        self.memories[memory.memory_id] = (
            memory.to_dict()
        )

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.index.add(vectors)

    # DELETE

    def delete(
        self,
        memory_id: str
    ):
        """
        Delete a memory from the vector store.
        """

        if memory_id not in self.memories:
            raise ValueError(
                f"Memory not found: {memory_id}"
            )

        delete_index = self.memory_ids.index(
            memory_id
        )

        vectors = self.index.reconstruct_n(
            0,
            self.index.ntotal
        )

        vectors = np.delete(
            vectors,
            delete_index,
            axis=0
        )

        self.memory_ids.pop(delete_index)

        del self.memories[memory_id]

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        if len(vectors) > 0:
            self.index.add(vectors)

    # SAVE

    def save(self):
        """
        Persist FAISS index and memory metadata.
        """

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

    # LOAD

    def load(self):
        """
        Load persisted FAISS index and metadata.
        """

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

        if self.index.ntotal != len(
            self.memory_ids
        ):
            raise ValueError(
                "FAISS index and memory metadata "
                "are out of sync."
            )
        
    # COUNT

    def count(self) -> int:
        return self.index.ntotal

    # INTERNAL VALIDATION

    def _validate_embedding(
        self,
        embedding: np.ndarray
    ) -> np.ndarray:
        """
        Validate and reshape an embedding.
        """

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

        return vector