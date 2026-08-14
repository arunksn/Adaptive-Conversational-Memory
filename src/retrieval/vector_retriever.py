from src.embeddings.embedding_model import EmbeddingModel
from src.models.memory import Memory
from src.storage.vector_store import VectorStore


class VectorRetriever:

    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        storage_dir: str = "data/vector_store"
    ):
        self.embedding_model = EmbeddingModel(
            model_name
        )

        self.vector_store = VectorStore(
            dimension=self.embedding_model.dimension,
            storage_dir=storage_dir
        )

    def add_memory(
        self,
        memory: Memory
    ):
        """
        Convert memory text into an embedding
        and store it in FAISS.
        """

        embedding = self.embedding_model.encode(
            memory.content
        )

        self.vector_store.add(
            memory,
            embedding
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None
    ):
        """
        Convert a text query into an embedding
        and retrieve semantically similar memories.
        """

        query_embedding = (
            self.embedding_model.encode(query)
        )

        return self.vector_store.search(
            query_embedding,
            top_k=top_k,
            min_score=min_score
        )

    def update_memory(
        self,
        memory: Memory
    ):
        """
        Update an existing memory.
        """

        embedding = self.embedding_model.encode(
            memory.content
        )

        self.vector_store.update(
            memory,
            embedding
        )

    def delete_memory(
        self,
        memory_id: str
    ):
        """
        Delete a memory.
        """

        self.vector_store.delete(
            memory_id
        )

    def save(self):
        """
        Persist the vector store.
        """

        self.vector_store.save()

    def load(self):
        """
        Load the persisted vector store.
        """

        self.vector_store.load()

    def count(self) -> int:
        return self.vector_store.count()