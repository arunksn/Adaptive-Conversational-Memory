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
        and store it in the vector store.
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

        The vector store remains responsible for
        semantic similarity ranking.
        """

        if not isinstance(
            query,
            str
        ) or not query.strip():

            raise ValueError(
                "query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        query_embedding = (
            self.embedding_model.encode(
                query
            )
        )

        return self.vector_store.search(
            query_embedding,
            top_k=top_k,
            min_score=min_score
        )

    
    # RETRIEVE
    # 
    #
    # Compatibility interface used by:
    #
    # EvaluationRunner
    # BaselineRAG
    # Benchmark
    #


    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None
    ):
        """
        Retrieve semantically similar memories.

        Internally delegates to search().
        """

        return self.search(
            query=query,
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
        Delete a memory from the vector store.
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
        """
        Return the number of stored memories.
        """

        return self.vector_store.count()