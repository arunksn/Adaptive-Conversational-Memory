from src.models.memory import Memory, MemoryType
from src.retrieval.vector_retriever import VectorRetriever


def test_semantic_retrieval(tmp_path):

    retriever = VectorRetriever(
        storage_dir=str(tmp_path)
    )

    memories = [

        Memory(
            content=(
                "I prefer Python for "
                "machine learning projects."
            ),
            memory_type=MemoryType.SEMANTIC
        ),

        Memory(
            content=(
                "I am learning Go for "
                "backend development."
            ),
            memory_type=MemoryType.SEMANTIC
        ),

        Memory(
            content=(
                "I enjoy playing football "
                "on weekends."
            ),
            memory_type=MemoryType.SEMANTIC
        )
    ]

    for memory in memories:
        retriever.add_memory(memory)

    results = retriever.search(
        (
            "What programming language do I "
            "prefer for machine learning?"
        ),
        top_k=2
    )

    assert len(results) == 2

    assert (
        results[0]["memory"]["content"]
        == (
            "I prefer Python for "
            "machine learning projects."
        )
    )