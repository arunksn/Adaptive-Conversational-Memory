import numpy as np

from src.models.memory import Memory, MemoryType
from src.storage.vector_store import VectorStore


def test_add_memory():

    store = VectorStore(
        dimension=3,
        storage_dir="data/test_vector_store"
    )

    memory = Memory(
        content="I prefer Python.",
        memory_type=MemoryType.SEMANTIC
    )

    embedding = np.array(
        [1.0, 0.0, 0.0],
        dtype="float32"
    )

    store.add(
        memory,
        embedding
    )

    assert store.count() == 1


def test_search_memory():

    store = VectorStore(
        dimension=3,
        storage_dir="data/test_vector_store"
    )

    memory1 = Memory(
        content="I prefer Python.",
        memory_type=MemoryType.SEMANTIC
    )

    memory2 = Memory(
        content="I enjoy football.",
        memory_type=MemoryType.SEMANTIC
    )

    store.add(
        memory1,
        np.array(
            [1.0, 0.0, 0.0],
            dtype="float32"
        )
    )

    store.add(
        memory2,
        np.array(
            [0.0, 1.0, 0.0],
            dtype="float32"
        )
    )

    query = np.array(
        [1.0, 0.0, 0.0],
        dtype="float32"
    )

    results = store.search(
        query,
        top_k=1
    )

    assert len(results) == 1

    assert (
        results[0]["memory"]["content"]
        == "I prefer Python."
    )


def test_empty_store_search():

    store = VectorStore(
        dimension=3,
        storage_dir="data/test_vector_store"
    )

    query = np.array(
        [1.0, 0.0, 0.0],
        dtype="float32"
    )

    results = store.search(
        query,
        top_k=5
    )

    assert results == []