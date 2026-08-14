import numpy as np

from src.models.memory import Memory, MemoryType
from src.storage.vector_store import VectorStore


def create_store(tmp_path):
    return VectorStore(
        dimension=3,
        storage_dir=str(tmp_path)
    )


def test_add_memory(tmp_path):

    store = create_store(tmp_path)

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


def test_search_memory(tmp_path):

    store = create_store(tmp_path)

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


def test_similarity_score(tmp_path):

    store = create_store(tmp_path)

    memory = Memory(
        content="Python memory.",
        memory_type=MemoryType.SEMANTIC
    )

    store.add(
        memory,
        np.array(
            [1.0, 0.0, 0.0],
            dtype="float32"
        )
    )

    results = store.search(
        np.array(
            [1.0, 0.0, 0.0],
            dtype="float32"
        ),
        top_k=1
    )

    assert results[0]["score"] > 0.99


def test_similarity_threshold(tmp_path):

    store = create_store(tmp_path)

    memory = Memory(
        content="Python memory.",
        memory_type=MemoryType.SEMANTIC
    )

    store.add(
        memory,
        np.array(
            [1.0, 0.0, 0.0],
            dtype="float32"
        )
    )

    results = store.search(
        np.array(
            [0.0, 1.0, 0.0],
            dtype="float32"
        ),
        top_k=1,
        min_score=0.8
    )

    assert results == []


def test_update_memory(tmp_path):

    store = create_store(tmp_path)

    memory = Memory(
        content="I prefer Python.",
        memory_type=MemoryType.SEMANTIC
    )

    store.add(
        memory,
        np.array(
            [1.0, 0.0, 0.0],
            dtype="float32"
        )
    )

    memory.content = "I prefer Go."

    store.update(
        memory,
        np.array(
            [0.0, 1.0, 0.0],
            dtype="float32"
        )
    )

    results = store.search(
        np.array(
            [0.0, 1.0, 0.0],
            dtype="float32"
        ),
        top_k=1
    )

    assert (
        results[0]["memory"]["content"]
        == "I prefer Go."
    )


def test_delete_memory(tmp_path):

    store = create_store(tmp_path)

    memory = Memory(
        content="Temporary memory.",
        memory_type=MemoryType.SEMANTIC
    )

    store.add(
        memory,
        np.array(
            [1.0, 0.0, 0.0],
            dtype="float32"
        )
    )

    assert store.count() == 1

    store.delete(
        memory.memory_id
    )

    assert store.count() == 0


def test_empty_store_search(tmp_path):

    store = create_store(tmp_path)

    query = np.array(
        [1.0, 0.0, 0.0],
        dtype="float32"
    )

    results = store.search(
        query,
        top_k=5
    )

    assert results == []


def test_save_and_load(tmp_path):

    store = create_store(tmp_path)

    memory = Memory(
        content="I use Python.",
        memory_type=MemoryType.SEMANTIC
    )

    store.add(
        memory,
        np.array(
            [1.0, 0.0, 0.0],
            dtype="float32"
        )
    )

    store.save()

    loaded_store = create_store(tmp_path)

    loaded_store.load()

    assert loaded_store.count() == 1

    results = loaded_store.search(
        np.array(
            [1.0, 0.0, 0.0],
            dtype="float32"
        ),
        top_k=1
    )

    assert (
        results[0]["memory"]["content"]
        == "I use Python."
    )