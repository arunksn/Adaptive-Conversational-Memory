from dataclasses import dataclass

import pytest

from experiments.vector_memory import (
    VectorMemory,
    VectorMemoryResult,
)


@dataclass
class FakeMemory:

    memory_id: str
    content: str


@dataclass
class FakeVectorResult:

    memory_id: str
    score: float
    memory: dict


class FakeEmbedder:

    def __init__(self):
        self.calls = []

    def embed(
        self,
        text,
    ):
        self.calls.append(
            text
        )

        return [1.0, 0.0, 0.0]


class FakeVectorStore:

    def __init__(self):
        self.add_calls = []
        self.search_calls = []

        self.results = [
            FakeVectorResult(
                memory_id="memory-python",
                score=0.95,
                memory={
                    "memory_id": "memory-python",
                    "content": "I prefer Python.",
                },
            ),
            FakeVectorResult(
                memory_id="memory-go",
                score=0.80,
                memory={
                    "memory_id": "memory-go",
                    "content": "I also use Go.",
                },
            ),
        ]

    def add(
        self,
        memory,
        embedding,
    ):
        self.add_calls.append(
            (
                memory,
                embedding,
            )
        )

    def search(
        self,
        query_embedding,
        top_k=5,
    ):
        self.search_calls.append(
            (
                query_embedding,
                top_k,
            )
        )

        return self.results[:top_k]


class FakeContext:

    def __init__(
        self,
        text,
    ):
        self.text = text


class FakeContextBuilder:

    def __init__(self):
        self.calls = []

    def build(
        self,
        results,
        query=None,
    ):
        self.calls.append(
            (
                results,
                query,
            )
        )

        return FakeContext(
            "vector memory context"
        )


def test_initialization():

    embedder = FakeEmbedder()
    store = FakeVectorStore()

    system = VectorMemory(
        vector_store=store,
        embedder=embedder,
    )

    assert system.vector_store is store
    assert system.embedder is embedder
    assert system.context_builder is None


def test_none_vector_store_rejected():

    with pytest.raises(ValueError):

        VectorMemory(
            vector_store=None,
            embedder=FakeEmbedder(),
        )


def test_none_embedder_rejected():

    with pytest.raises(ValueError):

        VectorMemory(
            vector_store=FakeVectorStore(),
            embedder=None,
        )


def test_add_memory():

    embedder = FakeEmbedder()
    store = FakeVectorStore()

    system = VectorMemory(
        vector_store=store,
        embedder=embedder,
    )

    memory = FakeMemory(
        memory_id="memory-python",
        content="I prefer Python.",
    )

    system.add_memory(
        memory
    )

    assert embedder.calls == [
        "I prefer Python."
    ]

    assert len(
        store.add_calls
    ) == 1

    stored_memory, embedding = (
        store.add_calls[0]
    )

    assert stored_memory is memory

    assert embedding == [
        1.0,
        0.0,
        0.0,
    ]


def test_add_none_memory_rejected():

    system = VectorMemory(
        vector_store=FakeVectorStore(),
        embedder=FakeEmbedder(),
    )

    with pytest.raises(ValueError):

        system.add_memory(
            None
        )


def test_retrieve():

    embedder = FakeEmbedder()
    store = FakeVectorStore()

    system = VectorMemory(
        vector_store=store,
        embedder=embedder,
    )

    results = system.retrieve(
        query="What do I prefer?",
        k=2,
    )

    assert len(results) == 2

    assert results[0].memory_id == (
        "memory-python"
    )

    assert results[1].memory_id == (
        "memory-go"
    )

    assert embedder.calls == [
        "What do I prefer?"
    ]

    assert store.search_calls == [
        (
            [1.0, 0.0, 0.0],
            2,
        )
    ]


def test_retrieve_limits_results():

    system = VectorMemory(
        vector_store=FakeVectorStore(),
        embedder=FakeEmbedder(),
    )

    results = system.retrieve(
        query="What do I prefer?",
        k=1,
    )

    assert len(results) == 1

    assert results[0].memory_id == (
        "memory-python"
    )


def test_retrieve_empty_query_rejected():

    system = VectorMemory(
        vector_store=FakeVectorStore(),
        embedder=FakeEmbedder(),
    )

    with pytest.raises(ValueError):

        system.retrieve(
            query="",
            k=5,
        )


def test_retrieve_invalid_k_rejected():

    system = VectorMemory(
        vector_store=FakeVectorStore(),
        embedder=FakeEmbedder(),
    )

    with pytest.raises(ValueError):

        system.retrieve(
            query="Python",
            k=0,
        )


def test_extract_memory_ids():

    results = [
        FakeVectorResult(
            memory_id="memory-1",
            score=0.9,
            memory={
                "memory_id": "memory-1",
                "content": "One",
            },
        ),
        FakeVectorResult(
            memory_id="memory-2",
            score=0.8,
            memory={
                "memory_id": "memory-2",
                "content": "Two",
            },
        ),
    ]

    ids = VectorMemory.extract_memory_ids(
        results
    )

    assert ids == [
        "memory-1",
        "memory-2",
    ]


def test_extract_memory_ids_from_dict():

    results = [
        {
            "memory_id": "memory-1",
            "memory": {
                "content": "One",
            },
        }
    ]

    ids = VectorMemory.extract_memory_ids(
        results
    )

    assert ids == [
        "memory-1"
    ]


def test_extract_memory_ids_missing():

    results = [
        {
            "score": 0.8,
            "memory": {
                "content": "No ID",
            },
        }
    ]

    ids = VectorMemory.extract_memory_ids(
        results
    )

    assert ids == []


def test_build_context_without_builder():

    store = FakeVectorStore()

    system = VectorMemory(
        vector_store=store,
        embedder=FakeEmbedder(),
    )

    results = system.retrieve(
        query="What do I prefer?",
        k=1,
    )

    context = system.build_context(
        results=results,
        query="What do I prefer?",
    )

    assert (
        "I prefer Python."
        in context
    )


def test_build_context_with_builder():

    context_builder = (
        FakeContextBuilder()
    )

    system = VectorMemory(
        vector_store=FakeVectorStore(),
        embedder=FakeEmbedder(),
        context_builder=context_builder,
    )

    results = [
        FakeVectorResult(
            memory_id="memory-python",
            score=0.95,
            memory={
                "memory_id": "memory-python",
                "content": "I prefer Python.",
            },
        )
    ]

    context = system.build_context(
        results=results,
        query="What do I prefer?",
    )

    assert context.text == (
        "vector memory context"
    )

    assert len(
        context_builder.calls
    ) == 1


def test_run():

    system = VectorMemory(
        vector_store=FakeVectorStore(),
        embedder=FakeEmbedder(),
    )

    result = system.run(
        query="What do I prefer?",
        k=2,
    )

    assert isinstance(
        result,
        VectorMemoryResult,
    )

    assert result.query == (
        "What do I prefer?"
    )

    assert result.retrieved_ids == [
        "memory-python",
        "memory-go",
    ]

    assert (
        "I prefer Python."
        in result.context
    )


def test_run_with_context_builder():

    context_builder = (
        FakeContextBuilder()
    )

    system = VectorMemory(
        vector_store=FakeVectorStore(),
        embedder=FakeEmbedder(),
        context_builder=context_builder,
    )

    result = system.run(
        query="What do I prefer?",
        k=1,
    )

    assert result.retrieved_ids == [
        "memory-python"
    ]

    assert result.context == (
        "vector memory context"
    )