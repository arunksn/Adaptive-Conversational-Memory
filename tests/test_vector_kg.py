from dataclasses import dataclass

import pytest

from experiments.vector_kg import (
    VectorKG,
    VectorKGRetrievalResult,
)


@dataclass
class FakeResult:
    memory_id: str


class FakeVectorRetriever:

    def __init__(self):
        self.calls = []

    def retrieve(
        self,
        query,
        top_k=5,
    ):
        self.calls.append(
            (
                query,
                top_k,
            )
        )

        return [
            FakeResult(
                memory_id="memory-python"
            ),
            FakeResult(
                memory_id="memory-other"
            ),
        ]


class FakeGraphRetriever:

    def __init__(self):
        self.calls = []

    def retrieve(
        self,
        query,
        top_k=5,
    ):
        self.calls.append(
            (
                query,
                top_k,
            )
        )

        return [
            FakeResult(
                memory_id="memory-python"
            ),
            FakeResult(
                memory_id="memory-graph"
            ),
        ]


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

        return "fake context"


def test_initialization():

    vector_retriever = FakeVectorRetriever()
    graph_retriever = FakeGraphRetriever()

    system = VectorKG(
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
    )

    assert system.vector_retriever is vector_retriever
    assert system.graph_retriever is graph_retriever


def test_none_vector_retriever_rejected():

    with pytest.raises(ValueError):

        VectorKG(
            vector_retriever=None,
            graph_retriever=FakeGraphRetriever(),
        )


def test_none_graph_retriever_rejected():

    with pytest.raises(ValueError):

        VectorKG(
            vector_retriever=FakeVectorRetriever(),
            graph_retriever=None,
        )


def test_retrieve():

    vector_retriever = FakeVectorRetriever()
    graph_retriever = FakeGraphRetriever()

    system = VectorKG(
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
    )

    result = system.retrieve(
        query="What programming language do I prefer?",
        k=3,
    )

    assert isinstance(
        result,
        VectorKGRetrievalResult,
    )

    assert result.retrieved_ids == [
        "memory-python",
        "memory-other",
        "memory-graph",
    ]

    assert result.result_count == 3


def test_retrieve_calls_both_retrievers():

    vector_retriever = FakeVectorRetriever()
    graph_retriever = FakeGraphRetriever()

    system = VectorKG(
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
    )

    system.retrieve(
        query="test query",
        k=2,
    )

    assert vector_retriever.calls == [
        (
            "test query",
            2,
        )
    ]

    assert graph_retriever.calls == [
        (
            "test query",
            2,
        )
    ]


def test_retrieve_limits_results_to_k():

    system = VectorKG(
        vector_retriever=FakeVectorRetriever(),
        graph_retriever=FakeGraphRetriever(),
    )

    result = system.retrieve(
        query="test query",
        k=2,
    )

    assert result.result_count == 2

    assert result.retrieved_ids == [
        "memory-python",
        "memory-other",
    ]


def test_duplicate_memory_ids_are_removed():

    system = VectorKG(
        vector_retriever=FakeVectorRetriever(),
        graph_retriever=FakeGraphRetriever(),
    )

    result = system.retrieve(
        query="test query",
        k=10,
    )

    assert result.retrieved_ids == [
        "memory-python",
        "memory-other",
        "memory-graph",
    ]

    assert result.result_count == 3


def test_vector_results_are_kept_before_graph_results():

    system = VectorKG(
        vector_retriever=FakeVectorRetriever(),
        graph_retriever=FakeGraphRetriever(),
    )

    result = system.retrieve(
        query="test query",
        k=10,
    )

    assert result.retrieved_ids[0] == (
        "memory-python"
    )

    assert result.retrieved_ids[1] == (
        "memory-other"
    )

    assert result.retrieved_ids[2] == (
        "memory-graph"
    )


def test_empty_query_rejected():

    system = VectorKG(
        vector_retriever=FakeVectorRetriever(),
        graph_retriever=FakeGraphRetriever(),
    )

    with pytest.raises(ValueError):

        system.retrieve(
            query="",
            k=5,
        )


def test_whitespace_query_rejected():

    system = VectorKG(
        vector_retriever=FakeVectorRetriever(),
        graph_retriever=FakeGraphRetriever(),
    )

    with pytest.raises(ValueError):

        system.retrieve(
            query="   ",
            k=5,
        )


def test_invalid_k_rejected():

    system = VectorKG(
        vector_retriever=FakeVectorRetriever(),
        graph_retriever=FakeGraphRetriever(),
    )

    with pytest.raises(ValueError):

        system.retrieve(
            query="test",
            k=0,
        )


def test_negative_k_rejected():

    system = VectorKG(
        vector_retriever=FakeVectorRetriever(),
        graph_retriever=FakeGraphRetriever(),
    )

    with pytest.raises(ValueError):

        system.retrieve(
            query="test",
            k=-1,
        )


def test_extract_memory_id_from_object():

    result = FakeResult(
        memory_id="memory-1"
    )

    assert VectorKG._extract_memory_id(
        result
    ) == "memory-1"


def test_extract_memory_id_from_dict():

    result = {
        "memory_id": "memory-1"
    }

    assert VectorKG._extract_memory_id(
        result
    ) == "memory-1"


def test_extract_memory_id_from_item():

    @dataclass
    class Item:
        memory_id: str

    @dataclass
    class Result:
        item: Item

    result = Result(
        item=Item(
            memory_id="memory-item"
        )
    )

    assert VectorKG._extract_memory_id(
        result
    ) == "memory-item"


def test_extract_memory_id_from_metadata():

    @dataclass
    class Result:
        metadata: dict

    result = Result(
        metadata={
            "memory_id": "memory-meta"
        }
    )

    assert VectorKG._extract_memory_id(
        result
    ) == "memory-meta"


def test_missing_memory_id_is_ignored():

    class Result:
        pass

    result = VectorKG._extract_memory_id(
        Result()
    )

    assert result is None


def test_extract_memory_ids():

    results = [
        FakeResult(
            memory_id="memory-1"
        ),
        FakeResult(
            memory_id="memory-2"
        ),
    ]

    assert VectorKG._extract_memory_ids(
        results
    ) == [
        "memory-1",
        "memory-2",
    ]


def test_build_context():

    vector_retriever = FakeVectorRetriever()
    graph_retriever = FakeGraphRetriever()
    context_builder = FakeContextBuilder()

    system = VectorKG(
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
    )

    context = system.build_context(
        query="test query",
        context_builder=context_builder,
        k=2,
    )

    assert context == "fake context"

    assert len(
        context_builder.calls
    ) == 1

    results, query = (
        context_builder.calls[0]
    )

    assert query == "test query"

    assert len(results) == 2


def test_build_context_requires_builder():

    system = VectorKG(
        vector_retriever=FakeVectorRetriever(),
        graph_retriever=FakeGraphRetriever(),
    )

    with pytest.raises(ValueError):

        system.build_context(
            query="test query",
            context_builder=None,
        )


def test_run_without_context_builder():

    system = VectorKG(
        vector_retriever=FakeVectorRetriever(),
        graph_retriever=FakeGraphRetriever(),
    )

    result = system.run(
        query="test query",
        k=2,
    )

    assert result["query"] == "test query"

    assert isinstance(
        result["retrieval"],
        VectorKGRetrievalResult,
    )

    assert result["context"] is None

    assert result["retrieved_ids"] == [
        "memory-python",
        "memory-other",
    ]


def test_run_with_context_builder():

    context_builder = FakeContextBuilder()

    system = VectorKG(
        vector_retriever=FakeVectorRetriever(),
        graph_retriever=FakeGraphRetriever(),
    )

    result = system.run(
        query="test query",
        k=2,
        context_builder=context_builder,
    )

    assert result["query"] == "test query"

    assert result["context"] == (
        "fake context"
    )

    assert result["retrieved_ids"] == [
        "memory-python",
        "memory-other",
    ]

    assert len(
        context_builder.calls
    ) == 1


def test_normalize_list_response():

    results = [
        FakeResult(
            memory_id="memory-1"
        )
    ]

    normalized = (
        VectorKG._normalize_retrieval_response(
            results
        )
    )

    assert normalized == results


def test_normalize_tuple_response():

    results = [
        FakeResult(
            memory_id="memory-1"
        )
    ]

    normalized = (
        VectorKG._normalize_retrieval_response(
            (
                None,
                results,
            )
        )
    )

    assert normalized == results


def test_normalize_none_response():

    assert (
        VectorKG._normalize_retrieval_response(
            None
        )
        == []
    )


def test_merge_results():

    vector_results = [
        FakeResult(
            memory_id="memory-1"
        ),
        FakeResult(
            memory_id="memory-2"
        ),
    ]

    graph_results = [
        FakeResult(
            memory_id="memory-2"
        ),
        FakeResult(
            memory_id="memory-3"
        ),
    ]

    merged = VectorKG._merge_results(
        vector_results=vector_results,
        graph_results=graph_results,
    )

    assert (
        VectorKG._extract_memory_ids(
            merged
        )
        == [
            "memory-1",
            "memory-2",
            "memory-3",
        ]
    )