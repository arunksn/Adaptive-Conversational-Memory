from dataclasses import dataclass

import pytest

from experiments.hybrid_memory import (
    HybridMemory,
    HybridMemoryResult,
)


@dataclass
class FakeResult:
    memory_id: str


class FakeRetriever:

    def __init__(
        self,
        results,
    ):
        self.results = results
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

        return self.results


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


def make_retrievers():

    return {
        "vector": FakeRetriever(
            [
                FakeResult(
                    memory_id="memory-python"
                ),
                FakeResult(
                    memory_id="memory-vector"
                ),
            ]
        ),
        "graph": FakeRetriever(
            [
                FakeResult(
                    memory_id="memory-python"
                ),
                FakeResult(
                    memory_id="memory-graph"
                ),
            ]
        ),
    }


def test_initialization():

    retrievers = make_retrievers()

    system = HybridMemory(
        retrievers
    )

    assert system.retrievers is retrievers


def test_none_retrievers_rejected():

    with pytest.raises(ValueError):

        HybridMemory(
            None
        )


def test_empty_retrievers_rejected():

    with pytest.raises(ValueError):

        HybridMemory(
            {}
        )


def test_non_dictionary_retrievers_rejected():

    with pytest.raises(ValueError):

        HybridMemory(
            []
        )


def test_none_individual_retriever_rejected():

    with pytest.raises(ValueError):

        HybridMemory(
            {
                "vector": None
            }
        )


def test_retrieve():

    retrievers = make_retrievers()

    system = HybridMemory(
        retrievers
    )

    result = system.retrieve(
        query="What programming language do I prefer?",
        k=5,
    )

    assert isinstance(
        result,
        HybridMemoryResult,
    )

    assert result.retrieved_ids == [
        "memory-python",
        "memory-vector",
        "memory-graph",
    ]

    assert result.result_count == 3


def test_retrieve_calls_all_sources():

    retrievers = make_retrievers()

    system = HybridMemory(
        retrievers
    )

    system.retrieve(
        query="test query",
        k=2,
    )

    assert retrievers[
        "vector"
    ].calls == [
        (
            "test query",
            2,
        )
    ]

    assert retrievers[
        "graph"
    ].calls == [
        (
            "test query",
            2,
        )
    ]


def test_retrieve_limits_results_to_k():

    system = HybridMemory(
        make_retrievers()
    )

    result = system.retrieve(
        query="test",
        k=2,
    )

    assert result.result_count == 2

    assert result.retrieved_ids == [
        "memory-python",
        "memory-vector",
    ]


def test_duplicate_results_are_removed():

    system = HybridMemory(
        make_retrievers()
    )

    result = system.retrieve(
        query="test",
        k=10,
    )

    assert result.retrieved_ids == [
        "memory-python",
        "memory-vector",
        "memory-graph",
    ]


def test_source_results_are_preserved():

    system = HybridMemory(
        make_retrievers()
    )

    result = system.retrieve(
        query="test",
        k=5,
    )

    assert set(
        result.source_results.keys()
    ) == {
        "vector",
        "graph",
    }

    assert len(
        result.source_results["vector"]
    ) == 2

    assert len(
        result.source_results["graph"]
    ) == 2


def test_empty_query_rejected():

    system = HybridMemory(
        make_retrievers()
    )

    with pytest.raises(ValueError):

        system.retrieve(
            query="",
            k=5,
        )


def test_whitespace_query_rejected():

    system = HybridMemory(
        make_retrievers()
    )

    with pytest.raises(ValueError):

        system.retrieve(
            query="   ",
            k=5,
        )


def test_invalid_k_rejected():

    system = HybridMemory(
        make_retrievers()
    )

    with pytest.raises(ValueError):

        system.retrieve(
            query="test",
            k=0,
        )


def test_negative_k_rejected():

    system = HybridMemory(
        make_retrievers()
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

    assert HybridMemory._extract_memory_id(
        result
    ) == "memory-1"


def test_extract_memory_id_from_dict():

    result = {
        "memory_id": "memory-1"
    }

    assert HybridMemory._extract_memory_id(
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

    assert HybridMemory._extract_memory_id(
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

    assert HybridMemory._extract_memory_id(
        result
    ) == "memory-meta"


def test_missing_memory_id_is_ignored():

    class Result:
        pass

    assert (
        HybridMemory._extract_memory_id(
            Result()
        )
        is None
    )


def test_normalize_list_response():

    results = [
        FakeResult(
            memory_id="memory-1"
        )
    ]

    assert (
        HybridMemory._normalize_response(
            results
        )
        == results
    )


def test_normalize_tuple_response():

    results = [
        FakeResult(
            memory_id="memory-1"
        )
    ]

    assert (
        HybridMemory._normalize_response(
            (
                None,
                results,
            )
        )
        == results
    )


def test_normalize_none_response():

    assert (
        HybridMemory._normalize_response(
            None
        )
        == []
    )


def test_merge_results():

    source_results = {
        "vector": [
            FakeResult(
                memory_id="memory-1"
            ),
            FakeResult(
                memory_id="memory-2"
            ),
        ],
        "graph": [
            FakeResult(
                memory_id="memory-2"
            ),
            FakeResult(
                memory_id="memory-3"
            ),
        ],
    }

    merged = HybridMemory._merge_results(
        source_results
    )

    assert (
        HybridMemory._extract_memory_ids(
            merged
        )
        == [
            "memory-1",
            "memory-2",
            "memory-3",
        ]
    )


def test_build_context():

    context_builder = FakeContextBuilder()

    system = HybridMemory(
        make_retrievers()
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

    system = HybridMemory(
        make_retrievers()
    )

    with pytest.raises(ValueError):

        system.build_context(
            query="test",
            context_builder=None,
        )


def test_run_without_context_builder():

    system = HybridMemory(
        make_retrievers()
    )

    result = system.run(
        query="test query",
        k=2,
    )

    assert result["query"] == (
        "test query"
    )

    assert isinstance(
        result["retrieval"],
        HybridMemoryResult,
    )

    assert result["context"] is None

    assert result["retrieved_ids"] == [
        "memory-python",
        "memory-vector",
    ]


def test_run_with_context_builder():

    context_builder = FakeContextBuilder()

    system = HybridMemory(
        make_retrievers()
    )

    result = system.run(
        query="test query",
        k=2,
        context_builder=context_builder,
    )

    assert result["context"] == (
        "fake context"
    )

    assert result["retrieved_ids"] == [
        "memory-python",
        "memory-vector",
    ]

    assert len(
        context_builder.calls
    ) == 1