from dataclasses import dataclass

import pytest

from experiments.baseline_rag import (
    BaselineRAG,
    BaselineRAGResult,
)


@dataclass
class FakeMemory:

    memory_id: str
    content: str


@dataclass
class FakeRetrievalResult:

    memory_id: str | None
    item: object | None = None


class FakeRetriever:

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

        query_lower = query.lower()

        # The test query asks about a programming
        # language preference, so return the Python
        # memory for that query.
        if (
            "python" in query_lower
            or "programming language" in query_lower
        ):

            return (
                None,
                [
                    FakeRetrievalResult(
                        memory_id="memory-python"
                    ),
                    FakeRetrievalResult(
                        memory_id="memory-other"
                    ),
                ],
            )

        return (
            None,
            []
        )


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
            text="retrieved memory context"
        )


class FakeGeneratedResponse:

    def __init__(
        self,
        text,
    ):
        self.text = text


class FakeResponseGenerator:

    def __init__(self):
        self.calls = []

    def generate(
        self,
        query,
        context,
    ):
        self.calls.append(
            (
                query,
                context,
            )
        )

        return FakeGeneratedResponse(
            text="generated answer"
        )


def test_initialization():

    retriever = FakeRetriever()

    system = BaselineRAG(
        retriever
    )

    assert system.retriever is retriever
    assert system.response_generator is None
    assert system.context_builder is None


def test_none_retriever_rejected():

    with pytest.raises(ValueError):

        BaselineRAG(
            None
        )


def test_retrieve():

    retriever = FakeRetriever()

    system = BaselineRAG(
        retriever
    )

    results = system.retrieve(
        query="What programming language do I prefer?",
        k=2,
    )

    assert len(results) == 2

    assert results[0].memory_id == (
        "memory-python"
    )

    assert results[1].memory_id == (
        "memory-other"
    )

    assert retriever.calls == [
        (
            "What programming language do I prefer?",
            2,
        )
    ]


def test_retrieve_limits_results_to_k():

    retriever = FakeRetriever()

    system = BaselineRAG(
        retriever
    )

    results = system.retrieve(
        query="What programming language do I prefer?",
        k=1,
    )

    assert len(results) == 1

    assert results[0].memory_id == (
        "memory-python"
    )


def test_retrieve_empty_query_rejected():

    system = BaselineRAG(
        FakeRetriever()
    )

    with pytest.raises(ValueError):

        system.retrieve(
            query="",
            k=5,
        )


def test_retrieve_invalid_k_rejected():

    system = BaselineRAG(
        FakeRetriever()
    )

    with pytest.raises(ValueError):

        system.retrieve(
            query="python",
            k=0,
        )


def test_extract_memory_ids():

    results = [
        FakeRetrievalResult(
            memory_id="memory-1"
        ),
        FakeRetrievalResult(
            memory_id="memory-2"
        ),
    ]

    ids = BaselineRAG.extract_memory_ids(
        results
    )

    assert ids == [
        "memory-1",
        "memory-2",
    ]


def test_extract_memory_ids_from_item():

    memory = FakeMemory(
        memory_id="memory-1",
        content="I prefer Python.",
    )

    result = FakeRetrievalResult(
        memory_id=None,
        item=memory,
    )

    ids = BaselineRAG.extract_memory_ids(
        [result]
    )

    assert ids == [
        "memory-1"
    ]


def test_extract_memory_ids_ignores_missing_ids():

    result = FakeRetrievalResult(
        memory_id=None
    )

    ids = BaselineRAG.extract_memory_ids(
        [result]
    )

    assert ids == []


def test_build_context_without_builder():

    memory = FakeMemory(
        memory_id="memory-python",
        content="I prefer Python.",
    )

    result = FakeRetrievalResult(
        memory_id="memory-python",
        item=memory,
    )

    system = BaselineRAG(
        FakeRetriever()
    )

    context = system.build_context(
        results=[result],
        query="What do I prefer?",
    )

    assert (
        "I prefer Python."
        in context
    )


def test_build_context_with_builder():

    retriever = FakeRetriever()

    context_builder = FakeContextBuilder()

    system = BaselineRAG(
        retriever=retriever,
        context_builder=context_builder,
    )

    result = FakeRetrievalResult(
        memory_id="memory-python"
    )

    context = system.build_context(
        results=[result],
        query="What do I prefer?",
    )

    assert context.text == (
        "retrieved memory context"
    )

    assert len(
        context_builder.calls
    ) == 1


def test_run_without_generation():

    system = BaselineRAG(
        FakeRetriever()
    )

    result = system.run(
        query="What programming language do I prefer?",
        k=2,
    )

    assert isinstance(
        result,
        BaselineRAGResult,
    )

    assert result.retrieved_ids == [
        "memory-python",
        "memory-other",
    ]

    assert result.response is None


def test_run_with_generation():

    retriever = FakeRetriever()

    context_builder = FakeContextBuilder()

    response_generator = (
        FakeResponseGenerator()
    )

    system = BaselineRAG(
        retriever=retriever,
        context_builder=context_builder,
        response_generator=response_generator,
    )

    result = system.run(
        query="What programming language do I prefer?",
        k=2,
    )

    assert result.retrieved_ids == [
        "memory-python",
        "memory-other",
    ]

    assert result.context == (
        "retrieved memory context"
    )

    assert result.response == (
        "generated answer"
    )

    assert len(
        response_generator.calls
    ) == 1