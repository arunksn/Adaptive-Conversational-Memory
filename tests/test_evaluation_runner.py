from dataclasses import dataclass

import pytest

from src.evaluation.evaluation_dataset import (
    EvaluationCase,
    EvaluationDataset
)

from src.evaluation.evaluation_runner import (
    EvaluationRunner
)


@dataclass
class FakeRetrievalResult:

    memory_id: str


class FakeRetriever:

    def __init__(self):
        self.calls = []

    def retrieve(
        self,
        query,
        top_k=5
    ):
        self.calls.append(
            (
                query,
                top_k
            )
        )

        if "programming language" in query.lower():

            return (
                None,
                [
                    FakeRetrievalResult(
                        memory_id="memory-python"
                    ),
                    FakeRetrievalResult(
                        memory_id="memory-other"
                    )
                ]
            )

        return (
            None,
            []
        )


def test_evaluate_case():

    retriever = FakeRetriever()

    runner = EvaluationRunner(
        retriever
    )

    case = EvaluationCase(
        case_id="case-1",
        query="What programming language do I prefer?",
        relevant_memory_ids=[
            "memory-python"
        ]
    )

    result = runner.evaluate_case(
        case,
        k=2
    )

    assert result.case_id == "case-1"

    assert result.query == (
        "What programming language do I prefer?"
    )

    assert result.retrieved_ids == [
        "memory-python",
        "memory-other"
    ]

    assert result.relevant_ids == [
        "memory-python"
    ]

    assert result.metrics.recall_at_k == 1.0

    assert result.metrics.hit_at_k == 1.0

    assert retriever.calls == [
        (
            "What programming language do I prefer?",
            2
        )
    ]


def test_evaluate_dataset():

    retriever = FakeRetriever()

    runner = EvaluationRunner(
        retriever
    )

    dataset = EvaluationDataset([
        EvaluationCase(
            case_id="case-1",
            query="What programming language do I prefer?",
            relevant_memory_ids=[
                "memory-python"
            ]
        )
    ])

    summary = runner.evaluate(
        dataset,
        k=2
    )

    assert summary.case_count == 1

    assert len(
        summary.results
    ) == 1

    assert summary.recall_at_k == 1.0

    assert summary.precision_at_k == 0.5

    assert summary.hit_at_k == 1.0

    assert summary.reciprocal_rank == 1.0


def test_evaluate_multiple_cases():

    retriever = FakeRetriever()

    runner = EvaluationRunner(
        retriever
    )

    dataset = EvaluationDataset([
        EvaluationCase(
            case_id="case-1",
            query="What programming language do I prefer?",
            relevant_memory_ids=[
                "memory-python"
            ]
        ),
        EvaluationCase(
            case_id="case-2",
            query="Something with no result",
            relevant_memory_ids=[
                "memory-missing"
            ]
        )
    ])

    summary = runner.evaluate(
        dataset,
        k=2
    )

    assert summary.case_count == 2

    assert len(
        summary.results
    ) == 2

    assert summary.recall_at_k == 0.5

    assert summary.hit_at_k == 0.5

    assert summary.reciprocal_rank == 0.5


def test_non_retrieval_cases_are_ignored():

    retriever = FakeRetriever()

    runner = EvaluationRunner(
        retriever
    )

    dataset = EvaluationDataset([
        EvaluationCase(
            case_id="retrieval",
            query="What programming language do I prefer?",
            relevant_memory_ids=[
                "memory-python"
            ]
        ),
        EvaluationCase(
            case_id="classification",
            query="I prefer Python.",
            expected_memory_type="semantic"
        )
    ])

    summary = runner.evaluate(
        dataset
    )

    assert summary.case_count == 1

    assert (
        summary.results[0].case_id
        == "retrieval"
    )


def test_empty_dataset():

    retriever = FakeRetriever()

    runner = EvaluationRunner(
        retriever
    )

    dataset = EvaluationDataset()

    summary = runner.evaluate(
        dataset
    )

    assert summary.case_count == 0

    assert summary.results == []

    assert summary.recall_at_k == 0.0
    assert summary.precision_at_k == 0.0
    assert summary.hit_at_k == 0.0
    assert summary.reciprocal_rank == 0.0
    assert summary.ndcg_at_k == 0.0


def test_empty_dataset_does_not_call_retriever():

    retriever = FakeRetriever()

    runner = EvaluationRunner(
        retriever
    )

    dataset = EvaluationDataset()

    runner.evaluate(
        dataset
    )

    assert retriever.calls == []


def test_k_must_be_positive():

    runner = EvaluationRunner(
        FakeRetriever()
    )

    dataset = EvaluationDataset([
        EvaluationCase(
            case_id="case-1",
            query="What do I prefer?",
            relevant_memory_ids=[
                "memory-1"
            ]
        )
    ])

    with pytest.raises(ValueError):

        runner.evaluate(
            dataset,
            k=0
        )


def test_none_dataset_rejected():

    runner = EvaluationRunner(
        FakeRetriever()
    )

    with pytest.raises(ValueError):

        runner.evaluate(
            None
        )


def test_none_retriever_rejected():

    with pytest.raises(ValueError):

        EvaluationRunner(
            None
        )


def test_case_with_missing_memory_ids_is_handled():

    retriever = FakeRetriever()

    runner = EvaluationRunner(
        retriever
    )

    case = EvaluationCase(
        case_id="case-1",
        query="What programming language do I prefer?",
        relevant_memory_ids=[
            "memory-python"
        ]
    )

    result = runner.evaluate_case(
        case,
        k=1
    )

    assert result.retrieved_ids == [
        "memory-python"
    ]

    assert result.metrics.hit_at_k == 1.0


def test_evaluate_case_requires_case():

    runner = EvaluationRunner(
        FakeRetriever()
    )

    with pytest.raises(ValueError):

        runner.evaluate_case(
            None
        )