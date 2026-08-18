import pytest
import math  # used for log2 in ndcg calculation

from src.evaluation.metrics import (
    ClassificationMetricsCalculator,
    QAMetricsCalculator,
    RetrievalMetricsCalculator
)



def test_recall_at_k():

    calculator = (
        RetrievalMetricsCalculator()
    )

    score = calculator.recall_at_k(
        retrieved_ids=[
            "A",
            "B",
            "C"
        ],
        relevant_ids=[
            "B",
            "C"
        ],
        k=3
    )

    assert score == 1.0


def test_recall_at_k_partial():

    calculator = (
        RetrievalMetricsCalculator()
    )

    score = calculator.recall_at_k(
        retrieved_ids=[
            "A",
            "B"
        ],
        relevant_ids=[
            "B",
            "C"
        ],
        k=2
    )

    assert score == 0.5


def test_precision_at_k():

    calculator = (
        RetrievalMetricsCalculator()
    )

    score = calculator.precision_at_k(
        retrieved_ids=[
            "A",
            "B",
            "C",
            "D"
        ],
        relevant_ids=[
            "B",
            "D"
        ],
        k=4
    )

    assert score == 0.5


def test_hit_at_k():

    calculator = (
        RetrievalMetricsCalculator()
    )

    score = calculator.hit_at_k(
        retrieved_ids=[
            "A",
            "B",
            "C"
        ],
        relevant_ids=[
            "C"
        ],
        k=3
    )

    assert score == 1.0


def test_hit_at_k_miss():

    calculator = (
        RetrievalMetricsCalculator()
    )

    score = calculator.hit_at_k(
        retrieved_ids=[
            "A",
            "B",
            "C"
        ],
        relevant_ids=[
            "D"
        ],
        k=3
    )

    assert score == 0.0


def test_reciprocal_rank_first_result():

    calculator = (
        RetrievalMetricsCalculator()
    )

    score = calculator.reciprocal_rank(
        retrieved_ids=[
            "A",
            "B",
            "C"
        ],
        relevant_ids=[
            "A"
        ],
        k=3
    )

    assert score == 1.0


def test_reciprocal_rank_second_result():

    calculator = (
        RetrievalMetricsCalculator()
    )

    score = calculator.reciprocal_rank(
        retrieved_ids=[
            "A",
            "B",
            "C"
        ],
        relevant_ids=[
            "B"
        ],
        k=3
    )

    assert score == 0.5


def test_reciprocal_rank_no_match():

    calculator = (
        RetrievalMetricsCalculator()
    )

    score = calculator.reciprocal_rank(
        retrieved_ids=[
            "A",
            "B"
        ],
        relevant_ids=[
            "C"
        ],
        k=2
    )

    assert score == 0.0


def test_ndcg_perfect_result():

    calculator = (
        RetrievalMetricsCalculator()
    )

    score = calculator.ndcg_at_k(
        retrieved_ids=[
            "A",
            "B",
            "C"
        ],
        relevant_ids=[
            "A",
            "B"
        ],
        k=3
    )

    assert score == 1.0


def test_retrieval_metrics_combined():

    calculator = (
        RetrievalMetricsCalculator()
    )

    metrics = calculator.evaluate(
        retrieved_ids=[
            "A",
            "B",
            "C"
        ],
        relevant_ids=[
            "B"
        ],
        k=3
    )

    assert metrics.recall_at_k == 1.0
    assert metrics.precision_at_k == pytest.approx(
        1 / 3
    )
    assert metrics.hit_at_k == 1.0
    assert metrics.reciprocal_rank == 0.5
    assert metrics.ndcg_at_k == pytest.approx(
        1 / math.log2(3) # is not equal to 1 / 3, though
    )


def test_empty_retrieval():

    calculator = (
        RetrievalMetricsCalculator()
    )

    metrics = calculator.evaluate(
        retrieved_ids=[],
        relevant_ids=["A"],
        k=5
    )

    assert metrics.recall_at_k == 0.0
    assert metrics.precision_at_k == 0.0
    assert metrics.hit_at_k == 0.0
    assert metrics.reciprocal_rank == 0.0
    assert metrics.ndcg_at_k == 0.0


def test_retrieval_invalid_k():

    calculator = (
        RetrievalMetricsCalculator()
    )

    with pytest.raises(
        ValueError
    ):
        calculator.recall_at_k(
            retrieved_ids=[],
            relevant_ids=[],
            k=0
        )




def test_classification_metrics():

    calculator = (
        ClassificationMetricsCalculator()
    )

    metrics = calculator.evaluate(
        predictions=[
            "semantic",
            "episodic",
            "procedural",
            "semantic"
        ],
        targets=[
            "semantic",
            "episodic",
            "procedural",
            "episodic"
        ]
    )

    assert metrics.accuracy == 0.75
    assert 0.0 <= metrics.macro_precision <= 1.0
    assert 0.0 <= metrics.macro_recall <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0


def test_classification_mismatched_lengths():

    calculator = (
        ClassificationMetricsCalculator()
    )

    with pytest.raises(
        ValueError
    ):
        calculator.evaluate(
            predictions=[
                "semantic"
            ],
            targets=[
                "semantic",
                "episodic"
            ]
        )


def test_empty_classification():

    calculator = (
        ClassificationMetricsCalculator()
    )

    metrics = calculator.evaluate(
        predictions=[],
        targets=[]
    )

    assert metrics.accuracy == 0.0
    assert metrics.macro_precision == 0.0
    assert metrics.macro_recall == 0.0
    assert metrics.macro_f1 == 0.0



def test_qa_exact_match():

    calculator = (
        QAMetricsCalculator()
    )

    result = calculator.evaluate(
        prediction="I prefer Python.",
        reference="I prefer Python."
    )

    assert result.correct is True
    assert result.exact_match == 1.0
    assert result.token_f1 == 1.0


def test_qa_token_f1_partial_match():

    calculator = (
        QAMetricsCalculator()
    )

    result = calculator.evaluate(
        prediction="I prefer Python",
        reference="I prefer Python programming"
    )

    assert result.correct is False
    assert result.exact_match == 0.0
    assert result.token_f1 > 0.0
    assert result.token_f1 < 1.0


def test_qa_no_overlap():

    calculator = (
        QAMetricsCalculator()
    )

    result = calculator.evaluate(
        prediction="I prefer Java",
        reference="I prefer Python"
    )

    assert result.correct is False
    assert result.exact_match == 0.0
    assert result.token_f1 > 0.0


def test_qa_empty_answers():

    calculator = (
        QAMetricsCalculator()
    )

    result = calculator.evaluate(
        prediction="",
        reference=""
    )

    assert result.correct is True
    assert result.exact_match == 1.0
    assert result.token_f1 == 1.0