from dataclasses import dataclass

import pytest

from src.evaluation.evaluation_report import (
    EvaluationReport,
    EvaluationReportGenerator
)

from src.evaluation.evaluation_runner import (
    RetrievalEvaluationResult,
    RetrievalEvaluationSummary
)


@dataclass
class FakeMetrics:

    recall_at_k: float
    precision_at_k: float
    hit_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


def create_summary():
    result = RetrievalEvaluationResult(
        case_id="case-1",
        query="What programming language do I prefer?",
        retrieved_ids=[
            "memory-python",
            "memory-other"
        ],
        relevant_ids=[
            "memory-python"
        ],
        metrics=FakeMetrics(
            recall_at_k=1.0,
            precision_at_k=0.5,
            hit_at_k=1.0,
            reciprocal_rank=1.0,
            ndcg_at_k=1.0
        )
    )

    return RetrievalEvaluationSummary(
        results=[result],
        recall_at_k=1.0,
        precision_at_k=0.5,
        hit_at_k=1.0,
        reciprocal_rank=1.0,
        ndcg_at_k=1.0,
        k=2,
        case_count=1
    )


def test_generate_returns_evaluation_report():

    generator = EvaluationReportGenerator()

    summary = create_summary()

    report = generator.generate(
        summary
    )

    assert isinstance(
        report,
        EvaluationReport
    )

    assert report.summary is summary


def test_report_contains_header():

    generator = EvaluationReportGenerator()

    report = generator.generate(
        create_summary()
    )

    assert (
        "Adaptive Conversational Memory"
        in report.text
    )


def test_report_contains_summary():

    generator = EvaluationReportGenerator()

    report = generator.generate(
        create_summary()
    )

    assert (
        "Cases evaluated: 1"
        in report.text
    )

    assert (
        "K: 2"
        in report.text
    )


def test_report_contains_all_metrics():

    generator = EvaluationReportGenerator()

    report = generator.generate(
        create_summary()
    )

    assert "Recall@2:       1.000" in report.text
    assert "Precision@2:    0.500" in report.text
    assert "Hit@2:          1.000" in report.text
    assert "MRR:                      1.000" in report.text
    assert "NDCG@2:         1.000" in report.text


def test_report_contains_case_results():

    generator = EvaluationReportGenerator()

    report = generator.generate(
        create_summary()
    )

    assert (
        "case-1"
        in report.text
    )

    assert (
        "Recall: 1.000"
        in report.text
    )

    assert (
        "Precision: 0.500"
        in report.text
    )


def test_generate_text_returns_string():

    generator = EvaluationReportGenerator()

    text = generator.generate_text(
        create_summary()
    )

    assert isinstance(
        text,
        str
    )

    assert (
        "Adaptive Conversational Memory"
        in text
    )


def test_none_summary_rejected():

    generator = EvaluationReportGenerator()

    with pytest.raises(ValueError):

        generator.generate(
            None
        )


def test_empty_results_report():

    generator = EvaluationReportGenerator()

    summary = RetrievalEvaluationSummary(
        results=[],
        recall_at_k=0.0,
        precision_at_k=0.0,
        hit_at_k=0.0,
        reciprocal_rank=0.0,
        ndcg_at_k=0.0,
        k=5,
        case_count=0
    )

    report = generator.generate(
        summary
    )

    assert (
        "Cases evaluated: 0"
        in report.text
    )

    assert (
        "No retrieval cases were evaluated."
        in report.text
    )


def test_report_preserves_original_summary():

    generator = EvaluationReportGenerator()

    summary = create_summary()

    report = generator.generate(
        summary
    )

    assert report.summary is summary


def test_report_metric_precision():

    generator = EvaluationReportGenerator()

    summary = RetrievalEvaluationSummary(
        results=[],
        recall_at_k=0.123456,
        precision_at_k=0.654321,
        hit_at_k=0.987654,
        reciprocal_rank=0.333333,
        ndcg_at_k=0.777777,
        k=10,
        case_count=0
    )

    report = generator.generate(
        summary
    )

    assert "Recall@10:       0.123" in report.text
    assert "Precision@10:    0.654" in report.text
    assert "Hit@10:          0.988" in report.text
    assert "MRR:                      0.333" in report.text
    assert "NDCG@10:         0.778" in report.text