from dataclasses import dataclass

import pytest

from src.evaluation.benchmark import (
    BenchmarkResult,
    MemoryBenchmark
)

from src.evaluation.evaluation_dataset import (
    EvaluationCase,
    EvaluationDataset
)


@dataclass
class FakeMetrics:

    recall_at_k: float
    precision_at_k: float
    hit_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


class FakeRetrievalResult:

    def __init__(
        self,
        memory_id: str
    ):
        self.memory_id = memory_id


class FakeRetriever:

    def __init__(
        self,
        results=None
    ):
        self.results = (
            results
            if results is not None
            else []
        )

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

        return (
            None,
            self.results[:top_k]
        )


# ORIGINAL TWO-SYSTEM TESTS


def test_benchmark_initialization():

    baseline = FakeRetriever()

    adaptive = FakeRetriever()

    benchmark = MemoryBenchmark(
        baseline,
        adaptive
    )

    assert benchmark.baseline_runner is not None
    assert benchmark.adaptive_runner is not None


def test_none_baseline_rejected():

    with pytest.raises(ValueError):

        MemoryBenchmark(
            None,
            FakeRetriever()
        )


def test_none_adaptive_rejected():

    with pytest.raises(ValueError):

        MemoryBenchmark(
            FakeRetriever(),
            None
        )


def test_run_evaluates_both_systems():

    baseline = FakeRetriever([
        FakeRetrievalResult(
            "memory-other"
        )
    ])

    adaptive = FakeRetriever([
        FakeRetrievalResult(
            "memory-python"
        )
    ])

    benchmark = MemoryBenchmark(
        baseline,
        adaptive
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

    result = benchmark.run(
        dataset,
        k=1
    )

    assert isinstance(
        result,
        BenchmarkResult
    )

    assert (
        result.baseline.recall_at_k
        == 0.0
    )

    assert (
        result.adaptive.recall_at_k
        == 1.0
    )


def test_same_dataset_is_used():

    baseline = FakeRetriever()

    adaptive = FakeRetriever()

    benchmark = MemoryBenchmark(
        baseline,
        adaptive
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

    benchmark.run(
        dataset,
        k=3
    )

    assert baseline.calls == [
        (
            "What do I prefer?",
            3
        )
    ]

    assert adaptive.calls == [
        (
            "What do I prefer?",
            3
        )
    ]


def test_k_must_be_positive():

    benchmark = MemoryBenchmark(
        FakeRetriever(),
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

        benchmark.run(
            dataset,
            k=0
        )


def test_none_dataset_rejected():

    benchmark = MemoryBenchmark(
        FakeRetriever(),
        FakeRetriever()
    )

    with pytest.raises(ValueError):

        benchmark.run(
            None
        )


def test_improvement():

    assert (
        MemoryBenchmark.improvement(
            0.60,
            0.80
        )
        == pytest.approx(0.20)
    )


def test_relative_improvement():

    assert (
        MemoryBenchmark.relative_improvement(
            0.50,
            0.75
        )
        == pytest.approx(50.0)
    )


def test_relative_improvement_zero_baseline():

    assert (
        MemoryBenchmark.relative_improvement(
            0.0,
            0.75
        )
        == 0.0
    )


def test_compare_metrics():

    baseline = FakeRetriever()
    adaptive = FakeRetriever()

    benchmark = MemoryBenchmark(
        baseline,
        adaptive
    )

    dataset = EvaluationDataset()

    result = BenchmarkResult(
        baseline=benchmark.baseline_runner.evaluate(
            dataset
        ),
        adaptive=benchmark.adaptive_runner.evaluate(
            dataset
        )
    )

    comparison = (
        MemoryBenchmark.compare_metrics(
            result
        )
    )

    assert comparison["recall_at_k"]["baseline"] == 0.0
    assert comparison["recall_at_k"]["adaptive"] == 0.0
    assert comparison["recall_at_k"]["improvement"] == 0.0


def test_compare_metrics_rejects_none():

    with pytest.raises(ValueError):

        MemoryBenchmark.compare_metrics(
            None
        )


# MULTI-SYSTEM BENCHMARK TESTS


def test_multi_system_initialization():

    systems = {
        "baseline_rag": FakeRetriever(),
        "vector_memory": FakeRetriever(),
        "vector_kg": FakeRetriever(),
        "hybrid_memory": FakeRetriever(),
        "adaptive_memory": FakeRetriever()
    }

    benchmark = MemoryBenchmark(
        systems=systems
    )

    assert set(
        benchmark.runners.keys()
    ) == {
        "baseline_rag",
        "vector_memory",
        "vector_kg",
        "hybrid_memory",
        "adaptive_memory"
    }


def test_multi_system_none_retriever_rejected():

    systems = {
        "baseline_rag": FakeRetriever(),
        "vector_memory": None,
        "vector_kg": FakeRetriever()
    }

    with pytest.raises(ValueError):

        MemoryBenchmark(
            systems=systems
        )


def test_multi_system_empty_rejected():

    with pytest.raises(ValueError):

        MemoryBenchmark(
            systems={}
        )


def test_multi_system_invalid_dictionary_rejected():

    with pytest.raises(ValueError):

        MemoryBenchmark(
            systems=[]
        )


def test_multi_system_benchmark_evaluates_all_systems():

    systems = {
        "baseline_rag": FakeRetriever([
            FakeRetrievalResult(
                "memory-other"
            )
        ]),
        "vector_memory": FakeRetriever([
            FakeRetrievalResult(
                "memory-python"
            )
        ]),
        "vector_kg": FakeRetriever([
            FakeRetrievalResult(
                "memory-python"
            )
        ]),
        "hybrid_memory": FakeRetriever([
            FakeRetrievalResult(
                "memory-python"
            )
        ]),
        "adaptive_memory": FakeRetriever([
            FakeRetrievalResult(
                "memory-python"
            )
        ])
    }

    benchmark = MemoryBenchmark(
        systems=systems
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

    result = benchmark.run(
        dataset,
        k=1
    )

    assert isinstance(
        result,
        BenchmarkResult
    )

    assert set(
        result.systems.keys()
    ) == {
        "baseline_rag",
        "vector_memory",
        "vector_kg",
        "hybrid_memory",
        "adaptive_memory"
    }

    assert (
        result.systems[
            "baseline_rag"
        ].recall_at_k
        == 0.0
    )

    assert (
        result.systems[
            "vector_memory"
        ].recall_at_k
        == 1.0
    )

    assert (
        result.systems[
            "vector_kg"
        ].recall_at_k
        == 1.0
    )

    assert (
        result.systems[
            "hybrid_memory"
        ].recall_at_k
        == 1.0
    )

    assert (
        result.systems[
            "adaptive_memory"
        ].recall_at_k
        == 1.0
    )


def test_multi_system_uses_same_dataset_and_k():

    systems = {
        "baseline_rag": FakeRetriever(),
        "vector_memory": FakeRetriever(),
        "vector_kg": FakeRetriever(),
        "hybrid_memory": FakeRetriever(),
        "adaptive_memory": FakeRetriever()
    }

    benchmark = MemoryBenchmark(
        systems=systems
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

    benchmark.run(
        dataset,
        k=3
    )

    for retriever in systems.values():

        assert retriever.calls == [
            (
                "What do I prefer?",
                3
            )
        ]


def test_compare_systems():

    systems = {
        "baseline_rag": FakeRetriever(),
        "vector_memory": FakeRetriever(),
        "vector_kg": FakeRetriever(),
        "hybrid_memory": FakeRetriever(),
        "adaptive_memory": FakeRetriever()
    }

    benchmark = MemoryBenchmark(
        systems=systems
    )

    dataset = EvaluationDataset()

    result = benchmark.run(
        dataset
    )

    comparison = (
        MemoryBenchmark.compare_systems(
            result
        )
    )

    assert set(
        comparison.keys()
    ) == {
        "baseline_rag",
        "vector_memory",
        "vector_kg",
        "hybrid_memory",
        "adaptive_memory"
    }

    for metrics in comparison.values():

        assert "recall_at_k" in metrics
        assert "precision_at_k" in metrics
        assert "hit_at_k" in metrics
        assert "reciprocal_rank" in metrics
        assert "ndcg_at_k" in metrics


def test_compare_systems_rejects_none():

    with pytest.raises(ValueError):

        MemoryBenchmark.compare_systems(
            None
        )