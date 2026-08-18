from src.evaluation.benchmark_dataset import (
    AdaptiveMemoryBenchmarkDataset
)

from src.evaluation.evaluation_dataset import (
    EvaluationDataset
)


def test_build_returns_dataset():

    dataset = (
        AdaptiveMemoryBenchmarkDataset.build()
    )

    assert isinstance(
        dataset,
        EvaluationDataset
    )


def test_dataset_contains_cases():

    dataset = (
        AdaptiveMemoryBenchmarkDataset.build()
    )

    cases = dataset.retrieval_cases()

    assert len(cases) == 20


def test_cases_have_unique_ids():

    dataset = (
        AdaptiveMemoryBenchmarkDataset.build()
    )

    cases = dataset.retrieval_cases()

    case_ids = [
        case.case_id
        for case in cases
    ]

    assert len(case_ids) == len(
        set(case_ids)
    )


def test_cases_have_queries():

    dataset = (
        AdaptiveMemoryBenchmarkDataset.build()
    )

    for case in dataset.retrieval_cases():

        assert case.query.strip()


def test_cases_have_relevant_memory_ids():

    dataset = (
        AdaptiveMemoryBenchmarkDataset.build()
    )

    for case in dataset.retrieval_cases():

        assert case.relevant_memory_ids


def test_semantic_cases_exist():

    dataset = (
        AdaptiveMemoryBenchmarkDataset.build()
    )

    case_ids = {
        case.case_id
        for case in dataset.retrieval_cases()
    }

    assert (
        "semantic-preference-001"
        in case_ids
    )

    assert (
        "semantic-goal-001"
        in case_ids
    )


def test_episodic_cases_exist():

    dataset = (
        AdaptiveMemoryBenchmarkDataset.build()
    )

    case_ids = {
        case.case_id
        for case in dataset.retrieval_cases()
    }

    assert (
        "episodic-event-001"
        in case_ids
    )


def test_procedural_cases_exist():

    dataset = (
        AdaptiveMemoryBenchmarkDataset.build()
    )

    case_ids = {
        case.case_id
        for case in dataset.retrieval_cases()
    }

    assert (
        "procedural-001"
        in case_ids
    )


def test_temporal_cases_exist():

    dataset = (
        AdaptiveMemoryBenchmarkDataset.build()
    )

    case_ids = {
        case.case_id
        for case in dataset.retrieval_cases()
    }

    assert (
        "temporal-001"
        in case_ids
    )

    assert (
        "temporal-002"
        in case_ids
    )


def test_conflict_cases_exist():

    dataset = (
        AdaptiveMemoryBenchmarkDataset.build()
    )

    case_ids = {
        case.case_id
        for case in dataset.retrieval_cases()
    }

    assert (
        "conflict-preference-001"
        in case_ids
    )


def test_consolidation_cases_exist():

    dataset = (
        AdaptiveMemoryBenchmarkDataset.build()
    )

    case_ids = {
        case.case_id
        for case in dataset.retrieval_cases()
    }

    assert (
        "consolidation-001"
        in case_ids
    )


def test_retrieval_cases_method_matches_build():

    dataset = (
        AdaptiveMemoryBenchmarkDataset.build()
    )

    cases = (
        AdaptiveMemoryBenchmarkDataset
        .retrieval_cases()
    )

    assert len(cases) == len(
        dataset.retrieval_cases()
    )