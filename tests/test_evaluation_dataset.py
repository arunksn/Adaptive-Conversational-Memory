import pytest

from src.evaluation.evaluation_dataset import (
    EvaluationCase,
    EvaluationDataset
)


def test_evaluation_case_creation():

    case = EvaluationCase(
        case_id="case-1",
        query="What programming language do I prefer?",
        relevant_memory_ids=[
            "memory-1"
        ],
        expected_memory_type="semantic",
        expected_answer="Python"
    )

    assert case.case_id == "case-1"
    assert case.query == (
        "What programming language do I prefer?"
    )
    assert case.relevant_memory_ids == [
        "memory-1"
    ]
    assert case.expected_memory_type == "semantic"
    assert case.expected_answer == "Python"


def test_retrieval_ground_truth():

    case = EvaluationCase(
        case_id="case-1",
        query="What do I prefer?",
        relevant_memory_ids=[
            "memory-1"
        ]
    )

    assert case.has_retrieval_ground_truth is True


def test_retrieval_ground_truth_missing():

    case = EvaluationCase(
        case_id="case-1",
        query="What do I prefer?"
    )

    assert case.has_retrieval_ground_truth is False


def test_answer_ground_truth():

    case = EvaluationCase(
        case_id="case-1",
        query="What do I prefer?",
        expected_answer="Python"
    )

    assert case.has_answer_ground_truth is True


def test_answer_ground_truth_missing():

    case = EvaluationCase(
        case_id="case-1",
        query="What do I prefer?"
    )

    assert case.has_answer_ground_truth is False


def test_empty_case_id_rejected():

    with pytest.raises(ValueError):

        EvaluationCase(
            case_id="",
            query="What do I prefer?"
        )


def test_empty_query_rejected():

    with pytest.raises(ValueError):

        EvaluationCase(
            case_id="case-1",
            query=""
        )


def test_empty_memory_type_rejected():

    with pytest.raises(ValueError):

        EvaluationCase(
            case_id="case-1",
            query="What do I prefer?",
            expected_memory_type=""
        )


def test_dataset_creation():

    cases = [
        EvaluationCase(
            case_id="case-1",
            query="What do I prefer?"
        ),
        EvaluationCase(
            case_id="case-2",
            query="What did I do yesterday?"
        )
    ]

    dataset = EvaluationDataset(
        cases
    )

    assert dataset.count() == 2


def test_dataset_add():

    dataset = EvaluationDataset()

    case = EvaluationCase(
        case_id="case-1",
        query="What do I prefer?"
    )

    dataset.add(
        case
    )

    assert dataset.count() == 1
    assert dataset.get(
        "case-1"
    ) is case


def test_dataset_duplicate_id_rejected():

    dataset = EvaluationDataset()

    case = EvaluationCase(
        case_id="case-1",
        query="What do I prefer?"
    )

    dataset.add(
        case
    )

    with pytest.raises(ValueError):

        dataset.add(
            EvaluationCase(
                case_id="case-1",
                query="Another question?"
            )
        )


def test_dataset_get_missing_case():

    dataset = EvaluationDataset()

    assert dataset.get(
        "missing"
    ) is None


def test_retrieval_cases():

    dataset = EvaluationDataset([
        EvaluationCase(
            case_id="retrieval",
            query="What do I prefer?",
            relevant_memory_ids=[
                "memory-1"
            ]
        ),
        EvaluationCase(
            case_id="other",
            query="Hello"
        )
    ])

    cases = dataset.retrieval_cases()

    assert len(cases) == 1
    assert cases[0].case_id == "retrieval"


def test_classification_cases():

    dataset = EvaluationDataset([
        EvaluationCase(
            case_id="semantic",
            query="What do I prefer?",
            expected_memory_type="semantic"
        ),
        EvaluationCase(
            case_id="other",
            query="Hello"
        )
    ])

    cases = dataset.classification_cases()

    assert len(cases) == 1
    assert cases[0].case_id == "semantic"


def test_answer_cases():

    dataset = EvaluationDataset([
        EvaluationCase(
            case_id="answer",
            query="What do I prefer?",
            expected_answer="Python"
        ),
        EvaluationCase(
            case_id="other",
            query="Hello"
        )
    ])

    cases = dataset.answer_cases()

    assert len(cases) == 1
    assert cases[0].case_id == "answer"


def test_dataset_rejects_duplicate_ids_on_creation():

    cases = [
        EvaluationCase(
            case_id="case-1",
            query="Question one"
        ),
        EvaluationCase(
            case_id="case-1",
            query="Question two"
        )
    ]

    with pytest.raises(ValueError):

        EvaluationDataset(
            cases
        )