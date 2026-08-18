from dataclasses import dataclass, field


@dataclass
class EvaluationCase:
    """
    Represents one ground-truth evaluation example.

    A case can be used to evaluate:
        - memory retrieval
        - memory classification
        - answer generation
    """

    case_id: str

    query: str

    relevant_memory_ids: list[str] = field(
        default_factory=list
    )

    expected_memory_type: str | None = None

    expected_answer: str | None = None

    metadata: dict = field(
        default_factory=dict
    )

    def __post_init__(self):
        if not self.case_id.strip():
            raise ValueError(
                "case_id cannot be empty"
            )

        if not self.query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if self.expected_memory_type is not None:
            if not self.expected_memory_type.strip():
                raise ValueError(
                    "expected_memory_type cannot be empty"
                )

    @property
    def has_retrieval_ground_truth(self) -> bool:
        """
        Return whether this case contains expected
        memory IDs for retrieval evaluation.
        """

        return bool(
            self.relevant_memory_ids
        )

    @property
    def has_answer_ground_truth(self) -> bool:
        """
        Return whether this case contains an expected
        answer for answer-quality evaluation.
        """

        return (
            self.expected_answer is not None
            and bool(
                self.expected_answer.strip()
            )
        )


class EvaluationDataset:
    """
    Collection of deterministic evaluation cases.

    The dataset is intentionally independent of the
    retrieval and LLM implementations.
    """

    def __init__(
        self,
        cases: list[EvaluationCase] | None = None
    ):
        self.cases = list(
            cases
            if cases is not None
            else []
        )

        self._validate_unique_ids()

    def add(
        self,
        case: EvaluationCase
    ) -> None:
        """
        Add one evaluation case.
        """

        if any(
            existing.case_id == case.case_id
            for existing in self.cases
        ):
            raise ValueError(
                f"Evaluation case already exists: "
                f"{case.case_id}"
            )

        self.cases.append(
            case
        )

    def get(
        self,
        case_id: str
    ) -> EvaluationCase | None:
        """
        Retrieve an evaluation case by ID.
        """

        for case in self.cases:
            if case.case_id == case_id:
                return case

        return None

    def count(self) -> int:
        """
        Return the number of evaluation cases.
        """

        return len(
            self.cases
        )

    def retrieval_cases(
        self
    ) -> list[EvaluationCase]:
        """
        Return cases that contain retrieval ground truth.
        """

        return [
            case
            for case in self.cases
            if case.has_retrieval_ground_truth
        ]

    def classification_cases(
        self
    ) -> list[EvaluationCase]:
        """
        Return cases that contain classification
        ground truth.
        """

        return [
            case
            for case in self.cases
            if case.expected_memory_type is not None
        ]

    def answer_cases(
        self
    ) -> list[EvaluationCase]:
        """
        Return cases that contain answer ground truth.
        """

        return [
            case
            for case in self.cases
            if case.has_answer_ground_truth
        ]

    def _validate_unique_ids(self) -> None:
        """
        Ensure every case has a unique identifier.
        """

        ids = [
            case.case_id
            for case in self.cases
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Evaluation case IDs must be unique."
            )