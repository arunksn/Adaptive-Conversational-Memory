from dataclasses import dataclass, field


@dataclass
class EvaluationCase:
    """
    Represents one ground-truth evaluation example.

    A case can be used to evaluate:
        - memory retrieval
        - procedural retrieval
        - memory classification
        - answer generation
    """

    case_id: str

    query: str

    # Memory-based retrieval ground truth.
    relevant_memory_ids: list[str] = field(
        default_factory=list
    )

    # Procedural retrieval ground truth.
    #
    # These are separate from relevant_memory_ids
    # because procedural retrieval operates on the
    # procedure/state graph.
    relevant_procedure_ids: list[str] = field(
        default_factory=list
    )

    relevant_state_ids: list[str] = field(
        default_factory=list
    )

    # Optional procedure context used during retrieval.
    procedure_id: str | None = None

    state_id: str | None = None

    # Classification ground truth.
    expected_memory_type: str | None = None

    # Answer-generation ground truth.
    expected_answer: str | None = None

    # Additional case metadata.
    metadata: dict = field(
        default_factory=dict
    )

    def __post_init__(self):
        if not isinstance(
            self.case_id,
            str
        ) or not self.case_id.strip():

            raise ValueError(
                "case_id cannot be empty"
            )

        if not isinstance(
            self.query,
            str
        ) or not self.query.strip():

            raise ValueError(
                "query cannot be empty"
            )

        if (
            self.expected_memory_type
            is not None
        ):

            if not isinstance(
                self.expected_memory_type,
                str
            ) or not self.expected_memory_type.strip():

                raise ValueError(
                    "expected_memory_type cannot be empty"
                )

    @property
    def has_retrieval_ground_truth(
        self
    ) -> bool:
        """
        Return whether this case contains any
        retrieval ground truth.
        """

        return (
            bool(
                self.relevant_memory_ids
            )
            or bool(
                self.relevant_procedure_ids
            )
            or bool(
                self.relevant_state_ids
            )
        )

    @property
    def has_memory_retrieval_ground_truth(
        self
    ) -> bool:
        """
        Return whether this case contains memory-ID
        retrieval ground truth.
        """

        return bool(
            self.relevant_memory_ids
        )

    @property
    def has_procedural_retrieval_ground_truth(
        self
    ) -> bool:
        """
        Return whether this case contains procedural
        graph retrieval ground truth.
        """

        return (
            bool(
                self.relevant_procedure_ids
            )
            or bool(
                self.relevant_state_ids
            )
        )

    @property
    def has_answer_ground_truth(
        self
    ) -> bool:
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

        if case is None:
            raise ValueError(
                "case cannot be None"
            )

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

    def memory_retrieval_cases(
        self
    ) -> list[EvaluationCase]:
        """
        Return cases that contain memory-ID retrieval
        ground truth.
        """

        return [
            case
            for case in self.cases
            if case.has_memory_retrieval_ground_truth
        ]

    def procedural_retrieval_cases(
        self
    ) -> list[EvaluationCase]:
        """
        Return cases that contain procedural graph
        retrieval ground truth.
        """

        return [
            case
            for case in self.cases
            if case.has_procedural_retrieval_ground_truth
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
        Return cases that contain an expected
        answer.
        """

        return [
            case
            for case in self.cases
            if case.has_answer_ground_truth
        ]

    def _validate_unique_ids(
        self
    ) -> None:
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