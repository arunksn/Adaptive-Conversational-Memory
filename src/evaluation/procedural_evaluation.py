from dataclasses import dataclass

from src.evaluation.evaluation_dataset import (
    EvaluationDataset,
)


@dataclass
class ProceduralEvaluationResult:
    """
    Result for one procedural retrieval case.
    """

    case_id: str
    query: str

    retrieved_procedure_ids: list[str]
    retrieved_state_ids: list[str]

    relevant_procedure_ids: list[str]
    relevant_state_ids: list[str]

    state_hit: bool
    procedure_hit: bool


@dataclass
class ProceduralEvaluationSummary:
    """
    Aggregated procedural retrieval evaluation.
    """

    results: list[ProceduralEvaluationResult]

    procedure_accuracy: float
    state_hit_rate: float

    case_count: int


class ProceduralEvaluationRunner:
    """
    Evaluates procedural graph retrieval independently
    from ordinary memory retrieval metrics.
    """

    def __init__(
        self,
        retriever,
    ):
        if retriever is None:
            raise ValueError(
                "retriever cannot be None"
            )

        self.retriever = retriever

    def evaluate(
        self,
        dataset: EvaluationDataset,
        k: int = 5,
    ) -> ProceduralEvaluationSummary:

        if dataset is None:
            raise ValueError(
                "dataset cannot be None"
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0"
            )

        cases = (
            dataset.procedural_retrieval_cases()
        )

        if not cases:

            return ProceduralEvaluationSummary(
                results=[],
                procedure_accuracy=0.0,
                state_hit_rate=0.0,
                case_count=0,
            )

        results = []

        for case in cases:

            results.append(
                self.evaluate_case(
                    case,
                    k=k,
                )
            )

        procedure_accuracy = (
            sum(
                result.procedure_hit
                for result in results
            )
            / len(results)
        )

        state_hit_rate = (
            sum(
                result.state_hit
                for result in results
            )
            / len(results)
        )

        return ProceduralEvaluationSummary(
            results=results,
            procedure_accuracy=procedure_accuracy,
            state_hit_rate=state_hit_rate,
            case_count=len(results),
        )

    def evaluate_case(
        self,
        case,
        k: int = 5,
    ) -> ProceduralEvaluationResult:

        if case is None:
            raise ValueError(
                "case cannot be None"
            )

        retrieval_result = (
            self.retriever.retrieve(
                query=case.query,
                top_k=k,
                procedure_id=case.procedure_id,
                state_id=case.state_id,
            )
        )

        if (
            isinstance(
                retrieval_result,
                tuple,
            )
            and len(retrieval_result) == 2
        ):
            _, retrieved = retrieval_result
        else:
            retrieved = retrieval_result

        if retrieved is None:
            retrieved = []

        retrieved = list(
            retrieved
        )[:k]

        procedure_ids = []
        state_ids = []

        for result in retrieved:

            metadata = getattr(
                result,
                "metadata",
                {},
            )

            if isinstance(
                metadata,
                dict,
            ):

                procedure_id = metadata.get(
                    "procedure_id"
                )

                state_id = metadata.get(
                    "state_id"
                )

                if (
                    procedure_id is not None
                    and procedure_id
                    not in procedure_ids
                ):

                    procedure_ids.append(
                        str(procedure_id)
                    )

                if (
                    state_id is not None
                    and state_id
                    not in state_ids
                ):

                    state_ids.append(
                        str(state_id)
                    )

            item = getattr(
                result,
                "item",
                None,
            )

            if item is not None:

                item_state_id = getattr(
                    item,
                    "state_id",
                    None,
                )

                if (
                    item_state_id is not None
                    and str(item_state_id)
                    not in state_ids
                ):

                    state_ids.append(
                        str(item_state_id)
                    )

        relevant_procedure_ids = list(
            case.relevant_procedure_ids
        )

        relevant_state_ids = list(
            case.relevant_state_ids
        )

        procedure_hit = bool(
            set(procedure_ids)
            & set(relevant_procedure_ids)
        )

        state_hit = bool(
            set(state_ids)
            & set(relevant_state_ids)
        )

        return ProceduralEvaluationResult(
            case_id=case.case_id,
            query=case.query,

            retrieved_procedure_ids=(
                procedure_ids
            ),

            retrieved_state_ids=(
                state_ids
            ),

            relevant_procedure_ids=(
                relevant_procedure_ids
            ),

            relevant_state_ids=(
                relevant_state_ids
            ),

            state_hit=state_hit,
            procedure_hit=procedure_hit,
        )