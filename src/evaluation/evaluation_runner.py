from dataclasses import dataclass

from src.evaluation.evaluation_dataset import (
    EvaluationCase,
    EvaluationDataset
)

from src.evaluation.metrics import (
    RetrievalMetricsCalculator
)


@dataclass
class RetrievalEvaluationResult:
    """
    Evaluation result for one retrieval case.
    """

    case_id: str
    query: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    metrics: object


@dataclass
class RetrievalEvaluationSummary:
    """
    Aggregated retrieval evaluation result.
    """

    results: list[RetrievalEvaluationResult]

    recall_at_k: float
    precision_at_k: float
    hit_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float

    k: int
    case_count: int


class EvaluationRunner:
    """
    Runs retrieval evaluation against an evaluation dataset.

    The runner does not modify the retrieval system.
    It only collects retrieved memory IDs and compares
    them against the ground-truth relevant IDs.
    """

    def __init__(
        self,
        retriever,
        metrics_calculator=None
    ):
        if retriever is None:
            raise ValueError(
                "retriever cannot be None"
            )

        self.retriever = retriever

        self.metrics_calculator = (
            metrics_calculator
            if metrics_calculator is not None
            else RetrievalMetricsCalculator()
        )

    # EVALUATE DATASET

    def evaluate(
        self,
        dataset: EvaluationDataset,
        k: int = 5
    ) -> RetrievalEvaluationSummary:
        """
        Evaluate all retrieval cases in a dataset.
        """

        if dataset is None:
            raise ValueError(
                "dataset cannot be None"
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0"
            )

        cases = dataset.retrieval_cases()

        if not cases:
            return RetrievalEvaluationSummary(
                results=[],
                recall_at_k=0.0,
                precision_at_k=0.0,
                hit_at_k=0.0,
                reciprocal_rank=0.0,
                ndcg_at_k=0.0,
                k=k,
                case_count=0
            )

        results = []

        for case in cases:

            result = self.evaluate_case(
                case=case,
                k=k
            )

            results.append(
                result
            )

        return self._aggregate(
            results=results,
            k=k
        )

    # SINGLE CASE

    def evaluate_case(
        self,
        case: EvaluationCase,
        k: int = 5
    ) -> RetrievalEvaluationResult:
        """
        Evaluate one retrieval case.

        Only the first k retrieved results are evaluated,
        even if the retriever returns more than k results.
        """

        if case is None:
            raise ValueError(
                "case cannot be None"
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0"
            )

        retrieval_result = self.retriever.retrieve(
            query=case.query,
            top_k=k
        )

        # The retriever normally returns:
        #
        #     (route, retrieved_results)
        #
        # Keep this defensive so the evaluation runner
        # can also handle a retriever that directly returns
        # the retrieved results.

        if (
            isinstance(
                retrieval_result,
                tuple
            )
            and len(retrieval_result) == 2
        ):
            _, retrieved = retrieval_result
        else:
            retrieved = retrieval_result

        # Evaluation at K must only consider the first K
        # retrieved memories.
        retrieved_ids = (
            self._extract_memory_ids(
                retrieved
            )[:k]
        )

        metrics = (
            self.metrics_calculator.evaluate(
                retrieved_ids=retrieved_ids,
                relevant_ids=list(
                    case.relevant_memory_ids
                ),
                k=k
            )
        )

        return RetrievalEvaluationResult(
            case_id=case.case_id,
            query=case.query,
            retrieved_ids=retrieved_ids,
            relevant_ids=list(
                case.relevant_memory_ids
            ),
            metrics=metrics
        )

    # MEMORY ID EXTRACTION

    @staticmethod
    def _extract_memory_ids(
        results
    ) -> list[str]:
        """
        Extract memory IDs from retrieval results.

        The normal RetrievalResult stores the ID directly
        in memory_id.

        This method also supports results where the ID is
        available from the underlying memory object.

        Results without an identifiable memory ID are
        ignored rather than causing evaluation to fail.
        """

        if results is None:
            return []

        memory_ids = []

        for result in results:

            memory_id = getattr(
                result,
                "memory_id",
                None
            )

            # Try the underlying memory object.
            if memory_id is None:

                item = getattr(
                    result,
                    "item",
                    None
                )

                if item is not None:

                    memory_id = getattr(
                        item,
                        "memory_id",
                        None
                    )

            # Finally check metadata.
            if memory_id is None:

                metadata = getattr(
                    result,
                    "metadata",
                    None
                )

                if isinstance(
                    metadata,
                    dict
                ):

                    memory_id = metadata.get(
                        "memory_id"
                    )

            if memory_id is not None:

                memory_ids.append(
                    str(memory_id)
                )

        return memory_ids

    # AGGREGATION

    @staticmethod
    def _aggregate(
        results: list[RetrievalEvaluationResult],
        k: int
    ) -> RetrievalEvaluationSummary:
        """
        Aggregate metrics across all evaluated cases.
        """

        if not results:
            return RetrievalEvaluationSummary(
                results=[],
                recall_at_k=0.0,
                precision_at_k=0.0,
                hit_at_k=0.0,
                reciprocal_rank=0.0,
                ndcg_at_k=0.0,
                k=k,
                case_count=0
            )

        recall = sum(
            result.metrics.recall_at_k
            for result in results
        ) / len(results)

        precision = sum(
            result.metrics.precision_at_k
            for result in results
        ) / len(results)

        hit = sum(
            result.metrics.hit_at_k
            for result in results
        ) / len(results)

        reciprocal_rank = sum(
            result.metrics.reciprocal_rank
            for result in results
        ) / len(results)

        ndcg = sum(
            result.metrics.ndcg_at_k
            for result in results
        ) / len(results)

        return RetrievalEvaluationSummary(
            results=results,
            recall_at_k=recall,
            precision_at_k=precision,
            hit_at_k=hit,
            reciprocal_rank=reciprocal_rank,
            ndcg_at_k=ndcg,
            k=k,
            case_count=len(results)
        )