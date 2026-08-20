from dataclasses import dataclass, field

from src.evaluation.evaluation_dataset import (
    EvaluationCase,
    EvaluationDataset,
)

from src.evaluation.metrics import (
    RetrievalMetricsCalculator,
)


@dataclass
class RetrievalEvaluationResult:
    """
    Evaluation result for one retrieval case.

    The original memory-ID fields remain compatible
    with the existing evaluation/report system.

    Procedural retrieval fields are optional and therefore
    do not break existing callers.
    """

    case_id: str

    query: str

    retrieved_ids: list[str]

    relevant_ids: list[str]

    metrics: object

    # Procedural retrieval results.
    retrieved_procedure_ids: list[str] = field(
        default_factory=list
    )

    retrieved_state_ids: list[str] = field(
        default_factory=list
    )

    relevant_procedure_ids: list[str] = field(
        default_factory=list
    )

    relevant_state_ids: list[str] = field(
        default_factory=list
    )


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

    It collects retrieved memory IDs and compares them
    against memory-based ground truth.

    Procedural ground truth is stored on the evaluation
    result for future procedural-specific metrics.
    """

    def __init__(
        self,
        retriever,
        metrics_calculator=None,
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

 

    def evaluate(
        self,
        dataset: EvaluationDataset,
        k: int = 5,
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
                case_count=0,
            )

        results = []

        for case in cases:

            result = self.evaluate_case(
                case=case,
                k=k,
            )

            results.append(
                result
            )

        return self._aggregate(
            results=results,
            k=k,
        )

  

    def evaluate_case(
        self,
        case: EvaluationCase,
        k: int = 5,
    ) -> RetrievalEvaluationResult:
        """
        Evaluate one retrieval case.

        Only the first k retrieved results are evaluated.
        """

        if case is None:

            raise ValueError(
                "case cannot be None"
            )

        if k <= 0:

            raise ValueError(
                "k must be greater than 0"
            )


        retrieval_kwargs = {
            "query": case.query,
            "top_k": k,
        }

        if case.procedure_id is not None:

            retrieval_kwargs[
                "procedure_id"
            ] = case.procedure_id

        if case.state_id is not None:

            retrieval_kwargs[
                "state_id"
            ] = case.state_id


        try:

            retrieval_result = (
                self.retriever.retrieve(
                    **retrieval_kwargs
                )
            )

        except TypeError:

            # Preserve compatibility with simple
            # retrievers that only accept query/top_k.

            retrieval_result = (
                self.retriever.retrieve(
                    query=case.query,
                    top_k=k,
                )
            )

        # Retrievers may return:
        #
        #     (metadata, results)
        #
        # or:
        #
        #     results

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
        )

     

        retrieved_ids = (
            self._extract_memory_ids(
                retrieved
            )[:k]
        )

        

        (
            retrieved_procedure_ids,
            retrieved_state_ids,
        ) = self._extract_procedural_ids(
            retrieved
        )

        retrieved_procedure_ids = (
            retrieved_procedure_ids[:k]
        )

        retrieved_state_ids = (
            retrieved_state_ids[:k]
        )

       
        # Memory retrieval metrics.
        #
        # Existing metrics remain unchanged.

        metrics = (
            self.metrics_calculator.evaluate(
                retrieved_ids=retrieved_ids,
                relevant_ids=list(
                    case.relevant_memory_ids
                ),
                k=k,
            )
        )

        return RetrievalEvaluationResult(
            case_id=case.case_id,

            query=case.query,

            retrieved_ids=retrieved_ids,

            relevant_ids=list(
                case.relevant_memory_ids
            ),

            metrics=metrics,

            retrieved_procedure_ids=(
                retrieved_procedure_ids
            ),

            retrieved_state_ids=(
                retrieved_state_ids
            ),

            relevant_procedure_ids=list(
                case.relevant_procedure_ids
            ),

            relevant_state_ids=list(
                case.relevant_state_ids
            ),
        )

 

    @staticmethod
    def _extract_memory_ids(
        results,
    ) -> list[str]:
        """
        Extract memory IDs from retrieval results.

        Supported result representations:

        1. result.memory_id
        2. result.item.memory_id
        3. result.metadata["memory_id"]
        4. result["memory_id"]
        5. result["item"]["memory_id"]
        6. result["memory"]["memory_id"]
        7. result["metadata"]["memory_id"]

        Results without an identifiable memory ID
        are ignored.
        """

        if results is None:
            return []

        memory_ids = []

        for result in results:

            memory_id = None

         

            memory_id = getattr(
                result,
                "memory_id",
                None,
            )

          

            if (
                memory_id is None
                and isinstance(
                    result,
                    dict,
                )
            ):

                memory_id = result.get(
                    "memory_id"
                )

            

            if memory_id is None:

                item = getattr(
                    result,
                    "item",
                    None,
                )

                if item is not None:

                    memory_id = getattr(
                        item,
                        "memory_id",
                        None,
                    )

                    if (
                        memory_id is None
                        and isinstance(
                            item,
                            dict,
                        )
                    ):

                        memory_id = item.get(
                            "memory_id"
                        )

           

            if (
                memory_id is None
                and isinstance(
                    result,
                    dict,
                )
            ):

                item = result.get(
                    "item"
                )

                if item is not None:

                    if isinstance(
                        item,
                        dict,
                    ):

                        memory_id = item.get(
                            "memory_id"
                        )

                    else:

                        memory_id = getattr(
                            item,
                            "memory_id",
                            None,
                        )

           

            if memory_id is None:

                metadata = getattr(
                    result,
                    "metadata",
                    None,
                )

                if isinstance(
                    metadata,
                    dict,
                ):

                    memory_id = metadata.get(
                        "memory_id"
                    )

         

            if (
                memory_id is None
                and isinstance(
                    result,
                    dict,
                )
            ):

                metadata = result.get(
                    "metadata"
                )

                if isinstance(
                    metadata,
                    dict,
                ):

                    memory_id = metadata.get(
                        "memory_id"
                    )

        

            if (
                memory_id is None
                and isinstance(
                    result,
                    dict,
                )
            ):

                memory = result.get(
                    "memory"
                )

                if memory is not None:

                    if isinstance(
                        memory,
                        dict,
                    ):

                        memory_id = memory.get(
                            "memory_id"
                        )

                    else:

                        memory_id = getattr(
                            memory,
                            "memory_id",
                            None,
                        )

           

            if memory_id is not None:

                memory_ids.append(
                    str(memory_id)
                )

        return memory_ids

   

    @staticmethod
    def _extract_procedural_ids(
        results,
    ) -> tuple[
        list[str],
        list[str]
    ]:
        """
        Extract procedure IDs and state IDs from
        procedural retrieval results.

        Supported representations include:

        - result.metadata
        - result.item
        - dictionary result
        - ProcedureState objects
        """

        procedure_ids = []

        state_ids = []

        if results is None:
            return (
                procedure_ids,
                state_ids,
            )

        for result in results:

            procedure_id = None
            state_id = None

           

            metadata = getattr(
                result,
                "metadata",
                None,
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
                isinstance(
                    result,
                    dict,
                )
                and isinstance(
                    result.get("metadata"),
                    dict,
                )
            ):

                metadata = result.get(
                    "metadata"
                )

                if procedure_id is None:

                    procedure_id = metadata.get(
                        "procedure_id"
                    )

                if state_id is None:

                    state_id = metadata.get(
                        "state_id"
                    )

        

            item = getattr(
                result,
                "item",
                None,
            )

            if item is not None:

                if state_id is None:

                    state_id = getattr(
                        item,
                        "state_id",
                        None,
                    )

                if procedure_id is None:

                    procedure_id = getattr(
                        item,
                        "procedure_id",
                        None,
                    )

         

            if (
                isinstance(
                    result,
                    dict,
                )
            ):

                item = result.get(
                    "item"
                )

                if isinstance(
                    item,
                    dict,
                ):

                    if state_id is None:

                        state_id = item.get(
                            "state_id"
                        )

                    if procedure_id is None:

                        procedure_id = item.get(
                            "procedure_id"
                        )

                elif item is not None:

                    if state_id is None:

                        state_id = getattr(
                            item,
                            "state_id",
                            None,
                        )

                    if procedure_id is None:

                        procedure_id = getattr(
                            item,
                            "procedure_id",
                            None,
                        )


            if isinstance(
                result,
                dict,
            ):

                if procedure_id is None:

                    procedure_id = result.get(
                        "procedure_id"
                    )

                if state_id is None:

                    state_id = result.get(
                        "state_id"
                    )

            # Append IDs

            if procedure_id is not None:

                procedure_id = str(
                    procedure_id
                )

                if procedure_id not in procedure_ids:

                    procedure_ids.append(
                        procedure_id
                    )

            if state_id is not None:

                state_id = str(
                    state_id
                )

                if state_id not in state_ids:

                    state_ids.append(
                        state_id
                    )

        return (
            procedure_ids,
            state_ids,
        )


    @staticmethod
    def _aggregate(
        results: list[
            RetrievalEvaluationResult
        ],
        k: int,
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
                case_count=0,
            )

        recall = (
            sum(
                result.metrics.recall_at_k
                for result in results
            )
            / len(results)
        )

        precision = (
            sum(
                result.metrics.precision_at_k
                for result in results
            )
            / len(results)
        )

        hit = (
            sum(
                result.metrics.hit_at_k
                for result in results
            )
            / len(results)
        )

        reciprocal_rank = (
            sum(
                result.metrics.reciprocal_rank
                for result in results
            )
            / len(results)
        )

        ndcg = (
            sum(
                result.metrics.ndcg_at_k
                for result in results
            )
            / len(results)
        )

        return RetrievalEvaluationSummary(
            results=results,

            recall_at_k=recall,

            precision_at_k=precision,

            hit_at_k=hit,

            reciprocal_rank=reciprocal_rank,

            ndcg_at_k=ndcg,

            k=k,

            case_count=len(results),
        )