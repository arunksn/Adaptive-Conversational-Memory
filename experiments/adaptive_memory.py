from dataclasses import dataclass

from src.retrieval.adaptive_retriever import (
    AdaptiveRetriever,
)


@dataclass
class AdaptiveMemoryResult:
    """
    Result produced by the Adaptive Memory experiment.

    The result preserves the routing decision, retrieved
    memories, and retrieved memory IDs so the experiment
    can be compared against the other retrieval systems.
    """

    routing: object

    results: list

    retrieved_ids: list[str]

    result_count: int


class AdaptiveMemory:
    """
    Adaptive Memory retrieval experiment.

    This experiment uses the project's existing
    AdaptiveRetriever.

    AdaptiveRetriever is responsible for:

    1. Query routing.
    2. Semantic retrieval.
    3. Episodic retrieval.
    4. Procedural retrieval.
    5. Hybrid fusion.
    6. Conflict detection.
    7. Conflict resolution.

    This experiment class only provides a clean interface
    for benchmarking and comparison with the other memory
    architectures.

    The underlying memory and retrieval systems are not
    modified.
    """

    def __init__(
        self,
        adaptive_retriever: AdaptiveRetriever,
    ):
        if adaptive_retriever is None:
            raise ValueError(
                "adaptive_retriever cannot be None"
            )

        self.adaptive_retriever = (
            adaptive_retriever
        )

    # RETRIEVE

    def retrieve(
        self,
        query: str,
        k: int = 5,
        start_time=None,
        end_time=None,
        procedure_id: str | None = None,
        state_id: str | None = None,
    ) -> AdaptiveMemoryResult:
        """
        Retrieve memories using the complete adaptive
        memory architecture.

        The routing and retrieval logic is delegated to
        AdaptiveRetriever.
        """

        if not isinstance(
            query,
            str,
        ) or not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0"
            )

        routing, results = (
            self.adaptive_retriever.retrieve(
                query=query,
                top_k=k,
                start_time=start_time,
                end_time=end_time,
                procedure_id=procedure_id,
                state_id=state_id,
            )
        )

        results = list(
            results or []
        )

        # AdaptiveRetriever normally already applies
        # Top-K selection. We enforce the experiment
        # contract here as well.
        results = results[:k]

        retrieved_ids = (
            self._extract_memory_ids(
                results
            )
        )

        return AdaptiveMemoryResult(
            routing=routing,
            results=results,
            retrieved_ids=retrieved_ids,
            result_count=len(results),
        )

    # MEMORY ID EXTRACTION

    @staticmethod
    def _extract_memory_id(
        result,
    ):
        """
        Extract a memory ID from a retrieval result.

        Supports the result representations used by the
        existing project.
        """

        if result is None:
            return None

        memory_id = getattr(
            result,
            "memory_id",
            None,
        )

        if memory_id is not None:
            return str(
                memory_id
            )

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

            if memory_id is not None:
                return str(
                    memory_id
                )

        if isinstance(
            result,
            dict,
        ):

            memory_id = result.get(
                "memory_id"
            )

            if memory_id is not None:
                return str(
                    memory_id
                )

            item = result.get(
                "item"
            )

            if isinstance(
                item,
                dict,
            ):

                memory_id = item.get(
                    "memory_id"
                )

                if memory_id is not None:
                    return str(
                        memory_id
                    )

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

            if memory_id is not None:
                return str(
                    memory_id
                )

        return None

    @classmethod
    def _extract_memory_ids(
        cls,
        results: list,
    ) -> list[str]:
        """
        Extract all identifiable memory IDs.
        """

        memory_ids = []

        for result in results:

            memory_id = (
                cls._extract_memory_id(
                    result
                )
            )

            if memory_id is not None:

                memory_ids.append(
                    memory_id
                )

        return memory_ids

    # CONTEXT

    def build_context(
        self,
        query: str,
        context_builder,
        k: int = 5,
        start_time=None,
        end_time=None,
        procedure_id: str | None = None,
        state_id: str | None = None,
    ):
        """
        Retrieve adaptive memories and construct an
        LLM-ready context using the existing
        ContextBuilder.
        """

        if context_builder is None:
            raise ValueError(
                "context_builder cannot be None"
            )

        retrieval = self.retrieve(
            query=query,
            k=k,
            start_time=start_time,
            end_time=end_time,
            procedure_id=procedure_id,
            state_id=state_id,
        )

        return context_builder.build(
            retrieval.results,
            query=query,
        )

    # RUN

    def run(
        self,
        query: str,
        k: int = 5,
        context_builder=None,
        start_time=None,
        end_time=None,
        procedure_id: str | None = None,
        state_id: str | None = None,
    ) -> dict:
        """
        Run the complete Adaptive Memory experiment.

        Context construction is optional so that the
        experiment can be used for retrieval benchmarking
        without requiring an LLM.
        """

        retrieval = self.retrieve(
            query=query,
            k=k,
            start_time=start_time,
            end_time=end_time,
            procedure_id=procedure_id,
            state_id=state_id,
        )

        context = None

        if context_builder is not None:

            context = context_builder.build(
                retrieval.results,
                query=query,
            )

        return {
            "query": query,
            "routing": retrieval.routing,
            "retrieval": retrieval,
            "context": context,
            "retrieved_ids": retrieval.retrieved_ids,
        }