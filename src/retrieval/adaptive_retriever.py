from datetime import datetime

from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    RetrievalResult
)

from src.routing.memory_router import (
    MemoryRoute,
    MemoryRouter,
    RoutingResult
)


class AdaptiveRetriever:

    def __init__(
        self,
        router: MemoryRouter,
        hybrid_retriever: HybridRetriever
    ):
        """
        Orchestrates routing and retrieval.

        MemoryRouter decides which memory sources are
        relevant. HybridRetriever executes and combines
        the resulting memories.
        """

        self.router = router
        self.hybrid_retriever = hybrid_retriever

    # MAIN RETRIEVAL PIPELINE

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        procedure_id: str | None = None,
        state_id: str | None = None
    ) -> tuple[
        RoutingResult,
        list[RetrievalResult]
    ]:
        """
        Route the query and retrieve memories from the
        selected memory sources.

        Temporal and procedural parameters are optional
        because their natural-language interpretation
        will be added later during LLM integration.
        """

        routing_result = self.router.route(
            query
        )

        results = []

        # SEMANTIC MEMORY

        if (
            MemoryRoute.SEMANTIC
            in routing_result.routes
        ):

            semantic_results = (
                self.hybrid_retriever.retrieve_semantic(
                    query=query,
                    top_k=top_k
                )
            )

            results.extend(
                semantic_results
            )

        # EPISODIC MEMORY

        if (
            MemoryRoute.EPISODIC
            in routing_result.routes
        ):

            temporal_results = (
                self._retrieve_temporal(
                    start_time=start_time,
                    end_time=end_time,
                    top_k=top_k
                )
            )

            results.extend(
                temporal_results
            )

        # PROCEDURAL MEMORY

        if (
            MemoryRoute.PROCEDURAL
            in routing_result.routes
        ):

            procedural_results = (
                self._retrieve_procedural(
                    procedure_id=procedure_id,
                    state_id=state_id
                )
            )

            results.extend(
                procedural_results
            )

        # FUSION

        fused_results = (
            self.hybrid_retriever.combine(
                results,
                top_k=top_k
            )
        )

        return (
            routing_result,
            fused_results
        )
    
    # TEMPORAL RETRIEVAL

    def _retrieve_temporal(
        self,
        start_time: datetime | None,
        end_time: datetime | None,
        top_k: int
    ) -> list[RetrievalResult]:
        """
        Retrieve episodic memories.

        A time range is required for explicit temporal
        retrieval. If it is not available, use recent
        episodic memories as a safe fallback.
        """

        retriever = (
            self.hybrid_retriever.temporal_retriever
        )

        if retriever is None:
            return []

        if (
            start_time is not None
            and end_time is not None
        ):

            return (
                self.hybrid_retriever.retrieve_temporal(
                    start_time=start_time,
                    end_time=end_time
                )
            )

        # No temporal range has been extracted yet.
        # Use recent episodic memories as a fallback.
        memories = retriever.recent(
            limit=top_k
        )

        results = []

        for memory in memories:

            results.append(
                RetrievalResult(
                    source=MemoryRoute.EPISODIC,
                    item=memory,
                    score=self.hybrid_retriever._temporal_score(
                        memory
                    ),
                    memory_id=memory.memory_id
                )
            )

        return results

    # PROCEDURAL RETRIEVAL

    def _retrieve_procedural(
        self,
        procedure_id: str | None,
        state_id: str | None
    ) -> list[RetrievalResult]:
        """
        Retrieve procedural memory from a known
        procedure and current state.

        Natural-language procedure/state extraction
        will be added later.
        """

        if (
            procedure_id is None
            or state_id is None
        ):
            return []

        return (
            self.hybrid_retriever.retrieve_procedural(
                procedure_id=procedure_id,
                state_id=state_id
            )
        )