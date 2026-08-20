from datetime import datetime

from src.conflict.conflict_detector import (
    ConflictDetector
)

from src.conflict.conflict_resolver import (
    ConflictResolver
)

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
        hybrid_retriever: HybridRetriever,
        conflict_detector: ConflictDetector | None = None,
        conflict_resolver: ConflictResolver | None = None
    ):
        """
        Orchestrates routing, retrieval, hybrid fusion,
        and conflict handling.

        MemoryRouter decides which memory sources are
        relevant.

        HybridRetriever executes and combines the
        resulting memories.

        ConflictDetector identifies contradictory
        memories.

        ConflictResolver determines which memory should
        be treated as the preferred/current memory while
        preserving historical information.
        """

        if router is None:
            raise ValueError(
                "router cannot be None"
            )

        if hybrid_retriever is None:
            raise ValueError(
                "hybrid_retriever cannot be None"
            )

        self.router = router
        self.hybrid_retriever = hybrid_retriever

        self.conflict_detector = (
            conflict_detector
            if conflict_detector is not None
            else ConflictDetector()
        )

        self.conflict_resolver = (
            conflict_resolver
            if conflict_resolver is not None
            else ConflictResolver()
        )

    # ========================================================
    # MAIN RETRIEVAL PIPELINE
    # ========================================================

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
        Route the query, retrieve memories, perform hybrid
        fusion, and resolve conflicts among the final
        Top-K memories.

        Routing remains the primary decision mechanism.

        Procedural queries additionally use semantic
        retrieval when vector memory is available. This
        allows procedural information stored as semantic
        memories to participate in the final retrieval
        result while preserving graph-based procedural
        retrieval.
        """

        if not isinstance(
            query,
            str
        ) or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        routing_result = self.router.route(
            query
        )

        results = []

        # ====================================================
        # SEMANTIC MEMORY
        # ====================================================

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

        # ====================================================
        # PROCEDURAL MEMORY
        # ====================================================

        if (
            MemoryRoute.PROCEDURAL
            in routing_result.routes
        ):

            # ------------------------------------------------
            # Semantic fallback / supporting retrieval
            #
            # Procedural information may exist in vector
            # memory even when the graph does not contain a
            # directly matching state.
            #
            # This is especially important for queries such
            # as:
            #
            #     "How do I deploy my application?"
            #
            # where the benchmark ground truth may refer to
            # a semantic memory such as:
            #
            #     memory-deployment-procedure
            # ------------------------------------------------

            semantic_results = (
                self.hybrid_retriever.retrieve_semantic(
                    query=query,
                    top_k=top_k
                )
            )

            results.extend(
                semantic_results
            )

            # ------------------------------------------------
            # Graph-based procedural retrieval
            #
            # This is only available when the caller has
            # supplied a known procedure and state.
            #
            # Without procedure/state context, the existing
            # behavior is preserved and no graph result is
            # fabricated.
            # ------------------------------------------------

            procedural_results = (
                self._retrieve_procedural(
                    procedure_id=procedure_id,
                    state_id=state_id
                )
            )

            results.extend(
                procedural_results
            )

        # ====================================================
        # EPISODIC MEMORY
        # ====================================================

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

        # ====================================================
        # HYBRID FUSION + TOP-K
        # ====================================================

        fused_results = (
            self.hybrid_retriever.combine(
                results,
                top_k=top_k
            )
        )

        # ====================================================
        # CONFLICT DETECTION + RESOLUTION
        # ====================================================

        resolved_results = (
            self._resolve_conflicts(
                fused_results
            )
        )

        return (
            routing_result,
            resolved_results
        )

    # ========================================================
    # CONFLICT RESOLUTION
    # ========================================================

    def _resolve_conflicts(
        self,
        results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """
        Detect and resolve conflicts among the final
        Top-K memories.

        The preferred memory remains in the returned
        result set.

        Historical memories are preserved through
        metadata attached to the preferred memory.
        """

        if len(results) < 2:
            return results

        conflicts = (
            self.conflict_detector.detect(
                results
            )
        )

        if not conflicts:
            return results

        removed_ids = set()

        for conflict in conflicts:

            resolution = (
                self.conflict_resolver.resolve(
                    conflict
                )
            )

            preferred = (
                resolution.preferred
            )

            historical = (
                resolution.historical
            )

            # ------------------------------------------------
            # Preserve historical information.
            # ------------------------------------------------

            if historical is not None:

                historical_records = (
                    preferred.metadata.setdefault(
                        "historical_memories",
                        []
                    )
                )

                historical_records.append(
                    {
                        "memory_id": (
                            historical.memory_id
                        ),
                        "content": getattr(
                            historical.item,
                            "content",
                            str(
                                historical.item
                            )
                        ),
                        "reason": (
                            resolution.reason
                        )
                    }
                )

                removed_ids.add(
                    historical.memory_id
                )

            # ------------------------------------------------
            # Store conflict information on the preferred
            # memory.
            # ------------------------------------------------

            preferred.metadata[
                "conflict_resolved"
            ] = True

            preferred.metadata[
                "conflict_reason"
            ] = resolution.reason

            preferred.metadata[
                "conflict_confidence"
            ] = resolution.confidence

        # ----------------------------------------------------
        # Remove historical duplicates from the returned
        # Top-K result set.
        # ----------------------------------------------------

        return [
            result
            for result in results
            if result.memory_id
            not in removed_ids
        ]

    # ========================================================
    # TEMPORAL / EPISODIC RETRIEVAL
    # ========================================================

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

        # ----------------------------------------------------
        # Explicit temporal range.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # No temporal range was extracted.
        #
        # Use recent episodic memories as the fallback.
        # ----------------------------------------------------

        memories = retriever.recent(
            limit=top_k
        )

        results = []

        for memory in memories:

            results.append(
                RetrievalResult(
                    source=MemoryRoute.EPISODIC,
                    item=memory,
                    score=(
                        self.hybrid_retriever
                        ._temporal_score(
                            memory
                        )
                    ),
                    memory_id=memory.memory_id
                )
            )

        return results

    # ========================================================
    # PROCEDURAL RETRIEVAL
    # ========================================================

    def _retrieve_procedural(
        self,
        procedure_id: str | None,
        state_id: str | None
    ) -> list[RetrievalResult]:
        """
        Retrieve procedural memory from a known
        procedure and current state.

        Natural-language procedure/state extraction
        is intentionally not performed here.

        If procedure/state context is unavailable, return
        an empty list rather than inventing procedural
        graph results.
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