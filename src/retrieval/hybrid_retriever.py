from dataclasses import dataclass, field
from typing import Any

from src.models.memory import Memory
from src.models.procedure import ProcedureState
from src.routing.memory_router import MemoryRoute


@dataclass
class RetrievalResult:
    """
    Normalized result returned by the hybrid
    retrieval system.
    """

    source: MemoryRoute
    item: Any

    score: float = 0.0

    memory_id: str | None = None

    metadata: dict = field(
        default_factory=dict
    )


class HybridRetriever:

    def __init__(
        self,
        vector_retriever=None,
        temporal_retriever=None,
        graph_retriever=None
    ):
        """
        Initialize the hybrid retriever.

        Retrievers are injected so each memory system
        remains independently testable.
        """

        self.vector_retriever = (
            vector_retriever
        )

        self.temporal_retriever = (
            temporal_retriever
        )

        self.graph_retriever = (
            graph_retriever
        )

    # VECTOR RETRIEVAL

    def retrieve_semantic(
        self,
        query: str,
        top_k: int = 5
    ) -> list[RetrievalResult]:
        """
        Retrieve semantic memories using the
        vector retriever.
        """

        if self.vector_retriever is None:
            return []

        results = self.vector_retriever.search(
            query,
            top_k=top_k
        )

        normalized = []

        for result in results:

            memory = result["memory"]

            memory_id = result.get(
                "memory_id"
            )

            normalized.append(
                RetrievalResult(
                    source=MemoryRoute.SEMANTIC,
                    item=memory,
                    score=float(
                        result.get(
                            "score",
                            0.0
                        )
                    ),
                    memory_id=memory_id
                )
            )

        return normalized

    # TEMPORAL RETRIEVAL

    def retrieve_temporal(
        self,
        start_time,
        end_time
    ) -> list[RetrievalResult]:
        """
        Retrieve episodic memories within a
        temporal range.
        """

        if self.temporal_retriever is None:
            return []

        memories = (
            self.temporal_retriever.search(
                start_time=start_time,
                end_time=end_time
            )
        )

        normalized = []

        for memory in memories:

            normalized.append(
                RetrievalResult(
                    source=MemoryRoute.EPISODIC,
                    item=memory,
                    score=self._temporal_score(
                        memory
                    ),
                    memory_id=memory.memory_id
                )
            )

        return normalized

    # PROCEDURAL RETRIEVAL

    def retrieve_procedural(
        self,
        procedure_id: str,
        state_id: str
    ) -> list[RetrievalResult]:
        """
        Retrieve the next procedural states from
        the current state.
        """

        if self.graph_retriever is None:
            return []

        states = (
            self.graph_retriever.get_next_states(
                procedure_id,
                state_id
            )
        )

        normalized = []

        for state in states:

            normalized.append(
                RetrievalResult(
                    source=MemoryRoute.PROCEDURAL,
                    item=state,
                    score=self._procedural_score(
                        state
                    ),
                    metadata={
                        "procedure_id": procedure_id,
                        "state_id": state.state_id
                    }
                )
            )

        return normalized

    # FUSION

    def fuse(
        self,
        results: list[RetrievalResult],
        top_k: int = 10
    ) -> list[RetrievalResult]:
        """
        Combine results from multiple memory sources.

        Results are sorted by normalized score.
        """

        if not results:
            return []

        normalized_scores = (
            self._normalize_scores(
                results
            )
        )

        for result, score in zip(
            results,
            normalized_scores
        ):
            result.score = score

        results = sorted(
            results,
            key=lambda result: result.score,
            reverse=True
        )

        return results[:top_k]

    # DEDUPLICATION

    def deduplicate(
        self,
        results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """
        Remove duplicate results.

        Memory IDs are used when available.
        """

        seen = set()
        unique_results = []

        for result in results:

            if result.memory_id is not None:

                key = (
                    result.source,
                    result.memory_id
                )

            else:

                key = (
                    result.source,
                    str(result.item)
                )

            if key in seen:
                continue

            seen.add(key)

            unique_results.append(
                result
            )

        return unique_results

    # COMPLETE FUSION PIPELINE

    def combine(
        self,
        results: list[RetrievalResult],
        top_k: int = 10
    ) -> list[RetrievalResult]:
        """
        Deduplicate and fuse retrieval results.
        """

        unique_results = self.deduplicate(
            results
        )

        return self.fuse(
            unique_results,
            top_k=top_k
        )

    # SCORE HELPERS

    @staticmethod
    def _temporal_score(
        memory: Memory
    ) -> float:
        """
        Initial temporal relevance score.

        For Phase 6, episodic results receive a
        baseline score based on memory importance.

        More sophisticated temporal ranking will be
        introduced later.
        """

        return max(
            0.0,
            min(
                1.0,
                memory.importance
            )
        )

    @staticmethod
    def _procedural_score(
        state: ProcedureState
    ) -> float:
        """
        Initial procedural relevance score.

        Terminal states receive a slightly higher
        score because they represent completed
        procedures.
        """

        if state.is_terminal:
            return 1.0

        return 0.8

    @staticmethod
    def _normalize_scores(
        results: list[RetrievalResult]
    ) -> list[float]:
        """
        Normalize scores into the [0, 1] range.
        """

        if not results:
            return []

        scores = [
            float(result.score)
            for result in results
        ]

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:

            return [
                1.0
                for _ in scores
            ]

        return [
            (
                score - min_score
            ) / (
                max_score - min_score
            )
            for score in scores
        ]