from dataclasses import dataclass

from src.retrieval.hybrid_retriever import (
    RetrievalResult
)


@dataclass
class RerankingWeights:
    """
    Weights used by the deterministic re-ranking model.

    The initial weights are intentionally simple.
    They can be tuned later during Phase 11 evaluation.
    """

    retrieval_score: float = 0.50
    importance: float = 0.20
    recency: float = 0.20
    source_priority: float = 0.10


class MemoryReranker:

    SOURCE_PRIORITIES = {
        "semantic": 1.0,
        "episodic": 0.9,
        "procedural": 0.95,
    }

    def __init__(
        self,
        weights: RerankingWeights | None = None
    ):
        self.weights = (
            weights
            if weights is not None
            else RerankingWeights()
        )

    # PUBLIC API

    def rerank(
        self,
        results: list[RetrievalResult],
        top_k: int = 5
    ) -> list[RetrievalResult]:
        """
        Re-rank retrieved memories using:

        - retrieval relevance
        - memory importance
        - recency
        - memory source priority
        """

        if not results:
            return []

        scored_results = []

        for result in results:

            score = self._calculate_score(
                result
            )

            result.metadata[
                "rerank_score"
            ] = score

            scored_results.append(
                result
            )

        scored_results.sort(
            key=lambda result: result.metadata[
                "rerank_score"
            ],
            reverse=True
        )

        return scored_results[:top_k]


    def _calculate_score(
        self,
        result: RetrievalResult
    ) -> float:
        """
        Calculate the final re-ranking score.
        """

        retrieval_score = self._clamp(
            result.score
        )

        importance = self._importance_score(
            result
        )

        recency = self._recency_score(
            result
        )

        source_priority = self._source_priority(
            result
        )

        score = (
            self.weights.retrieval_score
            * retrieval_score
            +
            self.weights.importance
            * importance
            +
            self.weights.recency
            * recency
            +
            self.weights.source_priority
            * source_priority
        )

        return self._clamp(
            score
        )

    # IMPORTANCE

    @staticmethod
    def _importance_score(
        result: RetrievalResult
    ) -> float:
        """
        Extract memory importance.

        If the retrieved item does not contain an
        importance value, use a neutral score.
        """

        item = result.item

        importance = getattr(
            item,
            "importance",
            0.5
        )

        return MemoryReranker._clamp(
            float(importance)
        )


    @staticmethod
    def _recency_score(
        result: RetrievalResult
    ) -> float:
        """
        Calculate a simple recency score.

        Memories with timestamps receive a decaying
        score based on their age.

        Untimestamped memories receive a neutral
        recency score.
        """

        item = result.item

        event_time = getattr(
            item,
            "event_time",
            None
        )

        created_at = getattr(
            item,
            "created_at",
            None
        )

        timestamp = (
            event_time
            if event_time is not None
            else created_at
        )

        if timestamp is None:
            return 0.5

        from datetime import datetime, timezone

        now = datetime.now(
            timezone.utc
        )

        # Handle naive timestamps.
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(
                tzinfo=timezone.utc
            )

        age_days = max(
            0.0,
            (
                now - timestamp
            ).total_seconds()
            / 86400.0
        )

        # Exponential decay.

        import math

        decay = math.exp(
            -age_days / 30.0
        )

        return MemoryReranker._clamp(
            decay
        )


    @classmethod
    def _source_priority(
        cls,
        result: RetrievalResult
    ) -> float:
        """
        Return the baseline priority of the memory
        source.
        """

        source = result.source.value

        return cls.SOURCE_PRIORITIES.get(
            source,
            0.5
        )


    @staticmethod
    def _clamp(
        value: float
    ) -> float:
        """
        Keep scores inside [0, 1].
        """

        return max(
            0.0,
            min(
                1.0,
                value
            )
        )