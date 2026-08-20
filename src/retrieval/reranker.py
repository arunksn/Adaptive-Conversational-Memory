from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re

from src.retrieval.hybrid_retriever import (
    RetrievalResult
)


@dataclass
class RerankingWeights:
    """
    Weights used by the deterministic re-ranking model.

    Retrieval relevance remains the dominant signal.

    Query relevance provides an additional signal when
    important query terms occur in the memory content.

    Importance, recency, and source priority provide
    secondary signals.
    """

    retrieval_score: float = 0.55
    query_relevance: float = 0.15
    importance: float = 0.12
    recency: float = 0.10
    source_priority: float = 0.08


class MemoryReranker:

    SOURCE_PRIORITIES = {
        "semantic": 1.0,
        "episodic": 0.9,
        "procedural": 0.95,
    }

    STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "am",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "me",
        "this",
        "that",
        "these",
        "those",
        "have",
        "has",
        "had",
        "can",
        "could",
        "should",
        "would",
        "will",
        "currently",
        "current",
        "previous",
        "latest",
        "recently",
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

  

    def rerank(
        self,
        results: list[RetrievalResult],
        top_k: int = 5,
        query: str | None = None
    ) -> list[RetrievalResult]:

        if not results:
            return []

        scored_results = []

        for result in results:

            score = self._calculate_score(
                result=result,
                query=query
            )

            result.metadata[
                "rerank_score"
            ] = score

            scored_results.append(
                result
            )

        scored_results.sort(
            key=lambda result: (
                result.metadata[
                    "rerank_score"
                ],
                result.score
            ),
            reverse=True
        )

        return scored_results[:top_k]

  

    def _calculate_score(
        self,
        result: RetrievalResult,
        query: str | None = None
    ) -> float:

        retrieval_score = self._clamp(
            result.score
        )

        query_relevance = (
            self._query_relevance(
                result=result,
                query=query
            )
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
            self.weights.query_relevance
            * query_relevance
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

        return self._clamp(score)


    @staticmethod
    def _field(
        item,
        name: str,
        default=None
    ):
        """
        Read a field from either:

            Memory object

        or:

            serialized dictionary
        """

        if isinstance(item, dict):
            return item.get(
                name,
                default
            )

        return getattr(
            item,
            name,
            default
        )



    @classmethod
    def _query_relevance(
        cls,
        result: RetrievalResult,
        query: str | None
    ) -> float:
        """
        Calculate deterministic query relevance.

        Combines:

        1. Direct lexical overlap.
        2. Intent-aware relevance.

        Intent groups help queries containing concepts such as:

            current
            latest
            previous
            frequently
            repeatedly
            project
            development
            learning
            interests
            career goal

        match memories whose content expresses the same
        intent, even when the exact query word does not
        appear in the memory content.
        """

        if not query:
            return 0.0

        query_terms = cls._tokenize(
            query
        )

        if not query_terms:
            return 0.0

        content = cls._field(
            result.item,
            "content",
            ""
        )

        if not isinstance(
            content,
            str
        ):
            content = str(
                content
            )

        content_terms = cls._tokenize(
            content
        )

        if not content_terms:
            return 0.0

        content_set = set(
            content_terms
        )

        

        matched = sum(
            1
            for term in query_terms
            if term in content_set
        )

        direct_score = (
            matched / len(query_terms)
        )

   

        intent_groups = {
            "current": {
                "current",
                "currently",
                "latest",
                "now",
                "present",
            },

            "historical": {
                "previous",
                "earlier",
                "old",
                "past",
                "formerly",
            },

            "repeated": {
                "frequently",
                "repeatedly",
                "usually",
                "often",
                "regularly",
            },

            "project": {
                "project",
                "application",
                "system",
            },

            "development": {
                "development",
                "develop",
                "building",
                "backend",
            },

            "learning": {
                "learning",
                "learn",
                "studying",
            },

            "interest": {
                "interest",
                "interests",
                "interested",
            },

            "goal": {
                "goal",
                "career",
                "future",
                "long-term",
            },
        }

        intent_score = 0.0
        intent_count = 0

        for group_terms in intent_groups.values():

            query_has_group = bool(
                set(query_terms)
                & group_terms
            )

            if not query_has_group:
                continue

            intent_count += 1

            content_has_group = bool(
                content_set
                & group_terms
            )

            if content_has_group:
                intent_score += 1.0

        if intent_count > 0:
            intent_score /= intent_count

      
        # Combine lexical and intent relevance.
        #
        # Direct lexical overlap remains the primary
        # lexical signal.
       

        score = (
            0.65 * direct_score
            +
            0.35 * intent_score
        )

        return cls._clamp(
            score
        )

    

    @classmethod
    def _tokenize(
        cls,
        text: str
    ) -> list[str]:

        tokens = re.findall(
            r"[a-zA-Z0-9]+",
            text.lower()
        )

        return [
            token
            for token in tokens
            if (
                token not in cls.STOP_WORDS
                and len(token) > 1
            )
        ]

   

    @staticmethod
    def _importance_score(
        result: RetrievalResult
    ) -> float:

        importance = MemoryReranker._field(
            result.item,
            "importance",
            0.5
        )

        try:
            importance = float(
                importance
            )
        except (
            TypeError,
            ValueError
        ):
            importance = 0.5

        return MemoryReranker._clamp(
            importance
        )


    @staticmethod
    def _recency_score(
        result: RetrievalResult
    ) -> float:

        event_time = MemoryReranker._field(
            result.item,
            "event_time",
            None
        )

        created_at = MemoryReranker._field(
            result.item,
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

        if isinstance(
            timestamp,
            str
        ):
            try:
                timestamp = datetime.fromisoformat(
                    timestamp
                )
            except ValueError:
                return 0.5

        if not isinstance(
            timestamp,
            datetime
        ):
            return 0.5

        now = datetime.now(
            timezone.utc
        )

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

        source = result.source.value

        return cls.SOURCE_PRIORITIES.get(
            source,
            0.5
        )

    

    @staticmethod
    def _clamp(
        value: float
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                float(value)
            )
        )