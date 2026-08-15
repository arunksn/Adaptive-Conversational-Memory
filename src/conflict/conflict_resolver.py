from dataclasses import dataclass
from datetime import datetime

from src.retrieval.hybrid_retriever import (
    RetrievalResult
)

from src.conflict.conflict_detector import (
    ConflictPair
)


@dataclass
class ConflictResolution:
    """
    Result of resolving a memory conflict.
    """

    conflict: ConflictPair

    preferred: RetrievalResult
    historical: RetrievalResult | None

    reason: str
    confidence: float


class ConflictResolver:

    # PUBLIC API

    def resolve(
        self,
        conflict: ConflictPair
    ) -> ConflictResolution:
        """
        Resolve a conflict using:

        1. Temporal recency
        2. Memory confidence
        3. Retrieval relevance

        Older memories are preserved as historical
        information rather than deleted.
        """

        first = conflict.first
        second = conflict.second

        preferred = self._choose_preferred(
            first,
            second
        )

        historical = (
            second
            if preferred is first
            else first
        )

        reason = self._build_reason(
            preferred,
            historical
        )

        confidence = self._resolution_confidence(
            preferred,
            historical
        )

        return ConflictResolution(
            conflict=conflict,
            preferred=preferred,
            historical=historical,
            reason=reason,
            confidence=confidence
        )


    def _choose_preferred(
        self,
        first: RetrievalResult,
        second: RetrievalResult
    ) -> RetrievalResult:
        """
        Choose the current/preferred memory.

        Priority:

        1. More recent timestamp
        2. Higher memory confidence
        3. Higher retrieval score
        """

        first_time = self._timestamp(
            first
        )

        second_time = self._timestamp(
            second
        )


        if (
            first_time is not None
            and second_time is not None
        ):

            if first_time > second_time:
                return first

            if second_time > first_time:
                return second


        first_confidence = (
            self._memory_confidence(
                first
            )
        )

        second_confidence = (
            self._memory_confidence(
                second
            )
        )

        if (
            first_confidence
            > second_confidence
        ):
            return first

        if (
            second_confidence
            > first_confidence
        ):
            return second

        # RETRIEVAL SCORE

        if (
            first.score
            >= second.score
        ):
            return first

        return second


    @staticmethod
    def _timestamp(
        result: RetrievalResult
    ) -> datetime | None:
        """
        Get the most relevant timestamp from
        a memory.
        """

        item = result.item

        event_time = getattr(
            item,
            "event_time",
            None
        )

        if event_time is not None:
            return event_time

        created_at = getattr(
            item,
            "created_at",
            None
        )

        return created_at


    @staticmethod
    def _memory_confidence(
        result: RetrievalResult
    ) -> float:
        """
        Extract confidence from memory metadata.

        If explicit confidence is unavailable,
        use a neutral confidence value.
        """

        item = result.item

        confidence = getattr(
            item,
            "confidence",
            None
        )

        if confidence is None:

            confidence = (
                result.metadata.get(
                    "confidence",
                    0.5
                )
            )

        return max(
            0.0,
            min(
                1.0,
                float(confidence)
            )
        )


    def _resolution_confidence(
        self,
        preferred: RetrievalResult,
        historical: RetrievalResult
    ) -> float:
        """
        Estimate confidence in the resolution.
        """

        preferred_time = self._timestamp(
            preferred
        )

        historical_time = self._timestamp(
            historical
        )

        confidence_gap = abs(
            self._memory_confidence(
                preferred
            )
            -
            self._memory_confidence(
                historical
            )
        )

        # Strong temporal ordering.
        if (
            preferred_time is not None
            and historical_time is not None
            and preferred_time != historical_time
        ):

            return min(
                1.0,
                0.75
                + confidence_gap * 0.25
            )

        # Same/unknown temporal position.
        return min(
            1.0,
            0.50
            + confidence_gap * 0.50
        )

    # EXPLANATION

    def _build_reason(
        self,
        preferred: RetrievalResult,
        historical: RetrievalResult
    ) -> str:
        """
        Explain why the preferred memory was selected.
        """

        preferred_time = self._timestamp(
            preferred
        )

        historical_time = self._timestamp(
            historical
        )

        if (
            preferred_time is not None
            and historical_time is not None
        ):

            if preferred_time > historical_time:

                return (
                    "Preferred the newer memory "
                    "because it has a more recent "
                    "timestamp. The older memory was "
                    "preserved as historical context."
                )

            if preferred_time < historical_time:

                return (
                    "Preferred the newer memory based "
                    "on temporal ordering. The other "
                    "memory was preserved as historical "
                    "context."
                )

        preferred_confidence = (
            self._memory_confidence(
                preferred
            )
        )

        historical_confidence = (
            self._memory_confidence(
                historical
            )
        )

        if (
            preferred_confidence
            > historical_confidence
        ):

            return (
                "Preferred the memory with higher "
                "confidence. The other memory was "
                "preserved as historical context."
            )

        return (
            "Preferred the memory with higher "
            "retrieval relevance. The other memory "
            "was preserved as historical context."
        )