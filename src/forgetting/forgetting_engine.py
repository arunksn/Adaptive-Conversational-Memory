from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher

from src.models.memory import (
    Memory,
    MemoryStatus
)


class ForgettingAction:

    KEEP = "keep"
    ARCHIVE = "archive"
    FORGET = "forget"


@dataclass
class ForgettingScore:
    """
    Represents the factors used to determine whether
    a memory should be retained.
    """

    importance: float
    recency: float
    access_frequency: float
    redundancy: float
    retention: float


@dataclass
class ForgettingDecision:
    """
    Decision produced by the forgetting engine.
    """

    memory: Memory
    action: str
    score: ForgettingScore
    reason: str


class ForgettingEngine:

    def __init__(
        self,
        keep_threshold: float = 0.65,
        archive_threshold: float = 0.35,
        recency_half_life_days: float = 30.0,
        redundancy_threshold: float = 0.85
    ):
        """
        Create a deterministic controlled forgetting engine.

        keep_threshold:
            Minimum retention score required to keep a
            memory active.

        archive_threshold:
            Minimum retention score required to archive
            rather than forget a memory.

        recency_half_life_days:
            Number of days after which the recency score
            falls to approximately 0.5.

        redundancy_threshold:
            Similarity above which a memory is considered
            highly redundant.
        """

        if not (
            0.0
            <= archive_threshold
            <= keep_threshold
            <= 1.0
        ):
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= archive_threshold <= "
                "keep_threshold <= 1"
            )

        if recency_half_life_days <= 0:
            raise ValueError(
                "recency_half_life_days must be positive"
            )

        if not (
            0.0
            <= redundancy_threshold
            <= 1.0
        ):
            raise ValueError(
                "redundancy_threshold must be "
                "between 0.0 and 1.0"
            )

        self.keep_threshold = (
            keep_threshold
        )

        self.archive_threshold = (
            archive_threshold
        )

        self.recency_half_life_days = (
            recency_half_life_days
        )

        self.redundancy_threshold = (
            redundancy_threshold
        )

    # MAIN DECISION

    def evaluate(
        self,
        memory: Memory,
        now: datetime | None = None,
        related_memories: list[Memory] | None = None
    ) -> ForgettingDecision:
        """
        Evaluate a memory and determine whether it
        should be kept, archived, or forgotten.
        """

        current_time = (
            now
            if now is not None
            else datetime.now(timezone.utc)
        )

        importance = (
            self._importance_score(
                memory
            )
        )

        recency = (
            self._recency_score(
                memory,
                current_time
            )
        )

        access_frequency = (
            self._access_frequency_score(
                memory
            )
        )

        redundancy = (
            self._redundancy_score(
                memory,
                related_memories or []
            )
        )

        retention = (
            self._retention_score(
                importance=importance,
                recency=recency,
                access_frequency=(
                    access_frequency
                ),
                redundancy=redundancy
            )
        )

        action = (
            self._determine_action(
                retention
            )
        )

        reason = (
            self._build_reason(
                action=action,
                importance=importance,
                recency=recency,
                access_frequency=(
                    access_frequency
                ),
                redundancy=redundancy,
                retention=retention
            )
        )

        return ForgettingDecision(
            memory=memory,
            action=action,
            score=ForgettingScore(
                importance=importance,
                recency=recency,
                access_frequency=(
                    access_frequency
                ),
                redundancy=redundancy,
                retention=retention
            ),
            reason=reason
        )


    def evaluate_all(
        self,
        memories: list[Memory],
        now: datetime | None = None
    ) -> list[ForgettingDecision]:
        """
        Evaluate a collection of memories.

        Each memory is compared against the other memories
        for redundancy.
        """

        decisions = []

        for memory in memories:

            related_memories = [
                other
                for other in memories
                if other.memory_id
                != memory.memory_id
            ]

            decisions.append(
                self.evaluate(
                    memory=memory,
                    now=now,
                    related_memories=(
                        related_memories
                    )
                )
            )

        return decisions


    def apply(
        self,
        decision: ForgettingDecision
    ) -> Memory:
        """
        Apply a forgetting decision to the memory.

        KEEP:
            Memory remains active.

        ARCHIVE:
            Memory is moved to archived state.

        FORGET:
            Memory is moved to forgotten state.
        """

        if decision.action == (
            ForgettingAction.KEEP
        ):

            return decision.memory

        if decision.action == (
            ForgettingAction.ARCHIVE
        ):

            decision.memory.archive()

            return decision.memory

        if decision.action == (
            ForgettingAction.FORGET
        ):

            decision.memory.forget()

            return decision.memory

        raise ValueError(
            f"Unknown forgetting action: "
            f"{decision.action}"
        )


    @staticmethod
    def _importance_score(
        memory: Memory
    ) -> float:
        """
        Importance is already normalized by the memory
        model, so it can be used directly.
        """

        return max(
            0.0,
            min(
                1.0,
                memory.importance
            )
        )

    # RECENCY, ACCESS FREQUENCY, REDUNDANCY

    def _recency_score(
        self,
        memory: Memory,
        now: datetime
    ) -> float:
        """
        Calculate recency using exponential decay.

        A memory at the half-life receives a score of
        approximately 0.5.

        Memories without an explicit event timestamp
        receive a neutral score of 0.5 because their
        actual temporal relevance cannot be reliably
        determined.
        """

        # Do not use created_at as a substitute for
        # event_time. Creation time tells us when the
        # memory was stored, not when the information
        # itself was relevant.

        timestamp = memory.event_time

        if timestamp is None:
            return 0.5

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(
                tzinfo=timezone.utc
            )

        if now.tzinfo is None:
            now = now.replace(
                tzinfo=timezone.utc
            )

        age_seconds = (
            now - timestamp
        ).total_seconds()

        age_days = max(
            0.0,
            age_seconds / 86400.0
        )

        return (
            0.5
            ** (
                age_days
                / self.recency_half_life_days
            )
        )

    @staticmethod
    def _access_frequency_score(
        memory: Memory
    ) -> float:
        """
        Convert access count into a bounded score.

        The logarithmic transformation prevents extremely
        high access counts from dominating the retention
        score.
        """

        import math

        if memory.access_count <= 0:
            return 0.0

        return min(
            1.0,
            math.log1p(
                memory.access_count
            ) / math.log1p(10)
        )


    def _redundancy_score(
        self,
        memory: Memory,
        related_memories: list[Memory]
    ) -> float:
        """
        Calculate how redundant a memory is compared
        with related memories.

        0.0 = no meaningful redundancy.

        1.0 = highly redundant.
        """

        if not related_memories:
            return 0.0

        best_similarity = 0.0

        for other in related_memories:

            if other.status != MemoryStatus.ACTIVE:
                continue

            similarity = (
                self._similarity(
                    memory,
                    other
                )
            )

            if similarity > best_similarity:
                best_similarity = similarity

        if (
            best_similarity
            >= self.redundancy_threshold
        ):
            return best_similarity

        return 0.0

    # RETENTION SCORE

    @staticmethod
    def _retention_score(
        importance: float,
        recency: float,
        access_frequency: float,
        redundancy: float
    ) -> float:
        """
        Calculate overall retention score.

        Importance:
            40%

        Recency:
            25%

        Access frequency:
            20%

        Redundancy penalty:
            15%
        """

        score = (
            (importance * 0.40)
            +
            (recency * 0.25)
            +
            (access_frequency * 0.20)
            +
            ((1.0 - redundancy) * 0.15)
        )

        return max(
            0.0,
            min(
                1.0,
                score
            )
        )


    def _determine_action(
        self,
        retention: float
    ) -> str:
        """
        Convert retention score into a lifecycle action.
        """

        if (
            retention
            >= self.keep_threshold
        ):
            return ForgettingAction.KEEP

        if (
            retention
            >= self.archive_threshold
        ):
            return ForgettingAction.ARCHIVE

        return ForgettingAction.FORGET

    # SIMILARITY

    @staticmethod
    def _normalize(
        text: str
    ) -> str:
        return " ".join(
            text.lower().strip().split()
        )

    def _similarity(
        self,
        first: Memory,
        second: Memory
    ) -> float:
        """
        Calculate normalized textual similarity
        for redundancy detection.
        """

        first_text = self._normalize(
            first.content
        )

        second_text = self._normalize(
            second.content
        )

        if not first_text or not second_text:
            return 0.0

        return SequenceMatcher(
            None,
            first_text,
            second_text
        ).ratio()


    @staticmethod
    def _build_reason(
        action: str,
        importance: float,
        recency: float,
        access_frequency: float,
        redundancy: float,
        retention: float
    ) -> str:
        """
        Produce a human-readable explanation for the
        forgetting decision.
        """

        return (
            f"Action={action}; "
            f"retention={retention:.3f}; "
            f"importance={importance:.3f}; "
            f"recency={recency:.3f}; "
            f"access_frequency="
            f"{access_frequency:.3f}; "
            f"redundancy={redundancy:.3f}."
        )