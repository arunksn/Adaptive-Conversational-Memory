from dataclasses import dataclass
from datetime import datetime

from src.consolidation.memory_consolidator import (
    ConsolidationResult,
    MemoryConsolidator
)

from src.forgetting.forgetting_engine import (
    ForgettingDecision,
    ForgettingEngine
)

from src.models.memory import (
    Memory,
    MemoryStatus
)


@dataclass
class LifecycleResult:
    """
    Result of one memory lifecycle cycle.
    """

    consolidation_results: list[
        ConsolidationResult
    ]

    forgetting_decisions: list[
        ForgettingDecision
    ]

    active_memories: list[Memory]

    archived_memories: list[Memory]

    forgotten_memories: list[Memory]


class MemoryLifecycle:

    def __init__(
        self,
        consolidator: MemoryConsolidator | None = None,
        forgetting_engine: ForgettingEngine | None = None
    ):
        """
        Coordinate memory consolidation and controlled
        forgetting.

        The lifecycle manager deliberately keeps the
        consolidation and forgetting components separate.
        This class only orchestrates them.
        """

        self.consolidator = (
            consolidator
            if consolidator is not None
            else MemoryConsolidator()
        )

        self.forgetting_engine = (
            forgetting_engine
            if forgetting_engine is not None
            else ForgettingEngine()
        )

    # FULL LIFECYCLE


    def process(
        self,
        memories: list[Memory],
        now: datetime | None = None
    ) -> LifecycleResult:
        """
        Run one complete memory lifecycle cycle.

        Steps:

        1. Separate semantic and episodic memories.
        2. Consolidate repeated episodic information.
        3. Reinforce matching semantic memories.
        4. Add newly created semantic memories.
        5. Evaluate forgetting decisions.
        6. Apply KEEP / ARCHIVE / FORGET actions.
        """

        semantic_memories = [
            memory
            for memory in memories
            if memory.memory_type.value
            == "semantic"
        ]

        consolidation_results = (
            self.consolidator.consolidate(
                memories=memories,
                existing_semantic_memories=(
                    semantic_memories
                )
            )
        )

        # Add newly created semantic memories to the lifecycle
        # collection.
     

        all_memories = list(
            memories
        )

        for result in consolidation_results:

            semantic_memory = (
                result.semantic_memory
            )

            if all(
                memory.memory_id
                != semantic_memory.memory_id
                for memory in all_memories
            ):

                all_memories.append(
                    semantic_memory
                )

        # Evaluate every memory.
        

        forgetting_decisions = (
            self.forgetting_engine.evaluate_all(
                all_memories,
                now=now
            )
        )

        # Apply lifecycle actions.
    

        for decision in forgetting_decisions:

            self.forgetting_engine.apply(
                decision
            )

        active_memories = [
            memory
            for memory in all_memories
            if memory.status
            == MemoryStatus.ACTIVE
        ]

        archived_memories = [
            memory
            for memory in all_memories
            if memory.status
            == MemoryStatus.ARCHIVED
        ]

        forgotten_memories = [
            memory
            for memory in all_memories
            if memory.status
            == MemoryStatus.FORGOTTEN
        ]

        return LifecycleResult(
            consolidation_results=(
                consolidation_results
            ),
            forgetting_decisions=(
                forgetting_decisions
            ),
            active_memories=(
                active_memories
            ),
            archived_memories=(
                archived_memories
            ),
            forgotten_memories=(
                forgotten_memories
            )
        )

    # CONSOLIDATION ONLY

    def consolidate(
        self,
        memories: list[Memory]
    ) -> list[ConsolidationResult]:
        """
        Run only the consolidation stage.

        Useful when consolidation needs to be executed
        independently from forgetting.
        """

        semantic_memories = [
            memory
            for memory in memories
            if memory.memory_type.value
            == "semantic"
        ]

        return (
            self.consolidator.consolidate(
                memories=memories,
                existing_semantic_memories=(
                    semantic_memories
                )
            )
        )

    # FORGETTING ONLY

    def evaluate_forgetting(
        self,
        memories: list[Memory],
        now: datetime | None = None
    ) -> list[ForgettingDecision]:
        """
        Run only the forgetting evaluation stage.

        No memory status is changed by this method.
        """

        return (
            self.forgetting_engine.evaluate_all(
                memories,
                now=now
            )
        )

    # APPLY FORGETTING

    def apply_forgetting(
        self,
        decisions: list[ForgettingDecision]
    ) -> list[Memory]:
        """
        Apply previously calculated forgetting decisions.
        """

        memories = []

        for decision in decisions:

            memory = (
                self.forgetting_engine.apply(
                    decision
                )
            )

            memories.append(
                memory
            )

        return memories