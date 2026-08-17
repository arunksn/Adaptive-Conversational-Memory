from datetime import datetime, timedelta, timezone

from src.consolidation.memory_consolidator import (
    MemoryConsolidator
)

from src.forgetting.forgetting_engine import (
    ForgettingAction,
    ForgettingEngine
)

from src.lifecycle.memory_lifecycle import (
    MemoryLifecycle
)

from src.models.memory import (
    Memory,
    MemoryStatus,
    MemoryType
)


def create_memory(
    content,
    memory_type,
    importance=0.5,
    event_time=None
):
    return Memory(
        content=content,
        memory_type=memory_type,
        importance=importance,
        event_time=event_time
    )



def test_lifecycle_initialization():

    lifecycle = MemoryLifecycle()

    assert lifecycle.consolidator is not None

    assert lifecycle.forgetting_engine is not None


def test_lifecycle_accepts_custom_components():

    consolidator = MemoryConsolidator(
        similarity_threshold=0.8
    )

    forgetting_engine = ForgettingEngine(
        keep_threshold=0.7
    )

    lifecycle = MemoryLifecycle(
        consolidator=consolidator,
        forgetting_engine=forgetting_engine
    )

    assert (
        lifecycle.consolidator
        is consolidator
    )

    assert (
        lifecycle.forgetting_engine
        is forgetting_engine
    )


def test_lifecycle_consolidates_repeated_episodic_memories():

    lifecycle = MemoryLifecycle(
        forgetting_engine=ForgettingEngine(
            keep_threshold=0.0,
            archive_threshold=0.0
        )
    )

    first = create_memory(
        content="I use Python.",
        memory_type=MemoryType.EPISODIC,
        importance=0.8
    )

    second = create_memory(
        content="I use Python.",
        memory_type=MemoryType.EPISODIC,
        importance=0.8
    )

    result = lifecycle.process(
        [
            first,
            second
        ],
        now=datetime(
            2026,
            8,
            18,
            tzinfo=timezone.utc
        )
    )

    assert (
        len(
            result.consolidation_results
        )
        == 1
    )

    semantic = (
        result.consolidation_results[
            0
        ].semantic_memory
    )

    assert (
        semantic.memory_type
        == MemoryType.SEMANTIC
    )

    assert (
        semantic.metadata[
            "reinforcement_count"
        ]
        == 2
    )


def test_lifecycle_creates_new_semantic_memory():

    lifecycle = MemoryLifecycle(
        forgetting_engine=ForgettingEngine(
            keep_threshold=0.0,
            archive_threshold=0.0
        )
    )

    first = create_memory(
        content="I use Python.",
        memory_type=MemoryType.EPISODIC
    )

    second = create_memory(
        content="I use Python.",
        memory_type=MemoryType.EPISODIC
    )

    result = lifecycle.process(
        [
            first,
            second
        ]
    )

    semantic = (
        result.consolidation_results[
            0
        ].semantic_memory
    )

    assert (
        semantic.memory_type
        == MemoryType.SEMANTIC
    )

    assert (
        semantic.content
        == "I use Python."
    )


# EXISTING SEMANTIC MEMORY

def test_lifecycle_reinforces_existing_semantic_memory():

    lifecycle = MemoryLifecycle(
        forgetting_engine=ForgettingEngine(
            keep_threshold=0.0,
            archive_threshold=0.0
        )
    )

    semantic = create_memory(
        content="I use Python.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.6
    )

    first = create_memory(
        content="I use Python.",
        memory_type=MemoryType.EPISODIC,
        importance=0.8
    )

    second = create_memory(
        content="I use Python.",
        memory_type=MemoryType.EPISODIC,
        importance=0.8
    )

    result = lifecycle.process(
        [
            semantic,
            first,
            second
        ]
    )

    assert (
        len(
            result.consolidation_results
        )
        == 1
    )

    consolidated = (
        result.consolidation_results[
            0
        ].semantic_memory
    )

    assert (
        consolidated
        is semantic
    )

    assert (
        result.consolidation_results[
            0
        ].created
        is False
    )



def test_lifecycle_preserves_source_episodic_memories():

    lifecycle = MemoryLifecycle(
        forgetting_engine=ForgettingEngine(
            keep_threshold=0.0,
            archive_threshold=0.0
        )
    )

    first = create_memory(
        content="I use Python.",
        memory_type=MemoryType.EPISODIC
    )

    second = create_memory(
        content="I use Python.",
        memory_type=MemoryType.EPISODIC
    )

    result = lifecycle.process(
        [
            first,
            second
        ]
    )

    semantic = (
        result.consolidation_results[
            0
        ].semantic_memory
    )

    source_ids = semantic.metadata[
        "source_memory_ids"
    ]

    assert first.memory_id in source_ids

    assert second.memory_id in source_ids



def test_lifecycle_evaluates_forgetting():

    lifecycle = MemoryLifecycle()

    memory = create_memory(
        content="Important information.",
        memory_type=MemoryType.SEMANTIC,
        importance=1.0,
        event_time=datetime(
            2026,
            8,
            18,
            tzinfo=timezone.utc
        )
    )

    decisions = lifecycle.evaluate_forgetting(
        [
            memory
        ],
        now=datetime(
            2026,
            8,
            18,
            tzinfo=timezone.utc
        )
    )

    assert len(decisions) == 1

    assert (
        decisions[0].memory
        is memory
    )


def test_evaluate_forgetting_does_not_change_status():

    lifecycle = MemoryLifecycle()

    memory = create_memory(
        content="Important information.",
        memory_type=MemoryType.SEMANTIC,
        importance=1.0
    )

    assert (
        memory.status
        == MemoryStatus.ACTIVE
    )

    lifecycle.evaluate_forgetting(
        [
            memory
        ]
    )

    assert (
        memory.status
        == MemoryStatus.ACTIVE
    )



def test_apply_forgetting_changes_status():

    lifecycle = MemoryLifecycle()

    memory = create_memory(
        content="Old information.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.0,
        event_time=(
            datetime(
                2026,
                8,
                18,
                tzinfo=timezone.utc
            )
            - timedelta(days=365)
        )
    )

    decisions = lifecycle.evaluate_forgetting(
        [
            memory
        ],
        now=datetime(
            2026,
            8,
            18,
            tzinfo=timezone.utc
        )
    )

    assert (
        decisions[0].action
        == ForgettingAction.FORGET
    )

    memories = lifecycle.apply_forgetting(
        decisions
    )

    assert len(memories) == 1

    assert (
        memories[0].status
        == MemoryStatus.FORGOTTEN
    )



def test_complete_lifecycle_returns_memory_groups():

    lifecycle = MemoryLifecycle(
        forgetting_engine=ForgettingEngine(
            keep_threshold=0.0,
            archive_threshold=0.0
        )
    )

    first = create_memory(
        content="I use Python.",
        memory_type=MemoryType.EPISODIC,
        importance=0.8
    )

    second = create_memory(
        content="I use Python.",
        memory_type=MemoryType.EPISODIC,
        importance=0.8
    )

    result = lifecycle.process(
        [
            first,
            second
        ]
    )

    assert (
        result.active_memories
        or result.archived_memories
        or result.forgotten_memories
    )


def test_complete_lifecycle_produces_forgetting_decisions():

    lifecycle = MemoryLifecycle()

    memory = create_memory(
        content="Test information.",
        memory_type=MemoryType.SEMANTIC
    )

    result = lifecycle.process(
        [
            memory
        ]
    )

    assert (
        len(
            result.forgetting_decisions
        )
        >= 1
    )



def test_consolidate_only_does_not_forget():

    lifecycle = MemoryLifecycle()

    first = create_memory(
        content="I use Python.",
        memory_type=MemoryType.EPISODIC
    )

    second = create_memory(
        content="I use Python.",
        memory_type=MemoryType.EPISODIC
    )

    results = lifecycle.consolidate(
        [
            first,
            second
        ]
    )

    assert len(results) == 1

    assert (
        first.status
        == MemoryStatus.ACTIVE
    )

    assert (
        second.status
        == MemoryStatus.ACTIVE
    )


def test_forgetting_evaluation_and_application_are_separate():

    lifecycle = MemoryLifecycle()

    memory = create_memory(
        content="Old information.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.0,
        event_time=(
            datetime(
                2026,
                8,
                18,
                tzinfo=timezone.utc
            )
            - timedelta(days=365)
        )
    )

    decisions = lifecycle.evaluate_forgetting(
        [
            memory
        ],
        now=datetime(
            2026,
            8,
            18,
            tzinfo=timezone.utc
        )
    )

    assert (
        memory.status
        == MemoryStatus.ACTIVE
    )

    lifecycle.apply_forgetting(
        decisions
    )

    assert (
        memory.status
        == MemoryStatus.FORGOTTEN
    )