from datetime import datetime, timedelta, timezone

import pytest

from src.forgetting.forgetting_engine import (
    ForgettingAction,
    ForgettingEngine
)

from src.models.memory import (
    Memory,
    MemoryStatus,
    MemoryType
)


def create_memory(
    content="Test memory.",
    importance=0.5,
    access_count=0,
    event_time=None
):
    memory = Memory(
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=importance,
        event_time=event_time
    )

    for _ in range(access_count):
        memory.access()

    return memory



def test_forgetting_engine_initialization():

    engine = ForgettingEngine()

    assert engine.keep_threshold == 0.65

    assert engine.archive_threshold == 0.35

    assert engine.recency_half_life_days == 30.0

    assert engine.redundancy_threshold == 0.85


def test_invalid_thresholds():

    with pytest.raises(
        ValueError
    ):
        ForgettingEngine(
            keep_threshold=0.30,
            archive_threshold=0.50
        )


def test_invalid_half_life():

    with pytest.raises(
        ValueError
    ):
        ForgettingEngine(
            recency_half_life_days=0
        )


def test_invalid_redundancy_threshold():

    with pytest.raises(
        ValueError
    ):
        ForgettingEngine(
            redundancy_threshold=1.5
        )



def test_importance_score():

    engine = ForgettingEngine()

    memory = create_memory(
        importance=0.8
    )

    score = engine._importance_score(
        memory
    )

    assert score == 0.8


def test_importance_score_is_bounded():

    engine = ForgettingEngine()

    memory = create_memory(
        importance=1.5
    )

    score = engine._importance_score(
        memory
    )

    assert score == 1.0



def test_recent_memory_has_high_recency():

    engine = ForgettingEngine()

    now = datetime(
        2026,
        8,
        18,
        tzinfo=timezone.utc
    )

    memory = create_memory(
        event_time=now
    )

    score = engine._recency_score(
        memory,
        now
    )

    assert score == 1.0


def test_half_life_memory_has_recency_near_half():

    engine = ForgettingEngine(
        recency_half_life_days=30
    )

    now = datetime(
        2026,
        8,
        18,
        tzinfo=timezone.utc
    )

    event_time = (
        now
        - timedelta(days=30)
    )

    memory = create_memory(
        event_time=event_time
    )

    score = engine._recency_score(
        memory,
        now
    )

    assert score == pytest.approx(
        0.5,
        abs=0.01
    )


def test_old_memory_has_lower_recency():

    engine = ForgettingEngine()

    now = datetime(
        2026,
        8,
        18,
        tzinfo=timezone.utc
    )

    old_time = (
        now
        - timedelta(days=180)
    )

    memory = create_memory(
        event_time=old_time
    )

    score = engine._recency_score(
        memory,
        now
    )

    assert score < 0.1


def test_untimestamped_memory_has_neutral_recency():

    engine = ForgettingEngine()

    now = datetime(
        2026,
        8,
        18,
        tzinfo=timezone.utc
    )

    memory = create_memory(
        event_time=None
    )

    score = engine._recency_score(
        memory,
        now
    )

    assert score == 0.5



def test_never_accessed_memory_has_zero_access_score():

    engine = ForgettingEngine()

    memory = create_memory(
        access_count=0
    )

    score = engine._access_frequency_score(
        memory
    )

    assert score == 0.0


def test_accessed_memory_has_positive_access_score():

    engine = ForgettingEngine()

    memory = create_memory(
        access_count=5
    )

    score = engine._access_frequency_score(
        memory
    )

    assert score > 0.0

    assert score <= 1.0


def test_access_frequency_is_bounded():

    engine = ForgettingEngine()

    memory = create_memory(
        access_count=100000
    )

    score = engine._access_frequency_score(
        memory
    )

    assert score <= 1.0



def test_no_related_memories_have_zero_redundancy():

    engine = ForgettingEngine()

    memory = create_memory(
        content="I use Python."
    )

    score = engine._redundancy_score(
        memory,
        []
    )

    assert score == 0.0


def test_duplicate_memory_has_high_redundancy():

    engine = ForgettingEngine()

    first = create_memory(
        content="I use Python."
    )

    second = create_memory(
        content="I use Python."
    )

    score = engine._redundancy_score(
        first,
        [second]
    )

    assert score == 1.0


def test_unrelated_memory_has_zero_redundancy():

    engine = ForgettingEngine()

    first = create_memory(
        content="I use Python."
    )

    second = create_memory(
        content="I live in Chennai."
    )

    score = engine._redundancy_score(
        first,
        [second]
    )

    assert score == 0.0



def test_retention_score_is_bounded():

    score = ForgettingEngine._retention_score(
        importance=1.0,
        recency=1.0,
        access_frequency=1.0,
        redundancy=0.0
    )

    assert 0.0 <= score <= 1.0


def test_high_quality_memory_has_high_retention():

    score = ForgettingEngine._retention_score(
        importance=1.0,
        recency=1.0,
        access_frequency=1.0,
        redundancy=0.0
    )

    assert score == 1.0


def test_redundancy_reduces_retention():

    without_redundancy = (
        ForgettingEngine._retention_score(
            importance=0.7,
            recency=0.7,
            access_frequency=0.7,
            redundancy=0.0
        )
    )

    with_redundancy = (
        ForgettingEngine._retention_score(
            importance=0.7,
            recency=0.7,
            access_frequency=0.7,
            redundancy=1.0
        )
    )

    assert (
        with_redundancy
        < without_redundancy
    )



def test_high_retention_means_keep():

    engine = ForgettingEngine()

    action = engine._determine_action(
        0.80
    )

    assert (
        action
        == ForgettingAction.KEEP
    )


def test_medium_retention_means_archive():

    engine = ForgettingEngine()

    action = engine._determine_action(
        0.50
    )

    assert (
        action
        == ForgettingAction.ARCHIVE
    )


def test_low_retention_means_forget():

    engine = ForgettingEngine()

    action = engine._determine_action(
        0.20
    )

    assert (
        action
        == ForgettingAction.FORGET
    )


def test_important_recent_memory_is_kept():

    engine = ForgettingEngine()

    now = datetime(
        2026,
        8,
        18,
        tzinfo=timezone.utc
    )

    memory = create_memory(
        importance=1.0,
        access_count=5,
        event_time=now
    )

    decision = engine.evaluate(
        memory,
        now=now
    )

    assert (
        decision.action
        == ForgettingAction.KEEP
    )

    assert (
        decision.score.retention
        >= engine.keep_threshold
    )


def test_old_low_value_memory_can_be_forgotten():

    engine = ForgettingEngine()

    now = datetime(
        2026,
        8,
        18,
        tzinfo=timezone.utc
    )

    old_time = (
        now
        - timedelta(days=365)
    )

    memory = create_memory(
        importance=0.0,
        access_count=0,
        event_time=old_time
    )

    decision = engine.evaluate(
        memory,
        now=now
    )

    assert (
        decision.action
        == ForgettingAction.FORGET
    )


def test_medium_value_memory_can_be_archived():

    engine = ForgettingEngine()

    now = datetime(
        2026,
        8,
        18,
        tzinfo=timezone.utc
    )

    memory = create_memory(
        importance=0.4,
        access_count=0,
        event_time=now
    )

    decision = engine.evaluate(
        memory,
        now=now
    )

    assert (
        decision.action
        in {
            ForgettingAction.ARCHIVE,
            ForgettingAction.FORGET
        }
    )



def test_apply_keep_preserves_active_status():

    engine = ForgettingEngine()

    memory = create_memory(
        importance=1.0
    )

    decision = engine.evaluate(
        memory,
        now=datetime(
            2026,
            8,
            18,
            tzinfo=timezone.utc
        )
    )

    assert (
        decision.action
        == ForgettingAction.KEEP
    )

    engine.apply(
        decision
    )

    assert (
        memory.status
        == MemoryStatus.ACTIVE
    )


def test_apply_archive_changes_status():

    engine = ForgettingEngine()

    memory = create_memory()

    decision = engine._determine_action(
        0.50
    )

    manual_decision = (
        decision
    )

    from src.forgetting.forgetting_engine import (
        ForgettingDecision,
        ForgettingScore
    )

    full_decision = ForgettingDecision(
        memory=memory,
        action=manual_decision,
        score=ForgettingScore(
            importance=0.4,
            recency=0.5,
            access_frequency=0.0,
            redundancy=0.0,
            retention=0.5
        ),
        reason="Archive test."
    )

    engine.apply(
        full_decision
    )

    assert (
        memory.status
        == MemoryStatus.ARCHIVED
    )


def test_apply_forget_changes_status():

    engine = ForgettingEngine()

    memory = create_memory()

    from src.forgetting.forgetting_engine import (
        ForgettingDecision,
        ForgettingScore
    )

    decision = ForgettingDecision(
        memory=memory,
        action=ForgettingAction.FORGET,
        score=ForgettingScore(
            importance=0.0,
            recency=0.0,
            access_frequency=0.0,
            redundancy=0.0,
            retention=0.0
        ),
        reason="Forget test."
    )

    engine.apply(
        decision
    )

    assert (
        memory.status
        == MemoryStatus.FORGOTTEN
    )



def test_evaluate_all_returns_decision_for_each_memory():

    engine = ForgettingEngine()

    memories = [
        create_memory(
            content="Memory one."
        ),
        create_memory(
            content="Memory two."
        ),
        create_memory(
            content="Memory three."
        )
    ]

    decisions = engine.evaluate_all(
        memories,
        now=datetime(
            2026,
            8,
            18,
            tzinfo=timezone.utc
        )
    )

    assert len(decisions) == 3


def test_forgetting_reason_contains_scores():

    engine = ForgettingEngine()

    memory = create_memory()

    decision = engine.evaluate(
        memory,
        now=datetime(
            2026,
            8,
            18,
            tzinfo=timezone.utc
        )
    )

    assert "retention=" in decision.reason

    assert "importance=" in decision.reason

    assert "recency=" in decision.reason

    assert "access_frequency=" in decision.reason

    assert "redundancy=" in decision.reason