from datetime import datetime

from src.conflict.conflict_detector import (
    ConflictDetector
)

from src.conflict.conflict_resolver import (
    ConflictResolver
)

from src.models.memory import (
    Memory,
    MemoryType
)

from src.retrieval.hybrid_retriever import (
    RetrievalResult
)

from src.routing.memory_router import (
    MemoryRoute
)


def create_memory(
    content,
    event_time=None,
    importance=0.5,
    confidence=0.5
):
    memory = Memory(
        content=content,
        memory_type=MemoryType.SEMANTIC,
        event_time=event_time,
        importance=importance
    )

    # Store explicit confidence on the memory.
    memory.confidence = confidence

    return memory


def create_result(
    content,
    event_time=None,
    importance=0.5,
    confidence=0.5,
    score=0.5
):
    memory = create_memory(
        content=content,
        event_time=event_time,
        importance=importance,
        confidence=confidence
    )

    return RetrievalResult(
        source=MemoryRoute.SEMANTIC,
        item=memory,
        score=score,
        memory_id=memory.memory_id
    )


# DETECTOR TESTS

def test_detect_explicit_contradiction():

    detector = ConflictDetector()

    first = create_result(
        "I use TensorFlow."
    )

    second = create_result(
        "I switched to PyTorch."
    )

    conflicts = detector.detect(
        [first, second]
    )

    assert len(conflicts) == 1


def test_detect_preference_conflict():

    detector = ConflictDetector()

    first = create_result(
        "I prefer Python."
    )

    second = create_result(
        "I no longer prefer Python."
    )

    conflicts = detector.detect(
        [first, second]
    )

    assert len(conflicts) == 1


def test_detect_boolean_conflict():

    detector = ConflictDetector()

    first = create_result(
        "The feature is enabled."
    )

    second = create_result(
        "The feature is disabled."
    )

    conflicts = detector.detect(
        [first, second]
    )

    assert len(conflicts) == 1


def test_no_conflict_for_unrelated_memories():

    detector = ConflictDetector()

    first = create_result(
        "I use Python."
    )

    second = create_result(
        "I live in Chennai."
    )

    conflicts = detector.detect(
        [first, second]
    )

    assert conflicts == []


def test_same_memory_is_not_conflict():

    detector = ConflictDetector()

    result = create_result(
        "I use Python."
    )

    conflicts = detector.detect(
        [result, result]
    )

    assert conflicts == []


def test_different_memory_types_are_not_compared():

    detector = ConflictDetector()

    semantic_memory = create_result(
        "I use Python."
    )

    procedural_memory = create_result(
        "I switched to Python."
    )

    procedural_memory.item.memory_type = (
        MemoryType.PROCEDURAL
    )

    conflicts = detector.detect(
        [
            semantic_memory,
            procedural_memory
        ]
    )

    assert conflicts == []


def test_conflict_contains_reason():

    detector = ConflictDetector()

    first = create_result(
        "I use TensorFlow."
    )

    second = create_result(
        "I switched to PyTorch."
    )

    conflicts = detector.detect(
        [first, second]
    )

    assert conflicts[0].reason

    assert (
        conflicts[0].confidence
        > 0.0
    )

# RESOLVER TESTS

def test_newer_memory_is_preferred():

    resolver = ConflictResolver()

    older = create_result(
        "I use TensorFlow.",
        event_time=datetime(
            2026,
            1,
            1
        ),
        confidence=0.8
    )

    newer = create_result(
        "I switched to PyTorch.",
        event_time=datetime(
            2026,
            8,
            1
        ),
        confidence=0.7
    )

    detector = ConflictDetector()

    conflict = detector.detect(
        [
            older,
            newer
        ]
    )[0]

    resolution = resolver.resolve(
        conflict
    )

    assert resolution.preferred is newer

    assert resolution.historical is older


def test_higher_confidence_wins_when_timestamp_equal():

    resolver = ConflictResolver()

    first = create_result(
        "I use Python.",
        event_time=datetime(
            2026,
            8,
            1
        ),
        confidence=0.9
    )

    second = create_result(
        "I use Java.",
        event_time=datetime(
            2026,
            8,
            1
        ),
        confidence=0.6
    )

    detector = ConflictDetector()

    conflict = detector.detect(
        [
            first,
            second
        ]
    )

    # These statements are not detected by the
    # baseline contradiction detector, so manually
    # construct the conflict for resolver testing.

    from src.conflict.conflict_detector import (
        ConflictPair
    )

    conflict_pair = ConflictPair(
        first=first,
        second=second,
        reason="Same attribute conflict.",
        confidence=0.8
    )

    resolution = resolver.resolve(
        conflict_pair
    )

    assert resolution.preferred is first


def test_retrieval_score_breaks_complete_tie():

    resolver = ConflictResolver()

    first = create_result(
        "I use Python.",
        event_time=None,
        confidence=0.5,
        score=0.9
    )

    second = create_result(
        "I use Java.",
        event_time=None,
        confidence=0.5,
        score=0.6
    )

    # Remove automatically generated creation timestamps.
    # This creates a genuine temporal tie so that the
    # retrieval score becomes the final tie-breaker.

    first.item.event_time = None
    first.item.created_at = None

    second.item.event_time = None
    second.item.created_at = None

    from src.conflict.conflict_detector import (
        ConflictPair
    )

    conflict = ConflictPair(
        first=first,
        second=second,
        reason="Same attribute conflict.",
        confidence=0.8
    )

    resolution = resolver.resolve(
        conflict
    )

    assert resolution.preferred is first


def test_resolution_preserves_historical_memory():

    resolver = ConflictResolver()

    older = create_result(
        "I use TensorFlow.",
        event_time=datetime(
            2026,
            1,
            1
        )
    )

    newer = create_result(
        "I switched to PyTorch.",
        event_time=datetime(
            2026,
            8,
            1
        )
    )

    from src.conflict.conflict_detector import (
        ConflictPair
    )

    conflict = ConflictPair(
        first=older,
        second=newer,
        reason="Technology change.",
        confidence=0.9
    )

    resolution = resolver.resolve(
        conflict
    )

    assert resolution.preferred is newer

    assert resolution.historical is older

    assert (
        resolution.historical
        is not None
    )


def test_resolution_has_reason():

    resolver = ConflictResolver()

    older = create_result(
        "I use TensorFlow.",
        event_time=datetime(
            2026,
            1,
            1
        )
    )

    newer = create_result(
        "I switched to PyTorch.",
        event_time=datetime(
            2026,
            8,
            1
        )
    )

    from src.conflict.conflict_detector import (
        ConflictPair
    )

    conflict = ConflictPair(
        first=older,
        second=newer,
        reason="Technology change.",
        confidence=0.9
    )

    resolution = resolver.resolve(
        conflict
    )

    assert resolution.reason

    assert (
        resolution.confidence
        > 0.0
    )


def test_resolution_confidence_is_bounded():

    resolver = ConflictResolver()

    first = create_result(
        "I use Python.",
        event_time=datetime(
            2026,
            1,
            1
        ),
        confidence=1.0
    )

    second = create_result(
        "I use Java.",
        event_time=datetime(
            2026,
            8,
            1
        ),
        confidence=0.0
    )

    from src.conflict.conflict_detector import (
        ConflictPair
    )

    conflict = ConflictPair(
        first=first,
        second=second,
        reason="Conflict.",
        confidence=1.0
    )

    resolution = resolver.resolve(
        conflict
    )

    assert 0.0 <= (
        resolution.confidence
    ) <= 1.0