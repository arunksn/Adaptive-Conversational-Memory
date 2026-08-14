from datetime import datetime

from src.models.memory import (
    Memory,
    MemoryType,
    MemoryStatus
)


def test_memory_creation():

    memory = Memory(
        content="I use Python.",
        memory_type=MemoryType.SEMANTIC
    )

    assert memory.content == "I use Python."
    assert memory.memory_type == MemoryType.SEMANTIC
    assert memory.status == MemoryStatus.ACTIVE


def test_memory_access():

    memory = Memory(
        content="I use Python.",
        memory_type=MemoryType.SEMANTIC
    )

    assert memory.access_count == 0

    memory.access()

    assert memory.access_count == 1
    assert memory.last_accessed is not None


def test_memory_archive():

    memory = Memory(
        content="Old information.",
        memory_type=MemoryType.SEMANTIC
    )

    memory.archive()

    assert memory.status == MemoryStatus.ARCHIVED


def test_memory_forget():

    memory = Memory(
        content="Low-value information.",
        memory_type=MemoryType.SEMANTIC
    )

    memory.forget()

    assert memory.status == MemoryStatus.FORGOTTEN


def test_memory_has_created_at():

    memory = Memory(
        content="I attended an AI workshop.",
        memory_type=MemoryType.EPISODIC
    )

    assert isinstance(
        memory.created_at,
        datetime
    )


def test_episodic_memory_event_time():

    event_time = datetime(
        2026,
        8,
        10,
        15,
        30
    )

    memory = Memory(
        content="I attended an AI workshop.",
        memory_type=MemoryType.EPISODIC,
        event_time=event_time
    )

    assert memory.event_time == event_time


def test_event_time_can_be_none():

    memory = Memory(
        content="I prefer Python.",
        memory_type=MemoryType.SEMANTIC
    )

    assert memory.event_time is None


def test_memory_serialization_contains_temporal_fields():

    event_time = datetime(
        2026,
        8,
        10,
        15,
        30
    )

    memory = Memory(
        content="I attended an AI workshop.",
        memory_type=MemoryType.EPISODIC,
        event_time=event_time
    )

    data = memory.to_dict()

    assert "created_at" in data
    assert "event_time" in data

    assert (
        data["event_time"]
        == event_time.isoformat()
    )