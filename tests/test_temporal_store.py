from datetime import datetime

import pytest

from src.models.memory import Memory, MemoryType
from src.storage.temporal_store import TemporalStore


def create_store(tmp_path):
    database_path = (
        tmp_path / "episodic_memory.db"
    )

    return TemporalStore(
        database_path=str(database_path)
    )


def create_episodic_memory(
    content: str,
    event_time: datetime
):
    return Memory(
        content=content,
        memory_type=MemoryType.EPISODIC,
        event_time=event_time,
        importance=0.8,
        confidence=0.95,
        source="conversation"
    )


def test_add_event(tmp_path):

    store = create_store(tmp_path)

    event_time = datetime(
        2026,
        8,
        10,
        15,
        30
    )

    memory = create_episodic_memory(
        "I attended an AI workshop.",
        event_time
    )

    store.add_event(memory)

    assert store.count() == 1

    store.close()


def test_get_event(tmp_path):

    store = create_store(tmp_path)

    event_time = datetime(
        2026,
        8,
        10,
        15,
        30
    )

    memory = create_episodic_memory(
        "I attended an AI workshop.",
        event_time
    )

    store.add_event(memory)

    result = store.get_event(
        memory.memory_id
    )

    assert result is not None
    assert result.content == (
        "I attended an AI workshop."
    )

    assert result.event_time == event_time

    store.close()


def test_get_event_not_found(tmp_path):

    store = create_store(tmp_path)

    result = store.get_event(
        "non-existent-id"
    )

    assert result is None

    store.close()


def test_only_episodic_memories_allowed(tmp_path):

    store = create_store(tmp_path)

    memory = Memory(
        content="I prefer Python.",
        memory_type=MemoryType.SEMANTIC
    )

    with pytest.raises(ValueError):
        store.add_event(memory)

    store.close()


def test_get_events_between(tmp_path):

    store = create_store(tmp_path)

    event1_time = datetime(
        2026,
        8,
        5,
        10,
        0
    )

    event2_time = datetime(
        2026,
        8,
        10,
        15,
        30
    )

    event3_time = datetime(
        2026,
        8,
        20,
        18,
        0
    )

    memory1 = create_episodic_memory(
        "I attended a workshop.",
        event1_time
    )

    memory2 = create_episodic_memory(
        "I worked on my AI project.",
        event2_time
    )

    memory3 = create_episodic_memory(
        "I went to a conference.",
        event3_time
    )

    store.add_event(memory1)
    store.add_event(memory2)
    store.add_event(memory3)

    results = store.get_events_between(
        datetime(2026, 8, 1),
        datetime(2026, 8, 15)
    )

    assert len(results) == 2

    assert results[0].content == (
        "I attended a workshop."
    )

    assert results[1].content == (
        "I worked on my AI project."
    )

    store.close()


def test_events_are_ordered_by_event_time(tmp_path):

    store = create_store(tmp_path)

    later_time = datetime(
        2026,
        8,
        20
    )

    earlier_time = datetime(
        2026,
        8,
        5
    )

    later_memory = create_episodic_memory(
        "Later event.",
        later_time
    )

    earlier_memory = create_episodic_memory(
        "Earlier event.",
        earlier_time
    )

    store.add_event(later_memory)
    store.add_event(earlier_memory)

    results = store.get_events_between(
        datetime(2026, 8, 1),
        datetime(2026, 8, 30)
    )

    assert results[0].content == "Earlier event."
    assert results[1].content == "Later event."

    store.close()


def test_get_recent_events(tmp_path):

    store = create_store(tmp_path)

    dates = [
        datetime(2026, 8, 5),
        datetime(2026, 8, 10),
        datetime(2026, 8, 20)
    ]

    for index, date in enumerate(dates):
        memory = create_episodic_memory(
            f"Event {index}",
            date
        )

        store.add_event(memory)

    results = store.get_recent_events(
        limit=2
    )

    assert len(results) == 2

    assert results[0].content == "Event 2"
    assert results[1].content == "Event 1"

    store.close()


def test_update_event(tmp_path):

    store = create_store(tmp_path)

    original_time = datetime(
        2026,
        8,
        10
    )

    updated_time = datetime(
        2026,
        8,
        15
    )

    memory = create_episodic_memory(
        "Original event.",
        original_time
    )

    store.add_event(memory)

    memory.content = "Updated event."
    memory.event_time = updated_time
    memory.importance = 0.9

    store.update_event(memory)

    result = store.get_event(
        memory.memory_id
    )

    assert result.content == "Updated event."
    assert result.event_time == updated_time
    assert result.importance == 0.9

    store.close()


def test_delete_event(tmp_path):

    store = create_store(tmp_path)

    memory = create_episodic_memory(
        "Temporary event.",
        datetime(2026, 8, 10)
    )

    store.add_event(memory)

    assert store.count() == 1

    store.delete_event(
        memory.memory_id
    )

    assert store.count() == 0

    assert (
        store.get_event(
            memory.memory_id
        )
        is None
    )

    store.close()


def test_delete_nonexistent_event(tmp_path):

    store = create_store(tmp_path)

    with pytest.raises(ValueError):
        store.delete_event(
            "non-existent-id"
        )

    store.close()


def test_persistent_database(tmp_path):

    database_path = (
        tmp_path / "episodic_memory.db"
    )

    event_time = datetime(
        2026,
        8,
        10,
        15,
        30
    )

    memory = create_episodic_memory(
        "Persistent episodic memory.",
        event_time
    )

    store = TemporalStore(
        database_path=str(database_path)
    )

    store.add_event(memory)

    store.close()

    # Open the same database again.
    new_store = TemporalStore(
        database_path=str(database_path)
    )

    result = new_store.get_event(
        memory.memory_id
    )

    assert result is not None
    assert result.content == (
        "Persistent episodic memory."
    )

    assert result.event_time == event_time

    new_store.close()