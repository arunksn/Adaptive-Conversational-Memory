from datetime import datetime

from src.models.memory import Memory, MemoryType
from src.retrieval.temporal_retriever import TemporalRetriever


def create_retriever(tmp_path):

    database_path = (
        tmp_path / "episodic_memory.db"
    )

    return TemporalRetriever(
        database_path=str(database_path)
    )


def create_memory(
    content: str,
    event_time: datetime
):

    return Memory(
        content=content,
        memory_type=MemoryType.EPISODIC,
        event_time=event_time,
        importance=0.8,
        confidence=0.95
    )


def test_temporal_search(tmp_path):

    retriever = create_retriever(
        tmp_path
    )

    memory1 = create_memory(
        "I attended an AI workshop.",
        datetime(2026, 8, 5)
    )

    memory2 = create_memory(
        "I worked on my project.",
        datetime(2026, 8, 10)
    )

    memory3 = create_memory(
        "I attended a conference.",
        datetime(2026, 8, 20)
    )

    retriever.add_memory(memory1)
    retriever.add_memory(memory2)
    retriever.add_memory(memory3)

    results = retriever.search(
        start_time=datetime(2026, 8, 1),
        end_time=datetime(2026, 8, 15)
    )

    assert len(results) == 2

    assert results[0].content == (
        "I attended an AI workshop."
    )

    assert results[1].content == (
        "I worked on my project."
    )

    retriever.close()


def test_recent_events(tmp_path):

    retriever = create_retriever(
        tmp_path
    )

    retriever.add_memory(
        create_memory(
            "Old event.",
            datetime(2026, 8, 1)
        )
    )

    retriever.add_memory(
        create_memory(
            "Recent event.",
            datetime(2026, 8, 20)
        )
    )

    results = retriever.recent(
        limit=1
    )

    assert len(results) == 1

    assert results[0].content == (
        "Recent event."
    )

    retriever.close()


def test_add_and_count(tmp_path):

    retriever = create_retriever(
        tmp_path
    )

    memory = create_memory(
        "I learned about RAG.",
        datetime(2026, 8, 10)
    )

    retriever.add_memory(memory)

    assert retriever.count() == 1

    retriever.close()


def test_update_memory(tmp_path):

    retriever = create_retriever(
        tmp_path
    )

    memory = create_memory(
        "I attended a workshop.",
        datetime(2026, 8, 10)
    )

    retriever.add_memory(memory)

    memory.content = (
        "I attended an AI workshop."
    )

    retriever.update_memory(
        memory
    )

    results = retriever.search(
        start_time=datetime(2026, 8, 1),
        end_time=datetime(2026, 8, 15)
    )

    assert results[0].content == (
        "I attended an AI workshop."
    )

    retriever.close()


def test_delete_memory(tmp_path):

    retriever = create_retriever(
        tmp_path
    )

    memory = create_memory(
        "Temporary event.",
        datetime(2026, 8, 10)
    )

    retriever.add_memory(memory)

    assert retriever.count() == 1

    retriever.delete_memory(
        memory.memory_id
    )

    assert retriever.count() == 0

    retriever.close()