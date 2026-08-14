from src.models.memory import Memory, MemoryType, MemoryStatus


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