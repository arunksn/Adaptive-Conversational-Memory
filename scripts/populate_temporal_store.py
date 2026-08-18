import json
from datetime import datetime

from src.models.memory import (
    Memory,
    MemoryType,
    MemoryStatus,
)

from src.storage.temporal_store import (
    TemporalStore,
)


SEED_PATH = "data/memories/seed_memories.json"

DATABASE_PATH = (
    "data/memories/episodic_memory.db"
)


def load_seed_memories():
    print("Loading seed memories...")

    with open(
        SEED_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    print(
        f"Loaded {len(data)} memories."
    )

    return data


def memory_from_dict(data: dict) -> Memory:
    event_time = data.get(
        "event_time"
    )

    created_at = data.get(
        "created_at"
    )

    last_accessed = data.get(
        "last_accessed"
    )

    return Memory(
        memory_id=data["memory_id"],
        content=data["content"],
        memory_type=MemoryType(
            data["memory_type"]
        ),
        created_at=(
            datetime.fromisoformat(
                created_at
            )
            if created_at
            else datetime.now()
        ),
        event_time=(
            datetime.fromisoformat(
                event_time
            )
            if event_time
            else None
        ),
        importance=data.get(
            "importance",
            0.5,
        ),
        confidence=data.get(
            "confidence",
            1.0,
        ),
        status=MemoryStatus(
            data.get(
                "status",
                "active",
            )
        ),
        source=data.get(
            "source"
        ),
        entities=data.get(
            "entities",
            [],
        ),
        metadata=data.get(
            "metadata",
            {},
        ),
        access_count=data.get(
            "access_count",
            0,
        ),
        last_accessed=(
            datetime.fromisoformat(
                last_accessed
            )
            if last_accessed
            else None
        ),
    )


def populate():
    seed_memories = load_seed_memories()

    episodic_memories = [
        memory_from_dict(memory)
        for memory in seed_memories
        if memory.get("memory_type")
        == "episodic"
    ]

    print(
        f"Found {len(episodic_memories)} "
        "episodic memories."
    )

    store = TemporalStore(
        database_path=DATABASE_PATH
    )

    print(
        "Populating temporal store..."
    )

    added = 0
    updated = 0

    for index, memory in enumerate(
        episodic_memories,
        start=1,
    ):

        print(
            f"[{index}/{len(episodic_memories)}] "
            f"{memory.memory_id} - "
            f"{memory.content}"
        )

        existing = store.get_event(
            memory.memory_id
        )

        if existing is None:

            store.add_event(
                memory
            )

            added += 1

        else:

            store.update_event(
                memory
            )

            updated += 1

    count = store.count()

    store.close()

    print()
    print(
        "Temporal store populated successfully."
    )
    print(
        f"Added memories: {added}"
    )
    print(
        f"Updated memories: {updated}"
    )
    print(
        f"Stored episodic memories: {count}"
    )
    print(
        f"Location: {DATABASE_PATH}"
    )


if __name__ == "__main__":
    populate()