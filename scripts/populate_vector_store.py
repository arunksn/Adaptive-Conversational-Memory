import json
from datetime import datetime
from pathlib import Path

from src.models.memory import (
    Memory,
    MemoryStatus,
    MemoryType,
)

from src.retrieval.vector_retriever import (
    VectorRetriever,
)


SEED_PATH = Path(
    "data/memories/seed_memories.json"
)

VECTOR_STORE_PATH = Path(
    "data/memories/vector_store"
)


def parse_datetime(
    value
):
    if value is None:
        return None

    return datetime.fromisoformat(
        value
    )


def memory_from_dict(
    data: dict
) -> Memory:

    return Memory(
        memory_id=data["memory_id"],
        content=data["content"],
        memory_type=MemoryType(
            data["memory_type"]
        ),
        created_at=parse_datetime(
            data["created_at"]
        ),
        event_time=parse_datetime(
            data.get("event_time")
        ),
        importance=data.get(
            "importance",
            0.5
        ),
        confidence=data.get(
            "confidence",
            1.0
        ),
        status=MemoryStatus(
            data.get(
                "status",
                "active"
            )
        ),
        source=data.get(
            "source"
        ),
        entities=data.get(
            "entities",
            []
        ),
        metadata=data.get(
            "metadata",
            {}
        ),
        access_count=data.get(
            "access_count",
            0
        ),
        last_accessed=parse_datetime(
            data.get("last_accessed")
        ),
    )


def load_seed_memories() -> list[Memory]:

    if not SEED_PATH.exists():
        raise FileNotFoundError(
            f"Seed memory file not found: "
            f"{SEED_PATH}"
        )

    with open(
        SEED_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if not isinstance(
        data,
        list
    ):
        raise ValueError(
            "seed_memories.json must contain "
            "a JSON list."
        )

    return [
        memory_from_dict(
            item
        )
        for item in data
    ]


def main():

    print(
        "Loading seed memories..."
    )

    memories = load_seed_memories()

    print(
        f"Loaded {len(memories)} memories."
    )

    VECTOR_STORE_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    retriever = VectorRetriever(
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        storage_dir=str(
            VECTOR_STORE_PATH
        )
    )

    print(
        "Creating embeddings and "
        "populating vector store..."
    )

    for index, memory in enumerate(
        memories,
        start=1
    ):

        print(
            f"[{index}/{len(memories)}] "
            f"{memory.memory_id} - "
            f"{memory.content}"
        )

        retriever.add_memory(
            memory
        )

    retriever.save()

    print()
    print(
        "Vector store populated successfully."
    )

    print(
        f"Stored memories: "
        f"{retriever.count()}"
    )

    print(
        f"Location: "
        f"{VECTOR_STORE_PATH}"
    )


if __name__ == "__main__":
    main()