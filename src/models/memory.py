from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import uuid


class MemoryType(Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class MemoryStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"


@dataclass
class Memory:
    content: str
    memory_type: MemoryType

    memory_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    # When the system created/stored this memory.
    created_at: datetime = field(
        default_factory=datetime.now
    )

    # When the actual event happened.
    # For semantic/procedural memories this can remain None.
    event_time: Optional[datetime] = None

    importance: float = 0.5
    confidence: float = 1.0

    status: MemoryStatus = MemoryStatus.ACTIVE

    source: Optional[str] = None

    entities: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    access_count: int = 0

    last_accessed: Optional[datetime] = None

    def access(self):
        """
        Record that this memory was retrieved or used.
        """

        self.access_count += 1
        self.last_accessed = datetime.now()

    def archive(self):
        """
        Move the memory to archived state.
        """

        self.status = MemoryStatus.ARCHIVED

    def forget(self):
        """
        Mark the memory as forgotten.
        """

        self.status = MemoryStatus.FORGOTTEN

    @property
    def timestamp(self) -> datetime:
        """
        Backward-compatible alias for created_at.

        Existing parts of the project that use
        'timestamp' can continue to work.
        """

        return self.created_at

    def to_dict(self) -> dict:
        """
        Convert memory into a serializable dictionary.
        """

        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "created_at": self.created_at.isoformat(),
            "event_time": (
                self.event_time.isoformat()
                if self.event_time
                else None
            ),
            "importance": self.importance,
            "confidence": self.confidence,
            "status": self.status.value,
            "source": self.source,
            "entities": self.entities,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": (
                self.last_accessed.isoformat()
                if self.last_accessed
                else None
            ),
        }


# For example:

# Current memory
# "I currently use Go for backend development."

# could have:

# importance = 0.9
# confidence = 0.95
# timestamp = 2026-08-14
# status = ACTIVE

# While an old low-value statement might eventually become:

# importance = 0.2
# status = ARCHIVED

# This is what allows us to implement controlled forgetting later.