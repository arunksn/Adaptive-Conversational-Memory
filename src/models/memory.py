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

    timestamp: datetime = field(
        default_factory=datetime.now
    )

    importance: float = 0.5
    confidence: float = 1.0

    status: MemoryStatus = MemoryStatus.ACTIVE

    source: Optional[str] = None

    entities: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    access_count: int = 0

    last_accessed: Optional[datetime] = None

    def access(self):
        """
        Record that this memory was retrieved/used.
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

    def to_dict(self) -> dict:
        """
        Convert memory into a serializable dictionary.
        """
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "timestamp": self.timestamp.isoformat(),
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