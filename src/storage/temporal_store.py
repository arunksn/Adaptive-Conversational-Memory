import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models.memory import Memory, MemoryType


class TemporalStore:

    def __init__(
        self,
        database_path: str = "data/memories/episodic_memory.db"
    ):
        self.database_path = Path(database_path)

        # Create the parent directory if it does not exist.
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.connection = sqlite3.connect(
            self.database_path
        )

        # Return rows that can be accessed by column name.
        self.connection.row_factory = sqlite3.Row

        self._create_table()

    # DATABASE SETUP

    def _create_table(self):
        """
        Create the episodic memory table if it does not exist.
        """

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS episodic_memories (
                memory_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                event_time TEXT,
                importance REAL NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                source TEXT,
                entities TEXT,
                metadata TEXT,
                access_count INTEGER NOT NULL,
                last_accessed TEXT
            )
            """
        )

        self.connection.commit()

    # ADD

    def add_event(
        self,
        memory: Memory
    ):
        """
        Store an episodic memory.
        """

        if memory.memory_type != MemoryType.EPISODIC:
            raise ValueError(
                "TemporalStore only accepts episodic memories."
            )

        self.connection.execute(
            """
            INSERT INTO episodic_memories (
                memory_id,
                content,
                memory_type,
                created_at,
                event_time,
                importance,
                confidence,
                status,
                source,
                entities,
                metadata,
                access_count,
                last_accessed
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.memory_id,
                memory.content,
                memory.memory_type.value,
                memory.created_at.isoformat(),
                (
                    memory.event_time.isoformat()
                    if memory.event_time
                    else None
                ),
                memory.importance,
                memory.confidence,
                memory.status.value,
                memory.source,
                self._serialize_list(
                    memory.entities
                ),
                self._serialize_dict(
                    memory.metadata
                ),
                memory.access_count,
                (
                    memory.last_accessed.isoformat()
                    if memory.last_accessed
                    else None
                ),
            )
        )

        self.connection.commit()

    # GET BY ID

    def get_event(
        self,
        memory_id: str
    ) -> Optional[Memory]:
        """
        Retrieve an episodic memory by ID.
        """

        cursor = self.connection.execute(
            """
            SELECT *
            FROM episodic_memories
            WHERE memory_id = ?
            """,
            (memory_id,)
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_memory(row)

    # TIME RANGE QUERY

    def get_events_between(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> list[Memory]:
        """
        Retrieve episodic memories whose event_time
        falls within the specified time range.
        """

        cursor = self.connection.execute(
            """
            SELECT *
            FROM episodic_memories
            WHERE event_time IS NOT NULL
              AND event_time >= ?
              AND event_time <= ?
            ORDER BY event_time ASC
            """,
            (
                start_time.isoformat(),
                end_time.isoformat()
            )
        )

        rows = cursor.fetchall()

        return [
            self._row_to_memory(row)
            for row in rows
        ]

    # RECENT EVENTS

    def get_recent_events(
        self,
        limit: int = 10
    ) -> list[Memory]:
        """
        Retrieve the most recent episodic events.
        """

        cursor = self.connection.execute(
            """
            SELECT *
            FROM episodic_memories
            WHERE event_time IS NOT NULL
            ORDER BY event_time DESC
            LIMIT ?
            """,
            (limit,)
        )

        rows = cursor.fetchall()

        return [
            self._row_to_memory(row)
            for row in rows
        ]

    # UPDATE

    def update_event(
        self,
        memory: Memory
    ):
        """
        Update an existing episodic memory.
        """

        if memory.memory_type != MemoryType.EPISODIC:
            raise ValueError(
                "TemporalStore only accepts episodic memories."
            )

        cursor = self.connection.execute(
            """
            UPDATE episodic_memories
            SET
                content = ?,
                event_time = ?,
                importance = ?,
                confidence = ?,
                status = ?,
                source = ?,
                entities = ?,
                metadata = ?,
                access_count = ?,
                last_accessed = ?
            WHERE memory_id = ?
            """,
            (
                memory.content,
                (
                    memory.event_time.isoformat()
                    if memory.event_time
                    else None
                ),
                memory.importance,
                memory.confidence,
                memory.status.value,
                memory.source,
                self._serialize_list(
                    memory.entities
                ),
                self._serialize_dict(
                    memory.metadata
                ),
                memory.access_count,
                (
                    memory.last_accessed.isoformat()
                    if memory.last_accessed
                    else None
                ),
                memory.memory_id
            )
        )

        self.connection.commit()

        if cursor.rowcount == 0:
            raise ValueError(
                f"Memory not found: {memory.memory_id}"
            )

    # DELETE

    def delete_event(
        self,
        memory_id: str
    ):
        """
        Delete an episodic memory.
        """

        cursor = self.connection.execute(
            """
            DELETE FROM episodic_memories
            WHERE memory_id = ?
            """,
            (memory_id,)
        )

        self.connection.commit()

        if cursor.rowcount == 0:
            raise ValueError(
                f"Memory not found: {memory_id}"
            )

    # COUNT

    def count(self) -> int:
        """
        Return the number of episodic memories.
        """

        cursor = self.connection.execute(
            """
            SELECT COUNT(*)
            FROM episodic_memories
            """
        )

        return cursor.fetchone()[0]

    # CLOSE

    def close(self):
        """
        Close the SQLite connection.
        """

        if self.connection:
            self.connection.close()
            self.connection = None

    # INTERNAL CONVERSION

    def _row_to_memory(
        self,
        row: sqlite3.Row
    ) -> Memory:
        """
        Convert a SQLite row back into a Memory object.
        """

        event_time = (
            datetime.fromisoformat(
                row["event_time"]
            )
            if row["event_time"]
            else None
        )

        created_at = datetime.fromisoformat(
            row["created_at"]
        )

        last_accessed = (
            datetime.fromisoformat(
                row["last_accessed"]
            )
            if row["last_accessed"]
            else None
        )

        memory = Memory(
            content=row["content"],
            memory_type=MemoryType(
                row["memory_type"]
            ),
            memory_id=row["memory_id"],
            created_at=created_at,
            event_time=event_time,
            importance=row["importance"],
            confidence=row["confidence"],
            source=row["source"],
            entities=self._deserialize_list(
                row["entities"]
            ),
            metadata=self._deserialize_dict(
                row["metadata"]
            ),
            access_count=row["access_count"],
            last_accessed=last_accessed
        )

        return memory

    # SERIALIZATION HELPERS

    @staticmethod
    def _serialize_list(
        values: list[str]
    ) -> str:
        """
        Store a list as JSON.
        """

        import json

        return json.dumps(values)

    @staticmethod
    def _deserialize_list(
        value: str
    ) -> list[str]:
        """
        Load a JSON list.
        """

        import json

        return json.loads(value)

    @staticmethod
    def _serialize_dict(
        values: dict
    ) -> str:
        """
        Store a dictionary as JSON.
        """

        import json

        return json.dumps(values)

    @staticmethod
    def _deserialize_dict(
        value: str
    ) -> dict:
        """
        Load a JSON dictionary.
        """

        import json

        return json.loads(value)