from datetime import datetime

from src.storage.temporal_store import TemporalStore


class TemporalRetriever:

    def __init__(
        self,
        database_path: str = (
            "data/memories/episodic_memory.db"
        )
    ):
        self.temporal_store = TemporalStore(
            database_path=database_path
        )

    def search(
        self,
        start_time: datetime,
        end_time: datetime
    ):
        """
        Retrieve episodic memories within
        a specific time range.
        """

        return self.temporal_store.get_events_between(
            start_time=start_time,
            end_time=end_time
        )

    def recent(
        self,
        limit: int = 10
    ):
        """
        Retrieve the most recent episodic events.
        """

        return self.temporal_store.get_recent_events(
            limit=limit
        )

    def add_memory(self, memory):
        """
        Add an episodic memory.
        """

        self.temporal_store.add_event(
            memory
        )

    def update_memory(self, memory):
        """
        Update an episodic memory.
        """

        self.temporal_store.update_event(
            memory
        )

    def delete_memory(
        self,
        memory_id: str
    ):
        """
        Delete an episodic memory.
        """

        self.temporal_store.delete_event(
            memory_id
        )

    def count(self) -> int:
        """
        Return number of episodic memories.
        """

        return self.temporal_store.count()

    def close(self):
        """
        Close the underlying SQLite connection.
        """

        self.temporal_store.close()