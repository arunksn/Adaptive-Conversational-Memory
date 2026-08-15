from datetime import datetime

from src.models.memory import (
    Memory,
    MemoryType
)

from src.models.procedure import (
    ProcedureState
)

from src.retrieval.adaptive_retriever import (
    AdaptiveRetriever
)

from src.retrieval.hybrid_retriever import (
    HybridRetriever
)

from src.routing.memory_router import (
    MemoryRoute,
    MemoryRouter
)



class FakeVectorRetriever:

    def __init__(self):
        self.calls = []

    def search(
        self,
        query,
        top_k=5
    ):

        self.calls.append(
            query
        )

        memory = Memory(
            content="I prefer Python.",
            memory_type=MemoryType.SEMANTIC
        )

        return [
            {
                "memory_id": memory.memory_id,
                "score": 0.95,
                "memory": memory
            }
        ]


class FakeConflictVectorRetriever:

    def __init__(self):
        self.calls = []

    def search(
        self,
        query,
        top_k=5
    ):

        self.calls.append(
            query
        )

        older = Memory(
            content="I use TensorFlow.",
            memory_type=MemoryType.SEMANTIC,
            event_time=datetime(
                2026,
                1,
                1
            ),
            importance=0.8
        )

        newer = Memory(
            content="I switched to PyTorch.",
            memory_type=MemoryType.SEMANTIC,
            event_time=datetime(
                2026,
                8,
                1
            ),
            importance=0.9
        )

        older.confidence = 0.8
        newer.confidence = 0.9

        return [
            {
                "memory_id": older.memory_id,
                "score": 0.90,
                "memory": older
            },
            {
                "memory_id": newer.memory_id,
                "score": 0.95,
                "memory": newer
            }
        ]


class FakeTemporalRetriever:

    def __init__(self):
        self.calls = []

    def search(
        self,
        start_time,
        end_time
    ):

        self.calls.append(
            (
                start_time,
                end_time
            )
        )

        memory = Memory(
            content="I attended an AI workshop.",
            memory_type=MemoryType.EPISODIC,
            event_time=datetime(
                2026,
                8,
                10
            ),
            importance=0.8
        )

        return [memory]

    def recent(
        self,
        limit=10
    ):

        self.calls.append(
            ("recent", limit)
        )

        memory = Memory(
            content="Recent conversation event.",
            memory_type=MemoryType.EPISODIC,
            event_time=datetime(
                2026,
                8,
                20
            ),
            importance=0.7
        )

        return [memory]


class FakeGraphRetriever:

    def __init__(self):
        self.calls = []

    def get_next_states(
        self,
        procedure_id,
        state_id
    ):

        self.calls.append(
            (
                procedure_id,
                state_id
            )
        )

        state = ProcedureState(
            name="Deploy"
        )

        return [state]



def create_adaptive_retriever():

    vector = FakeVectorRetriever()

    temporal = FakeTemporalRetriever()

    graph = FakeGraphRetriever()

    hybrid = HybridRetriever(
        vector_retriever=vector,
        temporal_retriever=temporal,
        graph_retriever=graph
    )

    adaptive = AdaptiveRetriever(
        router=MemoryRouter(),
        hybrid_retriever=hybrid
    )

    return (
        adaptive,
        vector,
        temporal,
        graph
    )



def create_conflict_adaptive_retriever():

    vector = FakeConflictVectorRetriever()

    temporal = FakeTemporalRetriever()

    graph = FakeGraphRetriever()

    hybrid = HybridRetriever(
        vector_retriever=vector,
        temporal_retriever=temporal,
        graph_retriever=graph
    )

    adaptive = AdaptiveRetriever(
        router=MemoryRouter(),
        hybrid_retriever=hybrid
    )

    return (
        adaptive,
        vector,
        temporal,
        graph
    )


def test_semantic_query():

    (
        adaptive,
        vector,
        temporal,
        graph
    ) = create_adaptive_retriever()

    routing, results = adaptive.retrieve(
        "What programming language do I prefer?"
    )

    assert routing.primary_route == (
        MemoryRoute.SEMANTIC
    )

    assert len(results) == 1

    assert results[0].source == (
        MemoryRoute.SEMANTIC
    )

    assert len(vector.calls) == 1

    assert len(temporal.calls) == 0

    assert len(graph.calls) == 0



def test_episodic_query():

    (
        adaptive,
        vector,
        temporal,
        graph
    ) = create_adaptive_retriever()

    routing, results = adaptive.retrieve(
        "What did I tell you yesterday?"
    )

    assert routing.primary_route == (
        MemoryRoute.EPISODIC
    )

    assert len(results) == 1

    assert results[0].source == (
        MemoryRoute.EPISODIC
    )

    assert len(vector.calls) == 0

    assert len(temporal.calls) == 1

    assert temporal.calls[0][0] == (
        "recent"
    )

    assert len(graph.calls) == 0


def test_episodic_explicit_time_range():

    (
        adaptive,
        vector,
        temporal,
        graph
    ) = create_adaptive_retriever()

    start = datetime(
        2026,
        8,
        1
    )

    end = datetime(
        2026,
        8,
        15
    )

    routing, results = adaptive.retrieve(
        "What did I tell you last month?",
        start_time=start,
        end_time=end
    )

    assert routing.primary_route == (
        MemoryRoute.EPISODIC
    )

    assert len(results) == 1

    assert (
        temporal.calls[0]
        == (start, end)
    )



def test_procedural_query():

    (
        adaptive,
        vector,
        temporal,
        graph
    ) = create_adaptive_retriever()

    routing, results = adaptive.retrieve(
        "How do I deploy my application?",
        procedure_id="procedure-1",
        state_id="state-1"
    )

    assert routing.primary_route == (
        MemoryRoute.PROCEDURAL
    )

    assert len(results) == 1

    assert results[0].source == (
        MemoryRoute.PROCEDURAL
    )

    assert len(vector.calls) == 0

    assert len(temporal.calls) == 0

    assert len(graph.calls) == 1

    assert graph.calls[0] == (
        "procedure-1",
        "state-1"
    )


def test_procedural_query_without_context():

    (
        adaptive,
        vector,
        temporal,
        graph
    ) = create_adaptive_retriever()

    routing, results = adaptive.retrieve(
        "How do I deploy my application?"
    )

    assert routing.primary_route == (
        MemoryRoute.PROCEDURAL
    )

    assert results == []

    assert len(graph.calls) == 0



def test_multi_memory_query():

    (
        adaptive,
        vector,
        temporal,
        graph
    ) = create_adaptive_retriever()

    routing, results = adaptive.retrieve(
        "What did I previously do when deploying "
        "my project?",
        procedure_id="procedure-1",
        state_id="state-1"
    )

    assert (
        MemoryRoute.EPISODIC
        in routing.routes
    )

    assert (
        MemoryRoute.PROCEDURAL
        in routing.routes
    )

    assert len(results) == 2

    sources = {
        result.source
        for result in results
    }

    assert (
        MemoryRoute.EPISODIC
        in sources
    )

    assert (
        MemoryRoute.PROCEDURAL
        in sources
    )



def test_top_k():

    (
        adaptive,
        vector,
        temporal,
        graph
    ) = create_adaptive_retriever()

    routing, results = adaptive.retrieve(
        "What programming language do I prefer?",
        top_k=1
    )

    assert len(results) <= 1


def test_empty_query():

    (
        adaptive,
        *_,
    ) = create_adaptive_retriever()

    try:

        adaptive.retrieve("")

        assert False

    except ValueError:

        assert True

def test_conflict_is_detected_and_resolved():

    (
        adaptive,
        vector,
        temporal,
        graph
    ) = create_conflict_adaptive_retriever()

    routing, results = adaptive.retrieve(
        "What technology do I use?"
    )

    assert routing.primary_route == (
        MemoryRoute.SEMANTIC
    )

    # Two conflicting memories were retrieved,
    # but only the preferred/current memory remains
    # in the final result set.

    assert len(results) == 1

    result = results[0]

    assert result.item.content == (
        "I switched to PyTorch."
    )

    assert result.metadata[
        "conflict_resolved"
    ] is True

    assert (
        "conflict_reason"
        in result.metadata
    )

    assert (
        "conflict_confidence"
        in result.metadata
    )


def test_historical_memory_is_preserved_after_resolution():

    (
        adaptive,
        vector,
        temporal,
        graph
    ) = create_conflict_adaptive_retriever()

    routing, results = adaptive.retrieve(
        "What technology do I use?"
    )

    assert len(results) == 1

    result = results[0]

    assert (
        "historical_memories"
        in result.metadata
    )

    historical = (
        result.metadata[
            "historical_memories"
        ]
    )

    assert len(historical) == 1

    assert historical[0][
        "content"
    ] == "I use TensorFlow."


def test_non_conflicting_memory_is_unchanged():

    (
        adaptive,
        vector,
        temporal,
        graph
    ) = create_adaptive_retriever()

    routing, results = adaptive.retrieve(
        "What programming language do I prefer?"
    )

    assert len(results) == 1

    result = results[0]

    assert result.item.content == (
        "I prefer Python."
    )

    assert (
        "conflict_resolved"
        not in result.metadata
    )


def test_conflict_does_not_change_routing():

    (
        adaptive,
        vector,
        temporal,
        graph
    ) = create_conflict_adaptive_retriever()

    routing, results = adaptive.retrieve(
        "What technology do I use?"
    )

    assert routing.primary_route == (
        MemoryRoute.SEMANTIC
    )

    assert (
        MemoryRoute.SEMANTIC
        in routing.routes
    )