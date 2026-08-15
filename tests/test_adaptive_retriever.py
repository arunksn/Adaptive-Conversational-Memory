from datetime import datetime

from src.models.memory import (
    Memory,
    MemoryType
)

from src.models.procedure import (
    Procedure,
    ProcedureState,
    ProcedureTransition
)

from src.retrieval.adaptive_retriever import (
    AdaptiveRetriever
)

from src.retrieval.graph_retriever import (
    GraphRetriever
)

from src.retrieval.hybrid_retriever import (
    HybridRetriever
)

from src.retrieval.temporal_retriever import (
    TemporalRetriever
)

from src.retrieval.vector_retriever import (
    VectorRetriever
)

from src.routing.memory_router import (
    MemoryRoute,
    MemoryRouter
)

# FAKE SEMANTIC RETRIEVER


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
            event_time=datetime(2026, 8, 10),
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
            event_time=datetime(2026, 8, 20),
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

# FACTORY

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

# MULTI-MEMORY ROUTING

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

# TOP-K

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