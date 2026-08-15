from datetime import datetime

from src.models.memory import (
    Memory,
    MemoryType
)

from src.models.procedure import (
    ProcedureState
)

from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    RetrievalResult
)

from src.routing.memory_router import (
    MemoryRoute
)


def test_retrieval_result_creation():

    memory = Memory(
        content="I prefer Python.",
        memory_type=MemoryType.SEMANTIC
    )

    result = RetrievalResult(
        source=MemoryRoute.SEMANTIC,
        item=memory,
        score=0.9,
        memory_id=memory.memory_id
    )

    assert result.source == (
        MemoryRoute.SEMANTIC
    )

    assert result.item == memory

    assert result.score == 0.9


def test_semantic_retrieval_normalization():

    class FakeVectorRetriever:

        def search(
            self,
            query,
            top_k=5
        ):
            return [
                {
                    "memory_id": "memory-1",
                    "score": 0.9,
                    "memory": "Python memory"
                },
                {
                    "memory_id": "memory-2",
                    "score": 0.7,
                    "memory": "Go memory"
                }
            ]

    retriever = HybridRetriever(
        vector_retriever=FakeVectorRetriever()
    )

    results = retriever.retrieve_semantic(
        "programming language"
    )

    assert len(results) == 2

    assert results[0].source == (
        MemoryRoute.SEMANTIC
    )

    assert results[0].score == 0.9


def test_temporal_retrieval_normalization():

    memory = Memory(
        content="I attended an AI workshop.",
        memory_type=MemoryType.EPISODIC,
        event_time=datetime(2026, 8, 10),
        importance=0.8
    )

    class FakeTemporalRetriever:

        def search(
            self,
            start_time,
            end_time
        ):
            return [memory]

    retriever = HybridRetriever(
        temporal_retriever=FakeTemporalRetriever()
    )

    results = retriever.retrieve_temporal(
        datetime(2026, 8, 1),
        datetime(2026, 8, 15)
    )

    assert len(results) == 1

    assert results[0].source == (
        MemoryRoute.EPISODIC
    )

    assert results[0].memory_id == (
        memory.memory_id
    )

    assert results[0].score == 0.8


def test_procedural_retrieval_normalization():

    state = ProcedureState(
        name="Deploy",
        is_terminal=False
    )

    class FakeGraphRetriever:

        def get_next_states(
            self,
            procedure_id,
            state_id
        ):
            return [state]

    retriever = HybridRetriever(
        graph_retriever=FakeGraphRetriever()
    )

    results = retriever.retrieve_procedural(
        "procedure-1",
        "state-1"
    )

    assert len(results) == 1

    assert results[0].source == (
        MemoryRoute.PROCEDURAL
    )

    assert results[0].score == 0.8


def test_score_normalization():

    results = [
        RetrievalResult(
            source=MemoryRoute.SEMANTIC,
            item="A",
            score=0.2
        ),
        RetrievalResult(
            source=MemoryRoute.EPISODIC,
            item="B",
            score=0.5
        ),
        RetrievalResult(
            source=MemoryRoute.PROCEDURAL,
            item="C",
            score=1.0
        )
    ]

    retriever = HybridRetriever()

    fused = retriever.fuse(
        results
    )

    assert fused[0].item == "C"
    assert fused[0].score == 1.0

    assert fused[-1].item == "A"
    assert fused[-1].score == 0.0


def test_fusion_orders_results():

    results = [
        RetrievalResult(
            source=MemoryRoute.SEMANTIC,
            item="Low",
            score=0.2
        ),
        RetrievalResult(
            source=MemoryRoute.EPISODIC,
            item="High",
            score=0.9
        ),
        RetrievalResult(
            source=MemoryRoute.PROCEDURAL,
            item="Medium",
            score=0.5
        )
    ]

    retriever = HybridRetriever()

    fused = retriever.fuse(
        results
    )

    assert [
        result.item
        for result in fused
    ] == [
        "High",
        "Medium",
        "Low"
    ]


def test_deduplication():

    results = [
        RetrievalResult(
            source=MemoryRoute.SEMANTIC,
            item="Python",
            score=0.9,
            memory_id="memory-1"
        ),
        RetrievalResult(
            source=MemoryRoute.SEMANTIC,
            item="Python duplicate",
            score=0.8,
            memory_id="memory-1"
        ),
        RetrievalResult(
            source=MemoryRoute.EPISODIC,
            item="Python",
            score=0.7,
            memory_id="memory-2"
        )
    ]

    retriever = HybridRetriever()

    unique = retriever.deduplicate(
        results
    )

    assert len(unique) == 2

    assert unique[0].item == "Python"

    assert unique[1].item == "Python"


def test_deduplication_without_memory_id():

    results = [
        RetrievalResult(
            source=MemoryRoute.PROCEDURAL,
            item="Deploy",
            score=0.9
        ),
        RetrievalResult(
            source=MemoryRoute.PROCEDURAL,
            item="Deploy",
            score=0.8
        )
    ]

    retriever = HybridRetriever()

    unique = retriever.deduplicate(
        results
    )

    assert len(unique) == 1


def test_combine():

    results = [
        RetrievalResult(
            source=MemoryRoute.SEMANTIC,
            item="A",
            score=0.9,
            memory_id="a"
        ),
        RetrievalResult(
            source=MemoryRoute.EPISODIC,
            item="B",
            score=0.7,
            memory_id="b"
        ),
        RetrievalResult(
            source=MemoryRoute.PROCEDURAL,
            item="C",
            score=0.5
        )
    ]

    retriever = HybridRetriever()

    combined = retriever.combine(
        results,
        top_k=2
    )

    assert len(combined) == 2

    assert combined[0].item == "A"
    assert combined[1].item == "B"


def test_empty_fusion():

    retriever = HybridRetriever()

    assert retriever.fuse([]) == []


def test_empty_deduplication():

    retriever = HybridRetriever()

    assert retriever.deduplicate([]) == []


def test_missing_retriever_returns_empty():

    retriever = HybridRetriever()

    assert (
        retriever.retrieve_semantic(
            "Python"
        )
        == []
    )

    assert (
        retriever.retrieve_temporal(
            datetime(2026, 8, 1),
            datetime(2026, 8, 10)
        )
        == []
    )

    assert (
        retriever.retrieve_procedural(
            "procedure",
            "state"
        )
        == []
    )