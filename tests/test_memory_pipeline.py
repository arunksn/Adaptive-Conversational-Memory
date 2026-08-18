from datetime import datetime

import pytest

from src.llm.context_builder import (
    ContextBuilder
)
from src.llm.llm_client import (
    LLMResponse
)
from src.llm.memory_extractor import (
    MemoryCandidate
)
from src.llm.response_generator import (
    ResponseGenerator
)
from src.models.memory import (
    Memory,
    MemoryType
)
from src.pipeline.memory_pipeline import (
    MemoryPipeline
)
from src.retrieval.hybrid_retriever import (
    RetrievalResult
)
from src.routing.memory_router import (
    MemoryRoute,
    RoutingResult
)


class FakeMemoryExtractor:

    def __init__(
        self,
        candidates=None
    ):
        self.candidates = (
            candidates
            if candidates is not None
            else []
        )

        self.calls = []

    def extract(
        self,
        conversation
    ):
        self.calls.append(
            conversation
        )

        return self.candidates


class FakeAdaptiveRetriever:

    def __init__(
        self,
        results=None,
        routes=None
    ):
        self.results = (
            results
            if results is not None
            else []
        )

        self.routes = (
            routes
            if routes is not None
            else [
                MemoryRoute.SEMANTIC
            ]
        )

        self.calls = []

    def retrieve(
        self,
        query,
        top_k=5,
        start_time=None,
        end_time=None,
        procedure_id=None,
        state_id=None
    ):
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "start_time": start_time,
            "end_time": end_time,
            "procedure_id": procedure_id,
            "state_id": state_id
        })

        routing = RoutingResult(
            routes=self.routes,
            confidence=1.0,
            reason="Fake routing result."
        )

        return (
            routing,
            self.results
        )


class FakeResponseLLM:

    def __init__(
        self,
        text="Generated answer."
    ):
        self.text = text

        self.calls = []

    def generate(
        self,
        prompt,
        system_prompt=None
    ):
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt
        })

        return LLMResponse(
            text=self.text,
            model="fake-model"
        )


def create_semantic_result():

    memory = Memory(
        content="I prefer Python.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.8
    )

    return RetrievalResult(
        source=MemoryRoute.SEMANTIC,
        item=memory,
        score=0.95,
        memory_id=memory.memory_id
    )


def create_pipeline(
    candidates=None,
    retrieval_results=None
):

    extractor = FakeMemoryExtractor(
        candidates=candidates
    )

    retriever = FakeAdaptiveRetriever(
        results=retrieval_results
    )

    llm = FakeResponseLLM()

    generator = ResponseGenerator(
        llm_client=llm
    )

    pipeline = MemoryPipeline(
        memory_extractor=extractor,
        adaptive_retriever=retriever,
        context_builder=ContextBuilder(),
        response_generator=generator
    )

    return (
        pipeline,
        extractor,
        retriever,
        llm
    )


# BACKWARD COMPATIBILITY


def test_process_creates_semantic_memory():

    pipeline = MemoryPipeline()

    memory = pipeline.process(
        "I prefer Python."
    )

    assert memory.content == (
        "I prefer Python."
    )

    assert memory.memory_type == (
        MemoryType.SEMANTIC
    )

    assert memory.importance > 0.5

    assert memory.source == (
        "conversation"
    )


def test_process_creates_episodic_memory():

    pipeline = MemoryPipeline()

    memory = pipeline.process(
        "Yesterday I attended an AI workshop."
    )

    assert memory.memory_type == (
        MemoryType.EPISODIC
    )


def test_process_creates_procedural_memory():

    pipeline = MemoryPipeline()

    memory = pipeline.process(
        "How to install and configure Docker."
    )

    assert memory.memory_type == (
        MemoryType.PROCEDURAL
    )


def test_process_rejects_empty_text():

    pipeline = MemoryPipeline()

    with pytest.raises(
        ValueError
    ):
        pipeline.process("")


# CANDIDATE PROCESSING


def test_process_candidate():

    pipeline = MemoryPipeline()

    event_time = datetime(
        2026,
        8,
        10
    )

    candidate = MemoryCandidate(
        content=(
            "Yesterday I attended "
            "an AI workshop."
        ),
        event_time=event_time,
        importance_hint=0.9,
        metadata={
            "topic": "AI"
        }
    )

    memory = pipeline.process_candidate(
        candidate
    )

    assert memory.memory_type == (
        MemoryType.EPISODIC
    )

    assert memory.event_time == (
        event_time
    )

    assert memory.metadata[
        "topic"
    ] == "AI"

    assert memory.metadata[
        "importance_hint"
    ] == 0.9


def test_process_candidate_uses_deterministic_importance():

    pipeline = MemoryPipeline()

    candidate = MemoryCandidate(
        content="I prefer Python.",
        importance_hint=0.1
    )

    memory = pipeline.process_candidate(
        candidate
    )

    assert memory.importance > 0.5

    assert memory.metadata[
        "importance_hint"
    ] == 0.1


# INGESTION


def test_ingest_extracts_and_processes_memories():

    candidates = [
        MemoryCandidate(
            content="I prefer Python."
        ),
        MemoryCandidate(
            content=(
                "Yesterday I attended "
                "an AI workshop."
            ),
            event_time=datetime(
                2026,
                8,
                10
            )
        )
    ]

    (
        pipeline,
        extractor,
        _,
        _
    ) = create_pipeline(
        candidates=candidates
    )

    result = pipeline.ingest(
        "Conversation text"
    )

    assert len(
        result.candidates
    ) == 2

    assert len(
        result.memories
    ) == 2

    assert result.memories[
        0
    ].memory_type == (
        MemoryType.SEMANTIC
    )

    assert result.memories[
        1
    ].memory_type == (
        MemoryType.EPISODIC
    )

    assert extractor.calls == [
        "Conversation text"
    ]


def test_ingest_handles_no_memories():

    (
        pipeline,
        *_,
    ) = create_pipeline(
        candidates=[]
    )

    result = pipeline.ingest(
        "Hello there."
    )

    assert result.candidates == []

    assert result.memories == []


def test_ingest_requires_extractor():

    pipeline = MemoryPipeline()

    with pytest.raises(
        RuntimeError
    ):
        pipeline.ingest(
            "I prefer Python."
        )


# RETRIEVAL


def test_retrieve_uses_adaptive_retriever():

    retrieval_result = (
        create_semantic_result()
    )

    (
        pipeline,
        _,
        retriever,
        _
    ) = create_pipeline(
        retrieval_results=[
            retrieval_result
        ]
    )

    routing, results = pipeline.retrieve(
        query=(
            "What programming language "
            "do I prefer?"
        ),
        top_k=3
    )

    assert routing.primary_route == (
        MemoryRoute.SEMANTIC
    )

    assert results == [
        retrieval_result
    ]

    assert retriever.calls[
        0
    ]["top_k"] == 3


def test_retrieve_forwards_temporal_parameters():

    (
        pipeline,
        _,
        retriever,
        _
    ) = create_pipeline()

    start = datetime(
        2026,
        8,
        1
    )

    end = datetime(
        2026,
        8,
        18
    )

    pipeline.retrieve(
        query="What happened recently?",
        start_time=start,
        end_time=end
    )

    call = retriever.calls[0]

    assert call[
        "start_time"
    ] == start

    assert call[
        "end_time"
    ] == end


# CONTEXT


def test_build_context():

    result = (
        create_semantic_result()
    )

    pipeline = MemoryPipeline()

    context = pipeline.build_context(
        query=(
            "What language do I prefer?"
        ),
        results=[result]
    )

    assert context.memory_count == 1

    assert (
        "I prefer Python."
        in context.text
    )

    assert (
        "[Semantic Memory]"
        in context.text
    )


# RESPONSE


def test_generate_response():

    result = (
        create_semantic_result()
    )

    (
        pipeline,
        _,
        _,
        llm
    ) = create_pipeline()

    context = pipeline.build_context(
        query=(
            "What language do I prefer?"
        ),
        results=[result]
    )

    response = (
        pipeline.generate_response(
            query=(
                "What language do I prefer?"
            ),
            context=context
        )
    )

    assert response.text == (
        "Generated answer."
    )

    assert len(
        llm.calls
    ) == 1


# COMPLETE QUERY PIPELINE


def test_answer_runs_complete_query_pipeline():

    retrieval_result = (
        create_semantic_result()
    )

    (
        pipeline,
        _,
        retriever,
        llm
    ) = create_pipeline(
        retrieval_results=[
            retrieval_result
        ]
    )

    result = pipeline.answer(
        "What programming language do I prefer?"
    )

    assert result.routing.primary_route == (
        MemoryRoute.SEMANTIC
    )

    assert len(
        result.retrieved_memories
    ) == 1

    assert result.context.memory_count == 1

    assert (
        "I prefer Python."
        in result.context.text
    )

    assert result.response.text == (
        "Generated answer."
    )

    assert len(
        retriever.calls
    ) == 1

    assert len(
        llm.calls
    ) == 1


def test_answer_with_no_retrieved_memories():

    (
        pipeline,
        *_,
    ) = create_pipeline(
        retrieval_results=[]
    )

    result = pipeline.answer(
        "What do you remember?"
    )

    assert (
        result.retrieved_memories
        == []
    )

    assert result.context.memory_count == 0

    assert result.context.text == (
        "No relevant memories were found."
    )


# COMPLETE CONVERSATIONAL TURN


def test_run_executes_ingestion_and_query_pipeline():

    candidate = MemoryCandidate(
        content="I prefer Python."
    )

    retrieval_result = (
        create_semantic_result()
    )

    (
        pipeline,
        extractor,
        retriever,
        llm
    ) = create_pipeline(
        candidates=[candidate],
        retrieval_results=[
            retrieval_result
        ]
    )

    result = pipeline.run(
        conversation=(
            "User: I prefer Python."
        ),
        query=(
            "What programming language "
            "do I prefer?"
        )
    )

    assert len(
        result.ingestion.memories
    ) == 1

    assert (
        result.ingestion.memories[
            0
        ].memory_type
        == MemoryType.SEMANTIC
    )

    assert (
        result.query.routing.primary_route
        == MemoryRoute.SEMANTIC
    )

    assert (
        result.query.context.memory_count
        == 1
    )

    assert result.query.response.text == (
        "Generated answer."
    )

    assert len(
        extractor.calls
    ) == 1

    assert len(
        retriever.calls
    ) == 1

    assert len(
        llm.calls
    ) == 1


def test_run_uses_conversation_as_default_query():

    (
        pipeline,
        _,
        retriever,
        _
    ) = create_pipeline(
        candidates=[]
    )

    pipeline.run(
        conversation=(
            "What do you remember about me?"
        )
    )

    assert (
        retriever.calls[
            0
        ]["query"]
        == "What do you remember about me?"
    )


# VALIDATION


def test_empty_conversation_rejected():

    (
        pipeline,
        *_,
    ) = create_pipeline()

    with pytest.raises(
        ValueError
    ):
        pipeline.run(
            conversation=""
        )


def test_answer_requires_retriever():

    pipeline = MemoryPipeline(
        response_generator=ResponseGenerator(
            llm_client=FakeResponseLLM()
        )
    )

    with pytest.raises(
        RuntimeError
    ):
        pipeline.answer(
            "What do you remember?"
        )


def test_answer_requires_response_generator():

    pipeline = MemoryPipeline(
        adaptive_retriever=(
            FakeAdaptiveRetriever()
        )
    )

    with pytest.raises(
        RuntimeError
    ):
        pipeline.answer(
            "What do you remember?"
        )