from dataclasses import dataclass
from datetime import datetime

from src.classification.importance_scorer import (
    ImportanceScorer
)
from src.classification.memory_classifier import (
    MemoryClassifier
)
from src.llm.context_builder import (
    ContextBuilder,
    MemoryContext
)
from src.llm.memory_extractor import (
    MemoryCandidate,
    MemoryExtractor
)
from src.llm.response_generator import (
    GeneratedResponse,
    ResponseGenerator
)
from src.models.memory import (
    Memory
)
from src.retrieval.adaptive_retriever import (
    AdaptiveRetriever
)
from src.retrieval.hybrid_retriever import (
    RetrievalResult
)
from src.routing.memory_router import (
    RoutingResult
)


@dataclass
class MemoryIngestionResult:
    """
    Result of processing conversational text into
    structured memories.

    candidates:
        Raw candidates produced by MemoryExtractor.

    memories:
        Final Memory objects after classification and
        importance scoring.
    """

    candidates: list[MemoryCandidate]

    memories: list[Memory]


@dataclass
class MemoryQueryResult:
    """
    Result of the retrieval/context/generation side of
    the conversational memory pipeline.
    """

    routing: RoutingResult

    retrieved_memories: list[RetrievalResult]

    context: MemoryContext

    response: GeneratedResponse


@dataclass
class ConversationResult:
    """
    Complete result of processing one conversational
    interaction.

    The user message can produce new memories while the
    query side independently retrieves existing memories
    and generates the final response.
    """

    ingestion: MemoryIngestionResult

    query: MemoryQueryResult


class MemoryPipeline:

    def __init__(
        self,
        classifier: MemoryClassifier | None = None,
        importance_scorer: ImportanceScorer | None = None,
        memory_extractor: MemoryExtractor | None = None,
        adaptive_retriever: AdaptiveRetriever | None = None,
        context_builder: ContextBuilder | None = None,
        response_generator: ResponseGenerator | None = None
    ):
        """
        Coordinate the main conversational memory
        pipeline.

        Components are injected so that the pipeline can
        be tested independently and can support different
        LLM/retrieval implementations.

        MemoryExtractor, AdaptiveRetriever, ContextBuilder,
        and ResponseGenerator are optional because the
        original deterministic process(text) API is kept
        for backward compatibility.
        """

        self.classifier = (
            classifier
            if classifier is not None
            else MemoryClassifier()
        )

        self.importance_scorer = (
            importance_scorer
            if importance_scorer is not None
            else ImportanceScorer()
        )

        self.memory_extractor = (
            memory_extractor
        )

        self.adaptive_retriever = (
            adaptive_retriever
        )

        self.context_builder = (
            context_builder
            if context_builder is not None
            else ContextBuilder()
        )

        self.response_generator = (
            response_generator
        )

    # BASIC MEMORY PROCESSING

    def process(
        self,
        text: str
    ) -> Memory:
        """
        Convert one already-extracted memory statement
        into a Memory object.

        This method preserves the original MemoryPipeline
        API used by earlier project stages.
        """

        if not text.strip():
            raise ValueError(
                "text cannot be empty"
            )

        memory_type = (
            self.classifier.classify(
                text
            )
        )

        importance = (
            self.importance_scorer.score(
                text
            )
        )

        return Memory(
            content=text.strip(),
            memory_type=memory_type,
            importance=importance,
            confidence=1.0,
            source="conversation"
        )

    # CANDIDATE PROCESSING

    def process_candidate(
        self,
        candidate: MemoryCandidate
    ) -> Memory:
        """
        Convert an extracted MemoryCandidate into the
        final internal Memory representation.

        Classification and final importance scoring remain
        the responsibility of the deterministic pipeline.

        The LLM importance hint is retained as metadata
        rather than silently replacing the deterministic
        importance score.
        """

        if candidate is None:
            raise ValueError(
                "candidate cannot be None"
            )

        if not candidate.content.strip():
            raise ValueError(
                "candidate content cannot be empty"
            )

        memory = self.process(
            candidate.content
        )

        memory.event_time = (
            candidate.event_time
        )

        memory.metadata.update(
            candidate.metadata or {}
        )

        if (
            candidate.importance_hint
            is not None
        ):
            memory.metadata[
                "importance_hint"
            ] = candidate.importance_hint

        return memory

    # MEMORY EXTRACTION / INGESTION

    def ingest(
        self,
        conversation: str
    ) -> MemoryIngestionResult:
        """
        Extract useful information from conversational
        text and convert the extracted candidates into
        Memory objects.

        This stage does not persist the memories yet.
        Storage synchronization is handled separately so
        the semantic, episodic, and procedural stores can
        keep their existing responsibilities.
        """

        if not conversation.strip():
            raise ValueError(
                "conversation cannot be empty"
            )

        extractor = (
            self._require_memory_extractor()
        )

        candidates = extractor.extract(
            conversation
        )

        memories = [
            self.process_candidate(
                candidate
            )
            for candidate in candidates
        ]

        return MemoryIngestionResult(
            candidates=candidates,
            memories=memories
        )

    # RETRIEVAL

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        procedure_id: str | None = None,
        state_id: str | None = None
    ) -> tuple[
        RoutingResult,
        list[RetrievalResult]
    ]:
        """
        Route a query and retrieve relevant memories
        through AdaptiveRetriever.
        """

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        retriever = (
            self._require_adaptive_retriever()
        )

        return retriever.retrieve(
            query=query,
            top_k=top_k,
            start_time=start_time,
            end_time=end_time,
            procedure_id=procedure_id,
            state_id=state_id
        )

    # CONTEXT CONSTRUCTION

    def build_context(
        self,
        query: str,
        results: list[RetrievalResult],
        max_memories: int | None = None
    ) -> MemoryContext:
        """
        Convert retrieved memories into the structured
        context supplied to the response model.
        """

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        return self.context_builder.build(
            results=results,
            query=query,
            max_memories=max_memories
        )

    # RESPONSE GENERATION

    def generate_response(
        self,
        query: str,
        context: MemoryContext
    ) -> GeneratedResponse:
        """
        Generate the final conversational response from
        an already-built MemoryContext.
        """

        generator = (
            self._require_response_generator()
        )

        return generator.generate(
            query=query,
            context=context
        )

    # COMPLETE QUERY PIPELINE

    def answer(
        self,
        query: str,
        top_k: int = 5,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        procedure_id: str | None = None,
        state_id: str | None = None,
        max_context_memories: int | None = None
    ) -> MemoryQueryResult:
        """
        Execute the complete query side of the memory
        architecture:

        Query
            ->
        Memory routing
            ->
        Adaptive / hybrid retrieval
            ->
        Context construction
            ->
        Response generation
        """

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        routing, results = self.retrieve(
            query=query,
            top_k=top_k,
            start_time=start_time,
            end_time=end_time,
            procedure_id=procedure_id,
            state_id=state_id
        )

        context = self.build_context(
            query=query,
            results=results,
            max_memories=max_context_memories
        )

        response = self.generate_response(
            query=query,
            context=context
        )

        return MemoryQueryResult(
            routing=routing,
            retrieved_memories=results,
            context=context,
            response=response
        )

    # COMPLETE CONVERSATIONAL TURN

    def run(
        self,
        conversation: str,
        query: str | None = None,
        top_k: int = 5,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        procedure_id: str | None = None,
        state_id: str | None = None,
        max_context_memories: int | None = None
    ) -> ConversationResult:
        """
        Execute one complete conversational-memory turn.

        1. Extract new memory candidates.
        2. Classify and score those candidates.
        3. Retrieve existing relevant memories.
        4. Build memory context.
        5. Generate the conversational response.

        New memories are returned to the caller for the
        storage/lifecycle layer.

        They are intentionally not inserted into storage
        inside this method yet because each memory type
        has a different persistence representation.
        """

        if not conversation.strip():
            raise ValueError(
                "conversation cannot be empty"
            )

        effective_query = (
            query
            if query is not None
            else conversation
        )

        if not effective_query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        ingestion_result = self.ingest(
            conversation
        )

        query_result = self.answer(
            query=effective_query,
            top_k=top_k,
            start_time=start_time,
            end_time=end_time,
            procedure_id=procedure_id,
            state_id=state_id,
            max_context_memories=(
                max_context_memories
            )
        )

        return ConversationResult(
            ingestion=ingestion_result,
            query=query_result
        )

    # COMPONENT VALIDATION

    def _require_memory_extractor(
        self
    ) -> MemoryExtractor:
        """
        Return the configured MemoryExtractor or raise a
        clear configuration error.
        """

        if self.memory_extractor is None:
            raise RuntimeError(
                "MemoryExtractor is not configured."
            )

        return self.memory_extractor

    def _require_adaptive_retriever(
        self
    ) -> AdaptiveRetriever:
        """
        Return the configured AdaptiveRetriever or raise a
        clear configuration error.
        """

        if self.adaptive_retriever is None:
            raise RuntimeError(
                "AdaptiveRetriever is not configured."
            )

        return self.adaptive_retriever

    def _require_response_generator(
        self
    ) -> ResponseGenerator:
        """
        Return the configured ResponseGenerator or raise a
        clear configuration error.
        """

        if self.response_generator is None:
            raise RuntimeError(
                "ResponseGenerator is not configured."
            )

        return self.response_generator