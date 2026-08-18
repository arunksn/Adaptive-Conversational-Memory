"""
Vector Memory Experiment
========================

A vector-based conversational memory baseline.

This experiment represents a system that stores memories with
embeddings and retrieves them using vector similarity.

It intentionally does NOT use:

- graph retrieval
- adaptive routing
- temporal reasoning
- conflict resolution
- memory consolidation
- controlled forgetting

The purpose is to provide a stronger baseline than simple RAG
while remaining substantially simpler than the final adaptive
memory architecture.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class VectorMemoryResult:
    """
    Result produced by one vector-memory query.
    """

    query: str

    retrieved_ids: list[str]

    results: list[Any]

    context: str


class VectorMemory:
    """
    Vector-memory experiment wrapper.

    The vector store is responsible for similarity search.

    The embedder is responsible for converting text into vectors.

    The experiment itself only coordinates these components.
    """

    def __init__(
        self,
        vector_store,
        embedder,
        context_builder=None,
    ):
        if vector_store is None:
            raise ValueError(
                "vector_store cannot be None"
            )

        if embedder is None:
            raise ValueError(
                "embedder cannot be None"
            )

        self.vector_store = vector_store
        self.embedder = embedder
        self.context_builder = context_builder


    def add_memory(
        self,
        memory,
    ):
        """
        Embed and store a memory.
        """

        if memory is None:
            raise ValueError(
                "memory cannot be None"
            )

        embedding = self._embed(
            memory.content
        )

        self.vector_store.add(
            memory=memory,
            embedding=embedding,
        )


    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> list[Any]:
        """
        Retrieve the top-k memories using vector similarity.
        """

        if not query or not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0"
            )

        query_embedding = self._embed(
            query
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
        )

        if results is None:
            return []

        return list(results)[:k]


    def build_context(
        self,
        results: list[Any],
        query: str | None = None,
    ):
        """
        Build context from retrieved vector memories.

        A project ContextBuilder can be supplied. Otherwise,
        a deterministic text context is created locally.
        """

        if self.context_builder is not None:

            return self.context_builder.build(
                results=results,
                query=query,
            )

        return self._simple_context(
            results
        )


    def run(
        self,
        query: str,
        k: int = 5,
    ) -> VectorMemoryResult:
        """
        Execute vector-memory retrieval and context construction.
        """

        results = self.retrieve(
            query=query,
            k=k,
        )

        context = self.build_context(
            results=results,
            query=query,
        )

        return VectorMemoryResult(
            query=query,
            retrieved_ids=self.extract_memory_ids(
                results
            ),
            results=results,
            context=self._context_text(
                context
            ),
        )


    @staticmethod
    def extract_memory_ids(
        results: list[Any],
    ) -> list[str]:
        """
        Extract memory IDs from vector-search results.

        Supports:

        1. Direct memory_id
        2. Nested memory object
        3. Serialized memory dictionary
        4. Metadata memory_id
        """

        memory_ids = []

        for result in results:

            memory_id = getattr(
                result,
                "memory_id",
                None,
            )

            if memory_id is None:

                if isinstance(
                    result,
                    dict,
                ):
                    memory_id = result.get(
                        "memory_id"
                    )

            if memory_id is None:

                item = getattr(
                    result,
                    "item",
                    None,
                )

                if item is not None:

                    memory_id = getattr(
                        item,
                        "memory_id",
                        None,
                    )

                    if (
                        memory_id is None
                        and isinstance(item, dict)
                    ):
                        memory_id = item.get(
                            "memory_id"
                        )

            if memory_id is None:

                memory = getattr(
                    result,
                    "memory",
                    None,
                )

                if memory is not None:

                    memory_id = getattr(
                        memory,
                        "memory_id",
                        None,
                    )

                    if (
                        memory_id is None
                        and isinstance(memory, dict)
                    ):
                        memory_id = memory.get(
                            "memory_id"
                        )

            if memory_id is None:

                metadata = getattr(
                    result,
                    "metadata",
                    None,
                )

                if isinstance(
                    metadata,
                    dict,
                ):

                    memory_id = metadata.get(
                        "memory_id"
                    )

            if memory_id is not None:

                memory_ids.append(
                    str(memory_id)
                )

        return memory_ids


    def _embed(
        self,
        text: str,
    ):
        """
        Generate an embedding using the supplied embedder.

        Supports common embedding interfaces used by the project.
        """

        if hasattr(
            self.embedder,
            "embed",
        ):

            return self.embedder.embed(
                text
            )

        if hasattr(
            self.embedder,
            "encode",
        ):

            return self.embedder.encode(
                text
            )

        if callable(
            self.embedder
        ):

            return self.embedder(
                text
            )

        raise TypeError(
            "embedder must provide an "
            "'embed', 'encode', or callable interface."
        )


    @staticmethod
    def _simple_context(
        results: list[Any],
    ) -> str:
        """
        Create deterministic context without a ContextBuilder.
        """

        if not results:
            return (
                "No relevant memories were found."
            )

        lines = []

        for result in results:

            content = None

            if isinstance(
                result,
                dict,
            ):

                memory = result.get(
                    "memory"
                )

                if isinstance(
                    memory,
                    dict,
                ):
                    content = memory.get(
                        "content"
                    )

                if content is None:
                    content = result.get(
                        "content"
                    )

            if content is None:

                item = getattr(
                    result,
                    "item",
                    None,
                )

                if item is not None:
                    content = getattr(
                        item,
                        "content",
                        None
                    )

                    if (
                        content is None
                        and isinstance(item, dict)
                    ):
                        content = item.get(
                            "content"
                        )

            if content is None:

                memory = getattr(
                    result,
                    "memory",
                    None,
                )

                if memory is not None:
                    content = getattr(
                        memory,
                        "content",
                        None
                    )

                    if (
                        content is None
                        and isinstance(memory, dict)
                    ):
                        content = memory.get(
                            "content"
                        )

            if content is not None:

                lines.append(
                    f"- {content}"
                )

        if not lines:
            return (
                "No relevant memories were found."
            )

        return "\n".join(
            lines
        )

    @staticmethod
    def _context_text(
        context,
    ) -> str:
        """
        Normalize a context object into plain text.
        """

        if context is None:
            return ""

        if isinstance(
            context,
            str,
        ):
            return context

        text = getattr(
            context,
            "text",
            None,
        )

        if text is not None:
            return str(text)

        return str(context)