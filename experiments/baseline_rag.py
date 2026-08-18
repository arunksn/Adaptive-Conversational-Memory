"""
Baseline RAG Experiment
========================

A simple retrieval-augmented generation baseline used as the
reference system for evaluating the progressively more advanced
memory architectures.

The baseline intentionally does NOT use:

- adaptive retrieval
- temporal retrieval
- graph retrieval
- conflict resolution
- memory consolidation
- controlled forgetting
- adaptive routing

It provides a simple vector-retrieval baseline against which the
other memory architectures can be compared.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class BaselineRAGResult:
    """
    Result produced by one baseline RAG query.
    """

    query: str

    retrieved_ids: list[str]

    context: str

    response: str | None = None


class BaselineRAG:
    """
    Simple vector-retrieval RAG baseline.

    The class is deliberately lightweight.

    Retrieval is delegated to the supplied retriever, while
    generation is delegated to the supplied response generator.

    This keeps the experiment independent from the implementation
    details of the production adaptive memory pipeline.
    """

    def __init__(
        self,
        retriever,
        response_generator=None,
        context_builder=None,
    ):
        if retriever is None:
            raise ValueError(
                "retriever cannot be None"
            )

        self.retriever = retriever
        self.response_generator = (
            response_generator
        )
        self.context_builder = (
            context_builder
        )


    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> list[Any]:
        """
        Retrieve the top-k memories for a query.

        The baseline uses only the supplied retriever.
        """

        if not query or not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0"
            )

        result = self.retriever.retrieve(
            query=query,
            top_k=k,
        )

        retrieved = self._extract_retrieval_results(
            result
        )

        return retrieved[:k]


    @staticmethod
    def extract_memory_ids(
        results: list[Any],
    ) -> list[str]:
        """
        Extract memory IDs from retrieval results.

        Supports the project's RetrievalResult structure as
        well as simple objects used by experiment tests.
        """

        memory_ids = []

        for result in results:

            memory_id = getattr(
                result,
                "memory_id",
                None,
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


    def build_context(
        self,
        results: list[Any],
        query: str | None = None,
    ):
        """
        Build context from retrieved memories.

        If a ContextBuilder is supplied, it is used.

        Otherwise, a simple deterministic text context is
        constructed directly from the retrieved results.
        """

        if self.context_builder is not None:

            return self.context_builder.build(
                results=results,
                query=query,
            )

        return self._simple_context(
            results
        )


    def generate(
        self,
        query: str,
        context,
    ):
        """
        Generate a response using the supplied response
        generator.

        Generation is optional because retrieval evaluation
        does not require an LLM.
        """

        if self.response_generator is None:
            return None

        return self.response_generator.generate(
            query=query,
            context=context,
        )


    def run(
        self,
        query: str,
        k: int = 5,
    ) -> BaselineRAGResult:
        """
        Execute the complete baseline RAG pipeline.

        1. Retrieve memories.
        2. Build context.
        3. Optionally generate a response.
        4. Return the experiment result.
        """

        results = self.retrieve(
            query=query,
            k=k,
        )

        context = self.build_context(
            results=results,
            query=query,
        )

        generated = self.generate(
            query=query,
            context=context,
        )

        response = self._extract_response_text(
            generated
        )

        context_text = self._extract_context_text(
            context
        )

        return BaselineRAGResult(
            query=query,
            retrieved_ids=self.extract_memory_ids(
                results
            ),
            context=context_text,
            response=response,
        )


    @staticmethod
    def _extract_retrieval_results(
        result,
    ) -> list[Any]:
        """
        Normalize different retriever return formats.

        The project's retrievers commonly return:

            (metadata, results)

        This helper also accepts a direct list of results.
        """

        if isinstance(
            result,
            tuple,
        ):

            if len(result) >= 2:
                retrieved = result[1]

                if retrieved is None:
                    return []

                return list(
                    retrieved
                )

            if len(result) == 1:
                retrieved = result[0]

                if retrieved is None:
                    return []

                if isinstance(
                    retrieved,
                    list,
                ):
                    return retrieved

        if result is None:
            return []

        if isinstance(
            result,
            list,
        ):
            return result

        return list(result)

    @staticmethod
    def _simple_context(
        results: list[Any],
    ) -> str:
        """
        Build a minimal deterministic context when no
        ContextBuilder is supplied.
        """

        if not results:
            return (
                "No relevant memories were found."
            )

        sections = []

        for result in results:

            item = getattr(
                result,
                "item",
                None,
            )

            if item is not None:

                content = getattr(
                    item,
                    "content",
                    None,
                )

                if content is not None:
                    sections.append(
                        f"- {content}"
                    )
                    continue

            content = getattr(
                result,
                "content",
                None,
            )

            if content is not None:
                sections.append(
                    f"- {content}"
                )

        if not sections:
            return (
                "No relevant memories were found."
            )

        return "\n".join(
            sections
        )

    @staticmethod
    def _extract_context_text(
        context,
    ) -> str:
        """
        Normalize MemoryContext or plain string into text.
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

    @staticmethod
    def _extract_response_text(
        generated,
    ) -> str | None:
        """
        Extract response text from GeneratedResponse or
        a plain string.
        """

        if generated is None:
            return None

        if isinstance(
            generated,
            str,
        ):
            return generated

        text = getattr(
            generated,
            "text",
            None,
        )

        if text is not None:
            return str(text)

        return str(generated)