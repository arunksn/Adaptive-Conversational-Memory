from dataclasses import dataclass

from src.retrieval.graph_retriever import (
    GraphRetriever,
)

from src.retrieval.vector_retriever import (
    VectorRetriever,
)


@dataclass
class VectorKGRetrievalResult:
    """
    Result produced by the Vector + Knowledge Graph
    retrieval experiment.
    """

    results: list

    vector_results: list

    graph_results: list

    retrieved_ids: list[str]

    result_count: int


class VectorKG:
    """
    Vector + Knowledge Graph retrieval experiment.

    This experiment combines:

    1. Vector-based semantic retrieval.
    2. Knowledge Graph-based retrieval.

    The two result sets are merged and deduplicated.

    This class does not modify the underlying retrieval
    systems or memory stores.
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever,
    ):
        if vector_retriever is None:
            raise ValueError(
                "vector_retriever cannot be None"
            )

        if graph_retriever is None:
            raise ValueError(
                "graph_retriever cannot be None"
            )

        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever

    # RETRIEVE

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> VectorKGRetrievalResult:
        """
        Retrieve memories using both vector and graph
        retrieval.
        """

        if not isinstance(
            query,
            str
        ) or not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0"
            )

        vector_results = self._vector_retrieve(
            query=query,
            k=k,
        )

        graph_results = self._graph_retrieve(
            query=query,
            k=k,
        )

        merged_results = self._merge_results(
            vector_results=vector_results,
            graph_results=graph_results,
        )

        merged_results = merged_results[
            :k
        ]

        retrieved_ids = self._extract_memory_ids(
            merged_results
        )

        return VectorKGRetrievalResult(
            results=merged_results,
            vector_results=vector_results,
            graph_results=graph_results,
            retrieved_ids=retrieved_ids,
            result_count=len(
                merged_results
            ),
        )

    # VECTOR RETRIEVAL

    def _vector_retrieve(
        self,
        query: str,
        k: int,
    ) -> list:
        """
        Execute vector retrieval.

        The method supports the existing retriever
        convention where retrieve() may return either
        a result list or a tuple containing results.
        """

        response = self.vector_retriever.retrieve(
            query=query,
            top_k=k,
        )

        return self._normalize_retrieval_response(
            response
        )

    # GRAPH RETRIEVAL

    def _graph_retrieve(
        self,
        query: str,
        k: int,
    ) -> list:
        """
        Execute knowledge graph retrieval.

        The graph retriever is treated as a separate
        retrieval source and its results are normalized
        before merging.
        """

        response = self.graph_retriever.retrieve(
            query=query,
            top_k=k,
        )

        return self._normalize_retrieval_response(
            response
        )

    # RESPONSE NORMALIZATION

    @staticmethod
    def _normalize_retrieval_response(
        response
    ) -> list:
        """
        Normalize common retrieval response formats.

        Supported formats:

        - list
        - (metadata, results)
        - (results, metadata)

        The actual result objects are preserved.
        """

        if response is None:
            return []

        if isinstance(
            response,
            list
        ):
            return response

        if isinstance(
            response,
            tuple
        ):

            if len(response) != 2:
                return []

            first, second = response

            if isinstance(
                first,
                list
            ):
                return first

            if isinstance(
                second,
                list
            ):
                return second

        return []

    # MERGING

    @staticmethod
    def _merge_results(
        vector_results: list,
        graph_results: list,
    ) -> list:
        """
        Merge vector and graph results.

        Duplicate memories are removed using their
        memory ID when available.

        Vector results are kept first, followed by
        graph results.
        """

        merged = []

        seen_ids = set()

        for result in (
            list(vector_results)
            + list(graph_results)
        ):

            memory_id = (
                VectorKG._extract_memory_id(
                    result
                )
            )

            if memory_id is not None:

                if memory_id in seen_ids:
                    continue

                seen_ids.add(
                    memory_id
                )

            merged.append(
                result
            )

        return merged

    # MEMORY ID EXTRACTION

    @staticmethod
    def _extract_memory_id(
        result
    ):
        """
        Extract a memory ID from a retrieval result.

        Supports the common result representations
        used throughout the project.
        """

        if result is None:
            return None

        memory_id = getattr(
            result,
            "memory_id",
            None
        )

        if memory_id is not None:
            return str(
                memory_id
            )

        item = getattr(
            result,
            "item",
            None
        )

        if item is not None:

            memory_id = getattr(
                item,
                "memory_id",
                None
            )

            if memory_id is not None:
                return str(
                    memory_id
                )

        if isinstance(
            result,
            dict
        ):

            memory_id = result.get(
                "memory_id"
            )

            if memory_id is not None:
                return str(
                    memory_id
                )

            item = result.get(
                "item"
            )

            if isinstance(
                item,
                dict
            ):

                memory_id = item.get(
                    "memory_id"
                )

                if memory_id is not None:
                    return str(
                        memory_id
                    )

        metadata = getattr(
            result,
            "metadata",
            None
        )

        if isinstance(
            metadata,
            dict
        ):

            memory_id = metadata.get(
                "memory_id"
            )

            if memory_id is not None:
                return str(
                    memory_id
                )

        return None

    @staticmethod
    def _extract_memory_ids(
        results: list
    ) -> list[str]:
        """
        Extract all identifiable memory IDs.
        """

        memory_ids = []

        for result in results:

            memory_id = (
                VectorKG._extract_memory_id(
                    result
                )
            )

            if memory_id is not None:

                memory_ids.append(
                    memory_id
                )

        return memory_ids

    # CONTEXT

    def build_context(
        self,
        query: str,
        context_builder,
        k: int = 5,
    ):
        """
        Retrieve memories and construct LLM-ready
        context using the existing ContextBuilder.
        """

        if context_builder is None:
            raise ValueError(
                "context_builder cannot be None"
            )

        retrieval = self.retrieve(
            query=query,
            k=k,
        )

        return context_builder.build(
            retrieval.results,
            query=query,
        )

    # RUN

    def run(
        self,
        query: str,
        k: int = 5,
        context_builder=None,
    ):
        """
        Execute the complete Vector + Knowledge Graph
        retrieval experiment.
        """

        retrieval = self.retrieve(
            query=query,
            k=k,
        )

        context = None

        if context_builder is not None:

            context = context_builder.build(
                retrieval.results,
                query=query,
            )

        return {
            "query": query,
            "retrieval": retrieval,
            "context": context,
            "retrieved_ids": retrieval.retrieved_ids,
        }