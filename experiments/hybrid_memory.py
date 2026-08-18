from dataclasses import dataclass


@dataclass
class HybridMemoryResult:
    """
    Result produced by the Hybrid Memory experiment.

    The experiment combines multiple memory retrieval
    sources and removes duplicate memories.
    """

    results: list

    source_results: dict[str, list]

    retrieved_ids: list[str]

    result_count: int


class HybridMemory:
    """
    Hybrid Memory retrieval experiment.

    This component combines results from multiple memory
    retrieval systems into one unified result set.

    The underlying retrieval systems are not modified.

    Each source is queried independently and the resulting
    memories are merged and deduplicated.
    """

    def __init__(
        self,
        retrievers: dict,
    ):
        if retrievers is None:
            raise ValueError(
                "retrievers cannot be None"
            )

        if not isinstance(
            retrievers,
            dict
        ):
            raise ValueError(
                "retrievers must be a dictionary"
            )

        if not retrievers:
            raise ValueError(
                "retrievers cannot be empty"
            )

        for name, retriever in retrievers.items():

            if not isinstance(
                name,
                str
            ) or not name.strip():

                raise ValueError(
                    "retriever names must be non-empty strings"
                )

            if retriever is None:

                raise ValueError(
                    f"retriever '{name}' cannot be None"
                )

        self.retrievers = retrievers

    # RETRIEVE

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> HybridMemoryResult:
        """
        Retrieve memories from every configured
        retrieval source.
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

        source_results = {}

        for name, retriever in self.retrievers.items():

            response = retriever.retrieve(
                query=query,
                top_k=k,
            )

            source_results[name] = (
                self._normalize_response(
                    response
                )
            )

        merged_results = self._merge_results(
            source_results
        )

        selected_results = merged_results[
            :k
        ]

        retrieved_ids = (
            self._extract_memory_ids(
                selected_results
            )
        )

        return HybridMemoryResult(
            results=selected_results,
            source_results=source_results,
            retrieved_ids=retrieved_ids,
            result_count=len(
                selected_results
            ),
        )

    # RESPONSE NORMALIZATION

    @staticmethod
    def _normalize_response(
        response
    ) -> list:
        """
        Normalize common retriever response formats.

        Supported formats:

        - list
        - (metadata, results)
        - (results, metadata)
        - None
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

    @classmethod
    def _merge_results(
        cls,
        source_results: dict[str, list],
    ) -> list:
        """
        Merge retrieval results from all sources.

        Sources are processed in insertion order.

        Duplicate memories are removed using memory ID.
        """

        merged = []

        seen_ids = set()

        for results in source_results.values():

            for result in results:

                memory_id = (
                    cls._extract_memory_id(
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

        Supports:

        - result.memory_id
        - result.item.memory_id
        - result["memory_id"]
        - result["item"]["memory_id"]
        - result.metadata["memory_id"]
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

    @classmethod
    def _extract_memory_ids(
        cls,
        results: list,
    ) -> list[str]:
        """
        Extract identifiable memory IDs.
        """

        memory_ids = []

        for result in results:

            memory_id = (
                cls._extract_memory_id(
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
        Retrieve hybrid memories and construct an
        LLM-ready context using ContextBuilder.
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
    ) -> dict:
        """
        Run the complete Hybrid Memory experiment.
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