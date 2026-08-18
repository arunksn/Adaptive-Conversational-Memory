from pathlib import Path
import json

from src.evaluation.benchmark import (
    MemoryBenchmark,
)

from src.evaluation.benchmark_dataset import (
    AdaptiveMemoryBenchmarkDataset,
)

from src.evaluation.evaluation_dataset import (
    EvaluationDataset,
)

from src.embeddings.embedding_model import (
    EmbeddingModel,
)

from src.storage.vector_store import (
    VectorStore,
)

from src.storage.temporal_store import (
    TemporalStore,
)

from src.storage.graph_store import (
    GraphStore,
)

from src.retrieval.vector_retriever import (
    VectorRetriever,
)

from src.retrieval.temporal_retriever import (
    TemporalRetriever,
)

from src.retrieval.graph_retriever import (
    GraphRetriever,
)

from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)

from src.retrieval.adaptive_retriever import (
    AdaptiveRetriever,
)

from src.routing.memory_router import (
    MemoryRouter,
)

from src.conflict.conflict_detector import (
    ConflictDetector,
)

from src.conflict.conflict_resolver import (
    ConflictResolver,
)

from src.models.procedure import (
    Procedure,
    ProcedureState,
    ProcedureTransition,
)


# ============================================================
# PATHS
# ============================================================

VECTOR_STORE_PATH = (
    "data/memories/vector_store"
)

TEMPORAL_DATABASE_PATH = (
    "data/memories/episodic_memory.db"
)

PROCEDURES_PATH = Path(
    "data/memories/procedures.json"
)


# ============================================================
# VECTOR RETRIEVER ADAPTER
# ============================================================

class VectorRetrieverAdapter:
    """
    Adapter that exposes the project's VectorRetriever
    through the retrieve() interface expected by
    EvaluationRunner.
    """

    def __init__(
        self,
        retriever: VectorRetriever,
    ):
        if retriever is None:
            raise ValueError(
                "retriever cannot be None"
            )

        self.retriever = retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ):
        """
        Convert VectorRetriever.search() into the
        (route, results) interface expected by
        EvaluationRunner.
        """

        results = self.retriever.search(
            query=query,
            top_k=top_k,
        )

        return (
            None,
            results,
        )


# ============================================================
# PROCEDURE LOADING
# ============================================================

def load_procedures(
    path: Path,
) -> list[dict]:
    """
    Load serialized procedural memories.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Procedures file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "procedures.json must contain a list"
        )

    return data


def build_procedure(
    data: dict,
) -> Procedure:
    """
    Convert serialized procedure data into a
    Procedure object.
    """

    procedure = Procedure(
        name=data["name"],
        description=data.get(
            "description",
            "",
        ),
        procedure_id=data[
            "procedure_id"
        ],
        version=data.get(
            "version",
            1,
        ),
    )

    # STATES

    for state_data in data.get(
        "states",
        [],
    ):

        state = ProcedureState(
            name=state_data["name"],
            description=state_data.get(
                "description",
                "",
            ),
            state_id=state_data[
                "state_id"
            ],
            is_terminal=state_data.get(
                "is_terminal",
                False,
            ),
        )

        procedure.add_state(
            state
        )

    # TRANSITIONS

    for transition_data in data.get(
        "transitions",
        [],
    ):

        transition = ProcedureTransition(
            from_state=transition_data[
                "from_state"
            ],
            to_state=transition_data[
                "to_state"
            ],
            action=transition_data[
                "action"
            ],
            condition=transition_data.get(
                "condition"
            ),
            transition_id=transition_data[
                "transition_id"
            ],
        )

        procedure.add_transition(
            transition
        )

    return procedure


def build_graph_store() -> GraphStore:
    """
    Build the procedural knowledge graph from
    procedures.json.
    """

    procedures = load_procedures(
        PROCEDURES_PATH
    )

    store = GraphStore()

    for data in procedures:

        procedure = build_procedure(
            data
        )

        store.add_procedure(
            procedure
        )

    return store


# ============================================================
# BUILD BASELINE
# ============================================================

def build_baseline():
    """
    Build the baseline vector retrieval system.

    The baseline uses only semantic/vector retrieval.
    """

    print(
        "Building baseline retriever..."
    )

    vector_retriever = VectorRetriever(
        storage_dir=VECTOR_STORE_PATH,
    )

    vector_retriever.load()

    print(
        f"Vector memories loaded: "
        f"{vector_retriever.count()}"
    )

    return VectorRetrieverAdapter(
        vector_retriever
    )


# ============================================================
# BUILD ADAPTIVE SYSTEM
# ============================================================

def build_adaptive():
    """
    Build the complete adaptive memory retrieval system.
    """

    print(
        "Building adaptive retriever..."
    )

    # VECTOR

    vector_retriever = VectorRetriever(
        storage_dir=VECTOR_STORE_PATH,
    )

    vector_retriever.load()

    print(
        f"Vector memories loaded: "
        f"{vector_retriever.count()}"
    )

    # TEMPORAL

    temporal_retriever = TemporalRetriever(
        database_path=TEMPORAL_DATABASE_PATH,
    )

    print(
        f"Episodic memories loaded: "
        f"{temporal_retriever.count()}"
    )

    # GRAPH

    graph_store = build_graph_store()

    print(
        f"Procedures loaded: "
        f"{graph_store.count()}"
    )

    graph_retriever = GraphRetriever(
        graph_store=graph_store,
    )

    # HYBRID

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        temporal_retriever=temporal_retriever,
        graph_retriever=graph_retriever,
    )

    # ROUTER

    router = MemoryRouter()

    # CONFLICT HANDLING

    conflict_detector = (
        ConflictDetector()
    )

    conflict_resolver = (
        ConflictResolver()
    )

    # ADAPTIVE RETRIEVER

    adaptive_retriever = AdaptiveRetriever(
        router=router,
        hybrid_retriever=hybrid_retriever,
        conflict_detector=conflict_detector,
        conflict_resolver=conflict_resolver,
    )

    return adaptive_retriever


# ============================================================
# DATASET
# ============================================================

def build_dataset():
    """
    Build the benchmark evaluation dataset.
    """

    print(
        "Loading benchmark dataset..."
    )

    benchmark_dataset = (
        AdaptiveMemoryBenchmarkDataset.build()
    )

    print(
        f"Benchmark cases: "
        f"{len(benchmark_dataset.cases)}"
    )

    return EvaluationDataset(
        cases=benchmark_dataset.cases
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    result,
):
    """
    Print baseline and adaptive evaluation results.
    """

    baseline = result.baseline
    adaptive = result.adaptive

    print()
    print(
        "======================================"
    )
    print(
        "Baseline Results"
    )
    print(
        "======================================"
    )

    print(
        f"Cases evaluated: {baseline.case_count}"
    )

    print(
        f"Recall@{baseline.k}: "
        f"{baseline.recall_at_k:.3f}"
    )

    print(
        f"Precision@{baseline.k}: "
        f"{baseline.precision_at_k:.3f}"
    )

    print(
        f"Hit@{baseline.k}: "
        f"{baseline.hit_at_k:.3f}"
    )

    print(
        f"MRR: "
        f"{baseline.reciprocal_rank:.3f}"
    )

    print(
        f"NDCG@{baseline.k}: "
        f"{baseline.ndcg_at_k:.3f}"
    )

    print()
    print(
        "======================================"
    )
    print(
        "Adaptive Results"
    )
    print(
        "======================================"
    )

    print(
        f"Cases evaluated: {adaptive.case_count}"
    )

    print(
        f"Recall@{adaptive.k}: "
        f"{adaptive.recall_at_k:.3f}"
    )

    print(
        f"Precision@{adaptive.k}: "
        f"{adaptive.precision_at_k:.3f}"
    )

    print(
        f"Hit@{adaptive.k}: "
        f"{adaptive.hit_at_k:.3f}"
    )

    print(
        f"MRR: "
        f"{adaptive.reciprocal_rank:.3f}"
    )

    print(
        f"NDCG@{adaptive.k}: "
        f"{adaptive.ndcg_at_k:.3f}"
    )


def print_comparison(
    result,
):
    """
    Print baseline vs adaptive improvement.
    """

    comparison = (
        MemoryBenchmark.compare_metrics(
            result
        )
    )

    print()
    print(
        "======================================"
    )
    print(
        "Baseline vs Adaptive"
    )
    print(
        "======================================"
    )

    for metric, values in comparison.items():

        print()
        print(
            metric
        )

        print(
            f"  Baseline: "
            f"{values['baseline']:.3f}"
        )

        print(
            f"  Adaptive: "
            f"{values['adaptive']:.3f}"
        )

        print(
            f"  Improvement: "
            f"{values['improvement']:.3f}"
        )

        print(
            f"  Relative improvement: "
            f"{values['relative_improvement']:.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "======================================"
    )
    print(
        "Adaptive Conversational Memory"
    )
    print(
        "Retrieval Benchmark"
    )
    print(
        "======================================"
    )
    print()

    # DATASET

    dataset = build_dataset()

    print()

    # BASELINE

    baseline = build_baseline()

    print()

    # ADAPTIVE

    adaptive = build_adaptive()

    print()

    # BENCHMARK

    benchmark = MemoryBenchmark(
        baseline_retriever=baseline,
        adaptive_retriever=adaptive,
    )

    print(
        "Running benchmark..."
    )

    print()

    result = benchmark.run(
        dataset=dataset,
        k=5,
    )

    # RESULTS

    print_results(
        result
    )

    print_comparison(
        result
    )

    print()
    print(
        "Benchmark completed successfully."
    )

    # CLOSE TEMPORAL STORE

    temporal_retriever = (
        adaptive
        .hybrid_retriever
        .temporal_retriever
    )

    if temporal_retriever is not None:

        temporal_retriever.close()


if __name__ == "__main__":
    main()