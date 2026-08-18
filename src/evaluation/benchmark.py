from dataclasses import dataclass

from src.evaluation.evaluation_dataset import (
    EvaluationDataset
)

from src.evaluation.evaluation_runner import (
    EvaluationRunner,
    RetrievalEvaluationSummary
)


@dataclass
class BenchmarkResult:
    """
    Result of comparing memory retrieval systems.

    The baseline and adaptive fields are preserved for
    backward compatibility with the original two-system
    benchmark.

    The systems dictionary contains evaluation summaries
    for multi-system benchmarks.
    """

    baseline: RetrievalEvaluationSummary

    adaptive: RetrievalEvaluationSummary

    systems: dict[str, RetrievalEvaluationSummary] | None = None


class MemoryBenchmark:
    """
    Benchmark multiple conversational memory systems.

    Supported systems can include:

        - baseline_rag
        - vector_memory
        - vector_kg
        - hybrid_memory
        - adaptive_memory

    Every system receives the same evaluation dataset
    and the same top-k value.

    The original two-system baseline/adaptive API remains
    supported for backward compatibility.
    """

    def __init__(
        self,
        baseline_retriever=None,
        adaptive_retriever=None,
        systems=None
    ):
        """
        Initialize the benchmark.

        Original usage:

            MemoryBenchmark(
                baseline_retriever,
                adaptive_retriever
            )

        Multi-system usage:

            MemoryBenchmark(
                systems={
                    "baseline_rag": baseline,
                    "vector_memory": vector,
                    "vector_kg": vector_kg,
                    "hybrid_memory": hybrid,
                    "adaptive_memory": adaptive
                }
            )
        """

        # MULTI-SYSTEM MODE

        if systems is not None:

            if not isinstance(
                systems,
                dict
            ):
                raise ValueError(
                    "systems must be a dictionary"
                )

            if not systems:
                raise ValueError(
                    "systems cannot be empty"
                )

            self.runners = {}

            for name, retriever in systems.items():

                if (
                    not isinstance(
                        name,
                        str
                    )
                    or not name.strip()
                ):
                    raise ValueError(
                        "system names must be non-empty strings"
                    )

                if retriever is None:
                    raise ValueError(
                        f"retriever for system "
                        f"'{name}' cannot be None"
                    )

                self.runners[name] = (
                    EvaluationRunner(
                        retriever
                    )
                )

            # Preserve access to the primary systems.

            self.baseline_runner = (
                self.runners.get(
                    "baseline_rag"
                )
            )

            if self.baseline_runner is None:

                self.baseline_runner = (
                    self.runners.get(
                        "baseline"
                    )
                )

            self.adaptive_runner = (
                self.runners.get(
                    "adaptive_memory"
                )
            )

            if self.adaptive_runner is None:

                self.adaptive_runner = (
                    self.runners.get(
                        "adaptive"
                    )
                )

            return

        # BACKWARD-COMPATIBLE TWO-SYSTEM MODE

        if baseline_retriever is None:

            raise ValueError(
                "baseline_retriever cannot be None"
            )

        if adaptive_retriever is None:

            raise ValueError(
                "adaptive_retriever cannot be None"
            )

        self.baseline_runner = (
            EvaluationRunner(
                baseline_retriever
            )
        )

        self.adaptive_runner = (
            EvaluationRunner(
                adaptive_retriever
            )
        )

        self.runners = {
            "baseline": self.baseline_runner,
            "adaptive": self.adaptive_runner
        }

    # RUN BENCHMARK

    def run(
        self,
        dataset: EvaluationDataset,
        k: int = 5
    ) -> BenchmarkResult:
        """
        Evaluate all configured memory systems on the
        same dataset and top-k value.
        """

        if dataset is None:

            raise ValueError(
                "dataset cannot be None"
            )

        if k <= 0:

            raise ValueError(
                "k must be greater than 0"
            )

        summaries = {}

        for name, runner in self.runners.items():

            summaries[name] = (
                runner.evaluate(
                    dataset,
                    k=k
                )
            )

        # Resolve baseline.

        baseline = self._get_primary_summary(
            summaries,
            (
                "baseline",
                "baseline_rag"
            )
        )

        # Resolve adaptive.

        adaptive = self._get_primary_summary(
            summaries,
            (
                "adaptive",
                "adaptive_memory"
            )
        )

        # Multi-system benchmarks may not contain
        # either primary system.

        if baseline is None:

            baseline = self._empty_summary(
                k
            )

        if adaptive is None:

            adaptive = self._empty_summary(
                k
            )

        return BenchmarkResult(
            baseline=baseline,
            adaptive=adaptive,
            systems=summaries
        )

    # PRIMARY SUMMARY HELPERS

    @staticmethod
    def _get_primary_summary(
        summaries: dict[str, RetrievalEvaluationSummary],
        names: tuple[str, ...]
    ):
        """
        Find the first matching summary from the
        supplied system names.
        """

        for name in names:

            if name in summaries:

                return summaries[name]

        return None

    @staticmethod
    def _empty_summary(
        k: int
    ) -> RetrievalEvaluationSummary:
        """
        Create an empty evaluation summary.
        """

        return RetrievalEvaluationSummary(
            results=[],
            recall_at_k=0.0,
            precision_at_k=0.0,
            hit_at_k=0.0,
            reciprocal_rank=0.0,
            ndcg_at_k=0.0,
            k=k,
            case_count=0
        )

    # IMPROVEMENT

    @staticmethod
    def improvement(
        baseline: float,
        adaptive: float
    ) -> float:
        """
        Calculate absolute improvement of the adaptive
        system over the baseline.
        """

        return adaptive - baseline

    # RELATIVE IMPROVEMENT

    @staticmethod
    def relative_improvement(
        baseline: float,
        adaptive: float
    ) -> float:
        """
        Calculate percentage improvement.

        Returns 0.0 when the baseline is zero.
        """

        if baseline == 0.0:

            return 0.0

        return (
            (adaptive - baseline)
            / baseline
        ) * 100.0

    # METRIC COMPARISON

    @classmethod
    def compare_metrics(
        cls,
        result: BenchmarkResult
    ) -> dict[str, dict[str, float]]:
        """
        Compare all retrieval metrics between the
        baseline and adaptive systems.

        This preserves the original benchmark API.
        """

        if result is None:

            raise ValueError(
                "result cannot be None"
            )

        baseline = result.baseline
        adaptive = result.adaptive

        metrics = {
            "recall_at_k": (
                baseline.recall_at_k,
                adaptive.recall_at_k
            ),
            "precision_at_k": (
                baseline.precision_at_k,
                adaptive.precision_at_k
            ),
            "hit_at_k": (
                baseline.hit_at_k,
                adaptive.hit_at_k
            ),
            "reciprocal_rank": (
                baseline.reciprocal_rank,
                adaptive.reciprocal_rank
            ),
            "ndcg_at_k": (
                baseline.ndcg_at_k,
                adaptive.ndcg_at_k
            )
        }

        comparison = {}

        for metric_name, values in metrics.items():

            baseline_value = values[0]
            adaptive_value = values[1]

            comparison[metric_name] = {
                "baseline": baseline_value,
                "adaptive": adaptive_value,
                "improvement": cls.improvement(
                    baseline_value,
                    adaptive_value
                ),
                "relative_improvement": (
                    cls.relative_improvement(
                        baseline_value,
                        adaptive_value
                    )
                )
            }

        return comparison

    # MULTI-SYSTEM COMPARISON

    @staticmethod
    def compare_systems(
        result: BenchmarkResult
    ) -> dict[str, dict[str, float]]:
        """
        Return retrieval metrics for every benchmarked
        system.

        Supports both the new multi-system benchmark
        and the original two-system benchmark.
        """

        if result is None:

            raise ValueError(
                "result cannot be None"
            )

        systems = result.systems

        # Backward-compatible fallback.

        if systems is None:

            systems = {
                "baseline": result.baseline,
                "adaptive": result.adaptive
            }

        comparison = {}

        for name, summary in systems.items():

            comparison[name] = {
                "recall_at_k": (
                    summary.recall_at_k
                ),
                "precision_at_k": (
                    summary.precision_at_k
                ),
                "hit_at_k": (
                    summary.hit_at_k
                ),
                "reciprocal_rank": (
                    summary.reciprocal_rank
                ),
                "ndcg_at_k": (
                    summary.ndcg_at_k
                )
            }

        return comparison