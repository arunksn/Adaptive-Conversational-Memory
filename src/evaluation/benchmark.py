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
    Result of comparing two memory systems.
    """

    baseline: RetrievalEvaluationSummary
    adaptive: RetrievalEvaluationSummary


class MemoryBenchmark:
    """
    Compare a baseline memory system against the
    adaptive conversational memory system.

    Both systems receive the same evaluation dataset
    and the same top-k value.

    This class does not modify either retrieval system.
    """

    def __init__(
        self,
        baseline_retriever,
        adaptive_retriever
    ):
        if baseline_retriever is None:
            raise ValueError(
                "baseline_retriever cannot be None"
            )

        if adaptive_retriever is None:
            raise ValueError(
                "adaptive_retriever cannot be None"
            )

        self.baseline_runner = EvaluationRunner(
            baseline_retriever
        )

        self.adaptive_runner = EvaluationRunner(
            adaptive_retriever
        )

    # RUN BENCHMARK

    def run(
        self,
        dataset: EvaluationDataset,
        k: int = 5
    ) -> BenchmarkResult:
        """
        Evaluate both memory systems on the same dataset.
        """

        if dataset is None:
            raise ValueError(
                "dataset cannot be None"
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0"
            )

        baseline = self.baseline_runner.evaluate(
            dataset,
            k=k
        )

        adaptive = self.adaptive_runner.evaluate(
            dataset,
            k=k
        )

        return BenchmarkResult(
            baseline=baseline,
            adaptive=adaptive
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

        Example:

            baseline = 0.60
            adaptive = 0.80

            improvement = 0.20
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