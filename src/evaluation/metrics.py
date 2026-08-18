from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass
class RetrievalMetrics:
    """
    Standard information-retrieval metrics calculated
    for a single retrieval query.
    """

    recall_at_k: float
    precision_at_k: float
    hit_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


class RetrievalMetricsCalculator:
    """
    Calculate deterministic retrieval metrics.

    The evaluator works with memory IDs so that it remains
    independent from any particular memory backend.

    Example:

        retrieved = ["A", "B", "C"]
        relevant = ["C"]

        calculator = RetrievalMetricsCalculator()

        metrics = calculator.evaluate(
            retrieved_ids=retrieved,
            relevant_ids=relevant,
            k=3
        )
    """

    def evaluate(
        self,
        retrieved_ids: Sequence[str],
        relevant_ids: Iterable[str],
        k: int = 5
    ) -> RetrievalMetrics:
        """
        Calculate retrieval metrics for one query.
        """

        self._validate_k(k)

        retrieved = list(
            retrieved_ids[:k]
        )

        relevant = set(
            relevant_ids
        )

        recall = self.recall_at_k(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant,
            k=k
        )

        precision = self.precision_at_k(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant,
            k=k
        )

        hit = self.hit_at_k(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant,
            k=k
        )

        reciprocal_rank = (
            self.reciprocal_rank(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant,
                k=k
            )
        )

        ndcg = self.ndcg_at_k(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant,
            k=k
        )

        return RetrievalMetrics(
            recall_at_k=recall,
            precision_at_k=precision,
            hit_at_k=hit,
            reciprocal_rank=reciprocal_rank,
            ndcg_at_k=ndcg
        )

    @staticmethod
    def recall_at_k(
        retrieved_ids: Sequence[str],
        relevant_ids: Iterable[str],
        k: int = 5
    ) -> float:
        """
        Recall@K:

            relevant retrieved in top-K
            --------------------------------
            total relevant items

        Returns 0.0 when there are no relevant items.
        """

        RetrievalMetricsCalculator._validate_k(k)

        relevant = set(
            relevant_ids
        )

        if not relevant:
            return 0.0

        retrieved = set(
            retrieved_ids[:k]
        )

        return (
            len(
                retrieved & relevant
            )
            / len(relevant)
        )

    @staticmethod
    def precision_at_k(
        retrieved_ids: Sequence[str],
        relevant_ids: Iterable[str],
        k: int = 5
    ) -> float:
        """
        Precision@K:

            relevant retrieved in top-K
            --------------------------------
            number of retrieved items in top-K
        """

        RetrievalMetricsCalculator._validate_k(k)

        relevant = set(
            relevant_ids
        )

        retrieved = list(
            retrieved_ids[:k]
        )

        if not retrieved:
            return 0.0

        relevant_count = sum(
            1
            for memory_id in retrieved
            if memory_id in relevant
        )

        return (
            relevant_count
            / len(retrieved)
        )

    @staticmethod
    def hit_at_k(
        retrieved_ids: Sequence[str],
        relevant_ids: Iterable[str],
        k: int = 5
    ) -> float:
        """
        Hit@K:

            1.0 if at least one relevant memory
            appears in the top-K results.

            0.0 otherwise.
        """

        RetrievalMetricsCalculator._validate_k(k)

        relevant = set(
            relevant_ids
        )

        retrieved = set(
            retrieved_ids[:k]
        )

        if retrieved & relevant:
            return 1.0

        return 0.0

    @staticmethod
    def reciprocal_rank(
        retrieved_ids: Sequence[str],
        relevant_ids: Iterable[str],
        k: int = 5
    ) -> float:
        """
        Reciprocal Rank of the first relevant result.

        Example:

            relevant item at rank 1 -> 1.0
            relevant item at rank 2 -> 0.5
            relevant item at rank 3 -> 0.333...
        """

        RetrievalMetricsCalculator._validate_k(k)

        relevant = set(
            relevant_ids
        )

        for index, memory_id in enumerate(
            retrieved_ids[:k],
            start=1
        ):

            if memory_id in relevant:
                return 1.0 / index

        return 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: Sequence[str],
        relevant_ids: Iterable[str],
        k: int = 5
    ) -> float:
        """
        Normalized Discounted Cumulative Gain@K.

        This implementation treats relevant memories
        as binary relevance:

            relevant     = 1
            non-relevant = 0
        """

        RetrievalMetricsCalculator._validate_k(k)

        relevant = set(
            relevant_ids
        )

        if not relevant:
            return 0.0

        retrieved = list(
            retrieved_ids[:k]
        )

        dcg = 0.0

        for rank, memory_id in enumerate(
            retrieved,
            start=1
        ):

            if memory_id in relevant:

                dcg += (
                    1.0
                    / math.log2(
                        rank + 1
                    )
                )

        ideal_count = min(
            len(relevant),
            k
        )

        ideal_dcg = sum(
            1.0
            / math.log2(rank + 1)
            for rank in range(
                1,
                ideal_count + 1
            )
        )

        if ideal_dcg == 0.0:
            return 0.0

        return dcg / ideal_dcg

    @staticmethod
    def _validate_k(k: int) -> None:
        """
        Validate the top-K parameter.
        """

        if not isinstance(
            k,
            int
        ):
            raise TypeError(
                "k must be an integer."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0."
            )


@dataclass
class ClassificationMetrics:
    """
    Metrics for evaluating memory-type classification.
    """

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float


class ClassificationMetricsCalculator:
    """
    Calculate deterministic classification metrics.

    This is useful for evaluating the MemoryClassifier
    independently from retrieval.
    """

    def evaluate(
        self,
        predictions: Sequence[str],
        targets: Sequence[str]
    ) -> ClassificationMetrics:
        """
        Calculate accuracy, macro precision, macro recall,
        and macro F1.
        """

        if len(predictions) != len(targets):
            raise ValueError(
                "predictions and targets must have "
                "the same length."
            )

        if not predictions:
            return ClassificationMetrics(
                accuracy=0.0,
                macro_precision=0.0,
                macro_recall=0.0,
                macro_f1=0.0
            )

        labels = set(
            predictions
        ) | set(
            targets
        )

        accuracy = sum(
            prediction == target
            for prediction, target in zip(
                predictions,
                targets
            )
        ) / len(predictions)

        precisions = []
        recalls = []
        f1_scores = []

        for label in labels:

            true_positive = 0
            false_positive = 0
            false_negative = 0

            for prediction, target in zip(
                predictions,
                targets
            ):

                if (
                    prediction == label
                    and target == label
                ):
                    true_positive += 1

                elif (
                    prediction == label
                    and target != label
                ):
                    false_positive += 1

                elif (
                    prediction != label
                    and target == label
                ):
                    false_negative += 1

            if (
                true_positive
                + false_positive
                == 0
            ):
                precision = 0.0
            else:
                precision = (
                    true_positive
                    /
                    (
                        true_positive
                        + false_positive
                    )
                )

            if (
                true_positive
                + false_negative
                == 0
            ):
                recall = 0.0
            else:
                recall = (
                    true_positive
                    /
                    (
                        true_positive
                        + false_negative
                    )
                )

            if (
                precision + recall
                == 0.0
            ):
                f1 = 0.0
            else:
                f1 = (
                    2
                    * precision
                    * recall
                    /
                    (
                        precision
                        + recall
                    )
                )

            precisions.append(
                precision
            )

            recalls.append(
                recall
            )

            f1_scores.append(
                f1
            )

        return ClassificationMetrics(
            accuracy=accuracy,
            macro_precision=(
                sum(precisions)
                / len(precisions)
            ),
            macro_recall=(
                sum(recalls)
                / len(recalls)
            ),
            macro_f1=(
                sum(f1_scores)
                / len(f1_scores)
            )
        )


@dataclass
class QAResult:
    """
    Result of evaluating one generated answer.
    """

    correct: bool
    exact_match: float
    token_f1: float


class QAMetricsCalculator:
    """
    Lightweight deterministic QA metrics.

    These metrics are useful for local experiments.

    LongMemEval's official evaluation can later be run
    using its own evaluation harness and judge setup.
    """

    def evaluate(
        self,
        prediction: str,
        reference: str
    ) -> QAResult:
        """
        Evaluate one generated answer.
        """

        if prediction is None:
            prediction = ""

        if reference is None:
            reference = ""

        normalized_prediction = (
            self._normalize(
                prediction
            )
        )

        normalized_reference = (
            self._normalize(
                reference
            )
        )

        exact_match = (
            1.0
            if (
                normalized_prediction
                == normalized_reference
            )
            else 0.0
        )

        f1 = self._token_f1(
            normalized_prediction,
            normalized_reference
        )

        return QAResult(
            correct=(
                exact_match == 1.0
            ),
            exact_match=exact_match,
            token_f1=f1
        )

    @staticmethod
    def _normalize(
        text: str
    ) -> str:
        """
        Normalize text for deterministic comparison.
        """

        return " ".join(
            text.lower().strip().split()
        )

    @staticmethod
    def _token_f1(
        prediction: str,
        reference: str
    ) -> float:
        """
        Calculate token-level F1.
        """

        prediction_tokens = (
            prediction.split()
        )

        reference_tokens = (
            reference.split()
        )

        if (
            not prediction_tokens
            or not reference_tokens
        ):
            return float(
                prediction_tokens
                == reference_tokens
            )

        prediction_counts = {}

        for token in prediction_tokens:
            prediction_counts[token] = (
                prediction_counts.get(
                    token,
                    0
                )
                + 1
            )

        reference_counts = {}

        for token in reference_tokens:
            reference_counts[token] = (
                reference_counts.get(
                    token,
                    0
                )
                + 1
            )

        common = 0

        for token in prediction_counts:

            common += min(
                prediction_counts[token],
                reference_counts.get(
                    token,
                    0
                )
            )

        if common == 0:
            return 0.0

        precision = (
            common
            / len(prediction_tokens)
        )

        recall = (
            common
            / len(reference_tokens)
        )

        if precision + recall == 0:
            return 0.0

        return (
            2
            * precision
            * recall
            /
            (
                precision
                + recall
            )
        )