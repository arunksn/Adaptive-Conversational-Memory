from dataclasses import dataclass

from src.evaluation.evaluation_runner import (
    RetrievalEvaluationResult,
    RetrievalEvaluationSummary
)


@dataclass
class EvaluationReport:
    """
    Human-readable evaluation report.

    The report contains both a formatted text representation
    and the original evaluation summary.
    """

    text: str
    summary: RetrievalEvaluationSummary


class EvaluationReportGenerator:
    """
    Converts retrieval evaluation results into a readable
    deterministic report.

    This component does not perform retrieval or calculate
    metrics. It only formats the results produced by
    EvaluationRunner.
    """

    def generate(
        self,
        summary: RetrievalEvaluationSummary
    ) -> EvaluationReport:
        """
        Generate a human-readable evaluation report.
        """

        if summary is None:
            raise ValueError(
                "summary cannot be None"
            )

        text = self._build_report(
            summary
        )

        return EvaluationReport(
            text=text,
            summary=summary
        )

    # REPORT BUILDING

    def _build_report(
        self,
        summary: RetrievalEvaluationSummary
    ) -> str:
        """
        Build the complete report.
        """

        sections = [
            self._header(),
            self._summary_section(
                summary
            ),
            self._metrics_section(
                summary
            ),
            self._case_results_section(
                summary.results
            )
        ]

        return "\n\n".join(
            sections
        )

    # HEADER

    @staticmethod
    def _header() -> str:
        return (
            "Adaptive Conversational Memory\n"
            "================================"
        )

    # SUMMARY

    @staticmethod
    def _summary_section(
        summary: RetrievalEvaluationSummary
    ) -> str:
        return (
            "Evaluation Summary\n"
            "------------------\n"
            f"Cases evaluated: {summary.case_count}\n"
            f"K: {summary.k}"
        )

    # METRICS

    @staticmethod
    def _metrics_section(
        summary: RetrievalEvaluationSummary
    ) -> str:
        return (
            "Retrieval Performance\n"
            "---------------------\n"
            f"Recall@{summary.k}:       "
            f"{summary.recall_at_k:.3f}\n"
            f"Precision@{summary.k}:    "
            f"{summary.precision_at_k:.3f}\n"
            f"Hit@{summary.k}:          "
            f"{summary.hit_at_k:.3f}\n"
            f"MRR:                      "
            f"{summary.reciprocal_rank:.3f}\n"
            f"NDCG@{summary.k}:         "
            f"{summary.ndcg_at_k:.3f}"
        )

    # PER-CASE RESULTS

    @staticmethod
    def _case_results_section(
        results: list[RetrievalEvaluationResult]
    ) -> str:
        """
        Format individual evaluation cases.
        """

        if not results:
            return (
                "Per-case Results\n"
                "----------------\n"
                "No retrieval cases were evaluated."
            )

        lines = [
            "Per-case Results",
            "----------------"
        ]

        for result in results:

            lines.append(
                (
                    f"{result.case_id} | "
                    f"Recall: "
                    f"{result.metrics.recall_at_k:.3f} | "
                    f"Precision: "
                    f"{result.metrics.precision_at_k:.3f} | "
                    f"Hit: "
                    f"{result.metrics.hit_at_k:.3f} | "
                    f"MRR: "
                    f"{result.metrics.reciprocal_rank:.3f} | "
                    f"NDCG: "
                    f"{result.metrics.ndcg_at_k:.3f}"
                )
            )

        return "\n".join(
            lines
        )

    # CONVENIENCE API

    def generate_text(
        self,
        summary: RetrievalEvaluationSummary
    ) -> str:
        """
        Generate only the formatted report text.
        """

        return self.generate(
            summary
        ).text