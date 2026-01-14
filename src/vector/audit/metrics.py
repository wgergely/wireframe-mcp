"""Search Quality Metrics for vector retrieval evaluation.

Implements standard IR metrics for measuring search quality:
- Precision@K: Fraction of top-K results that are relevant
- Recall@K: Fraction of relevant items found in top-K
- MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant result
- NDCG (Normalized Discounted Cumulative Gain): Position-weighted relevance
- Hit Rate@K: Fraction of queries with at least one relevant result in top-K
"""

import math
from dataclasses import dataclass, field
from typing import Sequence

from .lib import AuditReport, AuditResult


@dataclass
class SearchMetrics:
    """Collection of search quality metrics.

    Attributes:
        precision_at_k: Precision at various K values.
        recall_at_k: Recall at various K values (requires total relevant count).
        mrr: Mean Reciprocal Rank.
        ndcg_at_k: NDCG at various K values.
        hit_rate_at_k: Hit rate at various K values.
        query_count: Number of queries evaluated.
    """

    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: dict[int, float] = field(default_factory=dict)
    query_count: int = 0

    def summary(self) -> str:
        """Generate human-readable summary of metrics."""
        lines = [
            f"Search Quality Metrics (n={self.query_count})",
            "-" * 40,
            f"MRR: {self.mrr:.4f}",
        ]

        if self.precision_at_k:
            lines.append("Precision@K:")
            for k in sorted(self.precision_at_k.keys()):
                lines.append(f"  @{k}: {self.precision_at_k[k]:.4f}")

        if self.recall_at_k:
            lines.append("Recall@K:")
            for k in sorted(self.recall_at_k.keys()):
                lines.append(f"  @{k}: {self.recall_at_k[k]:.4f}")

        if self.ndcg_at_k:
            lines.append("NDCG@K:")
            for k in sorted(self.ndcg_at_k.keys()):
                lines.append(f"  @{k}: {self.ndcg_at_k[k]:.4f}")

        if self.hit_rate_at_k:
            lines.append("Hit Rate@K:")
            for k in sorted(self.hit_rate_at_k.keys()):
                lines.append(f"  @{k}: {self.hit_rate_at_k[k]:.4f}")

        return "\n".join(lines)


def precision_at_k(relevant: Sequence[bool], k: int) -> float:
    """Calculate precision at K.

    Precision@K = (# relevant in top K) / K

    Args:
        relevant: Boolean sequence indicating if each result is relevant.
        k: Number of top results to consider.

    Returns:
        Precision value between 0.0 and 1.0.
    """
    if k <= 0:
        return 0.0

    top_k = relevant[:k]
    if not top_k:
        return 0.0

    return sum(1 for r in top_k if r) / k


def recall_at_k(relevant: Sequence[bool], k: int, total_relevant: int) -> float:
    """Calculate recall at K.

    Recall@K = (# relevant in top K) / (total # relevant)

    Args:
        relevant: Boolean sequence indicating if each result is relevant.
        k: Number of top results to consider.
        total_relevant: Total number of relevant items in collection.

    Returns:
        Recall value between 0.0 and 1.0.
    """
    if total_relevant <= 0:
        return 0.0

    top_k = relevant[:k]
    relevant_in_k = sum(1 for r in top_k if r)

    return relevant_in_k / total_relevant


def reciprocal_rank(relevant: Sequence[bool]) -> float:
    """Calculate reciprocal rank.

    RR = 1 / (rank of first relevant result)
    Returns 0 if no relevant results.

    Args:
        relevant: Boolean sequence indicating if each result is relevant.

    Returns:
        Reciprocal rank between 0.0 and 1.0.
    """
    for i, is_relevant in enumerate(relevant):
        if is_relevant:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(relevance_scores: Sequence[float], k: int) -> float:
    """Calculate Discounted Cumulative Gain at K.

    DCG@K = sum(rel_i / log2(i + 2)) for i in 0..K-1

    Args:
        relevance_scores: Relevance score for each result (higher is better).
        k: Number of top results to consider.

    Returns:
        DCG value (unbounded positive).
    """
    if k <= 0:
        return 0.0

    top_k = relevance_scores[:k]
    dcg = 0.0

    for i, rel in enumerate(top_k):
        # Use log2(i + 2) so position 0 has discount log2(2) = 1
        dcg += rel / math.log2(i + 2)

    return dcg


def ndcg_at_k(relevance_scores: Sequence[float], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K.

    NDCG@K = DCG@K / IDCG@K where IDCG is DCG of ideal ordering.

    Args:
        relevance_scores: Relevance score for each result (higher is better).
        k: Number of top results to consider.

    Returns:
        NDCG value between 0.0 and 1.0.
    """
    actual_dcg = dcg_at_k(relevance_scores, k)

    # Ideal DCG: sort by relevance descending
    ideal_order = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg_at_k(ideal_order, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def hit_rate_at_k(relevant: Sequence[bool], k: int) -> float:
    """Calculate hit rate at K (binary: any relevant in top K?).

    Args:
        relevant: Boolean sequence indicating if each result is relevant.
        k: Number of top results to consider.

    Returns:
        1.0 if any relevant in top K, 0.0 otherwise.
    """
    top_k = relevant[:k]
    return 1.0 if any(top_k) else 0.0


class MetricsCalculator:
    """Calculates search quality metrics from audit results.

    Uses passed_required from evaluations as the relevance indicator.
    Also supports custom relevance functions for more control.

    Example:
        >>> calc = MetricsCalculator()
        >>> report = runner.run(queries)
        >>> metrics = calc.calculate(report, k_values=[1, 3, 5, 10])
        >>> print(metrics.summary())
    """

    def __init__(
        self,
        relevance_fn=None,
    ):
        """Initialize calculator.

        Args:
            relevance_fn: Optional function (ResultEvaluation) -> bool
                         to determine relevance. Defaults to passed_required.
        """
        self._relevance_fn = relevance_fn or (lambda e: e.passed_required)

    def calculate(
        self,
        report: AuditReport,
        k_values: list[int] | None = None,
        total_relevant_per_query: dict[str, int] | None = None,
    ) -> SearchMetrics:
        """Calculate all metrics from an audit report.

        Args:
            report: AuditReport from AuditRunner.
            k_values: K values for @K metrics. Defaults to [1, 3, 5, 10].
            total_relevant_per_query: Optional mapping of query name to total
                                      relevant items (for recall calculation).

        Returns:
            SearchMetrics with all computed values.
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        metrics = SearchMetrics(query_count=len(report.results))

        if not report.results:
            return metrics

        # Initialize accumulators
        precision_sums: dict[int, float] = {k: 0.0 for k in k_values}
        recall_sums: dict[int, float] = {k: 0.0 for k in k_values}
        ndcg_sums: dict[int, float] = {k: 0.0 for k in k_values}
        hit_rate_sums: dict[int, float] = {k: 0.0 for k in k_values}
        rr_sum = 0.0
        recall_query_count = 0

        for result in report.results:
            # Get relevance sequence for this query
            relevance = [self._relevance_fn(e) for e in result.evaluations]

            # Get relevance scores (using score_percentage for graded relevance)
            relevance_scores = [e.score_percentage / 100.0 for e in result.evaluations]

            # Calculate per-query metrics
            rr_sum += reciprocal_rank(relevance)

            for k in k_values:
                precision_sums[k] += precision_at_k(relevance, k)
                ndcg_sums[k] += ndcg_at_k(relevance_scores, k)
                hit_rate_sums[k] += hit_rate_at_k(relevance, k)

                # Recall requires total relevant count
                if total_relevant_per_query:
                    total_rel = total_relevant_per_query.get(result.query.name)
                    if total_rel:
                        recall_sums[k] += recall_at_k(relevance, k, total_rel)
                        if k == k_values[0]:
                            recall_query_count += 1

        # Compute averages
        n = len(report.results)
        metrics.mrr = rr_sum / n

        for k in k_values:
            metrics.precision_at_k[k] = precision_sums[k] / n
            metrics.ndcg_at_k[k] = ndcg_sums[k] / n
            metrics.hit_rate_at_k[k] = hit_rate_sums[k] / n

            if recall_query_count > 0:
                metrics.recall_at_k[k] = recall_sums[k] / recall_query_count

        return metrics

    def calculate_single(
        self, result: AuditResult, k_values: list[int] | None = None
    ) -> SearchMetrics:
        """Calculate metrics for a single query result.

        Args:
            result: Single AuditResult to evaluate.
            k_values: K values for @K metrics.

        Returns:
            SearchMetrics for this query.
        """
        report = AuditReport(results=[result])
        return self.calculate(report, k_values)


def calculate_metrics(
    report: AuditReport,
    k_values: list[int] | None = None,
) -> SearchMetrics:
    """Convenience function to calculate metrics from report.

    Args:
        report: AuditReport from AuditRunner.
        k_values: K values for @K metrics.

    Returns:
        SearchMetrics with all computed values.
    """
    calc = MetricsCalculator()
    return calc.calculate(report, k_values)


__all__ = [
    # Data classes
    "SearchMetrics",
    # Functions
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "dcg_at_k",
    "ndcg_at_k",
    "hit_rate_at_k",
    "calculate_metrics",
    # Classes
    "MetricsCalculator",
]
