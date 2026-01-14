"""Query-Result Audit Framework for vector search evaluation.

Provides infrastructure to define test queries with expected results,
run them against a VectorStore, and evaluate search quality.

Example:
    >>> from src.vector.audit import AuditRunner, AuditQuery, ExpectedCriteria, MatchCriterion
    >>> runner = AuditRunner(vector_store)
    >>> query = AuditQuery(
    ...     query="login form with email",
    ...     name="login_form",
    ...     expected=[
    ...         ExpectedCriteria(MatchCriterion.CONTAINS_COMPONENT, "input"),
    ...     ],
    ... )
    >>> report = runner.run([query])
    >>> print(f"Pass rate: {report.pass_rate:.1f}%")

    # Calculate search quality metrics
    >>> from src.vector.audit import calculate_metrics
    >>> metrics = calculate_metrics(report, k_values=[1, 3, 5])
    >>> print(metrics.summary())
"""

from .lib import (
    AuditQuery,
    AuditReport,
    AuditResult,
    AuditRunner,
    CriterionEvaluator,
    CriterionResult,
    ExpectedCriteria,
    MatchCriterion,
    ResultEvaluation,
    create_component_query,
)
from .metrics import (
    MetricsCalculator,
    SearchMetrics,
    calculate_metrics,
    dcg_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    # Enums
    "MatchCriterion",
    # Data classes
    "ExpectedCriteria",
    "AuditQuery",
    "CriterionResult",
    "ResultEvaluation",
    "AuditResult",
    "AuditReport",
    "SearchMetrics",
    # Classes
    "CriterionEvaluator",
    "AuditRunner",
    "MetricsCalculator",
    # Functions
    "create_component_query",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "dcg_at_k",
    "ndcg_at_k",
    "hit_rate_at_k",
    "calculate_metrics",
]
