"""Query-Result Audit Framework for vector search quality evaluation.

Provides infrastructure to define test queries with expected results,
run them against a VectorStore, and evaluate search quality metrics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from ..lib import VectorSearchResult, VectorStore


class MatchCriterion(str, Enum):
    """Types of matching criteria for expected results."""

    CONTAINS_COMPONENT = "contains_component"  # Result has specific component type
    CONTAINS_TEXT = "contains_text"  # Serialized text contains substring
    MIN_SCORE = "min_score"  # Similarity score above threshold
    SOURCE_EQUALS = "source_equals"  # Result from specific source
    MAX_DEPTH_LE = "max_depth_le"  # Layout depth at most N
    MIN_NODES = "min_nodes"  # At least N nodes in layout
    EXACT_ID = "exact_id"  # Specific item ID expected


@dataclass
class ExpectedCriteria:
    """Single criterion for evaluating a search result.

    Attributes:
        criterion: Type of matching criterion.
        value: Expected value (type depends on criterion).
        weight: Importance weight for scoring (0.0 to 1.0).
        required: If True, result fails if criterion not met.
    """

    criterion: MatchCriterion
    value: str | float | int
    weight: float = 1.0
    required: bool = False


@dataclass
class AuditQuery:
    """Test query specification with expected results.

    Attributes:
        query: Natural language query string.
        name: Human-readable name for this test case.
        description: Detailed description of what this tests.
        expected: List of criteria the top results should match.
        expected_top_k: How many top results to evaluate.
        tags: Optional tags for categorization.
    """

    query: str
    name: str
    description: str = ""
    expected: list[ExpectedCriteria] = field(default_factory=list)
    expected_top_k: int = 5
    tags: list[str] = field(default_factory=list)


@dataclass
class CriterionResult:
    """Result of evaluating a single criterion.

    Attributes:
        criterion: The criterion that was evaluated.
        passed: Whether the criterion was satisfied.
        actual_value: The actual value found.
        score: Weighted score for this criterion (0.0 to weight).
    """

    criterion: ExpectedCriteria
    passed: bool
    actual_value: str | float | int | None
    score: float


@dataclass
class ResultEvaluation:
    """Evaluation of a single search result.

    Attributes:
        result: The VectorSearchResult being evaluated.
        rank: Position in search results (0-indexed).
        criterion_results: Evaluation of each criterion.
        total_score: Sum of all criterion scores.
        max_score: Maximum possible score.
        passed_required: All required criteria passed.
    """

    result: VectorSearchResult
    rank: int
    criterion_results: list[CriterionResult]
    total_score: float
    max_score: float
    passed_required: bool

    @property
    def score_percentage(self) -> float:
        """Score as percentage of maximum."""
        if self.max_score == 0:
            return 0.0
        return (self.total_score / self.max_score) * 100


@dataclass
class AuditResult:
    """Complete result of running an audit query.

    Attributes:
        query: The original audit query.
        results: Search results from VectorStore.
        evaluations: Per-result evaluations.
        query_time_ms: Time to execute query in milliseconds.
    """

    query: AuditQuery
    results: list[VectorSearchResult]
    evaluations: list[ResultEvaluation]
    query_time_ms: float

    @property
    def top_result_passed(self) -> bool:
        """Whether the top result passed all required criteria."""
        if not self.evaluations:
            return False
        return self.evaluations[0].passed_required

    @property
    def average_score(self) -> float:
        """Average score across evaluated results."""
        if not self.evaluations:
            return 0.0
        return sum(e.total_score for e in self.evaluations) / len(self.evaluations)

    @property
    def best_match_rank(self) -> int | None:
        """Rank of first result passing all required criteria."""
        for eval_ in self.evaluations:
            if eval_.passed_required:
                return eval_.rank
        return None


@dataclass
class AuditReport:
    """Summary report of all audit queries.

    Attributes:
        results: All audit results.
        total_queries: Number of queries run.
        passed_queries: Number where top result passed.
        average_query_time_ms: Average query execution time.
    """

    results: list[AuditResult]

    @property
    def total_queries(self) -> int:
        """Total number of audit queries."""
        return len(self.results)

    @property
    def passed_queries(self) -> int:
        """Number of queries where top result passed all required criteria."""
        return sum(1 for r in self.results if r.top_result_passed)

    @property
    def pass_rate(self) -> float:
        """Percentage of queries that passed."""
        if not self.results:
            return 0.0
        return (self.passed_queries / self.total_queries) * 100

    @property
    def average_query_time_ms(self) -> float:
        """Average query execution time."""
        if not self.results:
            return 0.0
        return sum(r.query_time_ms for r in self.results) / len(self.results)

    @property
    def average_best_rank(self) -> float | None:
        """Average rank of first passing result."""
        ranks = [
            r.best_match_rank for r in self.results if r.best_match_rank is not None
        ]
        if not ranks:
            return None
        return sum(ranks) / len(ranks)

    def by_tag(self, tag: str) -> "AuditReport":
        """Filter results to those with a specific tag."""
        filtered = [r for r in self.results if tag in r.query.tags]
        return AuditReport(results=filtered)


class CriterionEvaluator:
    """Evaluates individual criteria against search results."""

    def __init__(self, store: VectorStore):
        """Initialize evaluator with VectorStore.

        Args:
            store: VectorStore instance for metadata access.
        """
        self._store = store

    def evaluate(
        self, criterion: ExpectedCriteria, result: VectorSearchResult
    ) -> CriterionResult:
        """Evaluate a single criterion against a result.

        Args:
            criterion: Criterion to evaluate.
            result: Search result to check.

        Returns:
            CriterionResult with evaluation details.
        """
        match criterion.criterion:
            case MatchCriterion.CONTAINS_COMPONENT:
                return self._eval_contains_component(criterion, result)
            case MatchCriterion.CONTAINS_TEXT:
                return self._eval_contains_text(criterion, result)
            case MatchCriterion.MIN_SCORE:
                return self._eval_min_score(criterion, result)
            case MatchCriterion.SOURCE_EQUALS:
                return self._eval_source_equals(criterion, result)
            case MatchCriterion.MAX_DEPTH_LE:
                return self._eval_max_depth(criterion, result)
            case MatchCriterion.MIN_NODES:
                return self._eval_min_nodes(criterion, result)
            case MatchCriterion.EXACT_ID:
                return self._eval_exact_id(criterion, result)
            case _:
                return CriterionResult(
                    criterion=criterion,
                    passed=False,
                    actual_value=None,
                    score=0.0,
                )

    def _eval_contains_component(
        self, criterion: ExpectedCriteria, result: VectorSearchResult
    ) -> CriterionResult:
        """Check if result contains a component type."""
        meta = self._store.get_metadata(result.id)
        if not meta:
            return CriterionResult(criterion, False, None, 0.0)

        summary = meta.get("component_summary", {})
        component_type = str(criterion.value).lower()
        count = summary.get(component_type, 0)
        passed = count > 0

        return CriterionResult(
            criterion=criterion,
            passed=passed,
            actual_value=count,
            score=criterion.weight if passed else 0.0,
        )

    def _eval_contains_text(
        self, criterion: ExpectedCriteria, result: VectorSearchResult
    ) -> CriterionResult:
        """Check if serialized text contains substring."""
        text = result.serialized_text or ""
        substring = str(criterion.value).lower()
        passed = substring in text.lower()

        return CriterionResult(
            criterion=criterion,
            passed=passed,
            actual_value=substring if passed else None,
            score=criterion.weight if passed else 0.0,
        )

    def _eval_min_score(
        self, criterion: ExpectedCriteria, result: VectorSearchResult
    ) -> CriterionResult:
        """Check if similarity score meets threshold."""
        threshold = float(criterion.value)
        passed = result.score >= threshold

        return CriterionResult(
            criterion=criterion,
            passed=passed,
            actual_value=result.score,
            score=criterion.weight if passed else 0.0,
        )

    def _eval_source_equals(
        self, criterion: ExpectedCriteria, result: VectorSearchResult
    ) -> CriterionResult:
        """Check if result is from expected source."""
        expected_source = str(criterion.value)
        passed = result.source == expected_source

        return CriterionResult(
            criterion=criterion,
            passed=passed,
            actual_value=result.source,
            score=criterion.weight if passed else 0.0,
        )

    def _eval_max_depth(
        self, criterion: ExpectedCriteria, result: VectorSearchResult
    ) -> CriterionResult:
        """Check if layout depth is within limit."""
        meta = self._store.get_metadata(result.id)
        if not meta:
            return CriterionResult(criterion, False, None, 0.0)

        max_depth = meta.get("max_depth", 0)
        threshold = int(criterion.value)
        passed = max_depth <= threshold

        return CriterionResult(
            criterion=criterion,
            passed=passed,
            actual_value=max_depth,
            score=criterion.weight if passed else 0.0,
        )

    def _eval_min_nodes(
        self, criterion: ExpectedCriteria, result: VectorSearchResult
    ) -> CriterionResult:
        """Check if layout has minimum node count."""
        meta = self._store.get_metadata(result.id)
        if not meta:
            return CriterionResult(criterion, False, None, 0.0)

        node_count = meta.get("node_count", 0)
        threshold = int(criterion.value)
        passed = node_count >= threshold

        return CriterionResult(
            criterion=criterion,
            passed=passed,
            actual_value=node_count,
            score=criterion.weight if passed else 0.0,
        )

    def _eval_exact_id(
        self, criterion: ExpectedCriteria, result: VectorSearchResult
    ) -> CriterionResult:
        """Check if result has exact expected ID."""
        expected_id = str(criterion.value)
        passed = result.id == expected_id

        return CriterionResult(
            criterion=criterion,
            passed=passed,
            actual_value=result.id,
            score=criterion.weight if passed else 0.0,
        )


class AuditRunner:
    """Executes audit queries and generates reports.

    Example:
        >>> runner = AuditRunner(vector_store)
        >>> queries = [
        ...     AuditQuery(
        ...         query="login form with email and password",
        ...         name="login_form_basic",
        ...         expected=[
        ...             ExpectedCriteria(MatchCriterion.CONTAINS_COMPONENT, "input"),
        ...             ExpectedCriteria(MatchCriterion.CONTAINS_COMPONENT, "button"),
        ...         ],
        ...     ),
        ... ]
        >>> report = runner.run(queries)
        >>> print(f"Pass rate: {report.pass_rate:.1f}%")
    """

    def __init__(
        self,
        store: VectorStore,
        on_query_complete: Callable[[AuditResult], None] | None = None,
    ):
        """Initialize runner.

        Args:
            store: VectorStore to query.
            on_query_complete: Optional callback after each query.
        """
        self._store = store
        self._evaluator = CriterionEvaluator(store)
        self._on_complete = on_query_complete

    def run(self, queries: list[AuditQuery]) -> AuditReport:
        """Run all audit queries and generate report.

        Args:
            queries: List of audit queries to run.

        Returns:
            AuditReport with all results.
        """
        results: list[AuditResult] = []

        for query in queries:
            result = self._run_query(query)
            results.append(result)

            if self._on_complete:
                self._on_complete(result)

        return AuditReport(results=results)

    def _run_query(self, query: AuditQuery) -> AuditResult:
        """Run a single audit query.

        Args:
            query: Audit query to run.

        Returns:
            AuditResult with evaluation.
        """
        import time

        start = time.perf_counter()
        results = self._store.search(query.query, k=query.expected_top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000

        evaluations: list[ResultEvaluation] = []

        for rank, result in enumerate(results):
            evaluation = self._evaluate_result(query, result, rank)
            evaluations.append(evaluation)

        return AuditResult(
            query=query,
            results=results,
            evaluations=evaluations,
            query_time_ms=elapsed_ms,
        )

    def _evaluate_result(
        self, query: AuditQuery, result: VectorSearchResult, rank: int
    ) -> ResultEvaluation:
        """Evaluate a single result against query criteria.

        Args:
            query: The audit query with criteria.
            result: Search result to evaluate.
            rank: Position in results.

        Returns:
            ResultEvaluation with criterion scores.
        """
        criterion_results: list[CriterionResult] = []
        total_score = 0.0
        max_score = 0.0
        passed_required = True

        for criterion in query.expected:
            crit_result = self._evaluator.evaluate(criterion, result)
            criterion_results.append(crit_result)
            total_score += crit_result.score
            max_score += criterion.weight

            if criterion.required and not crit_result.passed:
                passed_required = False

        return ResultEvaluation(
            result=result,
            rank=rank,
            criterion_results=criterion_results,
            total_score=total_score,
            max_score=max_score,
            passed_required=passed_required,
        )


def create_component_query(
    query: str,
    name: str,
    components: list[str],
    *,
    min_score: float = 0.5,
    required_components: list[str] | None = None,
) -> AuditQuery:
    """Helper to create an audit query with component expectations.

    Args:
        query: Natural language query string.
        name: Test case name.
        components: Component types expected in results.
        min_score: Minimum similarity score threshold.
        required_components: Components that must be present (subset of components).

    Returns:
        AuditQuery with configured criteria.
    """
    required_set = set(required_components or [])
    expected = [
        ExpectedCriteria(
            MatchCriterion.MIN_SCORE,
            min_score,
            weight=0.5,
            required=True,
        )
    ]

    for comp in components:
        expected.append(
            ExpectedCriteria(
                MatchCriterion.CONTAINS_COMPONENT,
                comp,
                weight=1.0,
                required=comp in required_set,
            )
        )

    return AuditQuery(
        query=query,
        name=name,
        expected=expected,
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
    # Classes
    "CriterionEvaluator",
    "AuditRunner",
    # Helpers
    "create_component_query",
]
