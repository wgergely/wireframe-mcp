"""Tests for Query-Result Audit Framework."""

import pytest

from src.vector.lib import VectorSearchResult

from .lib import (
    AuditQuery,
    AuditReport,
    AuditResult,
    AuditRunner,
    CriterionEvaluator,
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


class MockVectorStore:
    """Mock VectorStore for testing."""

    def __init__(self, results: list[VectorSearchResult], metadata: dict):
        """Initialize with fixed results and metadata."""
        self._results = results
        self._metadata = metadata

    def search(self, query: str, k: int = 5) -> list[VectorSearchResult]:
        """Return pre-configured results."""
        return self._results[:k]

    def get_metadata(self, item_id: str) -> dict | None:
        """Return pre-configured metadata."""
        return self._metadata.get(item_id)


class TestMatchCriterion:
    """Tests for MatchCriterion enum."""

    @pytest.mark.unit
    def test_criterion_values(self):
        """All criterion types have expected values."""
        assert MatchCriterion.CONTAINS_COMPONENT.value == "contains_component"
        assert MatchCriterion.CONTAINS_TEXT.value == "contains_text"
        assert MatchCriterion.MIN_SCORE.value == "min_score"
        assert MatchCriterion.SOURCE_EQUALS.value == "source_equals"


class TestExpectedCriteria:
    """Tests for ExpectedCriteria dataclass."""

    @pytest.mark.unit
    def test_criteria_creation(self):
        """Criteria can be created with required fields."""
        criteria = ExpectedCriteria(
            criterion=MatchCriterion.CONTAINS_COMPONENT,
            value="button",
        )
        assert criteria.criterion == MatchCriterion.CONTAINS_COMPONENT
        assert criteria.value == "button"
        assert criteria.weight == 1.0
        assert criteria.required is False

    @pytest.mark.unit
    def test_criteria_with_all_fields(self):
        """Criteria supports all optional fields."""
        criteria = ExpectedCriteria(
            criterion=MatchCriterion.MIN_SCORE,
            value=0.8,
            weight=2.0,
            required=True,
        )
        assert criteria.value == 0.8
        assert criteria.weight == 2.0
        assert criteria.required is True


class TestAuditQuery:
    """Tests for AuditQuery dataclass."""

    @pytest.mark.unit
    def test_query_creation(self):
        """Query can be created with basic fields."""
        query = AuditQuery(
            query="login form",
            name="test_login",
        )
        assert query.query == "login form"
        assert query.name == "test_login"
        assert query.expected == []
        assert query.expected_top_k == 5

    @pytest.mark.unit
    def test_query_with_criteria(self):
        """Query can include expected criteria."""
        query = AuditQuery(
            query="login form",
            name="test_login",
            expected=[
                ExpectedCriteria(MatchCriterion.CONTAINS_COMPONENT, "input"),
                ExpectedCriteria(MatchCriterion.CONTAINS_COMPONENT, "button"),
            ],
            expected_top_k=3,
            tags=["auth", "forms"],
        )
        assert len(query.expected) == 2
        assert query.expected_top_k == 3
        assert "auth" in query.tags


class TestCriterionEvaluator:
    """Tests for CriterionEvaluator."""

    @pytest.fixture
    def mock_store(self):
        """Create mock store with test data."""
        results = [
            VectorSearchResult(
                id="item_1",
                score=0.85,
                rank=0,
                source="rico",
                dataset="semantic",
                serialized_text="[CONTROL:button] submit Login",
            ),
        ]
        metadata = {
            "item_1": {
                "source": "rico",
                "dataset": "semantic",
                "node_count": 10,
                "max_depth": 3,
                "component_summary": {"button": 2, "input": 3, "container": 1},
            },
        }
        return MockVectorStore(results, metadata)

    @pytest.fixture
    def evaluator(self, mock_store):
        """Create evaluator with mock store."""
        return CriterionEvaluator(mock_store)

    @pytest.fixture
    def sample_result(self):
        """Sample search result for testing."""
        return VectorSearchResult(
            id="item_1",
            score=0.85,
            rank=0,
            source="rico",
            dataset="semantic",
            serialized_text="[CONTROL:button] submit Login",
        )

    @pytest.mark.unit
    def test_eval_contains_component_pass(self, evaluator, sample_result):
        """Contains component passes when component exists."""
        criterion = ExpectedCriteria(MatchCriterion.CONTAINS_COMPONENT, "button")
        result = evaluator.evaluate(criterion, sample_result)

        assert result.passed is True
        assert result.actual_value == 2
        assert result.score == 1.0

    @pytest.mark.unit
    def test_eval_contains_component_fail(self, evaluator, sample_result):
        """Contains component fails when component missing."""
        criterion = ExpectedCriteria(MatchCriterion.CONTAINS_COMPONENT, "checkbox")
        result = evaluator.evaluate(criterion, sample_result)

        assert result.passed is False
        assert result.actual_value == 0
        assert result.score == 0.0

    @pytest.mark.unit
    def test_eval_contains_text_pass(self, evaluator, sample_result):
        """Contains text passes when substring found."""
        criterion = ExpectedCriteria(MatchCriterion.CONTAINS_TEXT, "Login")
        result = evaluator.evaluate(criterion, sample_result)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.unit
    def test_eval_contains_text_case_insensitive(self, evaluator, sample_result):
        """Contains text is case insensitive."""
        criterion = ExpectedCriteria(MatchCriterion.CONTAINS_TEXT, "login")
        result = evaluator.evaluate(criterion, sample_result)

        assert result.passed is True

    @pytest.mark.unit
    def test_eval_min_score_pass(self, evaluator, sample_result):
        """Min score passes when score above threshold."""
        criterion = ExpectedCriteria(MatchCriterion.MIN_SCORE, 0.8)
        result = evaluator.evaluate(criterion, sample_result)

        assert result.passed is True
        assert result.actual_value == 0.85

    @pytest.mark.unit
    def test_eval_min_score_fail(self, evaluator, sample_result):
        """Min score fails when score below threshold."""
        criterion = ExpectedCriteria(MatchCriterion.MIN_SCORE, 0.9)
        result = evaluator.evaluate(criterion, sample_result)

        assert result.passed is False

    @pytest.mark.unit
    def test_eval_source_equals_pass(self, evaluator, sample_result):
        """Source equals passes when source matches."""
        criterion = ExpectedCriteria(MatchCriterion.SOURCE_EQUALS, "rico")
        result = evaluator.evaluate(criterion, sample_result)

        assert result.passed is True
        assert result.actual_value == "rico"

    @pytest.mark.unit
    def test_eval_max_depth(self, evaluator, sample_result):
        """Max depth passes when depth within limit."""
        criterion = ExpectedCriteria(MatchCriterion.MAX_DEPTH_LE, 5)
        result = evaluator.evaluate(criterion, sample_result)

        assert result.passed is True
        assert result.actual_value == 3

    @pytest.mark.unit
    def test_eval_min_nodes(self, evaluator, sample_result):
        """Min nodes passes when node count sufficient."""
        criterion = ExpectedCriteria(MatchCriterion.MIN_NODES, 5)
        result = evaluator.evaluate(criterion, sample_result)

        assert result.passed is True
        assert result.actual_value == 10

    @pytest.mark.unit
    def test_eval_exact_id(self, evaluator, sample_result):
        """Exact ID passes when ID matches."""
        criterion = ExpectedCriteria(MatchCriterion.EXACT_ID, "item_1")
        result = evaluator.evaluate(criterion, sample_result)

        assert result.passed is True


class TestResultEvaluation:
    """Tests for ResultEvaluation dataclass."""

    @pytest.mark.unit
    def test_score_percentage(self):
        """Score percentage calculated correctly."""
        result = VectorSearchResult(
            id="test", score=0.8, rank=0, source="test", dataset="test"
        )
        evaluation = ResultEvaluation(
            result=result,
            rank=0,
            criterion_results=[],
            total_score=3.0,
            max_score=4.0,
            passed_required=True,
        )
        assert evaluation.score_percentage == 75.0

    @pytest.mark.unit
    def test_score_percentage_zero_max(self):
        """Score percentage handles zero max score."""
        result = VectorSearchResult(
            id="test", score=0.8, rank=0, source="test", dataset="test"
        )
        evaluation = ResultEvaluation(
            result=result,
            rank=0,
            criterion_results=[],
            total_score=0.0,
            max_score=0.0,
            passed_required=True,
        )
        assert evaluation.score_percentage == 0.0


class TestAuditResult:
    """Tests for AuditResult dataclass."""

    @pytest.fixture
    def sample_audit_result(self):
        """Create sample audit result."""
        query = AuditQuery(query="test", name="test")
        results = [
            VectorSearchResult(
                id="item_1", score=0.9, rank=0, source="test", dataset="test"
            ),
            VectorSearchResult(
                id="item_2", score=0.7, rank=1, source="test", dataset="test"
            ),
        ]
        evaluations = [
            ResultEvaluation(
                result=results[0],
                rank=0,
                criterion_results=[],
                total_score=2.0,
                max_score=2.0,
                passed_required=True,
            ),
            ResultEvaluation(
                result=results[1],
                rank=1,
                criterion_results=[],
                total_score=1.0,
                max_score=2.0,
                passed_required=False,
            ),
        ]
        return AuditResult(
            query=query,
            results=results,
            evaluations=evaluations,
            query_time_ms=10.5,
        )

    @pytest.mark.unit
    def test_top_result_passed(self, sample_audit_result):
        """Top result passed property works."""
        assert sample_audit_result.top_result_passed is True

    @pytest.mark.unit
    def test_average_score(self, sample_audit_result):
        """Average score calculated correctly."""
        assert sample_audit_result.average_score == 1.5

    @pytest.mark.unit
    def test_best_match_rank(self, sample_audit_result):
        """Best match rank found correctly."""
        assert sample_audit_result.best_match_rank == 0


class TestAuditReport:
    """Tests for AuditReport dataclass."""

    @pytest.fixture
    def sample_report(self):
        """Create sample report with mixed results."""
        query1 = AuditQuery(query="test1", name="test1", tags=["forms"])
        query2 = AuditQuery(query="test2", name="test2", tags=["navigation"])
        query3 = AuditQuery(query="test3", name="test3", tags=["forms"])

        results = []
        for i, query in enumerate([query1, query2, query3]):
            passed = i != 1  # Second query fails
            search_result = VectorSearchResult(
                id=f"item_{i}", score=0.8, rank=0, source="test", dataset="test"
            )
            evaluation = ResultEvaluation(
                result=search_result,
                rank=0,
                criterion_results=[],
                total_score=1.0,
                max_score=1.0,
                passed_required=passed,
            )
            results.append(
                AuditResult(
                    query=query,
                    results=[search_result],
                    evaluations=[evaluation],
                    query_time_ms=10.0,
                )
            )
        return AuditReport(results=results)

    @pytest.mark.unit
    def test_total_queries(self, sample_report):
        """Total queries counted correctly."""
        assert sample_report.total_queries == 3

    @pytest.mark.unit
    def test_passed_queries(self, sample_report):
        """Passed queries counted correctly."""
        assert sample_report.passed_queries == 2

    @pytest.mark.unit
    def test_pass_rate(self, sample_report):
        """Pass rate calculated correctly."""
        assert abs(sample_report.pass_rate - 66.67) < 0.1

    @pytest.mark.unit
    def test_average_query_time(self, sample_report):
        """Average query time calculated correctly."""
        assert sample_report.average_query_time_ms == 10.0

    @pytest.mark.unit
    def test_by_tag(self, sample_report):
        """Filter by tag works correctly."""
        forms_report = sample_report.by_tag("forms")
        assert forms_report.total_queries == 2
        assert forms_report.passed_queries == 2


class TestAuditRunner:
    """Tests for AuditRunner."""

    @pytest.fixture
    def mock_store(self):
        """Create mock store with test data."""
        results = [
            VectorSearchResult(
                id="item_1",
                score=0.85,
                rank=0,
                source="rico",
                dataset="semantic",
                serialized_text="[CONTROL:button] submit Login",
            ),
            VectorSearchResult(
                id="item_2",
                score=0.75,
                rank=1,
                source="rico",
                dataset="semantic",
                serialized_text="[CONTROL:input] email",
            ),
        ]
        metadata = {
            "item_1": {
                "component_summary": {"button": 1},
                "node_count": 5,
                "max_depth": 2,
            },
            "item_2": {
                "component_summary": {"input": 1},
                "node_count": 3,
                "max_depth": 1,
            },
        }
        return MockVectorStore(results, metadata)

    @pytest.mark.unit
    def test_run_single_query(self, mock_store):
        """Runner executes single query correctly."""
        runner = AuditRunner(mock_store)
        query = AuditQuery(
            query="login button",
            name="test_login",
            expected=[
                ExpectedCriteria(MatchCriterion.CONTAINS_COMPONENT, "button"),
            ],
        )

        report = runner.run([query])

        assert report.total_queries == 1
        assert report.passed_queries == 1
        assert len(report.results[0].evaluations) == 2

    @pytest.mark.unit
    def test_run_with_callback(self, mock_store):
        """Runner calls callback after each query."""
        completed = []

        def on_complete(result):
            completed.append(result.query.name)

        runner = AuditRunner(mock_store, on_query_complete=on_complete)
        queries = [
            AuditQuery(query="test1", name="query1"),
            AuditQuery(query="test2", name="query2"),
        ]

        runner.run(queries)

        assert completed == ["query1", "query2"]

    @pytest.mark.unit
    def test_run_respects_k(self, mock_store):
        """Runner respects expected_top_k."""
        runner = AuditRunner(mock_store)
        query = AuditQuery(
            query="test",
            name="test",
            expected_top_k=1,
        )

        report = runner.run([query])

        assert len(report.results[0].results) == 1

    @pytest.mark.unit
    def test_required_criteria_affects_pass(self, mock_store):
        """Required criteria failing causes result to fail."""
        runner = AuditRunner(mock_store)
        query = AuditQuery(
            query="test",
            name="test",
            expected=[
                ExpectedCriteria(
                    MatchCriterion.CONTAINS_COMPONENT,
                    "checkbox",  # Not in results
                    required=True,
                ),
            ],
        )

        report = runner.run([query])

        assert report.passed_queries == 0


class TestCreateComponentQuery:
    """Tests for create_component_query helper."""

    @pytest.mark.unit
    def test_creates_query_with_components(self):
        """Helper creates query with component criteria."""
        query = create_component_query(
            query="login form",
            name="test_login",
            components=["button", "input"],
        )

        assert query.query == "login form"
        assert query.name == "test_login"
        assert len(query.expected) == 3  # min_score + 2 components

    @pytest.mark.unit
    def test_creates_query_with_required_components(self):
        """Helper marks required components correctly."""
        query = create_component_query(
            query="login form",
            name="test_login",
            components=["button", "input", "text"],
            required_components=["button"],
        )

        # Find the button criterion
        button_crit = None
        for crit in query.expected:
            if (
                crit.criterion == MatchCriterion.CONTAINS_COMPONENT
                and crit.value == "button"
            ):
                button_crit = crit
                break

        assert button_crit is not None
        assert button_crit.required is True

    @pytest.mark.unit
    def test_creates_query_with_custom_min_score(self):
        """Helper uses custom min_score."""
        query = create_component_query(
            query="test",
            name="test",
            components=[],
            min_score=0.7,
        )

        min_score_crit = query.expected[0]
        assert min_score_crit.criterion == MatchCriterion.MIN_SCORE
        assert min_score_crit.value == 0.7


# ============================================================================
# Search Quality Metrics Tests
# ============================================================================


class TestPrecisionAtK:
    """Tests for precision_at_k function."""

    @pytest.mark.unit
    def test_all_relevant(self):
        """Precision is 1.0 when all results are relevant."""
        relevant = [True, True, True, True, True]
        assert precision_at_k(relevant, 5) == 1.0

    @pytest.mark.unit
    def test_none_relevant(self):
        """Precision is 0.0 when no results are relevant."""
        relevant = [False, False, False, False, False]
        assert precision_at_k(relevant, 5) == 0.0

    @pytest.mark.unit
    def test_partial_relevant(self):
        """Precision calculated correctly for mixed results."""
        relevant = [True, False, True, False, False]
        assert precision_at_k(relevant, 5) == 0.4

    @pytest.mark.unit
    def test_k_smaller_than_results(self):
        """K limits calculation to top K."""
        relevant = [True, True, False, False, False]
        assert precision_at_k(relevant, 2) == 1.0

    @pytest.mark.unit
    def test_k_zero(self):
        """K=0 returns 0.0."""
        relevant = [True, True, True]
        assert precision_at_k(relevant, 0) == 0.0

    @pytest.mark.unit
    def test_empty_results(self):
        """Empty results returns 0.0."""
        assert precision_at_k([], 5) == 0.0


class TestRecallAtK:
    """Tests for recall_at_k function."""

    @pytest.mark.unit
    def test_all_found(self):
        """Recall is 1.0 when all relevant found in top K."""
        relevant = [True, True, True, False, False]
        assert recall_at_k(relevant, 5, total_relevant=3) == 1.0

    @pytest.mark.unit
    def test_partial_found(self):
        """Recall calculated correctly when some relevant found."""
        relevant = [True, False, True, False, False]
        assert recall_at_k(relevant, 5, total_relevant=4) == 0.5

    @pytest.mark.unit
    def test_none_found(self):
        """Recall is 0.0 when no relevant found."""
        relevant = [False, False, False]
        assert recall_at_k(relevant, 3, total_relevant=5) == 0.0

    @pytest.mark.unit
    def test_zero_total_relevant(self):
        """Zero total relevant returns 0.0."""
        relevant = [True, True]
        assert recall_at_k(relevant, 2, total_relevant=0) == 0.0


class TestReciprocalRank:
    """Tests for reciprocal_rank function."""

    @pytest.mark.unit
    def test_first_is_relevant(self):
        """RR is 1.0 when first result is relevant."""
        relevant = [True, False, False]
        assert reciprocal_rank(relevant) == 1.0

    @pytest.mark.unit
    def test_second_is_relevant(self):
        """RR is 0.5 when second result is first relevant."""
        relevant = [False, True, False]
        assert reciprocal_rank(relevant) == 0.5

    @pytest.mark.unit
    def test_third_is_relevant(self):
        """RR is 1/3 when third result is first relevant."""
        relevant = [False, False, True]
        assert abs(reciprocal_rank(relevant) - 1 / 3) < 0.001

    @pytest.mark.unit
    def test_none_relevant(self):
        """RR is 0.0 when no results are relevant."""
        relevant = [False, False, False]
        assert reciprocal_rank(relevant) == 0.0


class TestDCGAtK:
    """Tests for dcg_at_k function."""

    @pytest.mark.unit
    def test_perfect_relevance(self):
        """DCG calculated for perfect relevance."""
        # All scores 1.0
        scores = [1.0, 1.0, 1.0]
        dcg = dcg_at_k(scores, 3)
        # DCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.63 + 0.5
        assert dcg > 2.0

    @pytest.mark.unit
    def test_decreasing_relevance(self):
        """DCG higher when high relevance at top."""
        high_first = [1.0, 0.5, 0.0]
        low_first = [0.0, 0.5, 1.0]

        dcg_high = dcg_at_k(high_first, 3)
        dcg_low = dcg_at_k(low_first, 3)

        assert dcg_high > dcg_low

    @pytest.mark.unit
    def test_k_zero(self):
        """K=0 returns 0.0."""
        assert dcg_at_k([1.0, 1.0], 0) == 0.0


class TestNDCGAtK:
    """Tests for ndcg_at_k function."""

    @pytest.mark.unit
    def test_perfect_ordering(self):
        """NDCG is 1.0 for perfect ordering."""
        # Already sorted descending
        scores = [1.0, 0.8, 0.6, 0.4, 0.2]
        assert abs(ndcg_at_k(scores, 5) - 1.0) < 0.001

    @pytest.mark.unit
    def test_worst_ordering(self):
        """NDCG is low for reversed ordering."""
        # Worst possible order (ascending)
        scores = [0.2, 0.4, 0.6, 0.8, 1.0]
        ndcg = ndcg_at_k(scores, 5)
        # Should be less than 1.0
        assert ndcg < 1.0

    @pytest.mark.unit
    def test_all_zeros(self):
        """NDCG is 0.0 when all scores are zero."""
        scores = [0.0, 0.0, 0.0]
        assert ndcg_at_k(scores, 3) == 0.0


class TestHitRateAtK:
    """Tests for hit_rate_at_k function."""

    @pytest.mark.unit
    def test_hit_found(self):
        """Hit rate is 1.0 when any relevant in top K."""
        relevant = [False, False, True, False]
        assert hit_rate_at_k(relevant, 4) == 1.0

    @pytest.mark.unit
    def test_no_hit(self):
        """Hit rate is 0.0 when no relevant in top K."""
        relevant = [False, False, False, False]
        assert hit_rate_at_k(relevant, 4) == 0.0

    @pytest.mark.unit
    def test_hit_outside_k(self):
        """Hit rate is 0.0 when relevant outside top K."""
        relevant = [False, False, False, True]
        assert hit_rate_at_k(relevant, 3) == 0.0


class TestSearchMetrics:
    """Tests for SearchMetrics dataclass."""

    @pytest.mark.unit
    def test_summary_generation(self):
        """Summary generates readable output."""
        metrics = SearchMetrics(
            precision_at_k={1: 0.8, 5: 0.6},
            mrr=0.75,
            ndcg_at_k={1: 0.9, 5: 0.7},
            hit_rate_at_k={1: 0.9, 5: 1.0},
            query_count=10,
        )

        summary = metrics.summary()

        assert "MRR: 0.7500" in summary
        assert "Precision@K:" in summary
        assert "@1: 0.8000" in summary
        assert "n=10" in summary


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    @pytest.fixture
    def sample_report(self):
        """Create sample report for metrics testing."""
        results = []

        # Query 1: First result passes, second fails
        query1 = AuditQuery(query="test1", name="query1")
        search_results1 = [
            VectorSearchResult(
                id=f"item_1_{i}",
                score=0.9 - i * 0.1,
                rank=i,
                source="test",
                dataset="test",
            )
            for i in range(3)
        ]
        evaluations1 = [
            ResultEvaluation(
                result=sr,
                rank=i,
                criterion_results=[],
                total_score=1.0 if i == 0 else 0.0,
                max_score=1.0,
                passed_required=i == 0,
            )
            for i, sr in enumerate(search_results1)
        ]
        results.append(
            AuditResult(
                query=query1,
                results=search_results1,
                evaluations=evaluations1,
                query_time_ms=10.0,
            )
        )

        # Query 2: Second result passes
        query2 = AuditQuery(query="test2", name="query2")
        search_results2 = [
            VectorSearchResult(
                id=f"item_2_{i}",
                score=0.8 - i * 0.1,
                rank=i,
                source="test",
                dataset="test",
            )
            for i in range(3)
        ]
        evaluations2 = [
            ResultEvaluation(
                result=sr,
                rank=i,
                criterion_results=[],
                total_score=1.0 if i == 1 else 0.0,
                max_score=1.0,
                passed_required=i == 1,
            )
            for i, sr in enumerate(search_results2)
        ]
        results.append(
            AuditResult(
                query=query2,
                results=search_results2,
                evaluations=evaluations2,
                query_time_ms=10.0,
            )
        )

        return AuditReport(results=results)

    @pytest.mark.unit
    def test_calculate_mrr(self, sample_report):
        """MRR calculated correctly."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_report)

        # Query 1: first relevant at rank 0 -> RR = 1.0
        # Query 2: first relevant at rank 1 -> RR = 0.5
        # MRR = (1.0 + 0.5) / 2 = 0.75
        assert abs(metrics.mrr - 0.75) < 0.001

    @pytest.mark.unit
    def test_calculate_precision(self, sample_report):
        """Precision@K calculated correctly."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_report, k_values=[1, 3])

        # P@1: Query 1 = 1/1, Query 2 = 0/1 -> avg = 0.5
        assert abs(metrics.precision_at_k[1] - 0.5) < 0.001

        # P@3: Query 1 = 1/3, Query 2 = 1/3 -> avg = 1/3
        assert abs(metrics.precision_at_k[3] - 1 / 3) < 0.001

    @pytest.mark.unit
    def test_calculate_hit_rate(self, sample_report):
        """Hit rate@K calculated correctly."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_report, k_values=[1, 3])

        # HR@1: Query 1 = 1, Query 2 = 0 -> avg = 0.5
        assert abs(metrics.hit_rate_at_k[1] - 0.5) < 0.001

        # HR@3: Both have a hit -> avg = 1.0
        assert metrics.hit_rate_at_k[3] == 1.0

    @pytest.mark.unit
    def test_custom_relevance_fn(self, sample_report):
        """Custom relevance function works."""
        # Always consider relevant (should give perfect scores)
        calc = MetricsCalculator(relevance_fn=lambda e: True)
        metrics = calc.calculate(sample_report, k_values=[3])

        assert metrics.precision_at_k[3] == 1.0
        assert metrics.hit_rate_at_k[3] == 1.0

    @pytest.mark.unit
    def test_empty_report(self):
        """Empty report returns zero metrics."""
        calc = MetricsCalculator()
        metrics = calc.calculate(AuditReport(results=[]))

        assert metrics.query_count == 0
        assert metrics.mrr == 0.0


class TestCalculateMetrics:
    """Tests for calculate_metrics convenience function."""

    @pytest.mark.unit
    def test_convenience_function(self):
        """Convenience function works like MetricsCalculator."""
        query = AuditQuery(query="test", name="test")
        search_result = VectorSearchResult(
            id="item_1", score=0.9, rank=0, source="test", dataset="test"
        )
        evaluation = ResultEvaluation(
            result=search_result,
            rank=0,
            criterion_results=[],
            total_score=1.0,
            max_score=1.0,
            passed_required=True,
        )
        audit_result = AuditResult(
            query=query,
            results=[search_result],
            evaluations=[evaluation],
            query_time_ms=10.0,
        )
        report = AuditReport(results=[audit_result])

        metrics = calculate_metrics(report, k_values=[1])

        assert metrics.mrr == 1.0
        assert metrics.precision_at_k[1] == 1.0
