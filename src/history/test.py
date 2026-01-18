"""Tests for history management module.

Tests cover:
- Data models and serialization
- HistoryManager operations (CRUD)
- Storage size management and cleanup
- Variation generation and diversity metrics
- Session management
"""

import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

from .lib import HistoryManager
from .models import (
    ArtifactStatus,
    GenerationArtifact,
    GenerationStats,
    Session,
    StorageConfig,
    VariationRequest,
)
from .variations import (
    calculate_structural_diversity,
    calculate_temperature_spread,
    compare_artifacts,
    generate_rankings,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config():
    """Create test storage config with small limits."""
    return StorageConfig(
        max_total_size_mb=10,
        max_artifacts=100,
        max_artifact_age_days=30,
        max_session_artifacts=20,
        cleanup_batch_size=10,
        orphan_grace_period_hours=1,
        preview_cache_enabled=False,
    )


@pytest.fixture
def manager(temp_dir, config):
    """Create a HistoryManager with test config."""
    db_path = temp_dir / "test_history.db"
    mgr = HistoryManager(db_path=db_path, config=config, auto_cleanup=False)
    yield mgr
    mgr.close()


@pytest.fixture
def sample_layout():
    """Create a sample layout dict."""
    return {
        "type": "container",
        "id": "root",
        "orientation": "vertical",
        "children": [
            {"type": "header", "id": "header1", "label": "My App"},
            {
                "type": "card",
                "id": "card1",
                "children": [
                    {"type": "input", "id": "email", "label": "Email"},
                    {"type": "input", "id": "password", "label": "Password"},
                    {"type": "button", "id": "submit", "label": "Login"},
                ],
            },
        ],
    }


@pytest.fixture
def sample_stats():
    """Create sample generation stats."""
    return GenerationStats(
        attempts=1,
        validation_retries=0,
        json_repairs=0,
        total_tokens=500,
        final_model="gpt-4.1-mini",
    )


# =============================================================================
# Model Tests
# =============================================================================


class TestModels:
    """Tests for data models."""

    def test_generation_stats_defaults(self):
        """Test GenerationStats default values."""
        stats = GenerationStats()
        assert stats.attempts == 1
        assert stats.validation_retries == 0
        assert stats.json_repairs == 0
        assert stats.total_tokens == 0
        assert stats.final_model == ""

    def test_session_creation_via_factory(self):
        """Test Session model creation via factory."""
        session = Session.create(name="test-session")
        assert session.id is not None
        assert session.name == "test-session"
        assert session.artifact_count == 0
        assert session.total_tokens == 0
        assert isinstance(session.created_at, datetime)

    def test_generation_artifact_creation_via_factory(self, sample_layout):
        """Test GenerationArtifact model creation via factory."""
        artifact = GenerationArtifact.create(
            session_id="test-session",
            query="login form",
            layout=sample_layout,
            draft="Login [container]\n├── Header",
            model="gpt-4.1-mini",
            temperature=0.7,
            provider="d2",
        )
        assert artifact.id is not None
        assert artifact.query == "login form"
        assert artifact.layout == sample_layout
        assert artifact.status == ArtifactStatus.ACTIVE
        assert artifact.size_bytes > 0

    def test_storage_config_defaults(self):
        """Test StorageConfig default values."""
        config = StorageConfig()
        assert config.max_total_size_mb == 500
        assert config.max_artifacts == 10000
        assert config.max_artifact_age_days == 90

    def test_variation_request_creation(self):
        """Test VariationRequest model creation."""
        request = VariationRequest(query="dashboard", count=5)
        assert request.query == "dashboard"
        assert request.count == 5
        assert request.temperature_range is None
        assert request.include_rag is True


# =============================================================================
# HistoryManager Tests
# =============================================================================


class TestHistoryManager:
    """Tests for HistoryManager operations."""

    def test_create_session(self, manager):
        """Test session creation."""
        session = manager.get_or_create_session("test-session", "Test description")
        assert session.name == "test-session"
        assert session.description == "Test description"

        # Getting same session should return existing
        session2 = manager.get_or_create_session("test-session")
        assert session2.id == session.id

    def test_store_artifact(self, manager, sample_layout, sample_stats):
        """Test artifact storage."""
        session = manager.get_or_create_session("test-session")

        artifact = manager.store_artifact(
            query="login form",
            layout=sample_layout,
            draft="Login [container]",
            session_id=session.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            provider="d2",
            stats=sample_stats,
        )

        assert artifact.id is not None
        assert artifact.query == "login form"
        assert artifact.session_id == session.id

    def test_get_artifact(self, manager, sample_layout, sample_stats):
        """Test artifact retrieval."""
        session = manager.get_or_create_session("test-session")
        stored = manager.store_artifact(
            query="login form",
            layout=sample_layout,
            draft="Login [container]",
            session_id=session.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        retrieved = manager.get_artifact(stored.id)
        assert retrieved is not None
        assert retrieved.id == stored.id
        assert retrieved.query == "login form"

    def test_get_nonexistent_artifact(self, manager):
        """Test retrieval of nonexistent artifact."""
        artifact = manager.get_artifact("nonexistent-id")
        assert artifact is None

    def test_list_artifacts(self, manager, sample_layout, sample_stats):
        """Test artifact listing."""
        session = manager.get_or_create_session("test-session")

        # Store multiple artifacts
        for i in range(5):
            manager.store_artifact(
                query=f"query {i}",
                layout=sample_layout,
                draft=f"Draft {i}",
                session_id=session.id,
                model="gpt-4.1-mini",
                temperature=0.7,
                stats=sample_stats,
            )

        artifacts = manager.list_artifacts(session_id=session.id)
        assert len(artifacts) == 5

        # Test limit
        limited = manager.list_artifacts(session_id=session.id, limit=3)
        assert len(limited) == 3

    def test_delete_artifact(self, manager, sample_layout, sample_stats):
        """Test artifact deletion."""
        session = manager.get_or_create_session("test-session")
        artifact = manager.store_artifact(
            query="login form",
            layout=sample_layout,
            draft="Login [container]",
            session_id=session.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        assert manager.delete_artifact(artifact.id) is True
        assert manager.get_artifact(artifact.id) is None

    def test_lineage_tracking(self, manager, sample_layout, sample_stats):
        """Test parent-child relationship tracking."""
        session = manager.get_or_create_session("test-session")

        parent = manager.store_artifact(
            query="base form",
            layout=sample_layout,
            draft="Base",
            session_id=session.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        child = manager.store_artifact(
            query="refined form",
            layout=sample_layout,
            draft="Refined",
            session_id=session.id,
            parent_id=parent.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        lineage = manager.get_lineage(child.id)
        assert parent.id in [a.id for a in lineage["ancestors"]]

        parent_lineage = manager.get_lineage(parent.id)
        assert child.id in [a.id for a in parent_lineage["descendants"]]

    def test_get_stats(self, manager, sample_layout, sample_stats):
        """Test storage statistics."""
        session = manager.get_or_create_session("test-session")
        manager.store_artifact(
            query="test",
            layout=sample_layout,
            draft="Test",
            session_id=session.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        stats = manager.get_stats()
        assert stats.artifact_count == 1
        assert stats.session_count == 1
        assert stats.total_size_bytes > 0


# =============================================================================
# Cleanup Tests
# =============================================================================


class TestCleanup:
    """Tests for storage cleanup functionality."""

    def test_cleanup_empty_storage(self, manager):
        """Test cleanup on empty storage."""
        result = manager.cleanup()
        assert result.artifacts_deleted == 0
        assert result.sessions_deleted == 0

    def test_cleanup_respects_config(self, temp_dir, sample_layout, sample_stats):
        """Test that cleanup respects config limits."""
        config = StorageConfig(
            max_artifacts=5,
            max_artifact_age_days=30,
            orphan_grace_period_hours=0,  # Immediate orphan detection
        )
        db_path = temp_dir / "cleanup_test.db"
        mgr = HistoryManager(db_path=db_path, config=config, auto_cleanup=False)

        session = mgr.get_or_create_session("test-session")

        # Store more than limit
        for i in range(10):
            mgr.store_artifact(
                query=f"query {i}",
                layout=sample_layout,
                draft=f"Draft {i}",
                session_id=session.id,
                model="gpt-4.1-mini",
                temperature=0.7,
                stats=sample_stats,
            )

        result = mgr.cleanup(enforce_count_limit=True)
        assert result.artifacts_deleted >= 5  # Should delete excess

        mgr.close()


# =============================================================================
# Variation Tests
# =============================================================================


class TestVariations:
    """Tests for variation generation functionality."""

    def test_temperature_spread_single(self):
        """Test temperature spread for single variation."""
        temps = calculate_temperature_spread(1)
        assert len(temps) == 1
        assert temps[0] == 0.75  # Middle of default range

    def test_temperature_spread_multiple(self):
        """Test temperature spread for multiple variations."""
        temps = calculate_temperature_spread(5)
        assert len(temps) == 5
        assert temps[0] == 0.5  # Start of default range
        assert temps[-1] == 1.0  # End of default range

    def test_temperature_spread_custom_range(self):
        """Test temperature spread with custom range."""
        temps = calculate_temperature_spread(3, (0.2, 0.8))
        assert len(temps) == 3
        assert temps[0] == 0.2
        assert temps[-1] == 0.8

    def test_structural_diversity_identical(self, sample_layout):
        """Test diversity score for identical layouts."""
        artifacts = [
            GenerationArtifact.create(
                session_id="test",
                query="test",
                layout=sample_layout,
                draft="Test",
                model="gpt-4.1-mini",
                temperature=0.7,
            )
            for _ in range(3)
        ]

        diversity = calculate_structural_diversity(artifacts)
        assert diversity == 0.0  # Identical layouts

    def test_structural_diversity_different(self):
        """Test diversity score for different layouts."""
        layout1 = {
            "type": "container",
            "children": [{"type": "button"}],
        }
        layout2 = {
            "type": "card",
            "orientation": "horizontal",
            "children": [
                {"type": "input"},
                {"type": "input"},
                {"type": "checkbox"},
            ],
        }

        artifacts = [
            GenerationArtifact.create(
                session_id="test",
                query="test",
                layout=layout1,
                draft="Test 1",
                model="gpt-4.1-mini",
                temperature=0.5,
            ),
            GenerationArtifact.create(
                session_id="test",
                query="test",
                layout=layout2,
                draft="Test 2",
                model="gpt-4.1-mini",
                temperature=1.0,
            ),
        ]

        diversity = calculate_structural_diversity(artifacts)
        assert diversity > 0.0  # Different layouts

    def test_generate_rankings(self):
        """Test ranking generation."""
        # Create layouts with different characteristics
        simple = {
            "type": "container",
            "children": [{"type": "button"}],
        }
        complex_layout = {
            "type": "container",
            "children": [
                {
                    "type": "card",
                    "children": [
                        {"type": "input"},
                        {"type": "input"},
                        {
                            "type": "panel",
                            "children": [
                                {"type": "button"},
                                {"type": "checkbox"},
                            ],
                        },
                    ],
                },
            ],
        }

        artifacts = [
            GenerationArtifact(
                id="simple",
                session_id="test",
                query="test",
                layout=simple,
                draft="Simple",
                model="gpt-4.1-mini",
                temperature=0.5,
            ),
            GenerationArtifact(
                id="complex",
                session_id="test",
                query="test",
                layout=complex_layout,
                draft="Complex",
                model="gpt-4.1-mini",
                temperature=0.8,
            ),
        ]

        rankings = generate_rankings(artifacts)

        # Complex should rank higher in complexity, depth, variety, size
        assert rankings["complexity"][0] == "complex"
        assert rankings["depth"][0] == "complex"
        assert rankings["size"][0] == "complex"

    def test_compare_artifacts_empty(self):
        """Test comparison with empty list."""
        result = compare_artifacts([])
        assert result["count"] == 0
        assert result["diversity_score"] == 0.0

    def test_compare_artifacts(self):
        """Test artifact comparison."""
        layout1 = {"type": "container", "children": [{"type": "button"}]}
        layout2 = {"type": "card", "children": [{"type": "input"}]}

        artifacts = [
            GenerationArtifact.create(
                session_id="test",
                query="test",
                layout=layout1,
                draft="Test 1",
                model="gpt-4.1-mini",
                temperature=0.5,
            ),
            GenerationArtifact.create(
                session_id="test",
                query="test",
                layout=layout2,
                draft="Test 2",
                model="gpt-4.1-mini",
                temperature=0.8,
            ),
        ]

        result = compare_artifacts(artifacts)
        assert result["count"] == 2
        assert "diversity_score" in result
        assert "rankings" in result
        assert "feature_summary" in result


# =============================================================================
# Variation Set Tests
# =============================================================================


class TestVariationSets:
    """Tests for variation set management."""

    def test_store_variation_set(self, manager, sample_layout, sample_stats):
        """Test storing a variation set."""
        session = manager.get_or_create_session("test-session")

        # Store multiple artifacts for variation set
        artifact_ids = []
        for i in range(3):
            artifact = manager.store_artifact(
                query="login form",
                layout=sample_layout,
                draft=f"Draft {i}",
                session_id=session.id,
                model="gpt-4.1-mini",
                temperature=0.5 + (i * 0.25),
                stats=sample_stats,
                variation_group=str(uuid4()),
                variation_index=i,
            )
            artifact_ids.append(artifact.id)

        request = VariationRequest(query="login form", count=3)
        variation_set = manager.store_variation_set(
            request=request,
            artifact_ids=artifact_ids,
            diversity_score=0.5,
            rankings={"complexity": artifact_ids},
            session_id=session.id,
        )

        assert variation_set.id is not None
        assert len(variation_set.artifact_ids) == 3
        assert variation_set.diversity_score == 0.5

    def test_get_variation_set(self, manager, sample_layout, sample_stats):
        """Test retrieving a variation set."""
        session = manager.get_or_create_session("test-session")

        artifact_ids = []
        for i in range(2):
            artifact = manager.store_artifact(
                query="test",
                layout=sample_layout,
                draft=f"Draft {i}",
                session_id=session.id,
                model="gpt-4.1-mini",
                temperature=0.5,
                stats=sample_stats,
            )
            artifact_ids.append(artifact.id)

        request = VariationRequest(query="test", count=2)
        stored = manager.store_variation_set(
            request=request,
            artifact_ids=artifact_ids,
            diversity_score=0.3,
            rankings={},
            session_id=session.id,
        )

        retrieved = manager.get_variation_set(stored.id)
        assert retrieved is not None
        assert retrieved.id == stored.id

    def test_get_variation_artifacts(self, manager, sample_layout, sample_stats):
        """Test retrieving artifacts from a variation set."""
        session = manager.get_or_create_session("test-session")
        variation_group = str(uuid4())

        artifact_ids = []
        for i in range(3):
            artifact = manager.store_artifact(
                query="test",
                layout=sample_layout,
                draft=f"Draft {i}",
                session_id=session.id,
                model="gpt-4.1-mini",
                temperature=0.5 + (i * 0.2),
                stats=sample_stats,
                variation_group=variation_group,
                variation_index=i,
            )
            artifact_ids.append(artifact.id)

        # Create variation set to link artifacts
        request = VariationRequest(query="test", count=3)
        variation_set = manager.store_variation_set(
            request=request,
            artifact_ids=artifact_ids,
            diversity_score=0.4,
            rankings={},
            session_id=session.id,
        )

        artifacts = manager.get_variation_artifacts(variation_set.id)
        assert len(artifacts) == 3
