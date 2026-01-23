"""Integration tests for interaction logging workflow.

Tests the full interaction logging lifecycle:
1. Generate a layout -> interaction logged
2. Refine the layout -> interaction logged with feedback
3. Retrieve session history -> shows both interactions
4. Clear history -> data deleted
"""

import tempfile
from pathlib import Path

import pytest

from src.history import (
    GenerationStats,
    HistoryManager,
    StorageConfig,
    close_history_manager,
)
from src.mcp.tools.session_history import get_artifact_history, get_session_history


@pytest.fixture
def temp_history_db():
    """Create a temporary history database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_history.db"
        yield db_path


@pytest.fixture
def test_config():
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
def history_manager(temp_history_db, test_config):
    """Create a HistoryManager with test config."""
    # Close any existing global manager first
    close_history_manager()

    manager = HistoryManager(
        db_path=temp_history_db, config=test_config, auto_cleanup=False
    )
    yield manager
    manager.close()


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


@pytest.mark.integration
class TestInteractionWorkflow:
    """Integration tests for full interaction logging workflow."""

    def test_generate_layout_logs_interaction(
        self, history_manager, sample_layout, sample_stats
    ):
        """Test that simulated generate_layout logs an interaction."""
        session = history_manager.get_or_create_session("workflow-test")

        # Simulate generate_layout: store artifact and log interaction
        artifact = history_manager.store_artifact(
            query="login form with email and password",
            layout=sample_layout,
            draft=(
                "Login Form [container]\n"
                "  Email [input]\n"
                "  Password [input]\n"
                "  Submit [button]"
            ),
            session_id=session.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        # Log the interaction (as generate_layout tool would)
        interaction = history_manager.store_interaction(
            tool_name="generate_layout",
            request_params={
                "query": "login form with email and password",
                "model": "gpt-4.1-mini",
                "temperature": 0.7,
            },
            session_id=session.id,
            artifact_id=artifact.id,
            response_summary={
                "status": "success",
                "node_count": 5,
            },
        )

        # Verify interaction was stored
        assert interaction.id is not None
        assert interaction.tool_name == "generate_layout"
        assert interaction.artifact_id == artifact.id
        assert (
            interaction.request_params["query"] == "login form with email and password"
        )

        # Verify we can retrieve it
        retrieved = history_manager.get_interaction(interaction.id)
        assert retrieved is not None
        assert retrieved.id == interaction.id

    def test_refine_layout_logs_feedback(
        self, history_manager, sample_layout, sample_stats
    ):
        """Test that simulated refine_layout logs feedback separately."""
        session = history_manager.get_or_create_session("workflow-test")

        # First, generate original layout
        original_artifact = history_manager.store_artifact(
            query="login form",
            layout=sample_layout,
            draft="Login Form [container]",
            session_id=session.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        history_manager.store_interaction(
            tool_name="generate_layout",
            request_params={"query": "login form"},
            session_id=session.id,
            artifact_id=original_artifact.id,
        )

        # Simulate refine_layout with feedback
        refined_layout = {
            **sample_layout,
            "children": sample_layout["children"]
            + [{"type": "link", "id": "forgot", "label": "Forgot Password?"}],
        }
        refined_artifact = history_manager.store_artifact(
            query="Refine: login form with forgot password link",
            layout=refined_layout,
            draft="Login Form [container]\n  ...\n  Forgot Password? [link]",
            session_id=session.id,
            parent_id=original_artifact.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        feedback_text = "add a forgot password link below the form"
        refine_interaction = history_manager.store_interaction(
            tool_name="refine_layout",
            request_params={
                "artifact_id": original_artifact.id,
                "model": "gpt-4.1-mini",
                "temperature": 0.7,
            },
            session_id=session.id,
            artifact_id=refined_artifact.id,
            feedback=feedback_text,
            response_summary={
                "status": "success",
                "parent_id": original_artifact.id,
            },
        )

        # Verify feedback is stored
        assert refine_interaction.feedback == feedback_text
        assert refine_interaction.tool_name == "refine_layout"

        # Verify lineage is correct
        lineage = history_manager.get_lineage(refined_artifact.id)
        assert original_artifact.id in [a.id for a in lineage["ancestors"]]

    def test_full_workflow_generate_refine_timeline(
        self, history_manager, sample_layout, sample_stats
    ):
        """Test full workflow: generate -> refine -> get timeline."""
        session = history_manager.get_or_create_session("workflow-test")

        # Step 1: Generate layout
        artifact1 = history_manager.store_artifact(
            query="dashboard with sidebar",
            layout=sample_layout,
            draft="Dashboard [container]",
            session_id=session.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        history_manager.store_interaction(
            tool_name="generate_layout",
            request_params={"query": "dashboard with sidebar"},
            session_id=session.id,
            artifact_id=artifact1.id,
            response_summary={"status": "success"},
        )

        # Step 2: Refine layout
        artifact2 = history_manager.store_artifact(
            query="Refine: make sidebar collapsible",
            layout=sample_layout,
            draft="Dashboard [container]\n  Collapsible Sidebar [panel]",
            session_id=session.id,
            parent_id=artifact1.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        history_manager.store_interaction(
            tool_name="refine_layout",
            request_params={"artifact_id": artifact1.id},
            session_id=session.id,
            artifact_id=artifact2.id,
            feedback="make the sidebar collapsible",
            response_summary={"status": "success", "parent_id": artifact1.id},
        )

        # Step 3: Get session timeline
        timeline = history_manager.get_session_timeline(session_id=session.id)

        # Should have 4 events: 2 interactions + 2 artifacts
        assert len(timeline) == 4

        # Verify chronological order (timeline is sorted by timestamp)
        timestamps = [e["timestamp"] for e in timeline]
        assert timestamps == sorted(timestamps)

        # Verify we have both interaction types
        interaction_events = [e for e in timeline if e["type"] == "interaction"]
        assert len(interaction_events) == 2

        tool_names = [e["tool_name"] for e in interaction_events]
        assert "generate_layout" in tool_names
        assert "refine_layout" in tool_names

        # Verify feedback is in timeline
        refine_events = [e for e in interaction_events if e.get("feedback")]
        assert len(refine_events) == 1
        assert refine_events[0]["feedback"] == "make the sidebar collapsible"

        # Verify artifacts are in timeline
        artifact_events = [e for e in timeline if e["type"] == "artifact"]
        assert len(artifact_events) == 2

    def test_get_session_history_tool(
        self, history_manager, sample_layout, sample_stats, monkeypatch
    ):
        """Test the get_session_history MCP tool function."""
        session = history_manager.get_or_create_session("mcp-tool-test")

        # Store some artifacts and interactions
        artifact = history_manager.store_artifact(
            query="test query",
            layout=sample_layout,
            draft="Test [container]",
            session_id=session.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        history_manager.store_interaction(
            tool_name="generate_layout",
            request_params={"query": "test query"},
            session_id=session.id,
            artifact_id=artifact.id,
        )

        # Mock get_history_manager in src.history module (where it's imported from)
        monkeypatch.setattr(
            "src.history.get_history_manager",
            lambda: history_manager,
        )

        # Call the MCP tool function
        result = get_session_history(session_id=session.id)

        assert "timeline" in result
        assert "session_id" in result
        assert result["session_id"] == session.id
        assert result["artifact_count"] == 1
        assert result["interaction_count"] == 1
        assert len(result["timeline"]) >= 1

    def test_get_artifact_history_tool(
        self, history_manager, sample_layout, sample_stats, monkeypatch
    ):
        """Test the get_artifact_history MCP tool function."""
        session = history_manager.get_or_create_session("artifact-history-test")

        # Create a chain: original -> refine1 -> refine2
        original = history_manager.store_artifact(
            query="original form",
            layout=sample_layout,
            draft="Original [container]",
            session_id=session.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        history_manager.store_interaction(
            tool_name="generate_layout",
            request_params={"query": "original form"},
            session_id=session.id,
            artifact_id=original.id,
        )

        refine1 = history_manager.store_artifact(
            query="Refine: add header",
            layout=sample_layout,
            draft="Original with Header [container]",
            session_id=session.id,
            parent_id=original.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        history_manager.store_interaction(
            tool_name="refine_layout",
            request_params={"artifact_id": original.id},
            session_id=session.id,
            artifact_id=refine1.id,
            feedback="add a header section",
        )

        refine2 = history_manager.store_artifact(
            query="Refine: add footer",
            layout=sample_layout,
            draft="Original with Header and Footer [container]",
            session_id=session.id,
            parent_id=refine1.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        history_manager.store_interaction(
            tool_name="refine_layout",
            request_params={"artifact_id": refine1.id},
            session_id=session.id,
            artifact_id=refine2.id,
            feedback="add a footer section",
        )

        # Mock get_history_manager in src.history module (where it's imported from)
        monkeypatch.setattr(
            "src.history.get_history_manager",
            lambda: history_manager,
        )

        # Call the MCP tool function
        result = get_artifact_history(artifact_id=refine2.id)

        assert result["artifact_id"] == refine2.id
        assert len(result["lineage"]) == 3  # original -> refine1 -> refine2
        assert len(result["feedback_chain"]) == 2  # Two refinements with feedback
        assert "add a header section" in result["feedback_chain"]
        assert "add a footer section" in result["feedback_chain"]

    def test_clear_history_removes_data(
        self, history_manager, sample_layout, sample_stats
    ):
        """Test that clearing history removes all data."""
        session = history_manager.get_or_create_session("clear-test")

        # Store some data
        artifact = history_manager.store_artifact(
            query="test",
            layout=sample_layout,
            draft="Test",
            session_id=session.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        history_manager.store_interaction(
            tool_name="generate_layout",
            request_params={"query": "test"},
            session_id=session.id,
            artifact_id=artifact.id,
        )

        # Verify data exists
        assert history_manager.get_artifact(artifact.id) is not None
        assert len(history_manager.list_interactions(session_id=session.id)) == 1

        # Delete session (which cascades to artifacts and interactions)
        history_manager.delete_session(session.id)

        # Verify data is gone
        assert history_manager.get_session(session.id) is None
        assert history_manager.get_artifact(artifact.id) is None
        assert len(history_manager.list_interactions(session_id=session.id)) == 0

    def test_multiple_sessions_isolation(
        self, history_manager, sample_layout, sample_stats
    ):
        """Test that interactions are isolated between sessions."""
        # Use create_session to ensure separate sessions (not get_or_create)
        session1 = history_manager.create_session(name="session-1")
        session2 = history_manager.create_session(name="session-2")

        # Create artifacts in each session
        artifact1 = history_manager.store_artifact(
            query="session 1 query",
            layout=sample_layout,
            draft="Session 1",
            session_id=session1.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        history_manager.store_interaction(
            tool_name="generate_layout",
            request_params={"query": "session 1 query"},
            session_id=session1.id,
            artifact_id=artifact1.id,
        )

        artifact2 = history_manager.store_artifact(
            query="session 2 query",
            layout=sample_layout,
            draft="Session 2",
            session_id=session2.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        history_manager.store_interaction(
            tool_name="generate_layout",
            request_params={"query": "session 2 query"},
            session_id=session2.id,
            artifact_id=artifact2.id,
        )

        # Verify isolation
        session1_interactions = history_manager.list_interactions(
            session_id=session1.id
        )
        session2_interactions = history_manager.list_interactions(
            session_id=session2.id
        )

        assert len(session1_interactions) == 1
        assert len(session2_interactions) == 1
        assert session1_interactions[0].session_id == session1.id
        assert session2_interactions[0].session_id == session2.id

        # Verify timelines are isolated
        timeline1 = history_manager.get_session_timeline(session_id=session1.id)
        timeline2 = history_manager.get_session_timeline(session_id=session2.id)

        assert len(timeline1) == 2  # 1 artifact + 1 interaction
        assert len(timeline2) == 2

        # Verify artifact IDs don't overlap
        artifacts_in_timeline1 = [
            e["artifact_id"] for e in timeline1 if e["type"] == "artifact"
        ]
        artifacts_in_timeline2 = [
            e["artifact_id"] for e in timeline2 if e["type"] == "artifact"
        ]

        assert artifact1.id in artifacts_in_timeline1
        assert artifact2.id in artifacts_in_timeline2
        assert artifact1.id not in artifacts_in_timeline2
        assert artifact2.id not in artifacts_in_timeline1
