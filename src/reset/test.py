"""Tests for reset functionality.

Tests cover:
- clear_history() function
- reset_environment() with history flag
- CLI --history flag integration
"""

import tempfile
from pathlib import Path

import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def history_dir(temp_dir):
    """Create a mock history directory structure."""
    history_path = temp_dir / "data" / "history"
    history_path.mkdir(parents=True, exist_ok=True)

    # Create mock history database file
    db_file = history_path / "history.db"
    db_file.write_text("mock database content")

    # Create mock WAL and SHM files (SQLite WAL mode)
    wal_file = history_path / "history.db-wal"
    wal_file.write_text("mock wal content")

    shm_file = history_path / "history.db-shm"
    shm_file.write_text("mock shm content")

    # Create mock preview cache directory
    previews_dir = history_path / "previews"
    previews_dir.mkdir(exist_ok=True)
    (previews_dir / "preview1.png").write_bytes(b"fake png data")
    (previews_dir / "preview2.png").write_bytes(b"fake png data 2")

    return history_path


# =============================================================================
# clear_history() Tests
# =============================================================================


class TestClearHistory:
    """Tests for clear_history() function."""

    @pytest.mark.unit
    def test_clear_history_deletes_database(self, temp_dir, history_dir, monkeypatch):
        """Test that clear_history deletes the history database."""
        from src.reset.lib import clear_history

        # Mock find_repo_root to return our temp directory
        monkeypatch.setattr("src.reset.lib.find_repo_root", lambda: temp_dir)

        # Mock close_history_manager to avoid actual connection handling
        monkeypatch.setattr(
            "src.history.close_history_manager", lambda: None, raising=False
        )

        result = clear_history(verbose=False)

        assert result is True
        # Database should be deleted
        assert not (history_dir / "history.db").exists()
        # WAL files should be deleted
        assert not (history_dir / "history.db-wal").exists()
        assert not (history_dir / "history.db-shm").exists()

    @pytest.mark.unit
    def test_clear_history_deletes_preview_cache(
        self, temp_dir, history_dir, monkeypatch
    ):
        """Test that clear_history deletes the preview cache."""
        from src.reset.lib import clear_history

        monkeypatch.setattr("src.reset.lib.find_repo_root", lambda: temp_dir)
        monkeypatch.setattr(
            "src.history.close_history_manager", lambda: None, raising=False
        )

        result = clear_history(verbose=False)

        assert result is True
        # Preview cache should be deleted
        assert not (history_dir / "previews").exists()

    @pytest.mark.unit
    def test_clear_history_returns_true_when_nothing_to_clear(
        self, temp_dir, monkeypatch
    ):
        """Test clear_history returns True when history directory doesn't exist."""
        from src.reset.lib import clear_history

        monkeypatch.setattr("src.reset.lib.find_repo_root", lambda: temp_dir)
        monkeypatch.setattr(
            "src.history.close_history_manager", lambda: None, raising=False
        )

        # Don't create the history directory
        result = clear_history(verbose=False)

        assert result is True

    @pytest.mark.unit
    def test_clear_history_prints_status_when_verbose(
        self, temp_dir, history_dir, monkeypatch, capsys
    ):
        """Test that clear_history prints status messages when verbose."""
        from src.reset.lib import clear_history

        monkeypatch.setattr("src.reset.lib.find_repo_root", lambda: temp_dir)
        monkeypatch.setattr(
            "src.history.close_history_manager", lambda: None, raising=False
        )

        clear_history(verbose=True)

        captured = capsys.readouterr()
        assert "[reset] Clearing history database..." in captured.out


# =============================================================================
# reset_environment() with history Tests
# =============================================================================


class TestResetEnvironmentWithHistory:
    """Tests for reset_environment() with history flag."""

    @pytest.mark.unit
    def test_reset_environment_calls_clear_history(
        self, temp_dir, history_dir, monkeypatch
    ):
        """Test that reset_environment calls clear_history when history=True."""
        from src.reset.lib import reset_environment

        # Track if clear_history was called
        clear_history_called = False

        def mock_clear_history(verbose=True):
            nonlocal clear_history_called
            clear_history_called = True
            return True

        monkeypatch.setattr("src.reset.lib.clear_history", mock_clear_history)

        # Need to mock other functions that might be called
        monkeypatch.setattr("src.reset.lib.clear_indices", lambda v: True)

        reset_environment(history=True, verbose=False)

        assert clear_history_called is True

    @pytest.mark.unit
    def test_reset_environment_history_not_called_by_default(self, monkeypatch):
        """Test that clear_history is NOT called when history flag is False."""
        from src.reset.lib import reset_environment

        clear_history_called = False

        def mock_clear_history(verbose=True):
            nonlocal clear_history_called
            clear_history_called = True
            return True

        monkeypatch.setattr("src.reset.lib.clear_history", mock_clear_history)
        monkeypatch.setattr("src.reset.lib.clear_indices", lambda v: True)

        # Default behavior (index only)
        reset_environment(verbose=False)

        assert clear_history_called is False

    @pytest.mark.unit
    def test_reset_environment_all_includes_history(self, monkeypatch):
        """Test that --all should NOT automatically include history (per plan)."""
        # Note: Per the plan, --all expands to index + models + docker + temp
        # but NOT history. This is a conscious design choice to protect user data.
        from src.reset.lib import reset_environment

        clear_history_called = False

        def mock_clear_history(verbose=True):
            nonlocal clear_history_called
            clear_history_called = True
            return True

        monkeypatch.setattr("src.reset.lib.clear_history", mock_clear_history)
        monkeypatch.setattr("src.reset.lib.clear_indices", lambda v: True)
        monkeypatch.setattr("src.reset.lib.clear_models", lambda v: True)
        monkeypatch.setattr("src.reset.lib.clear_docker", lambda v: True)
        monkeypatch.setattr("src.reset.lib.clear_temp", lambda v: True)

        # Simulate --all by setting all flags except history
        reset_environment(
            index=True, models=True, docker=True, temp=True, verbose=False
        )

        # History should NOT be called unless explicitly requested
        assert clear_history_called is False


# =============================================================================
# CLI --history Flag Tests
# =============================================================================


class TestResetCommandHistoryFlag:
    """Tests for the --history CLI flag."""

    @pytest.mark.unit
    def test_reset_command_accepts_history_flag(self, monkeypatch):
        """Test that the reset command accepts --history flag."""
        import importlib.util
        import sys
        from pathlib import Path

        # Load the project's __main__.py as a module
        main_path = Path(__file__).parent.parent.parent / "__main__.py"
        spec = importlib.util.spec_from_file_location("project_main", main_path)
        project_main = importlib.util.module_from_spec(spec)
        sys.modules["project_main"] = project_main
        spec.loader.exec_module(project_main)

        # Mock reset_environment to track calls
        called_with = {}

        def mock_reset_environment(**kwargs):
            called_with.update(kwargs)
            return True

        monkeypatch.setattr("src.reset.reset_environment", mock_reset_environment)

        result = project_main.handle_reset_command(["--history"])

        assert result == 0
        assert called_with.get("history") is True

    @pytest.mark.unit
    def test_reset_command_short_flag_H(self, monkeypatch):
        """Test that the reset command accepts -H short flag."""
        import importlib.util
        import sys
        from pathlib import Path

        main_path = Path(__file__).parent.parent.parent / "__main__.py"
        spec = importlib.util.spec_from_file_location("project_main", main_path)
        project_main = importlib.util.module_from_spec(spec)
        sys.modules["project_main"] = project_main
        spec.loader.exec_module(project_main)

        called_with = {}

        def mock_reset_environment(**kwargs):
            called_with.update(kwargs)
            return True

        monkeypatch.setattr("src.reset.reset_environment", mock_reset_environment)

        result = project_main.handle_reset_command(["-H"])

        assert result == 0
        assert called_with.get("history") is True

    @pytest.mark.unit
    def test_reset_command_history_with_other_flags(self, monkeypatch):
        """Test --history can be combined with other flags."""
        import importlib.util
        import sys
        from pathlib import Path

        main_path = Path(__file__).parent.parent.parent / "__main__.py"
        spec = importlib.util.spec_from_file_location("project_main", main_path)
        project_main = importlib.util.module_from_spec(spec)
        sys.modules["project_main"] = project_main
        spec.loader.exec_module(project_main)

        called_with = {}

        def mock_reset_environment(**kwargs):
            called_with.update(kwargs)
            return True

        monkeypatch.setattr("src.reset.reset_environment", mock_reset_environment)

        result = project_main.handle_reset_command(["--index", "--history"])

        assert result == 0
        assert called_with.get("history") is True
        assert called_with.get("index") is True
