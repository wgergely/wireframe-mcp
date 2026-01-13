"""Unit tests for Corpus API."""

from typing import Iterator

import pytest

from src.corpus.api.lib import CorpusManager
from src.corpus.provider.base import BaseProvider, DataType, StandardizedData
from src.mid import ComponentType, LayoutNode


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    @property
    def name(self) -> str:
        return "mock_provider"

    def has_data(self, data_type: DataType | None = None) -> bool:
        """Check if data exists."""
        return False

    def fetch(self, force: bool = False) -> None:
        pass

    def process(self) -> Iterator[StandardizedData]:
        yield from []

    def to_layout(self, hierarchy: dict, item_id: str) -> LayoutNode:
        """Mock implementation - not used in these tests."""
        return LayoutNode(id=item_id, type=ComponentType.CONTAINER)


class TestCorpusManager:
    """Tests for CorpusManager."""

    @pytest.mark.unit
    def test_init_defaults(self, monkeypatch, tmp_path):
        """Initializes with default data path using repo root detection."""
        # Create fake repo structure with .gitignore
        (tmp_path / ".gitignore").touch()
        monkeypatch.chdir(tmp_path)
        # Clear env var to ensure default behavior
        monkeypatch.delenv("CORPUS_DATA_DIR", raising=False)

        manager = CorpusManager()

        # Should use {repo_root}/.corpus/data as default
        expected = tmp_path / ".corpus" / "data"
        assert manager.data_dir == expected
        assert "rico_semantic" in manager.providers  # Default registered

    @pytest.mark.unit
    def test_init_with_override(self, tmp_path):
        """Initializes with explicit override path."""
        manager = CorpusManager(tmp_path)
        assert manager.data_dir == tmp_path
        assert "rico_semantic" in manager.providers

    @pytest.mark.unit
    def test_register_get_provider(self, tmp_path):
        """Can register and retrieve a provider."""
        manager = CorpusManager(tmp_path)
        provider = MockProvider(tmp_path)
        manager.register_provider(provider)

        retrieved = manager.get_provider("mock_provider")
        assert retrieved == provider

    @pytest.mark.unit
    def test_get_nonexistent_provider(self, tmp_path):
        """Raises error for unknown provider."""
        manager = CorpusManager(tmp_path)
        with pytest.raises(KeyError):
            manager.get_provider("nonexistent")

    @pytest.mark.unit
    def test_list_providers(self, tmp_path):
        """Lists registered providers."""
        manager = CorpusManager(tmp_path)
        # Should have defaults
        providers = manager.list_providers()
        assert "rico_semantic" in providers
        assert "websight" in providers
        assert "egfe" in providers
        assert "showui" in providers
        assert "enrico" in providers
