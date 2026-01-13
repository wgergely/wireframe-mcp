"""Unit tests for Corpus API."""

import pytest

from src.corpus.api.lib import CorpusManager
from src.corpus.provider.base import BaseProvider


class MockProvider(BaseProvider):
    @property
    def name(self):
        return "mock_provider"

    def fetch(self, force=False):
        pass

    def process(self):
        yield from []


class TestCorpusManager:
    """Tests for CorpusManager."""

    @pytest.mark.unit
    def test_init_defaults(self, monkeypatch, tmp_path):
        """Initializes with default data path."""
        monkeypatch.chdir(tmp_path)
        manager = CorpusManager()
        assert manager.data_dir.name == "data"
        assert "rico_semantic" in manager.providers  # Default registered

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
