"""Unit tests for new providers (WebSight, EGFE, ShowUI, Enrico)."""

import pytest

from src.corpus.api import CorpusManager
from src.corpus.provider.egfe import Provider as EgfeProvider
from src.corpus.provider.enrico import Provider as EnricoProvider
from src.corpus.provider.showui import Provider as ShowUIProvider
from src.corpus.provider.websight import Provider as WebSightProvider


class TestNewProviders:
    """Tests for new provider implementations."""

    @pytest.mark.unit
    def test_websight_init(self, tmp_path):
        """WebSight initializes correctly."""
        provider = WebSightProvider(tmp_path)
        assert provider.name == "websight"
        assert provider.data_dir == tmp_path

    @pytest.mark.unit
    def test_egfe_init(self, tmp_path):
        """EGFE initializes correctly."""
        provider = EgfeProvider(tmp_path)
        assert provider.name == "egfe"

    @pytest.mark.unit
    def test_showui_init(self, tmp_path):
        """ShowUI initializes correctly."""
        provider = ShowUIProvider(tmp_path)
        assert provider.name == "showui"

    @pytest.mark.unit
    def test_enrico_init(self, tmp_path):
        """Enrico initializes correctly."""
        provider = EnricoProvider(tmp_path)
        assert provider.name == "enrico"

    @pytest.mark.unit
    def test_corpus_manager_registers_new(self, tmp_path):
        """CorpusManager registers all new providers by default."""
        manager = CorpusManager(tmp_path)
        providers = manager.list_providers()
        assert "websight" in providers
        assert "egfe" in providers
        assert "showui" in providers
        assert "enrico" in providers
