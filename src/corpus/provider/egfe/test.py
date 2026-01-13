"""Unit tests for EGFE provider."""

import pytest

from src.corpus.provider.base.test import BaseProviderTest
from src.corpus.provider.egfe import Provider


class TestEgfeProvider(BaseProviderTest):
    """TEST suites for EGFE Provider."""

    @pytest.fixture
    def provider_class(self):
        return Provider

    @pytest.mark.unit
    def test_name_is_egfe(self, provider):
        """Verify the provider name is correct."""
        assert provider.name == "egfe"
