"""Unit tests for WebSight provider."""

import pytest

from src.corpus.provider.base.test import BaseProviderTest
from src.corpus.provider.websight import Provider


class TestWebSightProvider(BaseProviderTest):
    """TEST suites for WebSight Provider."""

    @pytest.fixture
    def provider_class(self):
        return Provider

    @pytest.mark.unit
    def test_name_is_websight(self, provider):
        """Verify the provider name is correct."""
        assert provider.name == "websight"
