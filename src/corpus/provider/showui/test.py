"""Unit tests for ShowUI provider."""

import pytest

from src.corpus.provider.base.test import BaseProviderTest
from src.corpus.provider.showui import Provider


class TestShowUIProvider(BaseProviderTest):
    """TEST suites for ShowUI Provider."""

    @pytest.fixture
    def provider_class(self):
        return Provider

    @pytest.mark.unit
    def test_name_is_showui(self, provider):
        """Verify the provider name is correct."""
        assert provider.name == "showui"
