"""Unit tests for Enrico provider."""

import pytest

from src.corpus.provider.base.test import BaseProviderTest
from src.corpus.provider.enrico import Provider


class TestEnricoProvider(BaseProviderTest):
    """TEST suites for Enrico Provider."""

    @pytest.fixture
    def provider_class(self):
        return Provider

    @pytest.mark.unit
    def test_name_is_enrico(self, provider):
        """Verify the provider name is correct."""
        assert provider.name == "enrico"
