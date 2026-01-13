"""Base test class for provider implementations."""

import pytest

from src.corpus.provider.base.lib import BaseProvider


class BaseProviderTest:
    """Base test suite that all providers should inherit from.

    Subclasses must define a `provider_class` fixture.
    """

    @pytest.fixture
    def provider_class(self):
        """Return the provider class to be tested."""
        raise NotImplementedError

    @pytest.fixture
    def provider(self, provider_class, tmp_path):
        """Create a provider instance for testing."""
        return provider_class(tmp_path)

    @pytest.mark.unit
    def test_is_base_provider_subclass(self, provider):
        """Verify the provider inherits from BaseProvider."""
        assert isinstance(provider, BaseProvider)

    @pytest.mark.unit
    def test_has_name(self, provider):
        """Verify the provider has a name property."""
        assert isinstance(provider.name, str)
        assert len(provider.name) > 0

    @pytest.mark.unit
    def test_initialization(self, provider, tmp_path):
        """Verify the provider initializes with the correct data directory."""
        assert provider.data_dir == tmp_path
