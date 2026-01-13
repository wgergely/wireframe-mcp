"""Corpus API for managing providers and data access."""

from pathlib import Path
from typing import Iterator

from src.corpus.provider.base import BaseProvider, StandardizedData
from src.corpus.provider.egfe import Provider as EgfeProvider
from src.corpus.provider.enrico import Provider as EnricoProvider
from src.corpus.provider.rico import Provider as RicoProvider
from src.corpus.provider.rico.lib import RICO_DATASETS
from src.corpus.provider.showui import Provider as ShowUIProvider
from src.corpus.provider.websight import Provider as WebSightProvider


class CorpusManager:
    """Manages corpus providers and data access."""

    def __init__(self, data_dir: Path | str | None = None):
        """Initialize the CorpusManager.

        Args:
            data_dir: Root directory for corpus data. Defaults to ./data.
        """
        self.data_dir = Path(data_dir) if data_dir else Path.cwd() / "data"
        self.providers: dict[str, BaseProvider] = {}
        self._register_default_providers()

    def _register_default_providers(self) -> None:
        """Register known default providers."""
        for dataset_type in RICO_DATASETS:
            self.register_provider(
                RicoProvider(self.data_dir, dataset_type=dataset_type)
            )

        self.register_provider(WebSightProvider(self.data_dir))
        self.register_provider(EgfeProvider(self.data_dir))
        self.register_provider(ShowUIProvider(self.data_dir))
        self.register_provider(EnricoProvider(self.data_dir))

    def register_provider(self, provider: BaseProvider) -> None:
        """Register a new provider."""
        self.providers[provider.name] = provider

    def get_provider(self, name: str) -> BaseProvider:
        """Get a provider by name.

        Args:
            name: Provider name to look up.

        Returns:
            The provider instance.

        Raises:
            KeyError: If provider not found.
        """
        if name not in self.providers:
            available = list(self.providers.keys())
            raise KeyError(f"Provider '{name}' not found. Available: {available}")
        return self.providers[name]

    def fetch_dataset(self, provider_name: str, force: bool = False) -> None:
        """Fetch data for a specific provider.

        Args:
            provider_name: Name of the provider to fetch from.
            force: If True, force re-download even if data exists.
        """
        self.get_provider(provider_name).fetch(force=force)

    def stream_data(self, provider_name: str) -> Iterator[StandardizedData]:
        """Stream standardized data from a provider.

        Args:
            provider_name: Name of the provider to stream from.

        Yields:
            StandardizedData items from the provider.
        """
        return self.get_provider(provider_name).process()

    def list_providers(self) -> list[str]:
        """List available provider names."""
        return list(self.providers.keys())
