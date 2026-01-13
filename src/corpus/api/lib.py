"""Corpus API for managing providers and data access."""

from pathlib import Path
from typing import Iterator

from src.corpus.provider.base import BaseProvider, StandardizedData
from src.corpus.provider.egfe import Provider as EgfeProvider
from src.corpus.provider.enrico import Provider as EnricoProvider
from src.corpus.provider.rico import Provider as RicoProvider
from src.corpus.provider.showui import Provider as ShowUIProvider
from src.corpus.provider.websight import Provider as WebSightProvider


class CorpusManager:
    """Manages corpus providers and data access."""

    def __init__(self, data_dir: Path | str | None = None):
        """Initialize the CorpusManager.

        Args:
            data_dir: Root directory for corpus data. Defaults to ./data.
        """
        if data_dir is None:
            self.data_dir = Path.cwd() / "data"
        else:
            self.data_dir = Path(data_dir)

        self.data_dir.resolve()

        # Initialize providers
        self.providers: dict[str, BaseProvider] = {}
        self._register_default_providers()

    def _register_default_providers(self) -> None:
        """Register known default providers."""
        # Register all Rico dataset types
        from src.corpus.provider.rico.lib import RICO_DATASETS

        for dataset_type in RICO_DATASETS:
            provider = RicoProvider(self.data_dir, dataset_type=dataset_type)
            self.register_provider(provider)

        # Register additional providers
        self.register_provider(WebSightProvider(self.data_dir))
        self.register_provider(EgfeProvider(self.data_dir))
        self.register_provider(ShowUIProvider(self.data_dir))
        self.register_provider(EnricoProvider(self.data_dir))

    def register_provider(self, provider: BaseProvider) -> None:
        """Register a new provider."""
        self.providers[provider.name] = provider

    def get_provider(self, name: str) -> BaseProvider:
        """Get a provider by name."""
        if name not in self.providers:
            raise KeyError(
                f"Provider '{name}' not found. Available: {list(self.providers.keys())}"
            )
        return self.providers[name]

    def fetch_dataset(self, provider_name: str, force: bool = False) -> None:
        """Fetch data for a specific provider."""
        provider = self.get_provider(provider_name)
        provider.fetch(force=force)

    def stream_data(self, provider_name: str) -> Iterator[StandardizedData]:
        """Stream standardized data from a provider."""
        provider = self.get_provider(provider_name)
        return provider.process()

    def list_providers(self) -> list[str]:
        """List available provider names."""
        return list(self.providers.keys())
