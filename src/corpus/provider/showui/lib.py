"""ShowUI Desktop dataset provider."""

from typing import Iterator

from src.corpus.provider.base import BaseProvider, StandardizedData


class Provider(BaseProvider):
    """Provider for the ShowUI Desktop dataset."""

    @property
    def name(self) -> str:
        """Provider name."""
        return "showui"

    def fetch(self, force: bool = False) -> None:
        """Download ShowUI data.

        Placeholder for GitHub/Repo download.
        """
        dest_dir = self.data_dir / "showui"
        dest_dir.mkdir(parents=True, exist_ok=True)

    def process(self) -> Iterator[StandardizedData]:
        """Process ShowUI data."""
        src_dir = self.data_dir / "showui"
        if not src_dir.exists():
            return
        # Placeholder for processing logic
        yield from []
