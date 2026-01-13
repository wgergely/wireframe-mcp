"""WebSight dataset provider."""

from typing import Iterator

from src.core import get_logger
from src.corpus.provider.base import BaseProvider, StandardizedData

logger = get_logger("provider.websight")


class Provider(BaseProvider):
    """Provider for the WebSight dataset (HuggingFaceM4).

    Targeting the 'v0.1' or similar subsets.
    Since WebSight is a large HF dataset, this provider currently
    supports processing local parquet/jsonl dumps.
    """

    @property
    def name(self) -> str:
        """Provider name."""
        return "websight"

    def fetch(self, force: bool = False) -> None:
        """Download WebSight data.

        Currently a placeholder as WebSight requires HuggingFace authentication
        or 'datasets' library for efficient streaming.
        """
        dest_dir = self.data_dir / "websight"
        dest_dir.mkdir(parents=True, exist_ok=True)

        if not any(dest_dir.iterdir()):
            logger.warning(
                f"[{self.name}] Automatic download not yet implemented for WebSight "
                "(2M+ items)."
            )
            logger.info(f"Please manually place parquet/jsonl files in: {dest_dir}")

    def process(self) -> Iterator[StandardizedData]:
        """Process local WebSight data files."""
        src_dir = self.data_dir / "websight"
        if not src_dir.exists():
            return
        # Placeholder for processing logic
        # Would iterate over parquet/jsonl files and yield StandardizedData
        yield from []
