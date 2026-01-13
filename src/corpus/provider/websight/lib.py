"""WebSight dataset provider.

WebSight is a large synthetic dataset of web page screenshots + HTML.
Source: https://huggingface.co/datasets/HuggingFaceM4/WebSight

Since the full dataset is 317GB+ with 2M+ samples, this provider supports:
1. Streaming from HuggingFace (requires `datasets` library)
2. Processing local parquet/jsonl files
3. Processing a small sample subset for testing
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from src.core import get_logger
from src.corpus.normalizer import hierarchy_to_layout, normalize_html_to_hierarchy
from src.corpus.provider.base import BaseProvider, StandardizedData

if TYPE_CHECKING:
    pass

logger = get_logger("provider.websight")

# Sample HTML for testing when no data is available
SAMPLE_HTML = """
<html>
<body class="bg-gray-100">
    <header class="bg-blue-600 text-white p-4">
        <nav class="flex justify-between items-center">
            <h1 class="text-2xl font-bold">WebSight Sample</h1>
            <button class="btn">Menu</button>
        </nav>
    </header>
    <main class="container mx-auto p-4">
        <article class="card bg-white p-4 rounded shadow">
            <h2 class="text-xl">Welcome</h2>
            <p>This is a sample web page from WebSight dataset.</p>
            <a href="#" class="btn text-blue-600">Learn More</a>
        </article>
    </main>
    <footer class="bg-gray-800 text-white p-4 text-center">
        <p>Footer content</p>
    </footer>
</body>
</html>
"""


class Provider(BaseProvider):
    """Provider for the WebSight dataset (HuggingFaceM4).

    WebSight contains synthetic web page screenshots paired with HTML/CSS code.
    This provider converts HTML DOM to the standardized hierarchy format.
    """

    @property
    def name(self) -> str:
        """Provider name."""
        return "websight"

    @property
    def _dest_dir(self) -> Path:
        """Base directory for WebSight data."""
        return self.data_dir / "websight"

    def fetch(self, force: bool = False) -> None:
        """Download WebSight data.

        Due to the large size (317GB+), this method provides instructions
        for manual download or streaming setup.
        """
        self._dest_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing data
        parquet_files = list(self._dest_dir.glob("*.parquet"))
        jsonl_files = list(self._dest_dir.glob("*.jsonl"))
        json_files = list(self._dest_dir.glob("*.json"))

        if (parquet_files or jsonl_files or json_files) and not force:
            logger.info(f"[{self.name}] Data found at {self._dest_dir}")
            return

        # Create sample data for testing
        sample_file = self._dest_dir / "sample.jsonl"
        if not sample_file.exists():
            logger.info(f"[{self.name}] Creating sample data for testing...")
            self._create_sample_data(sample_file)

        logger.info(
            f"[{self.name}] WebSight is a large dataset (317GB+). "
            "For full dataset access:"
        )
        logger.info(
            "  Option 1: Install `datasets` library and stream from HuggingFace"
        )
        logger.info("    pip install datasets")
        logger.info("    from datasets import load_dataset")
        logger.info('    ds = load_dataset("HuggingFaceM4/WebSight", streaming=True)')
        logger.info("")
        logger.info("  Option 2: Download parquet files manually from:")
        logger.info("    https://huggingface.co/datasets/HuggingFaceM4/WebSight")
        logger.info(f"    Place files in: {self._dest_dir}")

    def _create_sample_data(self, output_path: Path) -> None:
        """Create sample JSONL data for testing.

        Args:
            output_path: Path where to write the sample JSONL file.
        """
        samples = [
            {
                "id": "sample_001",
                "text": SAMPLE_HTML,
                "llm_generated_idea": "A simple landing page with header, main content, and footer",
            },
            {
                "id": "sample_002",
                "text": """
                <html><body>
                <nav class="navbar"><a href="/">Home</a><a href="/about">About</a></nav>
                <div class="container">
                    <h1>Product Page</h1>
                    <img src="product.jpg" alt="Product Image">
                    <p class="price">$99.99</p>
                    <button class="btn-primary">Add to Cart</button>
                </div>
                </body></html>
                """,
                "llm_generated_idea": "An e-commerce product page",
            },
            {
                "id": "sample_003",
                "text": """
                <html><body>
                <header><h1>Blog Title</h1></header>
                <article>
                    <h2>Post Title</h2>
                    <p>Article content goes here...</p>
                    <ul><li>Point 1</li><li>Point 2</li><li>Point 3</li></ul>
                </article>
                <aside>
                    <h3>Related Posts</h3>
                    <a href="#">Post 1</a>
                    <a href="#">Post 2</a>
                </aside>
                </body></html>
                """,
                "llm_generated_idea": "A blog post page with sidebar",
            },
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        logger.info(f"[{self.name}] Created {len(samples)} sample items")

    def process(self) -> Iterator[StandardizedData]:
        """Process WebSight data and yield standardized items.

        Supports:
        - JSONL files with 'text' (HTML) field
        - Parquet files (if pyarrow available)
        - HuggingFace streaming (if datasets available)
        """
        if not self._dest_dir.exists():
            logger.warning(f"[{self.name}] No data directory. Run fetch() first.")
            return

        # Try JSONL files first
        for jsonl_path in self._dest_dir.glob("*.jsonl"):
            yield from self._process_jsonl(jsonl_path)

        # Try JSON files
        for json_path in self._dest_dir.glob("*.json"):
            if json_path.name.endswith(".jsonl"):
                continue
            item = self._process_json_file(json_path)
            if item:
                yield item

        # Try parquet files if pyarrow is available
        parquet_files = list(self._dest_dir.glob("*.parquet"))
        if parquet_files:
            yield from self._process_parquet_files(parquet_files)

    def _process_jsonl(self, jsonl_path: Path) -> Iterator[StandardizedData]:
        """Process a JSONL file.

        Args:
            jsonl_path: Path to the JSONL file.

        Yields:
            StandardizedData items from the JSONL file.
        """
        logger.info(f"[{self.name}] Processing {jsonl_path.name}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    item = self._convert_to_standardized(data, f"jsonl_{line_num}")
                    if item:
                        yield item
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"[{self.name}] Invalid JSON at line {line_num}: {e}"
                    )
                except Exception as e:
                    logger.error(f"[{self.name}] Error at line {line_num}: {e}")

    def _process_json_file(self, json_path: Path) -> StandardizedData | None:
        """Process a single JSON file.

        Args:
            json_path: Path to the JSON file.

        Returns:
            StandardizedData if successful, None if parsing fails.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._convert_to_standardized(data, json_path.stem)
        except json.JSONDecodeError:
            logger.warning(f"[{self.name}] Skipping invalid JSON: {json_path}")
            return None
        except Exception as e:
            logger.error(f"[{self.name}] Error reading {json_path}: {e}")
            return None

    def _process_parquet_files(
        self, parquet_files: list[Path]
    ) -> Iterator[StandardizedData]:
        """Process parquet files if pyarrow is available.

        Args:
            parquet_files: List of paths to parquet files.

        Yields:
            StandardizedData items from the parquet files.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning(
                f"[{self.name}] pyarrow not installed. "
                "Install with: pip install pyarrow"
            )
            return

        for parquet_path in parquet_files:
            logger.info(f"[{self.name}] Processing {parquet_path.name}")
            try:
                table = pq.read_table(parquet_path)
                for i, row in enumerate(table.to_pylist()):
                    item = self._convert_to_standardized(
                        row, f"{parquet_path.stem}_{i}"
                    )
                    if item:
                        yield item
            except Exception as e:
                logger.error(f"[{self.name}] Error reading {parquet_path}: {e}")

    def _convert_to_standardized(
        self, data: dict, item_id: str
    ) -> StandardizedData | None:
        """Convert WebSight data to StandardizedData.

        Args:
            data: Dict with 'text' (HTML), optionally 'image' and 'llm_generated_idea'
            item_id: Unique identifier for this item

        Returns:
            StandardizedData or None if conversion fails
        """
        html = data.get("text", "")
        if not html:
            return None

        try:
            # Convert HTML to hierarchy
            hierarchy = normalize_html_to_hierarchy(html)

            # Convert hierarchy to LayoutNode
            layout = hierarchy_to_layout(hierarchy, id_prefix=f"websight_{item_id}")

            return StandardizedData(
                id=item_id,
                source="websight",
                dataset="v0.2",
                hierarchy=hierarchy,
                layout=layout,
                metadata={
                    "llm_idea": data.get("llm_generated_idea", ""),
                    "html_length": len(html),
                },
                screenshot_path=None,  # WebSight images are in-memory PIL
            )
        except Exception as e:
            logger.error(f"[{self.name}] Error converting item {item_id}: {e}")
            return None
