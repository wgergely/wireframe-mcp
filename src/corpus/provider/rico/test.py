"""Unit tests for Rico provider."""

import pytest

from src.corpus.provider.base.test import BaseProviderTest
from src.corpus.provider.rico.lib import RICO_DATASETS
from src.corpus.provider.rico.lib import Provider as RicoProvider


class TestRicoDatasets:
    """Tests for RICO_DATASETS constant."""

    @pytest.mark.unit
    def test_all_urls_are_valid_format(self):
        """All URLs start with https."""
        for name, info in RICO_DATASETS.items():
            assert info["url"].startswith("https://"), f"{name} URL invalid"

    @pytest.mark.unit
    def test_all_datasets_have_metadata(self):
        """All datasets have required metadata."""
        for _name, info in RICO_DATASETS.items():
            assert "url" in info
            assert "size_mb" in info
            assert "description" in info


class TestRicoProvider(BaseProviderTest):
    """Tests for RicoProvider."""

    @pytest.fixture
    def provider_class(self):
        """Return the RicoProvider class."""
        # Create a wrapper that passes dataset_type
        return lambda data_dir: RicoProvider(data_dir, dataset_type="semantic")

    @pytest.mark.unit
    def test_init_with_semantic_dataset(self, tmp_path):
        """Provider initializes with semantic dataset type."""
        provider = RicoProvider(tmp_path, dataset_type="semantic")
        assert provider.name == "rico_semantic"
        assert provider.data_dir == tmp_path
        # Accessing private props for testing logic references
        assert provider._dest_dir == tmp_path / "rico"
        assert provider._extract_dir == tmp_path / "rico" / "semantic"

    @pytest.mark.unit
    def test_fetch_calls_download_and_extract(self, tmp_path, monkeypatch):
        """Fetch method calls download utility and extracts."""

        # Mock download_and_extract
        mock_called = False
        captured_args = {}

        def mock_download_and_extract(
            url, dest_dir, extract_dir, provider_name, expected_size_mb=None
        ):
            nonlocal mock_called, captured_args
            mock_called = True
            captured_args = {
                "url": url,
                "dest_dir": dest_dir,
                "extract_dir": extract_dir,
                "provider_name": provider_name,
                "expected_size_mb": expected_size_mb,
            }
            # Create the extract directory and a dummy file
            extract_dir.mkdir(parents=True, exist_ok=True)
            (extract_dir / "test.json").write_text("{}")

        monkeypatch.setattr(
            "src.corpus.provider.rico.lib.download_and_extract",
            mock_download_and_extract,
        )

        provider = RicoProvider(tmp_path, dataset_type="semantic")
        provider.fetch()

        assert mock_called
        assert (tmp_path / "rico" / "semantic").exists()
        assert captured_args["provider_name"] == "rico_semantic"
        assert captured_args["expected_size_mb"] == 150  # semantic dataset size

    @pytest.mark.unit
    def test_process_yields_data(self, tmp_path):
        """Process yields StandardizedData from files."""
        # Setup dummy data
        extract_dir = tmp_path / "rico" / "semantic"
        extract_dir.mkdir(parents=True)

        json_file = extract_dir / "screen1.json"
        json_file.write_text('{"id": "screen1", "children": []}', encoding="utf-8")

        provider = RicoProvider(tmp_path, dataset_type="semantic")

        items = list(provider.process())
        assert len(items) == 1
        item = items[0]
        assert item.id == "screen1"
        assert item.source == "rico"
        assert item.dataset == "semantic"
        assert item.hierarchy == {"id": "screen1", "children": []}


# =============================================================================
# Integration Tests (require actual data)
# Converted from scripts/audit_rico.py
# =============================================================================


class TestRicoProviderIntegration:
    """Integration tests for Rico provider pipeline.

    These tests verify the full data pipeline including:
    - Data existence and structure
    - Field validation
    - Screenshot file existence

    Requires actual Rico corpus data to be downloaded.
    """

    @pytest.fixture
    def rico_provider(self, tmp_path):
        """Get Rico provider, skip if data unavailable."""
        from src.config import get_data_dir

        data_dir = get_data_dir()
        provider = RicoProvider(data_dir, dataset_type="semantic")

        if not provider.has_data():
            pytest.skip("Rico data not available (run: python . corpus download rico)")

        return provider

    @pytest.mark.integration
    def test_provider_has_data(self, rico_provider):
        """Provider reports data availability correctly."""
        assert rico_provider.has_data()

    @pytest.mark.integration
    def test_stream_items(self, rico_provider):
        """Provider streams items successfully."""
        items = []
        for i, item in enumerate(rico_provider.process()):
            items.append(item)
            if i >= 50:
                break

        assert len(items) > 0, "Should stream at least some items"

    @pytest.mark.integration
    def test_standardized_data_has_required_fields(self, rico_provider):
        """StandardizedData items have all required fields."""
        sample_size = 100
        items = []
        for i, item in enumerate(rico_provider.process()):
            items.append(item)
            if i >= sample_size:
                break

        assert len(items) > 0, "No items returned from provider"

        # Check required fields on all items
        for item in items:
            assert item.id is not None, "Item missing id"
            assert item.source == "rico", f"Wrong source: {item.source}"
            assert item.dataset == "semantic", f"Wrong dataset: {item.dataset}"
            assert item.hierarchy is not None, "Item missing hierarchy"

    @pytest.mark.integration
    def test_hierarchy_structure(self, rico_provider):
        """Hierarchy data has expected structure."""
        sample_size = 50
        items_with_children = 0
        items_with_bounds = 0
        items_with_label = 0

        for i, item in enumerate(rico_provider.process()):
            if i >= sample_size:
                break

            hierarchy = item.hierarchy
            if "children" in hierarchy:
                items_with_children += 1
            if "bounds" in hierarchy:
                items_with_bounds += 1
            if "componentLabel" in hierarchy:
                items_with_label += 1

        # Most items should have children structure
        assert items_with_children / sample_size >= 0.5, (
            "Less than 50% of items have children in hierarchy"
        )

    @pytest.mark.integration
    def test_screenshot_files_exist(self, rico_provider):
        """Screenshot paths reference existing files."""
        sample_size = 50
        screenshots_found = 0
        screenshots_missing = 0
        no_screenshot = 0

        for i, item in enumerate(rico_provider.process()):
            if i >= sample_size:
                break

            if item.screenshot_path:
                if item.screenshot_path.exists():
                    screenshots_found += 1
                else:
                    screenshots_missing += 1
            else:
                no_screenshot += 1

        # Allow up to 10% missing screenshots
        if screenshots_found + screenshots_missing > 0:
            total = screenshots_found + screenshots_missing
            missing_rate = screenshots_missing / total
            assert missing_rate < 0.10, (
                f"Too many missing screenshots: {screenshots_missing}/{total}"
            )
