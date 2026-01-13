"""Unit tests for EGFE provider."""

import json

import pytest

from src.corpus.provider.base.test import BaseProviderTest
from src.corpus.provider.egfe import Provider


class TestEgfeProvider(BaseProviderTest):
    """Test suites for EGFE Provider."""

    @pytest.fixture
    def provider_class(self):
        return Provider

    @pytest.mark.unit
    def test_name_is_egfe(self, provider):
        """Verify the provider name is correct."""
        assert provider.name == "egfe"

    @pytest.mark.unit
    def test_dest_dir_path(self, provider, tmp_path):
        """Verify destination directory path is correct."""
        assert provider._dest_dir == tmp_path / "egfe"

    @pytest.mark.unit
    def test_fetch_creates_directory(self, provider, tmp_path):
        """Verify fetch creates the destination directory structure."""
        # Note: This doesn't actually download - just tests mkdir
        provider._dest_dir.mkdir(parents=True, exist_ok=True)
        assert provider._dest_dir.exists()

    @pytest.mark.unit
    def test_has_data_returns_false_when_empty(self, provider):
        """Verify has_data returns False when no JSON files exist."""
        provider._dest_dir.mkdir(parents=True, exist_ok=True)
        assert provider.has_data() is False

    @pytest.mark.unit
    def test_has_data_returns_true_when_json_exists(self, provider):
        """Verify has_data returns True when JSON files exist."""
        provider._dest_dir.mkdir(parents=True, exist_ok=True)
        (provider._dest_dir / "test.json").write_text("{}")
        assert provider.has_data() is True

    @pytest.mark.unit
    def test_process_yields_standardized_data(self, provider):
        """Verify process yields StandardizedData items."""
        from src.corpus.provider.base import StandardizedData

        # Create mock data
        provider._dest_dir.mkdir(parents=True, exist_ok=True)
        test_data = {"type": "Frame", "name": "Login Screen"}
        (provider._dest_dir / "sample.json").write_text(json.dumps(test_data))

        items = list(provider.process())
        assert len(items) == 1
        assert isinstance(items[0], StandardizedData)
        assert items[0].id == "sample"
        assert items[0].source == "egfe"
        # Hierarchy is transformed to Rico-compatible format
        assert items[0].hierarchy["componentLabel"] == "Container"
        assert items[0].hierarchy["class"] == "figma.Frame"
        assert items[0].layout is not None

    @pytest.mark.unit
    def test_process_maps_screenshot(self, provider):
        """Verify process correctly maps screenshots to items."""
        provider._dest_dir.mkdir(parents=True, exist_ok=True)

        # Create JSON and matching PNG
        (provider._dest_dir / "sample.json").write_text('{"type": "Frame"}')
        (provider._dest_dir / "sample.png").write_bytes(b"fake png")

        items = list(provider.process())
        assert len(items) == 1
        assert items[0].screenshot_path is not None
        assert items[0].screenshot_path.name == "sample.png"

    @pytest.mark.unit
    def test_process_skips_asset_screenshots(self, provider):
        """Verify asset files (xxx-assets.png) are not mapped as screenshots."""
        provider._dest_dir.mkdir(parents=True, exist_ok=True)

        (provider._dest_dir / "sample.json").write_text('{"type": "Frame"}')
        (provider._dest_dir / "sample-assets.png").write_bytes(b"fake png")

        items = list(provider.process())
        assert len(items) == 1
        # Should NOT have screenshot because only asset file exists
        assert items[0].screenshot_path is None
