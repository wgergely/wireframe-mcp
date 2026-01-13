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
    def test_fetch_calls_urlretrieve(self, tmp_path, monkeypatch):
        """Fetch method calls download and extracts."""

        # Mock urlretrieve
        mock_called = False

        def mock_urlretrieve(url, filename, reporthook=None):
            nonlocal mock_called
            mock_called = True
            # Create a dummy zip file
            import zipfile

            with zipfile.ZipFile(filename, "w") as zf:
                zf.writestr("test.json", "{}")

        monkeypatch.setattr(
            "src.corpus.provider.rico.lib.urlretrieve", mock_urlretrieve
        )

        provider = RicoProvider(tmp_path, dataset_type="semantic")
        provider.fetch()

        assert mock_called
        assert (tmp_path / "rico" / "semantic").exists()

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
