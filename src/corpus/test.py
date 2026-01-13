"""Unit tests for corpus module."""

import pytest

from src.corpus.lib import (
    RICO_DATASETS,
    CorpusDataset,
    get_data_dir,
    list_rico_datasets,
)


class TestGetDataDir:
    """Tests for get_data_dir function."""

    @pytest.mark.unit
    def test_default_creates_data_dir(self, tmp_path, monkeypatch):
        """Default creates ./data relative to cwd."""
        monkeypatch.chdir(tmp_path)
        result = get_data_dir()
        assert result.exists()
        assert result.name == "data"

    @pytest.mark.unit
    def test_custom_dir(self, tmp_path):
        """Custom directory is created and returned."""
        custom = tmp_path / "custom_data"
        result = get_data_dir(custom)
        assert result.exists()
        assert result == custom.resolve()

    @pytest.mark.unit
    def test_nested_dir_creation(self, tmp_path):
        """Nested directories are created."""
        nested = tmp_path / "a" / "b" / "c"
        result = get_data_dir(nested)
        assert result.exists()
        assert result == nested.resolve()


class TestListRicoDatasets:
    """Tests for list_rico_datasets function."""

    @pytest.mark.unit
    def test_returns_all_datasets(self):
        """All expected datasets are returned."""
        datasets = list_rico_datasets()
        assert "semantic" in datasets
        assert "ui_screenshots" in datasets
        assert "ui_metadata" in datasets
        assert "ui_vectors" in datasets

    @pytest.mark.unit
    def test_dataset_metadata_structure(self):
        """Each dataset has required metadata."""
        datasets = list_rico_datasets()
        for _name, info in datasets.items():
            assert "url" in info
            assert "size_mb" in info
            assert "description" in info

    @pytest.mark.unit
    def test_semantic_dataset_info(self):
        """Semantic dataset has correct metadata."""
        datasets = list_rico_datasets()
        semantic = datasets["semantic"]
        assert semantic["size_mb"] == 150
        assert "semantic" in semantic["description"].lower()


class TestCorpusDataset:
    """Tests for CorpusDataset model."""

    @pytest.mark.unit
    def test_create_dataset(self, tmp_path):
        """CorpusDataset can be created with valid data."""
        dataset = CorpusDataset(
            name="test_dataset",
            path=tmp_path,
            source="rico",
            dataset_type="semantic",
            downloaded=True,
        )
        assert dataset.name == "test_dataset"
        assert dataset.path == tmp_path
        assert dataset.downloaded is True

    @pytest.mark.unit
    def test_default_downloaded_false(self, tmp_path):
        """Downloaded defaults to False."""
        dataset = CorpusDataset(
            name="test",
            path=tmp_path,
            source="rico",
            dataset_type="semantic",
        )
        assert dataset.downloaded is False


class TestRicoDatasets:
    """Tests for RICO_DATASETS constant."""

    @pytest.mark.unit
    def test_all_urls_are_valid_format(self):
        """All URLs start with https."""
        for name, info in RICO_DATASETS.items():
            assert info["url"].startswith("https://"), f"{name} URL invalid"

    @pytest.mark.unit
    def test_all_sizes_are_positive(self):
        """All sizes are positive integers."""
        for name, info in RICO_DATASETS.items():
            assert info["size_mb"] > 0, f"{name} size invalid"
