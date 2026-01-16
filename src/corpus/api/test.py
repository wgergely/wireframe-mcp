"""Unit tests for Corpus API."""

from typing import Iterator

import pytest

from src.corpus.api.lib import CorpusManager
from src.corpus.provider.base import BaseProvider, DataType, StandardizedData
from src.mid import ComponentType, LayoutNode


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir)
        self.fetch_called = False
        self.fetch_force_arg = None
        self.process_called = False
        self._mock_data: list[StandardizedData] = []

    @property
    def name(self) -> str:
        return "mock_provider"

    def has_data(self, data_type: DataType | None = None) -> bool:
        """Check if data exists."""
        return len(self._mock_data) > 0

    def fetch(self, force: bool = False) -> None:
        """Track fetch calls for testing."""
        self.fetch_called = True
        self.fetch_force_arg = force

    def process(self) -> Iterator[StandardizedData]:
        """Yield mock data for testing."""
        self.process_called = True
        yield from self._mock_data

    def to_layout(self, hierarchy: dict, item_id: str) -> LayoutNode:
        """Mock implementation - not used in these tests."""
        return LayoutNode(id=item_id, type=ComponentType.CONTAINER)

    def add_mock_data(self, item: StandardizedData) -> None:
        """Add mock data for testing stream_data."""
        self._mock_data.append(item)


class TestCorpusManager:
    """Tests for CorpusManager."""

    @pytest.mark.unit
    def test_init_defaults(self, monkeypatch, tmp_path):
        """Initializes with default data path using repo root detection."""
        # Create fake repo structure with .gitignore
        (tmp_path / ".gitignore").touch()
        monkeypatch.chdir(tmp_path)
        # Clear env var to ensure default behavior
        monkeypatch.delenv("CORPUS_DATA_DIR", raising=False)

        manager = CorpusManager()

        # Should use {repo_root}/.corpus/data as default
        expected = tmp_path / ".corpus" / "data"
        assert manager.data_dir == expected
        assert "rico_semantic" in manager.providers  # Default registered

    @pytest.mark.unit
    def test_init_with_override(self, tmp_path):
        """Initializes with explicit override path."""
        manager = CorpusManager(tmp_path)
        assert manager.data_dir == tmp_path
        assert "rico_semantic" in manager.providers

    @pytest.mark.unit
    def test_register_get_provider(self, tmp_path):
        """Can register and retrieve a provider."""
        manager = CorpusManager(tmp_path)
        provider = MockProvider(tmp_path)
        manager.register_provider(provider)

        retrieved = manager.get_provider("mock_provider")
        assert retrieved == provider

    @pytest.mark.unit
    def test_get_nonexistent_provider(self, tmp_path):
        """Raises error for unknown provider."""
        manager = CorpusManager(tmp_path)
        with pytest.raises(KeyError):
            manager.get_provider("nonexistent")

    @pytest.mark.unit
    def test_list_providers(self, tmp_path):
        """Lists registered providers."""
        manager = CorpusManager(tmp_path)
        # Should have defaults
        providers = manager.list_providers()
        assert "rico_semantic" in providers
        assert "websight" in providers
        assert "egfe" in providers
        assert "showui" in providers
        assert "enrico" in providers


class TestFetchDataset:
    """Tests for CorpusManager.fetch_dataset method."""

    @pytest.mark.unit
    def test_fetch_dataset_calls_provider_fetch(self, tmp_path):
        """fetch_dataset delegates to provider.fetch()."""
        manager = CorpusManager(tmp_path)
        provider = MockProvider(tmp_path)
        manager.register_provider(provider)

        manager.fetch_dataset("mock_provider")

        assert provider.fetch_called
        assert provider.fetch_force_arg is False

    @pytest.mark.unit
    def test_fetch_dataset_with_force_flag(self, tmp_path):
        """fetch_dataset passes force flag to provider."""
        manager = CorpusManager(tmp_path)
        provider = MockProvider(tmp_path)
        manager.register_provider(provider)

        manager.fetch_dataset("mock_provider", force=True)

        assert provider.fetch_called
        assert provider.fetch_force_arg is True

    @pytest.mark.unit
    def test_fetch_dataset_nonexistent_provider(self, tmp_path):
        """fetch_dataset raises KeyError for unknown provider."""
        manager = CorpusManager(tmp_path)

        with pytest.raises(KeyError, match="nonexistent"):
            manager.fetch_dataset("nonexistent")


class TestStreamData:
    """Tests for CorpusManager.stream_data method."""

    @pytest.mark.unit
    def test_stream_data_yields_from_provider(self, tmp_path):
        """stream_data yields items from provider.process()."""
        manager = CorpusManager(tmp_path)
        provider = MockProvider(tmp_path)

        # Add mock data
        mock_item = StandardizedData(
            id="test-item-1",
            source="mock",
            dataset="test",
            hierarchy={"id": "root", "children": []},
            metadata={},
        )
        provider.add_mock_data(mock_item)
        manager.register_provider(provider)

        items = list(manager.stream_data("mock_provider"))

        assert provider.process_called
        assert len(items) == 1
        assert items[0].id == "test-item-1"
        assert items[0].source == "mock"

    @pytest.mark.unit
    def test_stream_data_multiple_items(self, tmp_path):
        """stream_data yields all items from provider."""
        manager = CorpusManager(tmp_path)
        provider = MockProvider(tmp_path)

        # Add multiple mock items
        for i in range(5):
            item = StandardizedData(
                id=f"item-{i}",
                source="mock",
                dataset="test",
                hierarchy={"id": f"node-{i}"},
                metadata={},
            )
            provider.add_mock_data(item)

        manager.register_provider(provider)

        items = list(manager.stream_data("mock_provider"))

        assert len(items) == 5
        assert [item.id for item in items] == [f"item-{i}" for i in range(5)]

    @pytest.mark.unit
    def test_stream_data_empty_provider(self, tmp_path):
        """stream_data yields nothing for empty provider."""
        manager = CorpusManager(tmp_path)
        provider = MockProvider(tmp_path)
        manager.register_provider(provider)

        items = list(manager.stream_data("mock_provider"))

        assert provider.process_called
        assert len(items) == 0

    @pytest.mark.unit
    def test_stream_data_nonexistent_provider(self, tmp_path):
        """stream_data raises KeyError for unknown provider."""
        manager = CorpusManager(tmp_path)

        with pytest.raises(KeyError, match="nonexistent"):
            list(manager.stream_data("nonexistent"))

    @pytest.mark.unit
    def test_stream_data_is_generator(self, tmp_path):
        """stream_data returns a generator (lazy evaluation)."""
        manager = CorpusManager(tmp_path)
        provider = MockProvider(tmp_path)
        provider.add_mock_data(
            StandardizedData(
                id="item",
                source="mock",
                dataset="test",
                hierarchy={},
                metadata={},
            )
        )
        manager.register_provider(provider)

        result = manager.stream_data("mock_provider")

        # Should be a generator, not yet called process
        assert hasattr(result, "__next__")
        # Process is not called until iteration
        assert not provider.process_called

        # Consume the generator
        list(result)
        assert provider.process_called


class TestGetProviderErrorMessages:
    """Tests for error message quality in CorpusManager."""

    @pytest.mark.unit
    def test_get_provider_error_lists_available(self, tmp_path):
        """KeyError message includes available providers."""
        manager = CorpusManager(tmp_path)
        provider = MockProvider(tmp_path)
        manager.register_provider(provider)

        with pytest.raises(KeyError) as exc_info:
            manager.get_provider("unknown_provider")

        error_msg = str(exc_info.value)
        assert "mock_provider" in error_msg or "rico" in error_msg
