"""LLM module test fixtures."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Generator

import pytest

if TYPE_CHECKING:
    from src.llm.backend.base import GenerationConfig, GenerationResult


# =============================================================================
# Mock LLM Backend
# =============================================================================


class MockLLMBackend:
    """Mock LLM backend for testing without API keys.

    Provides deterministic responses based on prompt keywords for
    testing the full generation pipeline.
    """

    MOCK_LOGIN_JSON = """{
    "id": "login-page",
    "type": "container",
    "orientation": "vertical",
    "children": [
        {"id": "email", "type": "input", "label": "Email"},
        {"id": "password", "type": "input", "label": "Password"},
        {"id": "submit", "type": "button", "label": "Log In"}
    ]
}"""

    MOCK_DASHBOARD_JSON = """{
    "id": "dashboard",
    "type": "container",
    "orientation": "horizontal",
    "children": [
        {"id": "sidebar", "type": "toolbar", "orientation": "vertical", "children": []},
        {"id": "main", "type": "container", "orientation": "vertical", "children": []}
    ]
}"""

    MOCK_FORM_JSON = """{
    "id": "form",
    "type": "card",
    "children": [
        {"id": "title", "type": "text", "label": "Contact Us"},
        {"id": "name", "type": "input", "label": "Name"},
        {"id": "email", "type": "input", "label": "Email"},
        {"id": "message", "type": "input", "label": "Message"},
        {"id": "submit", "type": "button", "label": "Send"}
    ]
}"""

    @property
    def model_name(self) -> str:
        """Return mock model name."""
        return "mock-model-v1"

    @property
    def provider(self) -> str:
        """Return mock provider name."""
        return "mock"

    @property
    def supports_json_mode(self) -> bool:
        """Mock supports JSON mode."""
        return True

    @property
    def context_window(self) -> int:
        """Return mock context window size."""
        return 4096

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate mock response based on prompt keywords.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            config: Generation config (ignored).

        Returns:
            Mock GenerationResult with appropriate layout JSON.
        """
        from src.llm.backend.base import GenerationResult

        prompt_lower = prompt.lower()

        if "dashboard" in prompt_lower:
            content = self.MOCK_DASHBOARD_JSON
        elif "form" in prompt_lower or "contact" in prompt_lower:
            content = self.MOCK_FORM_JSON
        elif "login" in prompt_lower or "sign" in prompt_lower:
            content = self.MOCK_LOGIN_JSON
        else:
            content = self.MOCK_DASHBOARD_JSON

        return GenerationResult(
            content=content,
            finish_reason="stop",
            model=self.model_name,
            usage={"total_tokens": 100},
        )

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
    ) -> dict[str, Any]:
        """Generate mock JSON response.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            config: Generation config (ignored).

        Returns:
            Parsed JSON response.
        """
        import json

        result = self.generate(prompt, system_prompt=system_prompt, config=config)
        return json.loads(result.content)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_backend() -> MockLLMBackend:
    """Create a mock LLM backend for testing.

    Returns:
        MockLLMBackend instance.
    """
    return MockLLMBackend()


@pytest.fixture
def mock_api_key() -> str:
    """Provide a mock API key for testing.

    Returns:
        A test API key string.
    """
    return "test-api-key-12345"


@pytest.fixture
def preserve_env_keys() -> Generator[None, None, None]:
    """Fixture to preserve and restore API keys during tests.

    Saves existing API key environment variables before test and
    restores them after, allowing tests to modify them safely.

    Yields:
        None (context manager style).
    """
    saved_keys: dict[str, str | None] = {}
    key_names = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "QWEN_API_KEY",
        "VOYAGE_API_KEY",
    ]

    # Save existing values
    for key in key_names:
        saved_keys[key] = os.environ.get(key)

    yield

    # Restore values
    for key, value in saved_keys.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def mock_llm_response() -> dict[str, Any]:
    """Create a mock LLM response dictionary for testing.

    Returns:
        Dictionary matching expected LLM JSON output format.
    """
    return {
        "id": "root",
        "type": "container",
        "orientation": "vertical",
        "children": [
            {"id": "header", "type": "toolbar", "label": "Header"},
            {"id": "content", "type": "container"},
        ],
    }


@pytest.fixture
def sample_prompts() -> list[str]:
    """Sample prompts for generation testing.

    Returns:
        List of test prompt strings.
    """
    return [
        "Create a login page with email and password fields",
        "Design a dashboard with sidebar navigation",
        "Build a contact form with name, email, and message",
        "Make a settings page with toggle switches",
    ]
