"""Kroki rendering client implementation.

Provides flexible diagram rendering via HTTP API with configurable
output format, themes, scaling, and sizing options.
"""

import base64
import os
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import httpx

from src.ir import LayoutNode
from src.providers import get_provider


class OutputFormat(Enum):
    """Supported output formats for rendering."""

    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    JPEG = "jpeg"


class D2Theme(Enum):
    """D2 diagram themes.

    See: https://d2lang.com/tour/themes/
    """

    DEFAULT = 0
    NEUTRAL_GRAY = 1
    FLAGSHIP_TERRASTRUCT = 3
    COOL_CLASSICS = 4
    MIXED_BERRY_BLUE = 5
    GRAPE_SODA = 6
    AUBERGINE = 7
    COLORBLIND_CLEAR = 8
    VANILLA_NITRO_COLA = 100
    ORANGE_CREAMSICLE = 101
    SHIRLEY_TEMPLE = 102
    EARTH_TONES = 103
    EVERGLADE_GREEN = 104
    BUTTERED_TOAST = 105
    TERMINAL = 300
    TERMINAL_GRAYSCALE = 301
    ORIGAMI = 302


class PlantUMLTheme(Enum):
    """PlantUML themes.

    See: https://plantuml.com/theme
    """

    DEFAULT = ""
    AMIGA = "amiga"
    AWS_ORANGE = "aws-orange"
    BLACK_KNIGHT = "black-knight"
    BLUEGRAY = "bluegray"
    BLUEPRINT = "blueprint"
    CERULEAN = "cerulean"
    CERULEAN_OUTLINE = "cerulean-outline"
    CRT_AMBER = "crt-amber"
    CRT_GREEN = "crt-green"
    CYBORG = "cyborg"
    CYBORG_OUTLINE = "cyborg-outline"
    HACKER = "hacker"
    LIGHTGRAY = "lightgray"
    MARS = "mars"
    MATERIA = "materia"
    MATERIA_OUTLINE = "materia-outline"
    METAL = "metal"
    MIMEOGRAPH = "mimeograph"
    MINTY = "minty"
    PLAIN = "plain"
    SANDSTONE = "sandstone"
    SILVER = "silver"
    SKETCHY = "sketchy"
    SKETCHY_OUTLINE = "sketchy-outline"
    SPACELAB = "spacelab"
    SPACELAB_WHITE = "spacelab-white"
    SUPERHERO = "superhero"
    SUPERHERO_OUTLINE = "superhero-outline"
    TOY = "toy"
    UNITED = "united"
    VIBRANT = "vibrant"


@dataclass
class RenderConfig:
    """Configuration for diagram rendering.

    Attributes:
        output_format: Image output format (PNG, SVG, PDF, JPEG).
        theme: Theme for the diagram (D2Theme or PlantUMLTheme).
        scale: Scale factor for the output (0.5 to 3.0).
        width: Override output width in pixels.
        height: Override output height in pixels.
        background: Background color (e.g., "#ffffff", "transparent").
        sketch: Enable sketch/hand-drawn style (D2 only).
    """

    output_format: OutputFormat = OutputFormat.PNG
    theme: D2Theme | PlantUMLTheme | None = None
    scale: float = 1.0
    width: int | None = None
    height: int | None = None
    background: str | None = None
    sketch: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.1 <= self.scale <= 5.0:
            raise ValueError(f"Scale must be between 0.1 and 5.0, got {self.scale}")
        if self.width is not None and self.width < 1:
            raise ValueError(f"Width must be positive, got {self.width}")
        if self.height is not None and self.height < 1:
            raise ValueError(f"Height must be positive, got {self.height}")


@dataclass
class RenderOutput:
    """Result of diagram rendering.

    Attributes:
        image_bytes: Raw image data.
        format: Output format used.
        diagram_type: Source diagram type (d2, plantuml).
        width: Image width in pixels (if available).
        height: Image height in pixels (if available).
        metadata: Additional rendering metadata.
    """

    image_bytes: bytes
    format: OutputFormat
    diagram_type: str
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        """Size of image in bytes."""
        return len(self.image_bytes)

    def save(self, path: str) -> None:
        """Save image to file.

        Args:
            path: File path to save to.
        """
        with open(path, "wb") as f:
            f.write(self.image_bytes)


class RenderError(Exception):
    """Error during rendering."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class RenderClient:
    """HTTP client for Kroki diagram rendering.

    Supports rendering D2 and PlantUML diagrams to PNG, SVG, PDF, and JPEG
    via a self-hosted or public Kroki instance.

    Example:
        >>> client = RenderClient()
        >>> if client.is_available():
        ...     result = client.render("root: Hello { }", "d2")
        ...     result.save("diagram.png")

    Attributes:
        base_url: Kroki service URL.
        timeout: Request timeout in seconds.
    """

    # Diagram type to Kroki endpoint mapping
    _DIAGRAM_ENDPOINTS = {
        "d2": "d2",
        "plantuml": "plantuml",
        "puml": "plantuml",
    }

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize render client.

        Args:
            base_url: Kroki service URL. Defaults to KROKI_URL env var
                     or http://localhost:8000.
            timeout: Request timeout in seconds.
        """
        self.base_url = (
            base_url or os.environ.get("KROKI_URL") or "http://localhost:8000"
        ).rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def __del__(self) -> None:
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()

    def is_available(self) -> bool:
        """Check if Kroki service is reachable.

        Returns:
            True if service responds, False otherwise.
        """
        try:
            response = self._client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    def render(
        self,
        dsl_code: str,
        diagram_type: Literal["d2", "plantuml", "puml"],
        config: RenderConfig | None = None,
    ) -> RenderOutput:
        """Render DSL code to an image.

        Args:
            dsl_code: DSL source code (D2 or PlantUML).
            diagram_type: Type of diagram ("d2", "plantuml").
            config: Rendering configuration.

        Returns:
            RenderOutput with image bytes and metadata.

        Raises:
            RenderError: If rendering fails.
        """
        config = config or RenderConfig()
        endpoint = self._DIAGRAM_ENDPOINTS.get(diagram_type.lower())
        if not endpoint:
            raise RenderError(f"Unsupported diagram type: {diagram_type}")

        # Inject theme/options into DSL code
        modified_dsl = self._inject_options(dsl_code, diagram_type, config)

        # Encode diagram for Kroki API
        encoded = self._encode_diagram(modified_dsl)

        # Build URL with format
        url = f"{self.base_url}/{endpoint}/{config.output_format.value}/{encoded}"

        try:
            response = self._client.get(url)

            if response.status_code != 200:
                raise RenderError(
                    f"Kroki returned {response.status_code}",
                    status_code=response.status_code,
                    response_body=response.text[:500],
                )

            return RenderOutput(
                image_bytes=response.content,
                format=config.output_format,
                diagram_type=diagram_type,
                metadata={
                    "url": url,
                    "scale": config.scale,
                    "theme": config.theme.value if config.theme else None,
                },
            )

        except httpx.TimeoutException as e:
            raise RenderError(f"Kroki request timed out: {e}") from e
        except httpx.RequestError as e:
            raise RenderError(f"Kroki request failed: {e}") from e

    def render_layout(
        self,
        node: LayoutNode,
        provider: str = "d2",
        config: RenderConfig | None = None,
    ) -> RenderOutput:
        """Render a LayoutNode to an image.

        Transpiles the node to DSL code and renders it.

        Args:
            node: LayoutNode to render.
            provider: DSL provider name ("d2" or "plantuml").
            config: Rendering configuration.

        Returns:
            RenderOutput with image bytes.
        """
        dsl_provider = get_provider(provider)
        dsl_code = dsl_provider.transpile(node)
        return self.render(dsl_code, provider, config)

    def _encode_diagram(self, dsl_code: str) -> str:
        """Encode diagram for Kroki URL.

        Uses zlib compression + base64 URL-safe encoding.

        Args:
            dsl_code: Raw DSL code.

        Returns:
            Encoded string for URL path.
        """
        compressed = zlib.compress(dsl_code.encode("utf-8"), level=9)
        encoded = base64.urlsafe_b64encode(compressed).decode("ascii")
        return encoded

    def _inject_options(
        self,
        dsl_code: str,
        diagram_type: str,
        config: RenderConfig,
    ) -> str:
        """Inject rendering options into DSL code.

        Args:
            dsl_code: Original DSL code.
            diagram_type: Type of diagram.
            config: Rendering configuration.

        Returns:
            Modified DSL code with injected options.
        """
        if diagram_type.lower() == "d2":
            return self._inject_d2_options(dsl_code, config)
        elif diagram_type.lower() in ("plantuml", "puml"):
            return self._inject_plantuml_options(dsl_code, config)
        return dsl_code

    def _inject_d2_options(self, dsl_code: str, config: RenderConfig) -> str:
        """Inject D2-specific rendering options.

        D2 options are specified via comments at the top of the file.
        """
        options: list[str] = []

        # Theme
        if config.theme and isinstance(config.theme, D2Theme):
            options.append(f"# d2-config: --theme {config.theme.value}")

        # Sketch mode
        if config.sketch:
            options.append("# d2-config: --sketch")

        # Scale/size (D2 uses --scale flag)
        if config.scale != 1.0:
            options.append(f"# d2-config: --scale {config.scale}")

        if options:
            return "\n".join(options) + "\n\n" + dsl_code
        return dsl_code

    def _inject_plantuml_options(self, dsl_code: str, config: RenderConfig) -> str:
        """Inject PlantUML-specific rendering options.

        PlantUML options are specified via !theme directive.
        """
        # Find @startsalt or @startuml to inject after
        injections: list[str] = []

        # Theme
        if config.theme and isinstance(config.theme, PlantUMLTheme):
            if config.theme.value:
                injections.append(f"!theme {config.theme.value}")

        # Scale
        if config.scale != 1.0:
            injections.append(f"scale {config.scale}")

        if not injections:
            return dsl_code

        # Inject after @start directive
        lines = dsl_code.split("\n")
        result: list[str] = []
        injected = False

        for line in lines:
            result.append(line)
            if not injected and line.strip().startswith("@start"):
                result.extend(injections)
                injected = True

        return "\n".join(result)


__all__ = [
    "D2Theme",
    "OutputFormat",
    "PlantUMLTheme",
    "RenderClient",
    "RenderConfig",
    "RenderError",
    "RenderOutput",
]
