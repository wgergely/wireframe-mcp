import logging
import sys
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.backend.base import GenerationConfig, GenerationResult, LLMBackend
from src.llm.generator import GeneratorConfig, LayoutGenerator
from src.mid import LayoutNode
from src.providers import get_provider
from src.render import OutputFormat, RenderClient, RenderConfig
from src.schema import ComponentType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("workbench")

# Sample JSON responses to mock LLM output
MOCK_LOGIN_JSON = """
{
    "id": "login-page",
    "type": "container",
    "orientation": "vertical",
    "align": "center",
    "justify": "center",
    "padding": 40,
    "children": [
        {
            "id": "login-card",
            "type": "card",
            "padding": 32,
            "gap": 16,
            "children": [
                {
                    "id": "header",
                    "type": "text",
                    "label": "Welcome Back",
                    "text_size": "title",
                    "text_weight": "bold",
                    "text_align": "center"
                },
                {
                    "id": "email-input",
                    "type": "input",
                    "label": "Email Address"
                },
                {
                    "id": "password-input",
                    "type": "input",
                    "label": "Password"
                },
                {
                    "id": "submit-btn",
                    "type": "button",
                    "label": "Log In"
                }
            ]
        }
    ]
}
"""

MOCK_DASHBOARD_JSON = """
{
    "id": "dashboard-root",
    "type": "container",
    "orientation": "horizontal",
    "children": [
        {
            "id": "sidebar",
            "type": "toolbar",
            "orientation": "vertical",
            "flex_ratio": 2,
            "padding": 16,
            "gap": 8,
            "children": [
                {"id": "menu-1", "type": "button", "label": "Overview"},
                {"id": "menu-2", "type": "button", "label": "Sales"},
                {"id": "menu-3", "type": "button", "label": "Settings"}
            ]
        },
        {
            "id": "main-area",
            "type": "container",
            "flex_ratio": 10,
            "orientation": "vertical",
            "padding": 24,
            "gap": 24,
            "children": [
                {
                    "id": "top-bar",
                    "type": "toolbar",
                    "orientation": "horizontal",
                    "justify": "between",
                    "children": [
                        {
                            "id": "page-title",
                            "type": "text",
                            "label": "Sales Overview",
                            "text_size": "heading"
                        },
                        {"id": "profile", "type": "icon", "label": "User"}
                    ]
                },
                {
                    "id": "stats-grid",
                    "type": "container",
                    "orientation": "horizontal",
                    "gap": 16,
                    "children": [
                        {
                            "id": "card-1",
                            "type": "card",
                            "label": "Total Revenue",
                            "flex_ratio": 4
                        },
                        {
                            "id": "card-2",
                            "type": "card",
                            "label": "New Users",
                            "flex_ratio": 4
                        },
                        {
                            "id": "card-3",
                            "type": "card",
                            "label": "Active Sessions",
                            "flex_ratio": 4
                        }
                    ]
                }
            ]
        }
    ]
}
"""


class MockLLMBackend(LLMBackend):
    """Mock backend for answering layout queries without API keys."""

    @property
    def model_name(self) -> str:
        return "mock-model-v1"

    @property
    def provider(self) -> str:
        return "mock"

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def context_window(self) -> int:
        return 4096

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        logger.info(f"MockLLM generating for prompt: {prompt[:50]}...")

        # Simple keyword matching to select response
        prompt_lower = prompt.lower()

        # Prioritize "dashboard" layout if explicitly requested in the
        # user query part (heuristic) or if conflicting
        # "login" might appear in schema examples, so we check dashboard
        # first or look for "brief" context

        if "dashboard" in prompt_lower:
            content = MOCK_DASHBOARD_JSON
        elif "login" in prompt_lower:
            content = MOCK_LOGIN_JSON
        else:
            # Default to dashboard
            content = MOCK_DASHBOARD_JSON

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
        import json

        result = self.generate(prompt, system_prompt=system_prompt, config=config)
        return json.loads(result.content)


def analyze_layout_compliance(node: LayoutNode, image_path: Path) -> dict[str, Any]:
    """Analyze the generated layout for standards compliance.

    Performs assertions on the structural properties of the layout
    and verifies the image generation.
    """
    report = {
        "status": "PASS",
        "checks": [],
        "image_valid": False,
        "ux_score": 0,
        "issues": [],
    }

    # 1. Image Check
    if image_path.exists() and image_path.stat().st_size > 0:
        report["image_valid"] = True
        report["checks"].append("Image generated successfully")
    else:
        report["status"] = "FAIL"
        report["issues"].append("Image file missing or empty")

    # 2. Container Padding/Gap Check
    # Heuristic: Root containers should usually have padding
    if node.padding and node.padding >= 16:
        report["checks"].append(f"Root padding sufficient ({node.padding}px)")
        report["ux_score"] += 10
    else:
        # Not a hard fail, but a warning
        if node.type == ComponentType.CONTAINER:
            report["issues"].append("Root container has low/no padding")

    # 3. Hierarchy Check
    # Check depth to ensure it's not too flat
    def get_depth(n: LayoutNode) -> int:
        if not n.children:
            return 1
        return 1 + max(get_depth(c) for c in n.children)

    depth = get_depth(node)
    if depth >= 3:
        report["checks"].append(f"Layout depth sufficient ({depth} levels)")
        report["ux_score"] += 10
    else:
        report["issues"].append("Layout is too flat (depth < 3)")

    # 4. Text Accessibility Check (Mock)
    # Check if text nodes have sizing properties
    def check_text_nodes(n: LayoutNode) -> int:
        count = 0
        if n.type == ComponentType.TEXT:
            if n.text_size or n.text_weight:
                count += 1
        for c in n.children:
            count += check_text_nodes(c)
        return count

    styled_text_count = check_text_nodes(node)
    report["checks"].append(f"Found {styled_text_count} styled text elements")
    report["ux_score"] += styled_text_count * 5

    return report


def main():
    print("🚀 Starting Deep Testing Workbench")
    print("================================")

    # Setup
    output_dir = Path("deep_testing/output")
    output_dir.mkdir(exist_ok=True)

    render_client = RenderClient()
    if not render_client.is_available():
        print("⚠️  Kroki not available, skipping render step")

    # Initialize Generator with Mock Backend
    generator = LayoutGenerator(
        backend=MockLLMBackend(), config=GeneratorConfig(max_retries=1)
    )

    briefs = [
        "A secure login page for a fintech app",
        "A sales analytics dashboard for a pet store",
    ]

    for i, brief in enumerate(briefs, 1):
        print(f"\n📋 Brief {i}: '{brief}'")
        print("-" * 40)

        try:
            # 1. Generate
            print("🤖 Generative Phase...")
            output = generator.generate(brief, target_provider="plantuml")
            node = output.context.node
            print(
                f"   ✓ Generated LayoutNode: {node.id} ({len(node.children)} children)"
            )

            # 2. Transpile
            print("code Transpilation Phase...")
            provider = get_provider("plantuml")
            dsl = provider.transpile(node)
            print(f"   ✓ Transpiled to PlantUML ({len(dsl)} chars)")

            # 3. Render
            image_path = output_dir / f"brief_{i}_{node.id}.png"
            if render_client.is_available():
                print("🎨 Rendering Phase...")
                render_result = render_client.render(
                    dsl,
                    "plantuml",
                    RenderConfig(scale=2.0, output_format=OutputFormat.PNG),
                )
                render_result.save(str(image_path))
                print(
                    f"   ✓ Rendered to {image_path} ({render_result.size_bytes} bytes)"
                )

            # 4. Analysis
            print("🔍 Analysis Phase...")
            compliance = analyze_layout_compliance(node, image_path)

            print(f"   Compliance Status: {compliance['status']}")
            print(f"   UX Score: {compliance['ux_score']}")
            for check in compliance["checks"]:
                print(f"   - [PASS] {check}")
            for issue in compliance["issues"]:
                print(f"   - [WARN] {issue}")

        except Exception as e:
            print(f"❌ Failed processing brief: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
