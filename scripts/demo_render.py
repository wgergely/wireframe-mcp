#!/usr/bin/env python
"""Demo script for visualizing rendered layouts.

This script demonstrates the render pipeline:
1. Takes a natural language description (or uses a pre-defined layout)
2. Creates a LayoutNode tree
3. Transpiles to D2 DSL
4. Renders to PNG via Kroki
5. Saves and opens the result for visual inspection
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mid import LayoutNode, Orientation
from src.providers import get_provider
from src.render import OutputFormat, RenderClient, RenderConfig
from src.schema import ComponentType


def create_dashboard_layout() -> LayoutNode:
    """Create a sample dashboard layout."""
    return LayoutNode(
        id="root",
        type=ComponentType.CONTAINER,
        label="Dashboard",
        orientation=Orientation.HORIZONTAL,
        gap=16,
        padding=24,
        children=[
            # Sidebar
            LayoutNode(
                id="sidebar",
                type=ComponentType.TOOLBAR,
                label="Navigation",
                orientation=Orientation.VERTICAL,
                flex_ratio=3,
                gap=8,
                padding=16,
                children=[
                    LayoutNode(
                        id="logo",
                        type=ComponentType.IMAGE,
                        label="Logo",
                    ),
                    LayoutNode(
                        id="nav-home",
                        type=ComponentType.BUTTON,
                        label="Home",
                    ),
                    LayoutNode(
                        id="nav-analytics",
                        type=ComponentType.BUTTON,
                        label="Analytics",
                    ),
                    LayoutNode(
                        id="nav-settings",
                        type=ComponentType.BUTTON,
                        label="Settings",
                    ),
                ],
            ),
            # Main content area
            LayoutNode(
                id="main",
                type=ComponentType.CONTAINER,
                label="Main Content",
                orientation=Orientation.VERTICAL,
                flex_ratio=9,
                gap=16,
                children=[
                    # Header
                    LayoutNode(
                        id="header",
                        type=ComponentType.TOOLBAR,
                        label="Header",
                        orientation=Orientation.HORIZONTAL,
                        justify="between",
                        align="center",
                        children=[
                            LayoutNode(
                                id="title",
                                type=ComponentType.TEXT,
                                label="Dashboard Overview",
                                text_size="heading",
                                text_weight="bold",
                            ),
                            LayoutNode(
                                id="user-menu",
                                type=ComponentType.BUTTON,
                                label="Profile",
                            ),
                        ],
                    ),
                    # Content grid
                    LayoutNode(
                        id="content",
                        type=ComponentType.CONTAINER,
                        label="Content",
                        orientation=Orientation.HORIZONTAL,
                        gap=16,
                        wrap="wrap",
                        children=[
                            LayoutNode(
                                id="card-revenue",
                                type=ComponentType.CARD,
                                label="Revenue",
                                flex_ratio=4,
                            ),
                            LayoutNode(
                                id="card-users",
                                type=ComponentType.CARD,
                                label="Active Users",
                                flex_ratio=4,
                            ),
                            LayoutNode(
                                id="card-orders",
                                type=ComponentType.CARD,
                                label="Orders",
                                flex_ratio=4,
                            ),
                        ],
                    ),
                    # Chart area
                    LayoutNode(
                        id="chart-section",
                        type=ComponentType.CONTAINER,
                        label="Analytics Section",
                        orientation=Orientation.HORIZONTAL,
                        gap=16,
                        children=[
                            LayoutNode(
                                id="main-chart",
                                type=ComponentType.CONTAINER,
                                label="Revenue Chart",
                                flex_ratio=8,
                            ),
                            LayoutNode(
                                id="metrics",
                                type=ComponentType.LIST_ITEM,
                                label="Key Metrics",
                                flex_ratio=4,
                                orientation=Orientation.VERTICAL,
                                gap=8,
                                children=[
                                    LayoutNode(
                                        id="metric-1",
                                        type=ComponentType.TEXT,
                                        label="Conversion: 3.2%",
                                    ),
                                    LayoutNode(
                                        id="metric-2",
                                        type=ComponentType.TEXT,
                                        label="Bounce Rate: 42%",
                                    ),
                                    LayoutNode(
                                        id="metric-3",
                                        type=ComponentType.TEXT,
                                        label="Avg. Session: 4m 32s",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def create_login_form_layout() -> LayoutNode:
    """Create a sample login form layout."""
    return LayoutNode(
        id="root",
        type=ComponentType.CONTAINER,
        label="Login Page",
        orientation=Orientation.VERTICAL,
        align="center",
        justify="center",
        children=[
            LayoutNode(
                id="form-container",
                type=ComponentType.CARD,
                label="Login Form",
                orientation=Orientation.VERTICAL,
                gap=24,
                padding=32,
                flex_ratio=4,
                children=[
                    LayoutNode(
                        id="logo",
                        type=ComponentType.IMAGE,
                        label="App Logo",
                    ),
                    LayoutNode(
                        id="title",
                        type=ComponentType.TEXT,
                        label="Welcome Back",
                        text_size="heading",
                        text_weight="bold",
                        text_align="center",
                    ),
                    LayoutNode(
                        id="email-field",
                        type=ComponentType.INPUT,
                        label="Email",
                    ),
                    LayoutNode(
                        id="password-field",
                        type=ComponentType.INPUT,
                        label="Password",
                    ),
                    LayoutNode(
                        id="options",
                        type=ComponentType.CONTAINER,
                        orientation=Orientation.HORIZONTAL,
                        justify="between",
                        children=[
                            LayoutNode(
                                id="remember",
                                type=ComponentType.CHECKBOX,
                                label="Remember me",
                            ),
                            LayoutNode(
                                id="forgot",
                                type=ComponentType.TEXT_BUTTON,
                                label="Forgot password?",
                            ),
                        ],
                    ),
                    LayoutNode(
                        id="login-btn",
                        type=ComponentType.BUTTON,
                        label="Sign In",
                    ),
                    LayoutNode(
                        id="signup-link",
                        type=ComponentType.TEXT,
                        label="Don't have an account? Sign up",
                        text_size="caption",
                        text_align="center",
                    ),
                ],
            ),
        ],
    )


LAYOUTS = {
    "dashboard": (
        "Dashboard with sidebar, header, cards, and charts",
        create_dashboard_layout,
    ),
    "login": ("Login form with email/password and options", create_login_form_layout),
}


def main():
    """Run the demo."""
    import argparse

    parser = argparse.ArgumentParser(description="Render demo layouts")
    parser.add_argument(
        "layout",
        choices=list(LAYOUTS.keys()),
        nargs="?",
        default="dashboard",
        help="Layout to render",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="plantuml",
        choices=["plantuml", "d2"],
        help="DSL provider (plantuml works with base Kroki, d2 requires kroki-d2)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "svg"],
        help="Output format",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.5,
        help="Scale factor (0.1-5.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't open the result",
    )
    parser.add_argument(
        "--show-dsl",
        action="store_true",
        help="Print the generated DSL code",
    )
    args = parser.parse_args()

    # Check Kroki availability
    client = RenderClient()
    if not client.is_available():
        print("ERROR: Kroki service is not available at", client.base_url)
        print(
            "Start Kroki with: docker run -d --name kroki -p 8000:8000 yuzutech/kroki"
        )
        sys.exit(1)

    print(f"✓ Kroki is available at {client.base_url}")

    # Create the layout
    description, layout_fn = LAYOUTS[args.layout]
    print(f"\n📐 Creating layout: {description}")
    layout = layout_fn()

    # Transpile to DSL
    provider = get_provider(args.provider)
    dsl_code = provider.transpile(layout)

    if args.show_dsl:
        print(f"\n📝 Generated {args.provider.upper()} DSL:")
        print("-" * 60)
        print(dsl_code)
        print("-" * 60)

    # Configure rendering
    config = RenderConfig(
        output_format=OutputFormat.PNG if args.format == "png" else OutputFormat.SVG,
        scale=args.scale,
    )

    print(f"\n🎨 Rendering with {args.provider}, scale: {args.scale}x")

    # Render
    result = client.render(dsl_code, args.provider, config)
    print(f"✓ Rendered {result.size_bytes:,} bytes")

    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            Path(tempfile.gettempdir()) / f"wireframe_{args.layout}.{args.format}"
        )

    result.save(str(output_path))
    print(f"✓ Saved to: {output_path}")

    # Open result
    if not args.no_open:
        print("\n🖼️ Opening rendered output...")
        if sys.platform == "win32":
            os.startfile(str(output_path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(output_path)])
        else:
            subprocess.run(["xdg-open", str(output_path)])


if __name__ == "__main__":
    main()
