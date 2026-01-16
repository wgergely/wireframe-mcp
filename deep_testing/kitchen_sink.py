import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mid import LayoutNode, Orientation
from src.providers import get_provider
from src.render import RenderClient, RenderConfig
from src.schema import ComponentType


def create_kitchen_sink() -> LayoutNode:
    """Create a layout containing every ComponentType."""

    # We'll group them by category for readability

    # Containers
    containers = LayoutNode(
        id="containers",
        type=ComponentType.CONTAINER,
        label="Containers",
        children=[
            LayoutNode(id="card", type=ComponentType.CARD, label="Card Component"),
            LayoutNode(id="modal", type=ComponentType.MODAL, label="Modal Component"),
            LayoutNode(id="webview", type=ComponentType.WEB_VIEW, label="Web View"),
        ],
    )

    # Navigation
    navs = LayoutNode(
        id="navs",
        type=ComponentType.CONTAINER,
        label="Navigation",
        children=[
            LayoutNode(id="toolbar", type=ComponentType.TOOLBAR, label="Toolbar"),
            LayoutNode(
                id="navbar",
                type=ComponentType.NAVBAR,
                children=[
                    LayoutNode(
                        id="nav1", type=ComponentType.TEXT_BUTTON, label="Item 1"
                    ),
                    LayoutNode(
                        id="nav2", type=ComponentType.TEXT_BUTTON, label="Item 2"
                    ),
                ],
            ),
            LayoutNode(
                id="bottom_nav",
                type=ComponentType.BOTTOM_NAV,
                children=[
                    LayoutNode(id="bn1", type=ComponentType.ICON, label="Home"),
                    LayoutNode(id="bn2", type=ComponentType.ICON, label="Search"),
                ],
            ),
            LayoutNode(id="drawer", type=ComponentType.DRAWER, label="Drawer"),
            LayoutNode(
                id="tab_bar",
                type=ComponentType.TAB_BAR,
                children=[
                    LayoutNode(id="tab1", type=ComponentType.TEXT, label="Tab 1"),
                    LayoutNode(id="tab2", type=ComponentType.TEXT, label="Tab 2"),
                ],
            ),
            LayoutNode(id="pager", type=ComponentType.PAGER_INDICATOR, label="Pager"),
        ],
    )

    # Content
    content = LayoutNode(
        id="content",
        type=ComponentType.CONTAINER,
        label="Content",
        children=[
            LayoutNode(id="text", type=ComponentType.TEXT, label="Plain Text"),
            LayoutNode(id="image", type=ComponentType.IMAGE, label="Image"),
            LayoutNode(id="list_item", type=ComponentType.LIST_ITEM, label="List Item"),
            LayoutNode(id="icon", type=ComponentType.ICON, label="Icon"),
            LayoutNode(id="ad", type=ComponentType.ADVERTISEMENT, label="Ad Prompt"),
        ],
    )

    # Controls
    controls = LayoutNode(
        id="controls",
        type=ComponentType.CONTAINER,
        label="Controls",
        children=[
            LayoutNode(id="btn", type=ComponentType.BUTTON, label="Button"),
            LayoutNode(
                id="txt_btn", type=ComponentType.TEXT_BUTTON, label="Text Button"
            ),
            LayoutNode(id="input", type=ComponentType.INPUT, label="Input"),
            LayoutNode(id="check", type=ComponentType.CHECKBOX, label="Checkbox"),
            LayoutNode(id="radio", type=ComponentType.RADIO_BUTTON, label="Radio"),
            LayoutNode(id="switch", type=ComponentType.SWITCH, label="Switch"),
            LayoutNode(id="slider", type=ComponentType.SLIDER, label="Slider"),
            LayoutNode(
                id="spinner", type=ComponentType.SPINNER, label="Spinner/Dropdown"
            ),
            LayoutNode(id="date", type=ComponentType.DATE_PICKER, label="Date Picker"),
            LayoutNode(id="step", type=ComponentType.NUMBER_STEPPER, label="Stepper"),
        ],
    )

    return LayoutNode(
        id="root",
        type=ComponentType.CONTAINER,
        label="Kitchen Sink of All Components",
        orientation=Orientation.VERTICAL,
        padding=10,
        children=[containers, navs, content, controls],
    )


def main():
    print("🍽️ Creating Kitchen Sink Layout...")
    layout = create_kitchen_sink()

    provider = get_provider("plantuml")
    dsl = provider.transpile(layout)

    print("\n📝 DSL Output (Snippet):")
    print(dsl[:500] + "...")

    render_client = RenderClient()
    if render_client.is_available():
        print("\n🎨 Rendering...")
        result = render_client.render(dsl, "plantuml", RenderConfig(scale=1.5))

        out_path = Path("deep_testing/output/kitchen_sink.png")
        out_path.parent.mkdir(exist_ok=True, parents=True)
        result.save(str(out_path))
        print(f"✓ Saved to {out_path}")

        # Try to open it
        try:
            import os

            os.startfile(str(out_path))
        except OSError:
            pass
    else:
        print("⚠️ Kroki unavailable")


if __name__ == "__main__":
    main()
