import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mid import LayoutNode, Orientation, TextSize, TextTransform, TextWeight
from src.providers import get_provider
from src.render import RenderClient
from src.schema import ComponentType


def main():
    print("🎨 Starting Style & Gap Test")
    print("===========================")

    # 1. Text Styling Node
    styled_node = LayoutNode(
        id="styled",
        type=ComponentType.CONTAINER,
        orientation=Orientation.VERTICAL,
        children=[
            LayoutNode(
                id="t1",
                type=ComponentType.TEXT,
                label="uppercase me",
                text_transform=TextTransform.UPPERCASE,
            ),
            LayoutNode(
                id="t2",
                type=ComponentType.TEXT,
                label="bold me",
                text_weight=TextWeight.BOLD,
            ),
            LayoutNode(
                id="t3",
                type=ComponentType.TEXT,
                label="Big Title",
                text_size=TextSize.TITLE,
            ),
        ],
    )

    # 2. Gap Node (Horizontal)
    gap_horiz_node = LayoutNode(
        id="gap_h",
        type=ComponentType.CONTAINER,
        orientation=Orientation.HORIZONTAL,
        gap=10,
        children=[
            LayoutNode(id="b1", type=ComponentType.BUTTON, label="A"),
            LayoutNode(id="b2", type=ComponentType.BUTTON, label="B"),
            LayoutNode(id="b3", type=ComponentType.BUTTON, label="C"),
        ],
    )

    # 3. Gap Node (Vertical)
    gap_vert_node = LayoutNode(
        id="gap_v",
        type=ComponentType.CONTAINER,
        orientation=Orientation.VERTICAL,
        label="Vertical Gap",
        gap=10,
        children=[
            LayoutNode(id="v1", type=ComponentType.TEXT, label="Top"),
            LayoutNode(id="v2", type=ComponentType.TEXT, label="Bottom"),
        ],
    )

    provider = get_provider("plantuml")

    # Test 1: Styling
    dsl_style = provider.transpile(styled_node)
    print("\n[Styling DSL Check]")
    print(dsl_style)

    if "UPPERCASE ME" in dsl_style:
        print("✅ Uppercase transform applied")
    else:
        print("❌ Uppercase transform FAILED")

    if "<b>bold me</b>" in dsl_style:
        print("✅ Bold weight applied")
    else:
        print("❌ Bold weight FAILED")

    if "<size:18>Big Title</size>" in dsl_style:
        print("✅ Title size applied")
    else:
        print("❌ Title size FAILED")

    # Test 2: Horizontal Gap
    dsl_gap_h = provider.transpile(gap_horiz_node)
    print("\n[Horizontal Gap DSL Check]")
    if "[A] | . | [B] | . | [C]" in dsl_gap_h:
        print("✅ Horizontal Gap spacers (.) present")
    else:
        print(f"❌ Horizontal Gap spacers missing. DSL: {dsl_gap_h}")

    # Test 3: Vertical Gap
    dsl_gap_v = provider.transpile(gap_vert_node)
    print("\n[Vertical Gap DSL Check]")
    if (
        '"Top"\n  "."\n"Bottom"' in dsl_gap_v
        or '"Top"\n    "."\n    "Bottom"' in dsl_gap_v.replace("  ", "    ")
    ):
        # Regex might be safer but strict string check is fine if we know indent
        print("✅ Vertical Gap spacers (.) present")
    else:
        # It has indentation, so simplistic check might fail.
        # Let's look for "." on its own line
        if '  "."' in dsl_gap_v:
            print("✅ Vertical Gap spacers (.) present")
        else:
            print(f"❌ Vertical Gap spacers missing. DSL: {dsl_gap_v}")

    # Render Visual Verification
    client = RenderClient()
    if client.is_available():
        print("\n🎨 Generating Verification Images...")
        client.render(dsl_style, "plantuml").save("deep_testing/output/style_test.png")
        client.render(dsl_gap_h, "plantuml").save("deep_testing/output/gap_h_test.png")
        client.render(dsl_gap_v, "plantuml").save("deep_testing/output/gap_v_test.png")
        print("✓ Saved debugging images to deep_testing/output/")


if __name__ == "__main__":
    main()
