import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mid import LayoutNode, Orientation, SemanticColor
from src.providers import get_provider
from src.render import RenderClient
from src.schema import ComponentType


def main():
    print("🚀 Starting Full Stack Alignment Test")
    print("===================================")

    # 1. Advanced Containers
    advanced = LayoutNode(
        id="adv",
        type=ComponentType.CONTAINER,
        orientation=Orientation.VERTICAL,
        gap=10,
        children=[
            # A. Scrollable
            LayoutNode(
                id="scroll",
                type=ComponentType.CONTAINER,
                label="Scrollable Area",
                scrollable=True,
                children=[
                    LayoutNode(id="cnt", type=ComponentType.TEXT, label="Content")
                ],
            ),
            # B. Titled Card
            LayoutNode(
                id="card",
                type=ComponentType.CONTAINER,
                label="My Titled Card",
                children=[
                    LayoutNode(id="c1", type=ComponentType.TEXT, label="Card Body")
                ],
            ),
            # C. Window
            LayoutNode(
                id="win",
                type=ComponentType.MODAL,
                label="My App Window",
                children=[
                    LayoutNode(id="w1", type=ComponentType.TEXT, label="Inside Modal")
                ],
            ),
        ],
    )

    # 2. New Component Types
    new_types = LayoutNode(
        id="new",
        type=ComponentType.CONTAINER,
        orientation=Orientation.VERTICAL,
        gap=5,
        children=[
            LayoutNode(
                id="div1",
                type=ComponentType.DIVIDER,
                orientation=Orientation.HORIZONTAL,
            ),
            LayoutNode(
                id="menu",
                type=ComponentType.MENU_BAR,
                children=[
                    LayoutNode(id="m1", type=ComponentType.TEXT_BUTTON, label="File"),
                    LayoutNode(id="m2", type=ComponentType.TEXT_BUTTON, label="Edit"),
                ],
            ),
            LayoutNode(
                id="tree",
                type=ComponentType.TREE,
                children=[
                    LayoutNode(id="t1", type=ComponentType.LIST_ITEM, label="Root"),
                    LayoutNode(id="t2", type=ComponentType.LIST_ITEM, label="Child"),
                ],
            ),
        ],
    )

    # 3. Colors
    colors = LayoutNode(
        id="cols",
        type=ComponentType.CONTAINER,
        orientation=Orientation.HORIZONTAL,
        children=[
            LayoutNode(
                id="c1",
                type=ComponentType.TEXT,
                label="Primary",
                semantic_color=SemanticColor.PRIMARY,
            ),
            LayoutNode(
                id="c2",
                type=ComponentType.TEXT,
                label="Danger",
                semantic_color=SemanticColor.DANGER,
            ),
        ],
    )

    # Check DSL
    provider = get_provider("plantuml")

    # Test 1
    dsl_adv = provider.transpile(advanced)
    print("\n[Advanced Containers DSL]")
    if "{S" in dsl_adv:
        print("✅ Scrollable detected")
    else:
        print("❌ Scrollable MISSING")

    if "{^" in dsl_adv and '"My Titled Card"' in dsl_adv:
        print("✅ Titled Card detected")
    else:
        print("❌ Titled Card MISSING")

    if "{*" in dsl_adv and '"My App Window"' in dsl_adv:
        print("✅ Window detected")
    else:
        print("❌ Window MISSING")

    # Test 2
    dsl_new = provider.transpile(new_types)
    print("\n[New Types DSL]")
    if ".." in dsl_new:
        print("✅ Divider detected")
    else:
        print("❌ Divider MISSING")

    if "{*" in dsl_new and "[File]" in dsl_new:
        print("✅ MenuBar detected")
    else:
        print("❌ MenuBar MISSING")

    # Tree might be tricky, it prints {+ then children.
    # Note: LayoutNode tree handling in provider just iterates children.
    # Salt expected: {+ \n + Root \n ++ Child }?
    # Currently my provider logic for children is:
    # _transpile_node(child, indent).
    # Child ListItems usually print "Label".
    # For Tree, existing logic prints `  "Root" \n   "Child"`.
    # This WON'T verify the `+` `++` hierarchy syntax required by
    # Salt Trees unless I updated `_transpile_node` logic for Tree
    # children!
    # I did NOT update child iteration logic for Tree.
    # So "Tree" test implies I missed a logic step in the Provider.
    print("⚠️ checking tree...")
    print(dsl_new)

    # Test 3
    dsl_col = provider.transpile(colors)
    print("\n[Colors DSL]")
    if "<color:Blue>" in dsl_col:
        print("✅ Primary Blue detected")
    else:
        print("❌ Primary Blue MISSING")

    # Render
    client = RenderClient()
    if client.is_available():
        client.render(dsl_adv + "\n" + dsl_new + "\n" + dsl_col, "plantuml").save(
            "deep_testing/output/full_stack.png"
        )
        print("✓ Saved deep_testing/output/full_stack.png")


if __name__ == "__main__":
    main()
