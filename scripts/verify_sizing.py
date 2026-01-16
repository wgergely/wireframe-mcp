from src.mid import (
    AlignContent,
    ComponentType,
    LayoutNode,
    Orientation,
    validate_layout,
)


def test_scenario():
    print("Testing 'Holy Grail' Layout (Fixed Sidebar + Fluid Content)")

    # PROBLEM: We cannot specify "Sidebar is 250px". Use flex_ratio=3 as proxy?
    # But 3/12 of screen depends on screen size. Not fixed.
    root = LayoutNode(
        id="root",
        type=ComponentType.CONTAINER,
        orientation=Orientation.HORIZONTAL,
        children=[
            LayoutNode(
                id="sidebar",
                type=ComponentType.CONTAINER,
                width=250,  # Fixed Width!
                label="Fixed Sidebar 250px",
            ),
            LayoutNode(
                id="content",
                type=ComponentType.CONTAINER,
                flex_ratio=1,  # Fill remaining
                label="Fluid Content",
            ),
        ],
    )

    errors = validate_layout(root)
    if errors:
        print("Validation Errors:", errors)
    else:
        print("Validation Passed! Fixed sidebar defined with width=250")

    print("\nTesting 'Toolbar with Fixed Spacer'")
    # PROBLEM: Specific 20px spacer
    toolbar = LayoutNode(
        id="toolbar",
        type=ComponentType.TOOLBAR,
        orientation=Orientation.HORIZONTAL,
        align_content=AlignContent.CENTER,
        children=[
            LayoutNode(id="btn_a", type=ComponentType.BUTTON, label="A"),
            # Fixed 20px spacer
            LayoutNode(id="spacer_20px", type=ComponentType.CONTAINER, width=20),
            LayoutNode(id="btn_b", type=ComponentType.BUTTON, label="B"),
            LayoutNode(
                id="spacer_spring",
                type=ComponentType.CONTAINER,
                flex_ratio=1,  # Grow to push C to end
            ),
            LayoutNode(id="btn_c", type=ComponentType.BUTTON, label="C"),
        ],
    )

    print(f"Spacer width: {toolbar.children[1].width}")
    print(f"Spring flex_ratio: {toolbar.children[3].flex_ratio}")

    errors = validate_layout(toolbar)
    if errors:
        print("Validation Errors:", errors)
    else:
        print("Validation Passed! Fixed spacer defined with width=20")


if __name__ == "__main__":
    test_scenario()
