import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mid import LayoutNode, validate_layout
from src.schema import ComponentType


def run_test(name: str, node: LayoutNode, expected_error_type: str | None):
    print(f"Running test: {name}")
    errors = validate_layout(node)

    if expected_error_type is None:
        if errors:
            print(f"❌ Expected VALID, but got errors: {errors}")
            return False
        print("✅ Passed (Valid)")
        return True

    # Expect specific error
    matched = [e for e in errors if e.error_type == expected_error_type]
    if matched:
        print(f"✅ Passed (Caught expected error: {expected_error_type})")
        return True
    else:
        if errors:
            error_types = [e.error_type for e in errors]
            print(
                f"❌ Expected {expected_error_type}, "
                f"but got different errors: {error_types}"
            )
        else:
            print(
                f"❌ Expected {expected_error_type}, "
                f"but validation PASSED unexpectedly."
            )
        return False


def main():
    print("🧪 Starting Stress/Edge Case Tests")
    print("=================================")

    failures = 0

    # 1. Invalid Flex Ratio (Existing Logic)
    # Should be caught by Pydantic OR validate_layout
    try:
        invalid_flex = LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            flex_ratio=15,  # Invalid > 12
        )
        if not run_test("Invalid Flex Ratio (>12)", invalid_flex, "invalid_flex_ratio"):
            failures += 1
    except Exception as e:
        print(f"✅ Passed (Caught Pydantic error for Flex Ratio: {e})")

    # 2. Component Constraints: Button with Children
    # Should FAIL, but currently PASSES (Bug)
    button_with_kids = LayoutNode(
        id="root",
        type=ComponentType.BUTTON,
        label="Click Me",
        children=[LayoutNode(id="child", type=ComponentType.ICON)],
    )
    # We EXPECT 'constraint_violation' (future) or similar.
    # For now, I'm asserting specifically what I WANT to happen.
    if not run_test(
        "Button with Children (Schema Constraint)",
        button_with_kids,
        "constraint_violation",
    ):
        failures += 1

    # 3. Component Constraints: Max Children
    # NAVBAR allows max 7 (from schema). Let's put 8.
    nav_children = [
        LayoutNode(id=f"item-{i}", type=ComponentType.ICON) for i in range(8)
    ]
    overpopulated_nav = LayoutNode(
        id="nav", type=ComponentType.NAVBAR, children=nav_children
    )
    if not run_test(
        "Navbar Max Children Exceeded", overpopulated_nav, "constraint_violation"
    ):
        failures += 1

    print("=================================")
    if failures > 0:
        print(f"💀 {failures} tests failed (as expected before fix).")
        sys.exit(1)
    else:
        print("🎉 All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
