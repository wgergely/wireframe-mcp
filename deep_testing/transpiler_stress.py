import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mid import LayoutNode
from src.providers import get_provider
from src.render import OutputFormat, RenderClient, RenderConfig
from src.schema import ComponentType, Orientation


def create_deep_layout(depth: int) -> LayoutNode:
    """Create a highly nested layout."""
    root = LayoutNode(
        id="root",
        type=ComponentType.CONTAINER,
        label="Root",
        orientation=Orientation.VERTICAL,
    )
    current = root

    for i in range(depth):
        child = LayoutNode(
            id=f"level-{i + 1}",
            type=ComponentType.CONTAINER,
            label=f"Level {i + 1}",
            padding=4,
        )
        current.children.append(child)
        current = child

    return root


def main():
    print("🏗️ Starting Transpiler Stress Test")
    print("=================================")

    depths = [10, 20, 50]
    provider = get_provider("plantuml")
    render_client = RenderClient()

    for d in depths:
        print(f"\n📏 Testing Depth: {d}")

        # 1. Creation
        start = time.time()
        node = create_deep_layout(d)
        print(f"   ✓ Created ({(time.time() - start) * 1000:.2f}ms)")

        # 2. Transpilation
        try:
            start = time.time()
            dsl = provider.transpile(node)
            elapsed_ms = (time.time() - start) * 1000
            print(
                f"   ✓ Transpiled to PlantUML "
                f"({len(dsl)} chars) ({elapsed_ms:.2f}ms)"
            )
        except RecursionError:
            print("   ❌ Transpilation Failed: RecursionError")
            continue
        except Exception as e:
            print(f"   ❌ Transpilation Failed: {e}")
            continue

        # 3. Rendering
        if render_client.is_available():
            try:
                start = time.time()
                # Use SVG for potentially better scaling with large docs
                result = render_client.render(
                    dsl, "plantuml", RenderConfig(output_format=OutputFormat.SVG)
                )
                elapsed_ms = (time.time() - start) * 1000
                print(
                    f"   ✓ Rendered SVG "
                    f"({result.size_bytes} bytes) ({elapsed_ms:.2f}ms)"
                )

                # Save just to satisfy "files generated" check
                out_path = Path(f"deep_testing/output/depth_{d}.svg")
                out_path.parent.mkdir(exist_ok=True, parents=True)
                result.save(str(out_path))
            except Exception as e:
                print(f"   ❌ Rendering Failed: {e}")
        else:
            print("   ⚠️ Rendering skipped (Kroki unavailable)")


if __name__ == "__main__":
    main()
