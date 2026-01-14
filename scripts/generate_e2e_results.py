"""Generate E2E test results document for manual verification.

Run with: python -m scripts.generate_e2e_results
"""

from dotenv import load_dotenv

load_dotenv()

from src.llm import LayoutGenerator
from src.output import OutputGenerator

# Test queries covering complexity spectrum
TEST_CASES = [
    ("empty container box", "MINIMAL"),
    ("login form with email and password input fields", "LOW"),
    ("dashboard with sidebar navigation and main content area", "MEDIUM"),
    (
        "complex app with toolbar header, sidebar with nested menu items, "
        "main content area with card grid containing multiple cards each with "
        "title, description, and action buttons",
        "HIGH",
    ),
    (
        "settings panel with three tabs: general settings with toggle switches, "
        "notifications with checkbox options, and account with profile card",
        "HIGH",
    ),
]


def count_nodes(node) -> int:
    """Count total nodes in tree."""
    return 1 + sum(count_nodes(child) for child in node.children)


def max_depth(node, current=1) -> int:
    """Calculate max depth."""
    if not node.children:
        return current
    return max(max_depth(child, current + 1) for child in node.children)


def main():
    print("=" * 80)
    print("E2E TEST INPUT/OUTPUT VERIFICATION")
    print("=" * 80)

    generator = LayoutGenerator()
    output_gen = OutputGenerator()

    results = []

    for query, complexity in TEST_CASES:
        print(f"\n[{complexity}] Generating: {query[:40]}...")

        try:
            result = generator.generate(query)
            output = output_gen.generate_from_context(result.context)

            node = result.context.node
            stats = {
                "nodes": count_nodes(node),
                "depth": max_depth(node),
                "tokens": result.stats.total_tokens,
                "model": result.stats.final_model,
            }

            results.append(
                {
                    "query": query,
                    "complexity": complexity,
                    "tree": output.text_tree,
                    "dsl": output.dsl_code,
                    "stats": stats,
                }
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(
                {
                    "query": query,
                    "complexity": complexity,
                    "error": str(e),
                }
            )

    # Print results document
    print("\n" + "=" * 80)
    print("RESULTS DOCUMENT")
    print("=" * 80)

    for r in results:
        print(f"\n## [{r['complexity']}] {r['query'][:50]}...")
        print("-" * 60)

        if "error" in r:
            print(f"ERROR: {r['error']}")
            continue

        print("INPUT QUERY:")
        print(f'  "{r["query"]}"')
        print("\nOUTPUT (Text Tree):")
        for line in r["tree"].split("\n"):
            print(f"  {line}")
        print("\nSTATISTICS:")
        print(f"  Nodes: {r['stats']['nodes']}")
        print(f"  Depth: {r['stats']['depth']}")
        print(f"  Tokens: {r['stats']['tokens']}")
        print(f"  Model: {r['stats']['model']}")


if __name__ == "__main__":
    main()
