import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.render import RenderClient


def main():
    print("Testing Salt Features...")

    # DSL testing various containers
    dsl = """
@startsalt
{
  {^"Group Box (Card)"
    Content inside card
  }
  .
  {* "Window Frame (Modal)"
    [Close]
    Are you sure?
    [Yes] | [No]
  }
  .
  {+
    Tree View:
    {T
     + Node 1
     ++ Node 1.1
     + Node 2
    }
  }
  .
  {
    MenuBar:
    {* File | Edit | Source | Refactor }
  }
  .
  "Password:" | "*"
}
@endsalt
    """

    client = RenderClient()
    if client.is_available():
        client.render(dsl, "plantuml").save("deep_testing/output/salt_features.png")
        print("created deep_testing/output/salt_features.png")
    else:
        print("Kroki skipped")


if __name__ == "__main__":
    main()
