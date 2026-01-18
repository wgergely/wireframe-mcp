# FastMCP Implementation Research

> **Purpose**: Reference document for implementing MCP server in wireframe-mcp
> **Last Updated**: 2026-01-18

---

## Overview

FastMCP is the recommended Python framework for building MCP (Model Context Protocol) servers. It provides a decorator-based API that handles protocol complexities automatically.

## Installation

```bash
pip install "fastmcp>=2.0,<3"
```

**Requirements**:
- Python 3.10+
- Pin to v2 to avoid breaking changes

## Basic Server Setup

```python
from fastmcp import FastMCP

mcp = FastMCP(name="wireframe-mcp")
```

## Defining Tools

### Simple Synchronous Tool

```python
@mcp.tool
def generate_layout(query: str, model: str = "gpt-4.1-mini") -> dict:
    """Generate UI layout from natural language description.

    Args:
        query: Natural language description of the desired layout
        model: LLM model to use for generation

    Returns:
        Dictionary containing the generated layout JSON and metadata
    """
    generator = LayoutGenerator()
    output = generator.generate(query)
    return {
        "layout": output.context.node.model_dump(),
        "stats": {
            "attempts": output.stats.attempts,
            "tokens": output.stats.total_tokens,
        }
    }
```

### Async Tool with Context

```python
from fastmcp import FastMCP, Context

@mcp.tool
async def search_layouts(
    query: str,
    k: int = 5,
    ctx: Context
) -> list[dict]:
    """Search for similar layouts in the vector database.

    Args:
        query: Search query describing desired layout
        k: Number of results to return

    Returns:
        List of similar layouts with similarity scores
    """
    await ctx.info(f"Searching for: {query}")

    store = VectorStore()
    store.load(get_index_dir())

    results = store.search(query, k=k)
    await ctx.info(f"Found {len(results)} results")

    return results
```

### Tool with Progress Tracking

```python
from fastmcp.dependencies import Progress

@mcp.tool(task=True)
async def build_index(
    provider: str,
    limit: int = 100,
    progress: Progress = Progress()
) -> dict:
    """Build vector index from corpus data.

    Args:
        provider: Corpus provider name
        limit: Maximum items to index

    Returns:
        Index statistics
    """
    await progress.set_total(limit)

    store = VectorStore()
    manager = CorpusManager()

    indexed = 0
    for item in manager.stream_data(provider):
        if item.layout:
            store.index_item(item)
            indexed += 1
            await progress.increment()

            if indexed >= limit:
                break

    return {"indexed": indexed}
```

## Defining Resources

### Static Resource

```python
@mcp.resource("schema://components")
def get_component_schema() -> str:
    """Get the component type catalog.

    Returns JSON schema describing all available UI component types.
    """
    from src.schema import export_llm_schema
    return export_llm_schema()
```

### Dynamic Resource Template

```python
@mcp.resource("corpus://{provider}/stats")
def get_provider_stats(provider: str) -> dict:
    """Get statistics for a corpus provider.

    Args:
        provider: Provider name (e.g., 'rico_semantic')

    Returns:
        Provider statistics including item count and coverage
    """
    manager = CorpusManager()
    provider_obj = manager.get_provider(provider)
    return {
        "name": provider,
        "has_data": provider_obj.has_data(),
        "description": provider_obj.description,
    }
```

## Transport Options

### STDIO (Claude Desktop)

```python
if __name__ == "__main__":
    mcp.run()  # Default STDIO
```

### HTTP (Web Deployment)

```python
if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=18080,
        path="/mcp"
    )
```

### SSE (Legacy)

```python
if __name__ == "__main__":
    mcp.run(
        transport="sse",
        host="127.0.0.1",
        port=8080
    )
```

## Testing Patterns

### Basic Test Fixture

```python
import pytest
from fastmcp import FastMCP, Client

@pytest.fixture
def mcp_server():
    from src.mcp.server import create_server
    return create_server()

@pytest.fixture
async def mcp_client(mcp_server):
    async with Client(mcp_server) as client:
        yield client
```

### Testing Tools

```python
@pytest.mark.asyncio
async def test_generate_layout(mcp_client):
    result = await mcp_client.call_tool(
        "generate_layout",
        {"query": "login form with email and password"}
    )

    assert "layout" in result.data
    assert result.data["layout"]["type"] == "container"
```

### Testing Resources

```python
@pytest.mark.asyncio
async def test_schema_resource(mcp_client):
    result = await mcp_client.read_resource("schema://components")

    schema = json.loads(result.data)
    assert "components" in schema
```

### Parameterized Tests

```python
@pytest.mark.parametrize("query,expected_type", [
    ("login form", "container"),
    ("navigation bar", "toolbar"),
    ("settings page", "container"),
])
async def test_layout_types(mcp_client, query, expected_type):
    result = await mcp_client.call_tool(
        "generate_layout",
        {"query": query}
    )
    assert result.data["layout"]["type"] == expected_type
```

## Dependencies Injection

```python
from fastmcp.dependencies import Depends

def get_vector_store() -> VectorStore:
    store = VectorStore()
    store.load(get_index_dir())
    return store

def get_generator() -> LayoutGenerator:
    return LayoutGenerator()

@mcp.tool
async def generate_with_rag(
    query: str,
    store: VectorStore = Depends(get_vector_store),
    generator: LayoutGenerator = Depends(get_generator),
) -> dict:
    """Generate layout with RAG context."""
    # Similar layouts for context
    similar = store.search(query, k=3)

    # Generate with context
    output = generator.generate(query)

    return {
        "layout": output.context.node.model_dump(),
        "similar_count": len(similar),
    }
```

## Error Handling

```python
from fastmcp.exceptions import ToolError

@mcp.tool
def validate_layout(layout_json: str) -> dict:
    """Validate a layout JSON structure.

    Args:
        layout_json: JSON string of layout to validate

    Returns:
        Validation result with any errors found

    Raises:
        ToolError: If JSON is unparseable
    """
    try:
        layout_dict = json.loads(layout_json)
    except json.JSONDecodeError as e:
        raise ToolError(f"Invalid JSON: {e}")

    try:
        node = LayoutNode.model_validate(layout_dict)
        errors = validate_layout(node)
        return {
            "valid": len(errors) == 0,
            "errors": [e.model_dump() for e in errors],
        }
    except ValidationError as e:
        return {
            "valid": False,
            "errors": [{"message": str(e)}],
        }
```

## Claude Desktop Configuration

```json
{
  "mcpServers": {
    "wireframe": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/wireframe-mcp",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "VOYAGE_API_KEY": "pa-...",
        "KROKI_URL": "http://localhost:18000"
      }
    }
  }
}
```

## Best Practices

1. **Write comprehensive docstrings** - FastMCP uses them as tool descriptions for LLMs
2. **Use type hints** - They generate JSON schemas for input validation
3. **Return structured data** - Always return dicts/lists, not plain strings
4. **Handle errors gracefully** - Use ToolError for user-facing errors
5. **Log operations** - Use Context.info() for progress updates
6. **Keep tools focused** - One tool, one responsibility
7. **Test with real client** - Use MCP Inspector during development

## Development Commands

```bash
# Run with MCP Inspector
fastmcp run src/mcp/server.py

# Or using uvx
uvx mcp dev src/mcp/server.py

# Test manually
python -m src.mcp.server
```

## Sources

- [FastMCP GitHub](https://github.com/jlowin/fastmcp)
- [FastMCP Documentation](https://gofastmcp.com/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Protocol Spec](https://modelcontextprotocol.io/)
