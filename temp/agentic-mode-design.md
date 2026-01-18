# Agentic Mode Architecture Design

> **Purpose**: Design document for server-side LLM orchestration
> **Last Updated**: 2026-01-18
> **Status**: DRAFT

---

## Overview

Agentic Mode enables multi-turn conversations with state persistence, allowing the MCP server to orchestrate complex layout design workflows.

---

## Use Cases

### 1. Iterative Refinement

```
User: Create a login page
Agent: [generates initial layout]
User: Make the form centered and add a forgot password link
Agent: [modifies layout with changes]
User: Add social login buttons below the form
Agent: [adds social login section]
```

### 2. Design Exploration

```
User: Show me 3 variations of a dashboard layout
Agent: [generates 3 different layouts]
User: I like the second one, but with a collapsible sidebar
Agent: [refines variation 2]
```

### 3. Component Library Building

```
User: Create a component library for an e-commerce app
Agent: [generates header, footer, product card, cart components]
User: Add a checkout flow
Agent: [generates checkout step components]
```

---

## Architecture

### State Model

```python
# src/mcp/agent/state.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from src.mid import LayoutNode


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    layout: LayoutNode | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSession:
    """Persistent conversation session."""
    session_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    turns: list[ConversationTurn] = field(default_factory=list)
    current_layout: LayoutNode | None = None
    layout_history: list[LayoutNode] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def add_turn(
        self,
        role: str,
        content: str,
        layout: LayoutNode | None = None,
    ) -> ConversationTurn:
        """Add a turn to the conversation."""
        turn = ConversationTurn(
            id=str(uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            layout=layout,
        )
        self.turns.append(turn)
        self.updated_at = datetime.now()

        if layout and role == "assistant":
            self.layout_history.append(self.current_layout)
            self.current_layout = layout

        return turn

    def get_context_prompt(self) -> str:
        """Build context prompt from conversation history."""
        lines = ["Previous conversation:"]
        for turn in self.turns[-10:]:  # Last 10 turns
            role = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role}: {turn.content[:500]}")
        return "\n".join(lines)
```

### Session Manager

```python
# src/mcp/agent/session.py
from typing import Dict
from .state import AgentSession


class SessionManager:
    """Manages agent sessions with optional persistence."""

    def __init__(self, persist_dir: Path | None = None):
        self._sessions: Dict[str, AgentSession] = {}
        self._persist_dir = persist_dir

    def create_session(self) -> AgentSession:
        """Create a new session."""
        session = AgentSession()
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> AgentSession | None:
        """Get existing session."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    def save_session(self, session_id: str) -> None:
        """Persist session to disk."""
        if not self._persist_dir:
            return
        # Implementation...

    def load_session(self, session_id: str) -> AgentSession | None:
        """Load session from disk."""
        if not self._persist_dir:
            return None
        # Implementation...
```

### Tool Chaining

```python
# src/mcp/agent/chain.py
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class ChainStep:
    """Single step in a tool chain."""
    tool_name: str
    args: dict[str, Any]
    transform: Callable[[Any], dict] | None = None


class ToolChain:
    """Chain multiple tools together."""

    def __init__(self, steps: list[ChainStep]):
        self.steps = steps
        self.results: list[Any] = []

    async def execute(self, mcp_client) -> list[Any]:
        """Execute the chain sequentially."""
        self.results = []
        prev_result = None

        for step in self.steps:
            # Transform args if needed
            args = step.args.copy()
            if step.transform and prev_result:
                args.update(step.transform(prev_result))

            # Execute tool
            result = await mcp_client.call_tool(step.tool_name, args)
            self.results.append(result)
            prev_result = result.data

        return self.results


# Predefined chains
GENERATE_RENDER_CHAIN = ToolChain([
    ChainStep(
        tool_name="generate_layout",
        args={"query": ""},  # Filled at runtime
    ),
    ChainStep(
        tool_name="validate_layout",
        args={},
        transform=lambda prev: {"layout": prev["layout"]},
    ),
    ChainStep(
        tool_name="render_layout",
        args={"format": "png"},
        transform=lambda prev: {"layout": prev["layout"] if prev["valid"] else None},
    ),
])
```

### Refinement Engine

```python
# src/mcp/agent/refine.py
from src.mid import LayoutNode
from src.llm import LayoutGenerator


class RefinementEngine:
    """Refines layouts based on user feedback."""

    def __init__(self, generator: LayoutGenerator):
        self._generator = generator

    def refine(
        self,
        current_layout: LayoutNode,
        feedback: str,
        session: AgentSession,
    ) -> LayoutNode:
        """Apply feedback to refine a layout.

        Args:
            current_layout: Layout to refine
            feedback: User feedback/instructions
            session: Conversation session for context

        Returns:
            Refined layout
        """
        # Build prompt with context
        context_prompt = session.get_context_prompt()
        current_json = current_layout.model_dump_json(indent=2)

        refinement_prompt = f"""
{context_prompt}

Current layout:
```json
{current_json}
```

User feedback: {feedback}

Generate an updated layout that addresses the feedback while preserving
the overall structure. Only modify the parts mentioned in the feedback.
"""

        output = self._generator.generate(refinement_prompt)
        return output.context.node

    def generate_variations(
        self,
        base_layout: LayoutNode,
        count: int = 3,
    ) -> list[LayoutNode]:
        """Generate variations of a layout.

        Args:
            base_layout: Base layout to vary
            count: Number of variations

        Returns:
            List of layout variations
        """
        variations = []
        base_json = base_layout.model_dump_json()

        for i in range(count):
            prompt = f"""
Generate variation {i + 1} of this layout, with a different approach:

Base layout:
```json
{base_json}
```

Create a distinct alternative that achieves the same purpose but with
different structure, component choices, or organization.
"""
            output = self._generator.generate(prompt)
            variations.append(output.context.node)

        return variations
```

---

## MCP Tool: Agent Session

```python
# src/mcp/tools/agent.py
from fastmcp import FastMCP, Context

from src.mcp.agent.session import SessionManager
from src.mcp.agent.refine import RefinementEngine


session_manager = SessionManager()


@mcp.tool
async def start_design_session(
    initial_prompt: str | None = None,
    ctx: Context,
) -> dict:
    """Start an interactive design session.

    Args:
        initial_prompt: Optional initial layout request

    Returns:
        Session ID and initial layout (if prompt provided)
    """
    session = session_manager.create_session()
    await ctx.info(f"Created session: {session.session_id}")

    result = {"session_id": session.session_id}

    if initial_prompt:
        # Generate initial layout
        gen_result = await generate_layout(initial_prompt)
        session.add_turn("user", initial_prompt)
        session.add_turn("assistant", "Generated initial layout", gen_result["layout"])
        result["layout"] = gen_result["layout"]

    return result


@mcp.tool
async def refine_design(
    session_id: str,
    feedback: str,
    ctx: Context,
) -> dict:
    """Refine a layout based on feedback.

    Args:
        session_id: Active session ID
        feedback: User feedback or modification request

    Returns:
        Updated layout
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise ToolError(f"Session not found: {session_id}")

    if not session.current_layout:
        raise ToolError("No layout to refine. Generate one first.")

    await ctx.info(f"Refining layout in session {session_id}")

    engine = RefinementEngine(get_generator())
    refined = engine.refine(session.current_layout, feedback, session)

    session.add_turn("user", feedback)
    session.add_turn("assistant", "Applied refinement", refined)

    return {
        "layout": refined.model_dump(),
        "session_id": session_id,
        "history_count": len(session.layout_history),
    }


@mcp.tool
async def get_design_variations(
    session_id: str,
    count: int = 3,
    ctx: Context,
) -> dict:
    """Generate variations of current design.

    Args:
        session_id: Active session ID
        count: Number of variations (1-5)

    Returns:
        List of layout variations
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise ToolError(f"Session not found: {session_id}")

    if not session.current_layout:
        raise ToolError("No layout to vary. Generate one first.")

    count = max(1, min(5, count))  # Clamp to 1-5
    await ctx.info(f"Generating {count} variations")

    engine = RefinementEngine(get_generator())
    variations = engine.generate_variations(session.current_layout, count)

    return {
        "variations": [v.model_dump() for v in variations],
        "session_id": session_id,
    }


@mcp.tool
async def undo_design_change(
    session_id: str,
    ctx: Context,
) -> dict:
    """Undo last design change.

    Args:
        session_id: Active session ID

    Returns:
        Previous layout
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise ToolError(f"Session not found: {session_id}")

    if not session.layout_history:
        raise ToolError("No history to undo")

    previous = session.layout_history.pop()
    session.current_layout = previous

    await ctx.info("Reverted to previous layout")

    return {
        "layout": previous.model_dump() if previous else None,
        "remaining_history": len(session.layout_history),
    }


@mcp.tool
async def end_design_session(
    session_id: str,
    ctx: Context,
) -> dict:
    """End a design session.

    Args:
        session_id: Session to end

    Returns:
        Session summary
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise ToolError(f"Session not found: {session_id}")

    summary = {
        "session_id": session_id,
        "total_turns": len(session.turns),
        "total_revisions": len(session.layout_history),
        "final_layout": session.current_layout.model_dump() if session.current_layout else None,
    }

    session_manager.delete_session(session_id)
    await ctx.info(f"Ended session {session_id}")

    return summary
```

---

## Workflow Examples

### Example 1: Dashboard Design Session

```
# Start session
tool: start_design_session
args: {initial_prompt: "dashboard for analytics app"}
result: {session_id: "abc123", layout: {...}}

# Refine header
tool: refine_design
args: {session_id: "abc123", feedback: "make header sticky with logo on left"}
result: {layout: {...}, history_count: 1}

# Add sidebar
tool: refine_design
args: {session_id: "abc123", feedback: "add collapsible sidebar with navigation"}
result: {layout: {...}, history_count: 2}

# Get variations
tool: get_design_variations
args: {session_id: "abc123", count: 3}
result: {variations: [...]}

# End session
tool: end_design_session
args: {session_id: "abc123"}
result: {total_turns: 4, total_revisions: 2, final_layout: {...}}
```

### Example 2: Error Recovery

```
# Start with complex request
tool: start_design_session
args: {initial_prompt: "complex multi-page app with 20 screens"}
result: {error: "Request too complex"}

# Simplify
tool: start_design_session
args: {initial_prompt: "simple home page"}
result: {session_id: "def456", layout: {...}}

# Build up incrementally
tool: refine_design
args: {session_id: "def456", feedback: "add navigation to 5 sections"}
result: {layout: {...}}
```

---

## Future Enhancements

1. **Persistent Sessions**: Save/restore sessions to disk
2. **Collaborative Editing**: Multiple users on same session
3. **Template Library**: Save refined layouts as templates
4. **A/B Testing**: Compare variations quantitatively
5. **Design Tokens**: Extract consistent styling
