# LLM Module

The LLM module provides a **unified interface** for integrating multiple LLM providers into the wireframe generation pipeline. It bridges `PromptBuilder` (RAG-enhanced prompts) with the MID layer (validated `LayoutNode` trees).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Integration Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │PromptBuilder│ -> │ LLMBackend  │ -> │   JSON Parser       │  │
│  │ (RAG + Schema)   │ (Provider   │    │   + Repair          │  │
│  │             │    │  specific)  │    │                     │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                            │                     │              │
│                            v                     v              │
│                    ┌─────────────┐    ┌─────────────────────┐   │
│                    │ Rate Limit  │    │    Validator        │   │
│                    │ + Retry     │    │  (validate_layout)  │   │
│                    └─────────────┘    └─────────────────────┘   │
│                                              │                  │
│                                              v                  │
│                                    ┌──────────────────────┐     │
│                                    │ TranspilationContext │     │
│                                    └──────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Usage

```python
from src.llm import LayoutGenerator, GeneratorConfig

config = GeneratorConfig(
    provider="openai",
    model="gpt-4.1-mini",
    max_retries=3,
)

generator = LayoutGenerator(config)

# Generate layout from natural language
layout = await generator.generate(
    "Create a dashboard with sidebar navigation and main content area"
)

# Result is a validated LayoutNode
print(layout.id, layout.type, len(layout.children))
```

---

## Provider Comparison

### OpenAI Models (Recommended for Development)

| Model | Context | Max Output | JSON Mode | Cost (per 1M tokens) |
|-------|---------|------------|-----------|----------------------|
| `gpt-4.1-mini` | 128K | 16K | ✅ Native | $0.15 in / $0.60 out |
| `gpt-4.1` | 128K | 16K | ✅ Native | Mid tier |
| `gpt-4.5` | 128K | 16K | ✅ Native | Higher tier |

**Key Features**:
- Native JSON mode via `response_format={"type": "json_object"}`
- Seed parameter for reproducibility
- Consistent latency (2-5s)

### Anthropic Claude Models (Recommended for Production)

| Model | Context | Max Output | JSON Mode | Cost (per 1M tokens) |
|-------|---------|------------|-----------|----------------------|
| `claude-sonnet-4-5` | 200K | 64K | Prompt-engineered | $3 in / $15 out |
| `claude-haiku-4-5` | 200K | 64K | Prompt-engineered | $1 in / $5 out |
| `claude-opus-4-5` | 200K | 64K | Prompt-engineered | $5 in / $25 out |

**Key Features**:
- Extended thinking for complex reasoning
- 1M token context window (beta) for Sonnet
- Requires prompt engineering for JSON (no native mode)

### DeepSeek Models (Budget Option)

| Model | Context | Max Output | JSON Mode | Cost (per 1M tokens) |
|-------|---------|------------|-----------|----------------------|
| `deepseek-chat` (V3.2) | 64K+ | 8K | ✅ Native | $0.14 in / $0.28 out |
| `deepseek-reasoner` (R1) | 64K | 8K | ✅ Native | $0.55 in / $2.19 out |

**Key Features**:
- OpenAI-compatible API at `https://api.deepseek.com/v1`
- 10-20x cheaper than GPT-4
- Excellent quality for structured output

### Qwen Models

| Model | Context | Max Output | JSON Mode |
|-------|---------|------------|-----------|
| `qwen3-max` | 131K | 8K | ✅ Native |
| `qwen3-coder-plus` | 131K | 8K | ✅ Native |

**Key Features**:
- OpenAI-compatible API
- Context caching for reduced costs
- Apache 2.0 licensed open-source versions

### Ollama (Offline)

| Model | Context | RAM Required |
|-------|---------|--------------|
| `deepseek-r1` | 64K | 16GB+ |
| `qwen3` | 32K | 8-32GB |
| `llama3.2` | 128K | 8-128GB |

**Key Features**:
- No API key required
- Fully offline operation
- JSON format via `format="json"`

---

## JSON Mode Comparison

| Provider | Native JSON Mode | Reliability | Notes |
|----------|------------------|-------------|-------|
| OpenAI | ✅ | 99%+ | `response_format={"type": "json_object"}` |
| DeepSeek | ✅ | 98%+ | OpenAI-compatible |
| Qwen | ✅ | 98%+ | OpenAI-compatible |
| Ollama | ✅ | 95%+ | `format="json"` parameter |
| Anthropic | ❌ | 90%+ | Requires prompt engineering + extraction |

### Claude JSON Strategy

Since Claude lacks native JSON mode, the implementation:

1. Appends instruction: "Respond with valid JSON only"
2. Extracts JSON from markdown code blocks if present
3. Uses regex to find balanced `{...}` blocks
4. Strips common prefixes ("Here is the JSON:", etc.)

---

## Prompt Engineering

### System Prompt Design

The system prompt emphasizes:

1. **JSON-only output**: Explicit instruction to output only JSON
2. **Schema compliance**: Reference to the LayoutNode schema
3. **Unique IDs**: Requirement for unique identifiers
4. **Valid types**: Restriction to 26 ComponentType values
5. **Flex ratio bounds**: 1-12 range enforcement

### Few-shot Examples

The PromptBuilder injects:
- Up to 3 similar layouts from VectorStore
- Serialized in INDENTED format for readability
- Truncated to 500 characters per example

### Error Feedback Loop

When validation fails:
1. Format errors as bullet points
2. Append to prompt: "PREVIOUS ATTEMPT HAD ERRORS: ..."
3. Re-generate with increased context

---

## Error Recovery

### JSON Repair Patterns

```python
# Remove markdown code blocks
r"^```json\s*" -> ""
r"\s*```$" -> ""

# Fix trailing commas
r",\s*}" -> "}"
r",\s*]" -> "]"

# Extract JSON from mixed content
r"\{[\s\S]*\}"  # Find balanced braces
```

### Retry Logic

| Error Type | Retry | Strategy |
|------------|-------|----------|
| Rate limit | ✅ | Exponential backoff |
| JSON parse | ✅ | Repair then retry |
| Validation | ✅ | Feedback loop |
| Auth error | ❌ | Fail fast |
| Context length | ❌ | Fail fast |

---

## Model Selection Recommendations

| Use Case | Recommended Model | Rationale |
|----------|-------------------|-----------|
| Development | `gpt-4.1-mini` | Cost-effective, reliable JSON |
| Production | `claude-sonnet-4-5` | Best quality/cost balance |
| Budget | `deepseek-chat` | 10x cheaper, excellent quality |
| Offline | `qwen3` (Ollama) | No API dependency |

---

## Best Practices

1. **Always validate**: Use `validate_layout()` before transpilation
2. **Enable retries**: JSON parse errors are often recoverable
3. **Use RAG**: Few-shot examples significantly improve output
4. **Monitor tokens**: Track usage for cost optimization
5. **Handle failures**: Implement graceful degradation

---

## Future Enhancements

1. **Streaming support**: Show partial layouts during generation
2. **Layout deserialization**: Reconstruct LayoutNode from VectorStore
3. **Model routing**: Auto-select model based on query complexity
4. **Feedback learning**: Use user corrections to improve prompts

---

## References

- [OpenAI Models Documentation](https://platform.openai.com/docs/models)
- [Anthropic Claude Models Overview](https://docs.anthropic.com/claude/docs/models-overview)
- [DeepSeek API Documentation](https://api-docs.deepseek.com/)
- [Qwen API Reference](https://www.alibabacloud.com/help/en/model-studio/qwen-api-reference)
- [Ollama Model Library](https://ollama.com/library)
