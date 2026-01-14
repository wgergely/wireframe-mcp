# LLM Integration Research Findings

**Date**: January 2026
**Author**: Wireframe-MCP Development Team

## Executive Summary

This document captures research findings for integrating LLM providers into the wireframe generation pipeline. The integration bridges the gap between `PromptBuilder` (which creates RAG-enhanced prompts) and `TranspilationContext` (which expects validated `LayoutNode` trees).

## Provider Comparison Matrix

### OpenAI Models (January 2026)

| Model | Context | Max Output | JSON Mode | Best For | Cost (per 1M tokens) |
|-------|---------|------------|-----------|----------|---------------------|
| `gpt-5.2` | 128K+ | 16K | Native | Professional work, agents | Higher tier |
| `gpt-5.1` | 128K | 16K | Native | Balanced speed/intelligence | Mid tier |
| `gpt-4.1` | 128K | 16K | Native | Coding tasks | Mid tier |
| `gpt-4.1-mini` | 128K | 16K | Native | Cost-effective, fast | $0.15 in / $0.60 out |
| `gpt-4.5` | 128K | 16K | Native | Creative, high EQ | Higher tier |

**Key Features**:
- Native JSON mode via `response_format={"type": "json_object"}`
- Seed parameter for reproducibility
- Vision and function calling support

### Anthropic Claude Models (January 2026)

| Model | Context | Max Output | JSON Mode | Best For | Cost (per 1M tokens) |
|-------|---------|------------|-----------|----------|---------------------|
| `claude-opus-4-5` | 200K | 64K | Prompt-engineered | Maximum intelligence | $5 in / $25 out |
| `claude-sonnet-4-5` | 200K (1M beta) | 64K | Prompt-engineered | Balanced, coding, agents | $3 in / $15 out |
| `claude-haiku-4-5` | 200K | 64K | Prompt-engineered | Fastest, cost-effective | $1 in / $5 out |
| `claude-opus-4-1` | 200K | 32K | Prompt-engineered | Legacy premium | $15 in / $75 out |

**Key Features**:
- Extended thinking for complex reasoning
- 1M token context window (beta) for Sonnet
- No native JSON mode - requires prompt engineering

### DeepSeek Models (January 2026)

| Model | Context | Max Output | JSON Mode | Best For | Cost (per 1M tokens) |
|-------|---------|------------|-----------|----------|---------------------|
| `deepseek-chat` (V3.2) | 64K+ | 8K | Native | General use, GPT-5 level | $0.14 in / $0.28 out |
| `deepseek-reasoner` (R1) | 64K | 8K | Native | Step-by-step reasoning | $0.55 in / $2.19 out |
| `deepseek-coder` | 64K | 8K | Native | Code generation | $0.14 in / $0.28 out |

**Key Features**:
- OpenAI-compatible API at `https://api.deepseek.com/v1`
- Open-source models available on HuggingFace
- Extremely cost-effective (10-20x cheaper than GPT-4)
- V4 expected mid-February 2026

### Qwen Models (January 2026)

| Model | Context | Max Output | JSON Mode | Best For | Cost |
|-------|---------|------------|-----------|----------|------|
| `qwen3-max` | 131K | 8K | Native | Flagship, agents | Premium |
| `qwen3-coder-plus` | 131K | 8K | Native | Code with cache | Mid tier |
| `qwen-turbo` | 131K | 8K | Native | Fast inference | Budget |
| `qwen-plus` | 131K | 8K | Native | General balanced | Mid tier |

**Key Features**:
- OpenAI-compatible API at `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- Context caching for reduced costs
- Apache 2.0 licensed open-source versions

### Ollama Local Models (January 2026)

| Model | Context | RAM Required | Best For |
|-------|---------|--------------|----------|
| `deepseek-r1` | 64K | 16GB+ | Reasoning tasks |
| `qwen3` | 32K | 8-32GB | General purpose |
| `llama3.2` | 128K | 8-128GB | Various sizes |
| `gemma3` | 32K | 8GB+ | Lightweight |
| `dolphin3` | 32K | 16GB | General purpose |

**Key Features**:
- No API key required
- Fully offline operation
- JSON format support via `format="json"`

## JSON Mode Comparison

| Provider | Native JSON Mode | Reliability | Notes |
|----------|------------------|-------------|-------|
| OpenAI | Yes | 99%+ | `response_format={"type": "json_object"}` |
| DeepSeek | Yes | 98%+ | OpenAI-compatible |
| Qwen | Yes | 98%+ | OpenAI-compatible |
| Ollama | Yes | 95%+ | `format="json"` parameter |
| Anthropic | No | 90%+ | Requires prompt engineering + extraction |

### Anthropic JSON Strategy

Since Claude lacks native JSON mode, our implementation:
1. Appends instruction: "Respond with valid JSON only"
2. Extracts JSON from markdown code blocks if present
3. Uses regex to find balanced `{...}` blocks
4. Strips common prefixes ("Here is the JSON:", etc.)

## Prompt Engineering Findings

### System Prompt Design

Our system prompt emphasizes:
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

When validation fails, we:
1. Format errors as bullet points
2. Append to prompt: "PREVIOUS ATTEMPT HAD ERRORS: ..."
3. Re-generate with increased context

## Error Recovery Strategies

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
| Rate limit | Yes | Exponential backoff |
| JSON parse | Yes | Repair then retry |
| Validation | Yes | Feedback loop |
| Auth error | No | Fail fast |
| Context length | No | Fail fast |

## Performance Considerations

### Token Optimization

1. **Prompt caching**: Qwen supports implicit caching at 20% cost
2. **Batch API**: OpenAI/DeepSeek offer batch processing at reduced cost
3. **Model selection**: Use smaller models for simple layouts

### Latency

| Provider | Typical Latency | Notes |
|----------|-----------------|-------|
| OpenAI | 2-5s | Consistent |
| Anthropic | 3-8s | Varies by complexity |
| DeepSeek | 2-6s | Good value |
| Qwen | 3-7s | Regional variance |
| Ollama | 5-30s | Hardware dependent |

## Implementation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      LayoutGenerator                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │ PromptBuilder │ -> │  LLMBackend  │ -> │  Validator  │  │
│  └───────────────┘    └──────────────┘    └─────────────┘  │
│         │                    │                   │          │
│         v                    v                   v          │
│  ┌───────────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │  VectorStore  │    │ JSON Parser  │    │  LayoutNode │  │
│  │  (RAG)        │    │ + Repair     │    │             │  │
│  └───────────────┘    └──────────────┘    └─────────────┘  │
│                                                  │          │
│                                                  v          │
│                                      ┌────────────────────┐ │
│                                      │TranspilationContext│ │
│                                      └────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Recommendations

### Default Model Selection

| Use Case | Recommended Model | Rationale |
|----------|-------------------|-----------|
| Development | `gpt-4.1-mini` | Cost-effective, reliable JSON |
| Production | `claude-sonnet-4-5` | Best quality/cost balance |
| Budget | `deepseek-chat` | 10x cheaper, excellent quality |
| Offline | `qwen3` (Ollama) | No API dependency |

### Best Practices

1. **Always validate**: Use `validate_layout()` before transpilation
2. **Enable retries**: JSON parse errors are often recoverable
3. **Use RAG**: Few-shot examples significantly improve output
4. **Monitor tokens**: Track usage for cost optimization
5. **Handle failures**: Implement graceful degradation

## Future Enhancements

1. **Streaming support**: Show partial layouts during generation
2. **Layout deserialization**: Reconstruct LayoutNode from VectorStore
3. **Model routing**: Auto-select model based on query complexity
4. **Feedback learning**: Use user corrections to improve prompts

## Sources

- [OpenAI Models Documentation](https://platform.openai.com/docs/models)
- [Anthropic Claude Models Overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [DeepSeek V3.2 Release Notes](https://api-docs.deepseek.com/news/news251201)
- [Qwen API Reference](https://www.alibabacloud.com/help/en/model-studio/qwen-api-reference)
- [Ollama Model Library](https://ollama.com/library)
