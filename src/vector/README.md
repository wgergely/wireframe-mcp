# Vector Module

The Vector module implements the **RAG (Retrieval-Augmented Generation) system** for UI layouts. It provides vector storage, similarity search, and layout serialization to ground LLM generation in real-world UI examples.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Vector Module                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │LayoutSerial │ -> │ Embedding   │ -> │   VectorStore       │  │
│  │   izer      │    │  Backend    │    │   (FAISS Index)     │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│         │                 │                      │              │
│         v                 v                      v              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ Text repr   │    │ Voyage AI / │    │  Similarity Search  │  │
│  │ of layout   │    │ Local Model │    │  (Top-K retrieval)  │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

| Component | Path | Description |
|-----------|------|-------------|
| `VectorStore` | `vector/lib.py` | Main orchestrator for indexing and search |
| `FAISSIndex` | `vector/index.py` | GPU/CPU vector index with fallback |
| `LayoutSerializer` | `vector/serializer.py` | Schema-aware text serialization |
| Embedding Backends | `vector/backend/` | Voyage AI + Local model options |

## Usage

```python
from src.vector import VectorStore, VectorStoreConfig

# Initialize store
config = VectorStoreConfig(
    index_path=".corpus/index",
    embedding_backend="voyage",  # or "local"
)
store = VectorStore(config)

# Index a layout
from src.mid import LayoutNode
layout = LayoutNode(id="sample", type="container")
store.add(layout, sample_id="sample_001")

# Search for similar layouts
query = "dashboard with sidebar navigation"
results = store.search(query, top_k=5)

for result in results:
    print(f"{result.id}: score={result.score:.3f}")
```

---

## RAG Strategy

### Why RAG for UI Layouts?

Research validates that **Retrieval-Augmented Generation** is highly effective for UI generation:

1. **Reduces Hallucinations**: Grounds generation in real-world examples
2. **Design Consistency**: Enables "style transfer" from existing premium designs
3. **Few-shot Learning**: Retrieved layouts serve as context for LLM generation

### Methodology

The system implements a **two-step RAG process**:

1. **Retrieve**: Find relevant layouts from the corpus based on user query
2. **Generate**: Use retrieved layouts as few-shot examples for LLM generation

```
User Query → Text Embedding → FAISS Search → Top-K Layouts → LLM Prompt Context
```

---

## Vector Database Selection

### FAISS-GPU (Primary)

Selected for **maximum performance** on high-dimensional vector search:

| Feature | Value |
|---------|-------|
| Backend | FAISS (Facebook AI) |
| GPU Support | CUDA-accelerated when available |
| Fallback | Automatic CPU fallback |
| Index Types | Flat, IVF, HNSW |
| Distance Metric | Cosine Similarity |

### Comparison Matrix

| Feature | FAISS | Qdrant | Chroma | Milvus |
|---------|:-----:|:------:|:------:|:------:|
| GPU Acceleration | ✅ | ❌ | ❌ | ✅ |
| Local Deployment | ✅ | ✅ | ✅ | ⚠️ |
| Python Integration | ✅ | ✅ | ✅ | ✅ |
| JSON Payloads | ⚠️ | ✅ | ✅ | ✅ |
| Hybrid Search | ⚠️ | ✅ | ✅ | ✅ |
| Setup Complexity | Low | Medium | Low | High |

**Decision**: FAISS-GPU for search performance, with metadata stored separately.

### Alternative Recommendations

- **Qdrant**: Best for JSON payload storage alongside vectors
- **Chroma**: Best for zero-setup prototyping
- **pgvector**: Best if already using PostgreSQL

---

## Embedding Backends

### Voyage AI (Cloud)

High-quality embeddings via API:

```python
config = VectorStoreConfig(embedding_backend="voyage")
# Requires VOYAGE_API_KEY environment variable
```

| Model | Dimensions | Use Case |
|-------|------------|----------|
| `voyage-3` | 1024 | General purpose |
| `voyage-3-lite` | 512 | Cost-effective |
| `voyage-code-3` | 1024 | Code/structure focus |

### Local Models (Offline)

GPU-accelerated local embeddings:

```python
config = VectorStoreConfig(
    embedding_backend="local",
    local_model="sentence-transformers/all-MiniLM-L6-v2",
)
```

| Model | Dimensions | Size |
|-------|------------|------|
| `all-MiniLM-L6-v2` | 384 | 80MB |
| `all-mpnet-base-v2` | 768 | 420MB |

---

## Layout Serialization

The `LayoutSerializer` converts `LayoutNode` trees to text for embedding:

### Serialization Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `TREE` | Indented hierarchy | Human readability |
| `FLAT` | Single-line compact | Token efficiency |
| `METADATA` | With component counts | Structural similarity |

### Example Output

**TREE format**:
```
container[dashboard] (horizontal)
  ├── drawer[sidebar] flex=3
  └── container[main] flex=9
      ├── text[header]
      └── list_item[items]
```

**METADATA format**:
```
{
  "depth": 3,
  "components": {"container": 2, "drawer": 1, "text": 1, "list_item": 1},
  "structure": "container > drawer + container > text + list_item"
}
```

---

## Index Configuration

### Recommended Settings

```python
from src.vector import FAISSIndex, IndexConfig

config = IndexConfig(
    dimensions=1024,        # Match embedding model
    index_type="IVF",       # Good balance for 10k-1M vectors
    nlist=100,              # Number of clusters
    nprobe=10,              # Clusters to search
    use_gpu=True,           # Enable GPU if available
)

index = FAISSIndex(config)
```

### Index Types

| Type | Vectors | Speed | Accuracy |
|------|---------|-------|----------|
| `Flat` | <10k | Slow | 100% |
| `IVF` | 10k-1M | Fast | 95%+ |
| `HNSW` | Any | Very Fast | 90%+ |

---

## Integration with Prompt Builder

The VectorStore integrates with `src/prompt` for RAG-enhanced generation:

```python
from src.prompt import PromptBuilder
from src.vector import VectorStore

store = VectorStore(...)
builder = PromptBuilder(vector_store=store)

# Build prompt with RAG context
prompt = builder.build(
    query="Create a login form with social auth",
    top_k=3,  # Retrieve 3 similar layouts
)

# Prompt now includes:
# 1. System instructions
# 2. Schema definition
# 3. Retrieved similar layouts as few-shot examples
# 4. User query
```

---

## References

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Voyage AI API](https://docs.voyageai.com/)
- [Rico Dataset](http://interactionmining.org/rico)
- [Sentence Transformers](https://www.sbert.net/)
