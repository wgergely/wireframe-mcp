# Research: RAG System for UI Layouts

## Executive Summary
This document validates the "RAG for UI" strategy and evaluates technical options for a local Windows implementation.
**Conclusion**: RAG is a highly effective strategy for UI generation. The Rico dataset is the industry standard for this. For the vector database, **FAISS-GPU** is the firm choice for high-performance retrieval, with a graceful fallback to CPU if hardware is unavailable.

## 1. Validity of RAG for UI Layouts
Research confirms that Retrieval-Augmented Generation (RAG) is increasingly applied to UI/UX workflows.
- **Methodology**: "LayoutRAG" and similar approaches use a two-step process:
    1.  **Retrieve**: Find relevant layouts (from Rico/WebUI) based on user query.
    2.  **Generate**: Use the retrieved layouts as "few-shot" examples or context for the LLM to generate the final code/structure.
- **Benefits**: Reduces hallucinations, ensures design consistency, and allows "style transfer" from existing premium designs.

## 2. Dataset Suitability (Rico)
- **Rico Dataset**: specifically created for data-driven UI generation.
- **Enhancement**: The "Semantic" subset is crucial. Raw screenshots are less useful for RAG than the **View Hierarchies** and **Semantic Annotations** (Button, Input, Modal) which effectively act as the "text" for the LLM.

## 3. Vector Database Recommendation
We evaluated options for a local Windows environment and selected **FAISS-GPU** as the priority.

| Option | Pros | Cons | Recommendation |
| :--- | :--- | :--- | :--- |
| **faiss-gpu** | • **Maximum Performance** for batch queries<br>• Scalable to millions of vectors<br>• Future-proof for larger datasets | • Requires NVIDIA GPU + CUDA<br>• Larger install size | **✅ PRIMARY CHOICE** |
| **faiss-cpu** | • Simple install (`pip install faiss-cpu`)<br>• Good fallback | • Slower for massive parallel searches | ⚠️ FALLBACK ONLY |
| **sqlite-vss** | • Single-file database | • Slower write precision<br>• Less mature | ❌ Not Recommended |

**Decision**: Use **`faiss-gpu`** (with `faiss-cpu` fallback).
- **Architecture**: The application will attempt to load the GPU index first. If CUDA is unavailable, it will gracefully degrade to CPU mode.
- **Benefits**: Ensures the "Premium/High-Performance" feel requested for the Agent, while maintaining compatibility.

## 4. Proposed Architecture Refinement
1.  **Ingest**: Load Rico `semantic_annotations.zip`.
2.  **Embed**: Use Rico's pre-computed 64-dim vectors OR generate new embeddings using a GPU-accelerated local model (e.g., `sentence-transformers`).
3.  **Index**: Store vectors in a persistent FAISS index.
    - **Primary**: GPU Index (`faiss.GpuIndexFlatIP` etc.)
    - **Secondary**: CPU Index (if GPU fails).
4.  **Retrieve**:
    - **Metric**: Cosine Similarity.
    - **Query**: "User Profile Page" -> Text Embedding -> FAISS -> Top 5 Screens.
    - **Context**: Pass the simplified JSON hierarchy of the Top 5 screens to the LLM.
