# Vector Store

This directory contains the **persistent vector store** used by the Academic Research RAG Helper for semantic retrieval over indexed academic literature.

## Purpose

The vector store contains the vector representations and associated metadata of processed document chunks.

During query time, the system uses this store for **dense semantic retrieval**, which is combined with BM25 lexical retrieval and later fused using Reciprocal Rank Fusion (RRF).

## Retrieval Pipeline

```text
Academic PDFs
      ↓
PDF Parsing
      ↓
Text Chunking
      ↓
BGE-M3 Embeddings
      ↓
Vector Store
      ↓
Dense Retrieval
      ↓
BM25 + Dense Retrieval
      ↓
RRF Fusion
      ↓
Cross-Encoder Reranking
      ↓
Grounded Generation
      ↓
Cited Answer
```

## Embedding Model

The project uses:

```text
BAAI/bge-m3
```

Embeddings are generated from processed academic document chunks and stored for semantic similarity search.

## Stored Metadata

Vector store records are associated with metadata such as:

* `paper_id`
* `page`
* `section`
* `title`
* `authors`
* `year`
* Source identifiers / URLs

This metadata supports filtering, retrieval, and source citation.

## Persistence

The vector store is persisted locally so that the indexed corpus can be reused across application restarts without recomputing all embeddings.

## Rebuilding the Vector Store

The vector store should be generated through the project's ingestion and indexing pipeline.

The general process is:

```text
PDF Corpus
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embedding Generation
    ↓
Vector Store Creation
```

Refer to the main [`README.md`](../README.md) for the complete setup and ingestion instructions.

## Important

The contents of this directory are **generated artifacts** and should not normally be modified manually.

If the source documents, chunking configuration, or embedding model changes, the vector store should be regenerated to keep it consistent with the indexed corpus.

## Version Control

Large generated vector-store files are treated separately from the source code.

If the vector store is not committed to the repository, a fresh copy must be generated locally before running the retrieval backend.

> **Note:** Do not store API keys, credentials, or other secrets in this directory.

---

## Academic Research RAG Helper

**Source-grounded retrieval for academic literature.**

The vector store is a core component of the retrieval layer and works together with lexical search, RRF fusion, and cross-encoder reranking to improve the relevance of retrieved evidence.
