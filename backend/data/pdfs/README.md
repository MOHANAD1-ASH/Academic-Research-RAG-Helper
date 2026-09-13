# Academic PDFs

This directory contains the **academic PDF documents** used as the source corpus for the Academic Research RAG Helper.

## Purpose

The PDF corpus serves as the primary source of knowledge for the RAG system.

During the ingestion process, each PDF is processed to extract its text and relevant metadata before being transformed into searchable document chunks.

## Ingestion Pipeline

```text
Academic PDFs
      ↓
PDF Validation
      ↓
Text Extraction
      ↓
Metadata Extraction
      ↓
Text Chunking
      ↓
BGE-M3 Embeddings
      ↓
Vector Store
      ↓
Hybrid Retrieval
```

## Document Processing

The ingestion pipeline extracts information such as:

* Document text
* Page numbers
* Titles
* Authors
* Sections
* Publication year
* DOI / arXiv identifiers when available
* Source URLs when available

The extracted content is then divided into smaller chunks for efficient retrieval.

## Chunking

The current ingestion configuration uses:

* **Target chunk size:** 220 words
* **Overlap:** 40 words

The overlap helps preserve contextual continuity between neighboring chunks while keeping individual retrieval units focused.

## Supported Documents

The corpus is primarily intended for:

* Research papers
* Academic publications
* Technical papers
* Scientific literature
* Other text-based academic PDFs

## Adding New PDFs

To add a new document:

1. Place the PDF inside this directory.
2. Run the project's ingestion/indexing pipeline.
3. The system will extract and process the document.
4. New chunks and embeddings will be generated.
5. The vector store will be updated with the new records.

> **Important:** Adding or removing PDFs requires re-running the appropriate ingestion process to keep the generated retrieval artifacts synchronized with the source corpus.

## Duplicate Protection

The ingestion pipeline includes duplicate detection using available document identifiers such as:

* SHA-256
* DOI
* arXiv ID
* Normalized title

This helps prevent the same academic paper from being indexed multiple times.

## Version Control

The PDF corpus may contain large files and is therefore **not required to be committed to the Git repository**.

If the PDFs are excluded from version control, users should provide their own academic corpus and run the ingestion pipeline before starting the retrieval system.

> **Note:** Only include documents that you are permitted to redistribute. Respect the copyright and licensing terms of academic publications.

---

## Academic Research RAG Helper

**From academic PDFs to source-grounded answers.**

The PDF corpus is the starting point of the system's offline ingestion pipeline and provides the evidence used by the retrieval and generation layers.
