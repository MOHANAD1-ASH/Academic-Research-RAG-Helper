# Academic Research Helper

A FastAPI + Gradio Retrieval-Augmented Generation (RAG) application for answering questions from a fixed corpus of academic papers.

## Architecture

```text
Gradio frontend
      |
      | POST /query
      v
FastAPI backend
      |
      +-- BM25 top 30
      +-- BGE-M3 dense retrieval top 30
      +-- Reciprocal Rank Fusion (k=60) top 40
      +-- CrossEncoder reranking top 10
      +-- Grounded generation from top 6 evidence chunks
      |
      +-- GET /papers
      +-- POST /papers/ingest (PDF)
```

## Important runtime contract

The backend expects the notebook-exported artifacts in `backend/data/artifacts/`:

- `chunks.json`
- `embeddings.npy`
- `manifest.json`
- `papers.json`

The ZIP intentionally does **not** contain the original Gemini key or generated vector-store data. Put the real artifacts in the artifact directory before starting the backend.

## Configuration

Copy `backend/.env.example` to `backend/.env` and set `GEMINI_API_KEY` when using Gemini. Never commit `.env`.

Default retrieval configuration:

| Setting | Value |
|---|---|
| Embedding model | `BAAI/bge-m3` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| BM25 | top 30 |
| Dense | top 30 |
| RRF | `k=60`, top 40 |
| Reranker output | top 10 |
| Generation context | top 6, max 150 words/chunk |
| Chunking for new PDFs | 220 words, 40 overlap, 40-word minimum |

## Setup on Windows

Use Python 3.13 (or another version supported by the pinned dependency ranges):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python -m pip install -r frontend\requirements.txt
Copy-Item backend\.env.example backend\.env
Copy-Item frontend\.env.example frontend\.env
```

Then set `GEMINI_API_KEY` in `backend\.env`.

## Run

Backend:

```powershell
Set-Location backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend (second terminal):

```powershell
Set-Location frontend
python app.py
```

## API

### Health

`GET /health`

Returns readiness for retrieval, generation, and ingestion services.

### Query

`POST /query`

```json
{"question": "What limitations of transformer-based models appear in the corpus?"}
```

### Papers

`GET /papers`

Returns the same paper metadata used by the backend retrieval corpus. The frontend refreshes this list from the backend and falls back to local `papers.json` if the backend is unavailable.

### PDF ingestion

`POST /papers/ingest` with multipart field `file`.

The ingestion path validates the PDF, extracts text, chunks it using the notebook-compatible settings, embeds it with the active embedding model, updates Chroma, and persists `chunks.json`, `papers.json`, `embeddings.npy`, and `manifest.json`. It uses a lock and rollback strategy so a failed ingestion does not intentionally leave a partial Chroma update.

## Tests

Run from `backend`:

```powershell
pytest
```

The API tests use fake services and therefore do not download ML models, initialize Chroma, or call Gemini. This keeps unit/API tests independent from the heavyweight inference stack.

## Security

Do not place API keys in source control, `.env.example`, Docker images, or ZIP archives. The provided project contains only the empty key field in `.env.example`.
