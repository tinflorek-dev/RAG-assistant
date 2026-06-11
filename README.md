# RAG Assistant

A small, fully local **Retrieval-Augmented Generation** service. Upload documents
(PDF / Markdown / text), and ask questions that are answered **grounded in those
documents**, with inline source citations. Everything runs on your machine — no
external APIs. Embeddings and the LLM are served by [Ollama](https://ollama.com/),
vectors are stored in [Chroma](https://www.trychroma.com/), and the HTTP layer is
[FastAPI](https://fastapi.tiangolo.com/).

## How it works

```
          ┌──────────── ingest ────────────┐         ┌──────────── query ───────────┐
 PDF/MD/TXT → chunk (512/50) → embed ──────►  Chroma  ◄──── embed(question) ── retrieve top-3
                              (nomic-embed-text)  │                                  │
                                                  └──── chunks ──► LLM (qwen2.5:3b) ─┘
                                                                   → answer + citations
```

1. **Ingest** — a document is loaded, split into ~512-character chunks (50-char
   overlap), embedded with `nomic-embed-text`, and stored in a Chroma collection
   named `documents` (cosine similarity).
2. **Query** — the question is embedded, the top 3 most similar chunks are
   retrieved, and `qwen2.5:3b` answers using only that context, citing sources as
   `[filename, chunk N]`.

## Stack

| Component        | Choice                          |
| ---------------- | ------------------------------- |
| API              | FastAPI + Uvicorn               |
| Embeddings       | `nomic-embed-text` (via Ollama) |
| LLM              | `qwen2.5:3b` (via Ollama)       |
| Vector store     | Chroma `0.6.3`                  |
| Package manager  | [uv](https://docs.astral.sh/uv/) |
| Python           | 3.12                            |

## Project layout

```
.
├── app/
│   ├── main.py        # FastAPI app: /health, /ingest, /query
│   ├── ingest.py      # load → chunk → embed → store in Chroma
│   └── query.py       # embed question → retrieve → LLM answer
├── docs/              # documents to ingest (mounted into the container)
├── Dockerfile         # uv-based image for the API
├── docker-compose.yml # ollama + chromadb + rag-api
├── pyproject.toml     # dependencies (managed by uv)
└── uv.lock            # pinned, reproducible lockfile
```

## API

| Method | Path      | Description                                                        |
| ------ | --------- | ----------------------------------------------------------------- |
| GET    | `/health` | Liveness check → `{"status": "ok"}`                               |
| POST   | `/ingest` | Upload a `.pdf` / `.md` / `.txt` file (multipart) and index it     |
| POST   | `/query`  | Body `{"question": "..."}` → `{"answer": ..., "sources": [...]}`   |

Interactive docs are available at `/docs` (Swagger) once the API is running.

## Running with Docker (recommended)

```bash
# 1. Start the stack
docker compose up -d --build

# 2. Pull the models into the Ollama container (first run only; persisted in a volume)
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull qwen2.5:3b

# 3. Use the API (exposed on host port 8080)
curl -F "file=@docs/test.pdf" http://localhost:8080/ingest
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

> The models live in the `ollama_data` named volume, so they survive restarts.
> Use `docker compose down` to stop; avoid `down -v`, which **wipes the volumes**
> (models and the vector store).

## Running locally (without Docker)

Requires [uv](https://docs.astral.sh/uv/) and a running Ollama with both models
pulled (`ollama pull nomic-embed-text && ollama pull qwen2.5:3b`).

```bash
# Install dependencies into a project venv
uv sync

# Start a local Chroma server (separate terminal)
uv run chroma run --host localhost --port 8000 --path ./chroma_data

# Ingest a document (scripts run from the app/ directory)
cd app && uv run python ingest.py            # ingests ../docs/test.pdf

# Ask a question from the CLI
uv run python query.py "What is RAG?"

# Or run the API
uv run uvicorn main:app --reload             # http://localhost:8000
```

> When running the API locally, set `DOCS_DIR` to a real path (it defaults to the
> container path `/app/docs`), e.g. `DOCS_DIR=../docs uv run uvicorn main:app`.

## Configuration

The scripts read these environment variables (with host-friendly defaults), so the
same code works both locally and inside the compose network:

| Variable      | Default     | Purpose                          |
| ------------- | ----------- | -------------------------------- |
| `CHROMA_HOST` | `localhost` | Chroma server host               |
| `CHROMA_PORT` | `8000`      | Chroma server port               |
| `OLLAMA_HOST` | `localhost` | Ollama host                      |
| `OLLAMA_PORT` | `11434`     | Ollama port                      |
| `DOCS_DIR`    | `/app/docs` | Where uploaded files are written |

## Notes

- **Chroma is pinned to `0.6.3`** to match the Python client. A newer server image
  breaks the `0.6.3` client (`KeyError: '_type'`).
- The collection uses **cosine** similarity, so `query.py` reports a similarity
  score in `[0, 1]` (computed as `1 - cosine_distance`).
- In `docker-compose.yml`, Chroma's host port `8000` is published for convenience
  (host-based testing). For a production-style setup it can be removed — `rag-api`
  reaches Chroma over the internal compose network as `chromadb:8000`.
