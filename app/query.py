# app/query.py
import os
import logging
import ollama
import chromadb
from dataclasses import dataclass

# chromadb 0.6.3 wywołuje posthog.capture() z niezgodną sygnaturą i loguje błąd
# telemetryczny przy każdym starcie klienta — wyciszamy ten konkretny logger.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# Domyślnie localhost (testy z hosta); w compose nadpisywane na chromadb/ollama
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"
TOP_K = 3

ollama_client = ollama.Client(host=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided context.
For every claim in your answer, cite the source using [filename, chunk N] format.
If the context does not contain enough information to answer, say so explicitly — do not make up facts."""

@dataclass
class QueryResult:
    answer: str
    sources: list[dict]

def embed(text: str) -> list[float]:
    return ollama_client.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]

def retrieve(query_embedding: list[float], collection) -> dict:
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

def build_context(results: dict) -> str:
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(f"[{meta['source']}, chunk {meta['chunk_index']}]\n{doc}")
    return "\n\n---\n\n".join(chunks)

def query(question: str) -> QueryResult:
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_or_create_collection("documents", metadata={"hnsw:space": "cosine"})

    query_embedding = embed(question)
    results = retrieve(query_embedding, collection)

    context = build_context(results)

    response = ollama_client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    sources = [
        {
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "score": round(1 - dist, 4),  # cosine distance → similarity
        }
        for meta, dist in zip(results["metadatas"][0], results["distances"][0])
    ]

    return QueryResult(
        answer=response["message"]["content"],
        sources=sources,
    )

if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "What is this document about?"
    result = query(question)

    print("\n=== ANSWER ===")
    print(result.answer)
    print("\n=== SOURCES ===")
    for s in result.sources:
        print(f"  {s['source']} | chunk {s['chunk_index']} | similarity {s['score']}")