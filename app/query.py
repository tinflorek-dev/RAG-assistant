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
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.4"))
NO_CONTEXT_ANSWER = "I don't have relevant information on that in the indexed documents."

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

def build_context(matches: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{m['source']}, chunk {m['chunk_index']}]\n{m['document']}" for m in matches
    )

def query(question: str) -> QueryResult:
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_or_create_collection("documents", metadata={"hnsw:space": "cosine"})

    query_embedding = embed(question)
    results = retrieve(query_embedding, collection)

    matches = [
        {
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "score": round(1 - dist, 4),  # cosine distance → similarity
            "document": doc,
        }
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
        if 1 - dist >= SIMILARITY_THRESHOLD
    ]

    if not matches:
        return QueryResult(answer=NO_CONTEXT_ANSWER, sources=[])

    context = build_context(matches)

    response = ollama_client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    sources = [
        {"source": m["source"], "chunk_index": m["chunk_index"], "score": m["score"]}
        for m in matches
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