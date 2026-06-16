# app/query.py
import os
import json
import logging
import ollama
import chromadb
from dataclasses import dataclass

from ingest import list_documents

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
MAX_STEPS = 5  # cap on agent tool-calling rounds before forcing a final answer

ollama_client = ollama.Client(host=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")

SYSTEM_PROMPT = """You are a research assistant answering questions strictly from an indexed document knowledge base.

You have tools:
- search_documents(query): retrieve chunks relevant to a query. Call it to gather context BEFORE answering. Call it multiple times — with different queries — for multi-part questions or to refine a weak result.
- list_documents(): see which documents are currently indexed and their chunk counts.

Rules:
- Always gather context with the tools before answering; never answer from prior knowledge.
- For every claim in your answer, cite the source using [filename, chunk N] format.
- If the tools return nothing relevant, say so explicitly — do not make up facts."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the indexed knowledge base for chunks relevant to a query. "
                "Call multiple times with different queries for multi-part questions "
                "or to refine a weak result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List the documents currently indexed and their chunk counts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

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

def search_documents(query_text: str, collection, sources_acc: list[dict]) -> str:
    """Tool: retrieve relevant chunks and record their sources for citation."""
    results = retrieve(embed(query_text), collection)

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

    seen = {(s["source"], s["chunk_index"]) for s in sources_acc}
    for m in matches:
        key = (m["source"], m["chunk_index"])
        if key not in seen:
            seen.add(key)
            sources_acc.append(
                {"source": m["source"], "chunk_index": m["chunk_index"], "score": m["score"]}
            )

    if not matches:
        return "No relevant chunks found for that query."
    return build_context(matches)


def query(question: str) -> QueryResult:
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_or_create_collection("documents", metadata={"hnsw:space": "cosine"})

    sources_acc: list[dict] = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_STEPS):
        response = ollama_client.chat(model=LLM_MODEL, messages=messages, tools=TOOLS)
        msg = response["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return QueryResult(answer=msg["content"], sources=sources_acc)

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            if name == "search_documents":
                result = search_documents(args["query"], collection, sources_acc)
            elif name == "list_documents":
                result = json.dumps(list_documents())
            else:
                result = f"Unknown tool: {name}"
            messages.append({"role": "tool", "name": name, "content": result})

    # Tool budget exhausted — force one final answer without tools.
    final = ollama_client.chat(model=LLM_MODEL, messages=messages)
    return QueryResult(answer=final["message"]["content"], sources=sources_acc)

if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "What is this document about?"
    result = query(question)

    print("\n=== ANSWER ===")
    print(result.answer)
    print("\n=== SOURCES ===")
    for s in result.sources:
        print(f"  {s['source']} | chunk {s['chunk_index']} | similarity {s['score']}")