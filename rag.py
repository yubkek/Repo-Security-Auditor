import os
import numpy as np
from config import client, EMBEDDING_MODEL, TOP_K, CHUNK_SIZE, CODEBASE_DIR

vector_store: dict[str, dict] = {}

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding

def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))  # 1.0 = same meaning, 0.0 = unrelated

def load_codebase(directory: str) -> dict[str, str]:
    # walk the directory, chunk every .py file into CHUNK_SIZE-line blocks
    chunks = {}
    for root, _, files in os.walk(directory):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for i in range(0, len(lines), CHUNK_SIZE):
                block = lines[i:i + CHUNK_SIZE]
                chunk_id = f"{path}:{i + 1}-{i + len(block)}"
                chunks[chunk_id] = "".join(block)
    return chunks

def build_vector_store() -> None:
    # runs once at startup - reads real files, embeds every chunk, stores vectors
    global vector_store
    print("[RAG] Building vector store...")
    chunks = load_codebase(CODEBASE_DIR)
    for chunk_id, code in chunks.items():
        vector_store[chunk_id] = {"code": code, "embedding": get_embedding(code)}
    print(f"[RAG] Indexed {len(vector_store)} chunks from '{CODEBASE_DIR}'.\n")

def run_rag_search(user_query: str) -> str:
    # embed the question, score every chunk, hand back the top-k winners
    print("[RAG] Embedding query and searching vector store...")
    query_vec = get_embedding(user_query)

    scored = [
        (chunk_id, cosine_similarity(query_vec, data["embedding"]), data["code"])
        for chunk_id, data in vector_store.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    snippets = []
    for chunk_id, score, code in scored[:TOP_K]:
        print(f"[RAG] Retrieved '{chunk_id}' (similarity: {score:.3f})")
        snippets.append(f"--- {chunk_id} (similarity: {score:.3f}) ---\n{code}")

    return "\n\n".join(snippets) if snippets else "No relevant code found."
