import ast
import os
import numpy as np
from config import client, EMBEDDING_MODEL, TOP_K, CODEBASE_DIR

SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", ".mypy_cache", "dist", "build"}

vector_store: dict[str, dict] = {}


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))  # 1.0 = same meaning, 0.0 = unrelated


def chunk_file(path: str) -> dict[str, str]:
    # split on actual function/class boundaries so each chunk is semantically complete
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    lines = source.splitlines(keepends=True)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {f"{path}:1-{len(lines)}": source}  # unparseable file, index it whole

    chunks = {}
    covered = set()

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start, end = node.lineno - 1, node.end_lineno  # slice indices into lines[]
        chunks[f"{path}:{node.lineno}-{node.end_lineno}"] = "".join(lines[start:end])
        covered.update(range(start, end))

    # module-level code (imports, constants, etc.) that sits outside any function/class
    module_level = "".join(line for i, line in enumerate(lines) if i not in covered)
    if module_level.strip():
        chunks[f"{path}:module"] = module_level

    return chunks


def load_codebase(directory: str) -> dict[str, str]:
    chunks = {}
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]  # prune junk dirs before descending
        for fname in files:
            if not fname.endswith(".py"):
                continue
            chunks.update(chunk_file(os.path.join(root, fname)))
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
