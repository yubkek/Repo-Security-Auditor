import ast
import hashlib
import json
import os
import numpy as np
from config import client, EMBEDDING_MODEL, TOP_K, CHUNK_SIZE, CODEBASE_DIR, CACHE_PATH, STORE_PATH, FILE_EXTENSIONS

SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", ".mypy_cache", "dist", "build"}

vector_store: dict[str, dict] = {}


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def codebase_fingerprint(chunks: dict[str, str]) -> str:
    # single hash representing the entire codebase state - changes if any file does
    combined = "".join(f"{k}:{content_hash(v)}" for k, v in sorted(chunks.items()))
    return hashlib.md5(combined.encode()).hexdigest()


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def load_store_from_disk() -> tuple[dict | None, str | None]:
    if not os.path.exists(STORE_PATH):
        return None, None
    with open(STORE_PATH) as f:
        data = json.load(f)
    return data["store"], data["fingerprint"]


def save_store_to_disk(store: dict, fingerprint: str) -> None:
    with open(STORE_PATH, "w") as f:
        json.dump({"store": store, "fingerprint": fingerprint}, f)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))  # 1.0 = same meaning, 0.0 = unrelated


def chunk_python(path: str, source: str, lines: list) -> dict[str, str]:
    # split on actual function/class boundaries so each chunk is semantically complete
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {f"{path}:1-{len(lines)}": source}  # unparseable, index whole file

    chunks = {}
    covered = set()

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start, end = node.lineno - 1, node.end_lineno
        chunks[f"{path}:{node.lineno}-{node.end_lineno}"] = "".join(lines[start:end])
        covered.update(range(start, end))

    module_level = "".join(line for i, line in enumerate(lines) if i not in covered)
    if module_level.strip():
        chunks[f"{path}:module"] = module_level

    return chunks


def chunk_generic(path: str, lines: list) -> dict[str, str]:
    chunks = {}
    for i in range(0, len(lines), CHUNK_SIZE):
        block = lines[i:i + CHUNK_SIZE]
        chunks[f"{path}:{i + 1}-{i + len(block)}"] = "".join(block)
    return chunks


def chunk_file(path: str) -> dict[str, str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()
    lines = source.splitlines(keepends=True)
    return chunk_python(path, source, lines) if path.endswith(".py") else chunk_generic(path, lines)


def load_codebase(directory: str) -> dict[str, str]:
    chunks = {}
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]  # prune junk dirs before descending
        for fname in files:
            if any(fname.endswith(ext) for ext in FILE_EXTENSIONS):
                chunks.update(chunk_file(os.path.join(root, fname)))
    return chunks


def build_vector_store() -> None:
    global vector_store
    print("[RAG] Scanning codebase...")
    chunks = load_codebase(CODEBASE_DIR)
    fingerprint = codebase_fingerprint(chunks)

    saved_store, saved_fingerprint = load_store_from_disk()
    if saved_store and saved_fingerprint == fingerprint:
        # codebase hasn't changed - load straight from disk, no API calls needed
        vector_store = saved_store
        print(f"[RAG] Loaded {len(vector_store)} chunks from disk (no changes detected).\n")
        return

    # first run or files changed - embed whatever is new, pull the rest from cache
    print("[RAG] Changes detected, rebuilding vector store...")
    cache = load_cache()
    new_count = 0

    for chunk_id, code in chunks.items():
        h = content_hash(code)
        if h in cache:
            embedding = cache[h]  # cache hit - skip the API call
        else:
            embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=code).data[0].embedding
            cache[h] = embedding
            new_count += 1
        vector_store[chunk_id] = {"code": code, "embedding": embedding}

    save_cache(cache)
    save_store_to_disk(vector_store, fingerprint)
    print(f"[RAG] Indexed {len(vector_store)} chunks ({new_count} new, {len(chunks) - new_count} from cache).\n")


def run_rag_search(user_query: str) -> str:
    # embed the question, score every chunk, hand back the top-k winners
    print("[RAG] Embedding query and searching vector store...")
    query_vec = client.embeddings.create(model=EMBEDDING_MODEL, input=user_query).data[0].embedding

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
