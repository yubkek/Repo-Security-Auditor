import ast
import hashlib
import json
import os
import pickle
import numpy as np
from config import client, PROVIDER, MODEL_NAME, EMBEDDING_MODEL, TOP_K, RERANK_CANDIDATES, CHUNK_SIZE, SCORE_THRESHOLD, CODEBASE_DIR, CACHE_PATH, STORE_PATH, FILE_EXTENSIONS

SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", ".mypy_cache", "dist", "build"}

# load the local embedding model if using groq - openai embeddings go through the api instead
if PROVIDER == "groq":
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBEDDING_MODEL)

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
    with open(STORE_PATH, "rb") as f:
        data = pickle.load(f)
    return data["store"], data["fingerprint"]


def save_store_to_disk(store: dict, fingerprint: str) -> None:
    with open(STORE_PATH, "wb") as f:
        pickle.dump({"store": store, "fingerprint": fingerprint}, f)


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
        # codebase hasn't changed - load straight from disk, no computation needed
        vector_store = saved_store
        print(f"[RAG] Loaded {len(vector_store)} chunks from disk (no changes detected).\n")
        return

    print("[RAG] Changes detected, rebuilding vector store")
    cache = load_cache()

    uncached = [(cid, code) for cid, code in chunks.items() if content_hash(code) not in cache]
    if uncached:
        print(f"[RAG] Embedding {len(uncached)} new chunks...")
        if PROVIDER == "groq":
            # local batch encode - no api calls, free
            embeddings = embedder.encode([code for _, code in uncached], batch_size=64, show_progress_bar=True)
            for (_, code), emb in zip(uncached, embeddings):
                cache[content_hash(code)] = emb.tolist()
        else:
            # openai batch api - 500 chunks per call
            for i in range(0, len(uncached), 500):
                batch = uncached[i:i + 500]
                response = client.embeddings.create(model=EMBEDDING_MODEL, input=[code for _, code in batch])
                for (_, code), emb in zip(batch, response.data):
                    cache[content_hash(code)] = emb.embedding

    for chunk_id, code in chunks.items():
        vector_store[chunk_id] = {"code": code, "embedding": cache[content_hash(code)]}

    save_cache(cache)
    save_store_to_disk(vector_store, fingerprint)
    print(f"[RAG] Indexed {len(vector_store)} chunks ({len(uncached)} new, {len(chunks) - len(uncached)} from cache).\n")


def rerank(query: str, candidates: list[tuple]) -> list[tuple]:
    # ask the LLM to pick the most relevant chunks from the cosine shortlist
    formatted = "\n\n".join(
        f"[{i}] {chunk_id}\n{code[:400]}"
        for i, (chunk_id, _, code) in enumerate(candidates)
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": (
                f'Query: "{query}"\n\n'
                f"Rank these code chunks by relevance. "
                f'Return ONLY a JSON object with key "order" containing an array of indices '
                f"sorted from most to least relevant.\n\n{formatted}"
            )
        }],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    try:
        order = json.loads(response.choices[0].message.content)["order"]
        return [candidates[i] for i in order if i < len(candidates)]
    except (json.JSONDecodeError, KeyError, IndexError):
        return candidates  # fall back to cosine order if parsing fails


def run_rag_search(user_query: str) -> str:
    # embed the question, score every chunk, re-rank the shortlist, apply threshold
    print("[RAG] Embedding query and searching vector store")
    if PROVIDER == "groq":
        query_vec = embedder.encode(user_query).tolist()
    else:
        query_vec = client.embeddings.create(model=EMBEDDING_MODEL, input=user_query).data[0].embedding

    scored = [
        (chunk_id, cosine_similarity(query_vec, data["embedding"]), data["code"])
        for chunk_id, data in vector_store.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    candidates = [s for s in scored[:RERANK_CANDIDATES] if s[1] >= SCORE_THRESHOLD]
    if not candidates:
        return "No relevant code found"

    ranked = rerank(user_query, candidates)

    snippets = []
    for chunk_id, score, code in ranked[:TOP_K]:
        print(f"[RAG] Retrieved '{chunk_id}' (similarity: {score:.3f})")
        snippets.append(f"--- {chunk_id} (similarity: {score:.3f}) ---\n{code}")

    return "\n\n".join(snippets)
