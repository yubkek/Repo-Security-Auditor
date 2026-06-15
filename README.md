# AI Codebase Security Auditor

Ask questions about any codebase in plain english and it reads your code, indexes it semantically, and answers with full context, across multiple turns, without losing track of what was said earlier.

---

## How it works

```
Your Repo
    |
    v
Chunk files (AST boundaries for Python, line-based for everything else)
    |
    v
Embed chunks with text-embedding-3-small -> store vectors to disk
    |
    v
You ask a question
    |
    v
Embed query -> cosine similarity search -> re-rank top candidates with LLM
    |
    v
Inject relevant chunks + conversation history into GPT-4o-mini
    |
    v
Streamed answer, history compacted automatically as conversation grows
```

The first run indexes your repo and saves everything to disk. Every run after that loads instantly - only re-indexes if files actually changed.

---

## Features

**RAG Pipeline**
- Reads 20+ file types: `.py`, `.js`, `.ts`, `.tsx`, `.go`, `.java`, `.rs`, `.c`, `.cpp`, `.cs`, `.swift`, `.kt`, `.json`, `.yaml`, and more
- AST-based chunking for Python | splits on actual function and class boundaries, not arbitrary line counts
- Fixed line-based chunking for all other languages
- Skips junk directories automatically: `.git`, `__pycache__`, `venv`, `node_modules`, `dist`, etc.
- Semantic search via cosine similarity | finds relevant code by meaning, not keyword matching
- Re-ranking | fetches 9 candidates, re-ranks with the LLM, returns the top 3 most relevant
- Score threshold | discards chunks below 0.3 similarity to prevent hallucination from irrelevant context

**Persistence**
- Embedding cache (`.embedding_cache.json`) | embeddings are stored by content hash; the same code is never embedded twice
- Vector store (`.vector_store.pkl`) | full store saved to disk after every build
- Fingerprint invalidation | hashes the entire codebase state; only rebuilds when files actually change
- Both files are gitignored and generated per user on first run

**Context Management**
- Full multi-turn conversation | remembers everything said in the session
- Token counting via `tiktoken`
- Automatic context compaction at 150k tokens | compresses old turns into a dense summary, keeps the 2 most recent turns verbatim
- Rolling summary injected into every prompt so the model never fully forgets earlier turns

**Interface**
- Streaming responses | answers print word by word, not all at once after a long wait
- Interactive REPL | type questions naturally, exit with `exit`, `quit`, `q`, or Ctrl+C
- Batch embedding on first run | all new chunks embedded in batches of 500, not one call at a time

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set your API key**

Supports OpenAI and Groq. Set whichever you have:
```bash
# OpenAI (default)
export OPENAI_API_KEY=sk-...

# Groq
export GROQ_API_KEY=gsk_...
```

Then set `PROVIDER` in `config.py` to match:
```python
PROVIDER = "openai"  # or "groq"
```

When using Groq, chat runs through `llama-3.3-70b-versatile` and embeddings run locally via `sentence-transformers` - no second API key needed. If you switch providers, delete `.embedding_cache.json` and `.vector_store.pkl` since the embedding dimensions differ.

**3. Point it at your repo**

Edit `CODEBASE_DIR` in `config.py`:
```python
CODEBASE_DIR = "/path/to/your/repo"
```

**4. Run**
```bash
python main.py
```

First run indexes the repo and saves to disk. Subsequent runs load instantly.

---

## Example session

```
[RAG] Scanning codebase...
[RAG] Changes detected, rebuilding vector store
[RAG] Embedding 42 new chunks in batches
[RAG] Indexed 42 chunks (42 new, 0 from cache).

Ready. Ask anything about the codebase, or type 'exit' to quit.

You: are there any places where user input isn't validated before hitting the database?
[RAG] Embedding query and searching vector store
[RAG] Retrieved './api/routes.py:45-89' (similarity: 0.821)
[RAG] Retrieved './db/queries.py:12-34' (similarity: 0.774)
[RAG] Retrieved './auth/middleware.py:1-28' (similarity: 0.751)
[SYSTEM] Active History Tokens: 1820 / 150000

[AI]: Yes - in routes.py on line 67, the user_id parameter is passed directly
into the SQL query without sanitisation...
--------------------------------------------------

You: what about the auth middleware, is the JWT verified properly?
[RAG] Embedding query and searching vector store
[RAG] Retrieved './auth/middleware.py:1-28' (similarity: 0.893)
[RAG] Retrieved './auth/tokens.py:12-45' (similarity: 0.811)
[RAG] Retrieved './api/routes.py:45-89' (similarity: 0.743)
[SYSTEM] Active History Tokens: 3204 / 150000

[AI]: No - in middleware.py the token is decoded with verify_signature set to
False, meaning any token will pass regardless of whether it was signed correctly...
--------------------------------------------------

You: exit
```

---

## Configuration

All tuning knobs are in `config.py`:

| Variable | Default | What it does |
|---|---|---|
| `CODEBASE_DIR` | `"."` | Directory to scan |
| `MODEL_NAME` | `gpt-4o-mini` | Model used for answers and re-ranking |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model used for embeddings |
| `TOP_K` | `3` | Chunks returned per query after re-ranking |
| `RERANK_CANDIDATES` | `9` | Candidates fetched before re-ranking |
| `SCORE_THRESHOLD` | `0.3` | Minimum similarity score to include a chunk |
| `CHUNK_SIZE` | `60` | Lines per chunk for non-Python files |
| `TOKEN_THRESHOLD` | `150,000` | Token count that triggers context compaction |

---

## Project structure

```
├── main.py        | REPL loop, conversation management, streaming
├── rag.py         | chunking, embedding, vector store, search, re-ranking
├── context.py     | token counting and context compaction
├── config.py      | all constants and configuration
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- OpenAI API key (default) or Groq API key
