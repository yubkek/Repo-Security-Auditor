import os
from openai import OpenAI

# grab the key or die early - better than a cryptic auth error 3 calls in
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

client = OpenAI(api_key=api_key)

MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"  # cheap + good enough for code search

TOKEN_THRESHOLD = 150_000  # ~80% of gpt-4o-mini's 128k context window
TOP_K = 3                  # how many chunks to return after re-ranking
RERANK_CANDIDATES = 9      # how many cosine candidates to fetch before re-ranking down to TOP_K
CHUNK_SIZE = 60            # lines per chunk for non-Python files
SCORE_THRESHOLD = 0.3      # discard chunks below this similarity - stops hallucination from irrelevant context

CODEBASE_DIR = "."                    # point this at whatever repo you want to audit
CACHE_PATH = ".embedding_cache.json"  # persists embeddings between runs
STORE_PATH = ".vector_store.pkl"      # persists the full vector store between runs (pickle, not JSON)

FILE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".java", ".rb", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".cs",
    ".swift", ".kt", ".php",
    ".json", ".yaml", ".yml", ".toml",
}
