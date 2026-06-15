import os

# switch between "openai" and "groq" here
PROVIDER = "openai"

if PROVIDER == "openai":
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set")
    client = OpenAI(api_key=api_key)
    MODEL_NAME = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"  # api-based, billed per token
    SCORE_THRESHOLD = 0.3  # openai embeddings score higher, 0.3 filters noise well

elif PROVIDER == "groq":
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set")
    client = Groq(api_key=api_key)
    MODEL_NAME = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # runs locally via sentence-transformers, no api key needed
    SCORE_THRESHOLD = 0.15  # minilm scores lower than openai, needs a looser threshold

else:
    raise ValueError(f"Unknown provider '{PROVIDER}' - use 'openai' or 'groq'")

TOKEN_THRESHOLD = 150_000
TOP_K = 3
RERANK_CANDIDATES = 9
CHUNK_SIZE = 60

CODEBASE_DIR = r"."
CACHE_PATH = ".embedding_cache.json"
STORE_PATH = ".vector_store.pkl"

FILE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".java", ".rb", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".cs",
    ".swift", ".kt", ".php",
    ".json", ".yaml", ".yml", ".toml",
}
