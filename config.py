import os
from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

client = OpenAI(api_key=api_key)

MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"  # cheap + good enough for code search

TOKEN_THRESHOLD = 450  # kept low so compaction kicks in fast during the demo
TOP_K = 2 # how many code chunks to pull per query
CHUNK_SIZE = 40 # lines per chunk when splitting files

CODEBASE_DIR = "." # point this at whatever repo you want to audit
