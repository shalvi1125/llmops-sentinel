import os

from dotenv import load_dotenv

load_dotenv()


def _get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default


def _normalize_local_loopback(url: str) -> str:
    """Resolve localhost URLs to IPv4 127.0.0.1 to prevent 2000ms Windows ::1 fallback stalls."""
    if url.endswith("://localhost"):
        return url[:-9] + "127.0.0.1"
    return url.replace("://localhost:", "://127.0.0.1:").replace("://localhost/", "://127.0.0.1/")


os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ.setdefault("MLFLOW_HOST", "mlflow")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
os.environ.setdefault("MLFLOW_HTTP_TCP_KEEPALIVE", "false")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "mistral")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = _normalize_local_loopback(os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"))
MLFLOW_TRACKING_URI = _normalize_local_loopback(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/tmp/chroma")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "ragscope_collection")
API_KEY = os.getenv("API_KEY", "")
MAX_UPLOAD_SIZE_BYTES = _get_int_env("MAX_UPLOAD_SIZE_BYTES", 10 * 1024 * 1024)
MAX_TOP_K = _get_int_env("MAX_TOP_K", 20)
MAX_CONTEXT_CHARS = _get_int_env("MAX_CONTEXT_CHARS", 20000)

if not API_KEY:
    raise RuntimeError("API_KEY is required and must not be empty.")
