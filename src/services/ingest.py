import os
import tempfile
from collections import Counter
from pathlib import Path

from fastapi import HTTPException, UploadFile
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import IngestResponse
from src.utils.env import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    MAX_UPLOAD_SIZE_BYTES,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
)
from src.utils.log_manager import logger

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
ALLOWED_CONTENT_TYPES = {
    ".pdf": {"application/pdf"},
    ".txt": {"text/plain", "application/octet-stream"},
}


def _get_ingest_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def get_indexed_documents() -> list[dict[str, str | int]]:
    """Return all distinct documents indexed in ChromaDB with their chunk counts."""
    vector_store = _get_ingest_vectorstore()
    try:
        res = vector_store._collection.get(include=["metadatas"])
        metas = res.get("metadatas", []) or []
        counts: Counter[str] = Counter()
        for m in metas:
            if m and isinstance(m, dict):
                name = m.get("filename") or m.get("source") or "unknown"
                clean_name = os.path.basename(name) if ("/" in name or "\\" in name) else name
                counts[clean_name] += 1
        return [{"filename": name, "chunks": count} for name, count in sorted(counts.items())]
    except Exception as e:
        logger.error(f"Error listing indexed documents: {e}")
        return []


def clear_all_documents() -> dict[str, str | int]:
    """Clear all documents from ChromaDB collection."""
    vector_store = _get_ingest_vectorstore()
    try:
        res = vector_store._collection.get()
        all_ids = res.get("ids", []) or []
        if all_ids:
            vector_store._collection.delete(ids=all_ids)
        try:
            from src.services.query import clear_vectorstore_cache
            clear_vectorstore_cache()
        except ImportError:
            pass
        return {"status": "ok", "deleted_chunks": len(all_ids)}
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {e}")


def delete_document(filename: str) -> dict[str, str | int]:
    """Delete all chunks for a specific document from ChromaDB."""
    vector_store = _get_ingest_vectorstore()
    try:
        res = vector_store._collection.get(include=["metadatas"])
        metas = res.get("metadatas", []) or []
        ids = res.get("ids", []) or []
        to_delete = []
        for doc_id, m in zip(ids, metas):
            if m and isinstance(m, dict):
                raw_name = m.get("filename") or m.get("source") or ""
                clean_name = os.path.basename(raw_name) if ("/" in raw_name or "\\" in raw_name) else raw_name
                if clean_name == filename or raw_name == filename:
                    to_delete.append(doc_id)
        if to_delete:
            vector_store._collection.delete(ids=to_delete)
        try:
            from src.services.query import clear_vectorstore_cache
            clear_vectorstore_cache()
        except ImportError:
            pass
        return {"status": "ok", "deleted_chunks": len(to_delete), "filename": filename}
    except Exception as e:
        logger.error(f"Error deleting document {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")


async def ingest_document(file: UploadFile, clear_existing: bool = False) -> IngestResponse:
    if clear_existing:
        clear_all_documents()

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Only .pdf and .txt are accepted.",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        total_bytes = 0

        with open(tmp_path, "wb") as tmp:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Uploaded file is too large. "
                            f"Maximum allowed size is {MAX_UPLOAD_SIZE_BYTES} bytes."
                        ),
                    )

                tmp.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        content_type = (file.content_type or "").lower()
        if content_type and content_type not in ALLOWED_CONTENT_TYPES.get(suffix, set()):
            raise HTTPException(
                status_code=400,
                detail=(f"Unsupported content type '{content_type}' for '{suffix}' files."),
            )

        embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )

        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
            pages = loader.load_and_split()
        else:
            loader = TextLoader(tmp_path)
            pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = splitter.split_documents(pages)

        for chunk in chunks:
            if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                chunk.metadata["filename"] = file.filename or ""

        vector_store.add_documents(documents=chunks)

        try:
            from src.services.query import clear_vectorstore_cache
            clear_vectorstore_cache()
        except ImportError:
            pass

        return IngestResponse(
            status="ok",
            chunks_stored=len(chunks),
            filename=file.filename or "",
        )
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
