import os
import time
from unittest.mock import MagicMock
from fastapi import BackgroundTasks, HTTPException
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.models import (
    QueryRequest,
    QueryResponse,
    RetrievalDiagnostics,
    SourceChunk,
)
from src.prompts.catalog import get_prompt_strategy
from src.services.evaluate import (
    cancel_current_evaluation,
    is_evaluation_running,
    run_judge_evaluations,
)
from src.utils.env import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    MAX_CONTEXT_CHARS,
    MAX_TOP_K,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_MODEL,
)
from src.utils.log_manager import logger

_llm_cache: dict[str, ChatOllama] = {}
_vectorstore: Chroma | None = None
_embeddings: OllamaEmbeddings | None = None


def clear_vectorstore_cache() -> None:
    """Clear the cached vectorstore instance so fresh data is loaded."""
    global _vectorstore
    _vectorstore = None


def get_llm(model: str = OLLAMA_MODEL) -> ChatOllama:
    """Return a cached ChatOllama instance for the requested model."""
    global _llm_cache
    if model not in _llm_cache:
        _llm_cache[model] = ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            num_predict=220,
            num_thread=8,
            keep_alive="15m",
        )
    return _llm_cache[model]


def _get_vectorstore() -> Chroma:
    """Return a cached singleton Chroma vectorstore."""
    global _vectorstore, _embeddings
    if _vectorstore is None:
        if _embeddings is None:
            _embeddings = OllamaEmbeddings(
                model=OLLAMA_EMBED_MODEL,
                base_url=OLLAMA_BASE_URL,
                keep_alive=300,
            )
        _vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=_embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
    return _vectorstore


async def handle_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks | None = None,
) -> QueryResponse:
    answer = ""
    sources: list[str] = []
    source_chunks: list[SourceChunk] = []

    try:
        if is_evaluation_running():
            logger.info("Preempting ongoing judge evaluation for incoming query...")
            cancel_current_evaluation()

        vectorstore = _get_vectorstore()

        try:
            collection_count = vectorstore._collection.count()
        except Exception:
            collection_count = 0

        if collection_count == 0:
            raise HTTPException(
                status_code=404,
                detail="No documents found. Please ingest documents first.",
            )

        safe_top_k = min(request.top_k, MAX_TOP_K)

        # Retrieval with latency measurement & score extraction
        t_retrieval_start = time.perf_counter()
        source_docs = []
        doc_scores: list[float | None] = []

        if hasattr(vectorstore, "similarity_search_with_score"):
            try:
                results = vectorstore.similarity_search_with_score(
                    request.question, k=safe_top_k
                )
                if isinstance(results, list) and (
                    len(results) == 0 or isinstance(results[0], tuple)
                ):
                    for doc, score in results:
                        source_docs.append(doc)
                        try:
                            doc_scores.append(float(score))
                        except (ValueError, TypeError):
                            doc_scores.append(None)
            except Exception:
                pass

        if not source_docs:
            retriever = vectorstore.as_retriever(search_kwargs={"k": safe_top_k})
            source_docs = retriever.invoke(request.question)
            doc_scores = [None] * len(source_docs)

        retrieval_latency_ms = (time.perf_counter() - t_retrieval_start) * 1000.0

        sources = [str(doc.page_content) for doc in source_docs]

        # Build structured source chunks with attribution & token estimates
        for idx, (doc, score) in enumerate(zip(source_docs, doc_scores)):
            meta = getattr(doc, "metadata", {})
            clean_doc_name = "Document"
            page_num: int | None = None

            if isinstance(meta, dict):
                raw_name = meta.get("filename") or meta.get("source")
                if raw_name and isinstance(raw_name, str):
                    clean_doc_name = (
                        os.path.basename(raw_name)
                        if ("/" in raw_name or "\\" in raw_name)
                        else raw_name
                    )
                raw_page = meta.get("page")
                if raw_page is not None:
                    try:
                        page_num = int(raw_page) + 1  # 1-indexed for users
                    except (ValueError, TypeError):
                        page_num = None

            text_content = str(doc.page_content)
            token_est = max(1, len(text_content) // 4)

            source_chunks.append(
                SourceChunk(
                    chunk_id=f"chunk-{idx + 1}",
                    text=text_content,
                    score=round(score, 4) if score is not None else None,
                    document_name=clean_doc_name,
                    page_number=page_num,
                    token_count_estimate=token_est,
                )
            )

        top_score = doc_scores[0] if doc_scores and doc_scores[0] is not None else None
        # Cosine distance: lower is more similar; > 0.75 indicates poor retrieval match
        low_confidence = top_score is not None and top_score > 0.75
        diagnostics = RetrievalDiagnostics(
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            top_score=round(top_score, 4) if top_score is not None else None,
            low_confidence=low_confidence,
            score_metric="cosine_distance",
        )

        # Context assembly & length guarding: keep context focused (< 3500 chars) for fast CPU inference
        max_safe_context = min(MAX_CONTEXT_CHARS, 3500)
        context = "\n\n".join(sources)
        if len(context) > max_safe_context:
            context = context[:max_safe_context]

        # Versioned prompt strategy selection
        strategy = get_prompt_strategy(request.prompt_id)
        selected_model = request.model or OLLAMA_MODEL
        llm = get_llm(model=selected_model)

        prompt = PromptTemplate(
            input_variables=["contexto", "question"], template=strategy.template
        )
        sequence = RunnableSequence(prompt | llm)

        t_gen_start = time.perf_counter()
        response = sequence.invoke({"contexto": context, "question": request.question})
        gen_latency_ms = (time.perf_counter() - t_gen_start) * 1000.0
        answer = response.content

        # Background judge evaluation
        if sources:
            if background_tasks is not None:
                background_tasks.add_task(
                    run_judge_evaluations,
                    question=request.question,
                    answer=answer,
                    context_chunks=sources,
                )
            else:
                run_judge_evaluations(
                    question=request.question,
                    answer=answer,
                    context_chunks=sources,
                )

        return QueryResponse(
            answer=answer,
            sources=sources,
            source_chunks=source_chunks,
            retrieval_diagnostics=diagnostics,
            prompt_id=strategy.id,
            prompt_version=strategy.version,
            model=selected_model,
            generation_latency_ms=round(gen_latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Query pipeline error: {exc}")
        raise HTTPException(status_code=500, detail="Internal query pipeline error.")
