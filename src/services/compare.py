import os
import time
from fastapi import HTTPException
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from src.models import (
    CompareRequest,
    CompareResponse,
    ModelComparisonItem,
    RetrievalDiagnostics,
    SourceChunk,
)
from src.prompts.catalog import get_prompt_strategy
from src.services.evaluate import (
    cancel_current_evaluation,
    is_evaluation_running,
)
from src.services.query import _get_vectorstore, get_llm
from src.utils.env import MAX_CONTEXT_CHARS, MAX_TOP_K
from src.utils.log_manager import logger

COMPARE_MODELS = ["llama3.2", "mistral"]


async def handle_compare(request: CompareRequest) -> CompareResponse:
    """Execute model comparison across llama3.2 and mistral sequentially on identical retrieved context."""
    try:
        if is_evaluation_running():
            logger.info("Preempting ongoing judge evaluation for incoming model comparison...")
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

        # 1. Retrieval (shared across both models)
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
        source_chunks: list[SourceChunk] = []

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
                        page_num = int(raw_page) + 1
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
        low_confidence = top_score is not None and top_score > 0.75
        diagnostics = RetrievalDiagnostics(
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            top_score=round(top_score, 4) if top_score is not None else None,
            low_confidence=low_confidence,
            score_metric="cosine_distance",
        )

        context = "\n\n".join(sources)
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]

        strategy = get_prompt_strategy(request.prompt_id)
        compare_template = (
            strategy.template
            + "\nBe concise, direct, and factual in 2-3 sentences. Do not add introductory filler."
        )
        prompt = PromptTemplate(
            input_variables=["contexto", "question"], template=compare_template
        )

        # 2. Sequential Model Invocations (strictly non-concurrent to preserve CPU & memory)
        model_results: list[ModelComparisonItem] = []

        for model_name in COMPARE_MODELS:
            logger.info(f"Running sequential comparison inference on model: {model_name}")
            from langchain_ollama import ChatOllama
            from src.utils.env import OLLAMA_BASE_URL
            llm = ChatOllama(
                model=model_name,
                base_url=OLLAMA_BASE_URL,
                temperature=0,
                num_predict=120,
                keep_alive="5m",
            )
            sequence = RunnableSequence(prompt | llm)

            t0 = time.perf_counter()
            response = sequence.invoke({"contexto": context, "question": request.question})
            gen_time_sec = max(0.001, time.perf_counter() - t0)
            gen_latency_ms = gen_time_sec * 1000.0

            ans_text = response.content
            token_count_est = max(1, len(ans_text) // 4)
            tok_per_sec = round(token_count_est / gen_time_sec, 2)

            model_results.append(
                ModelComparisonItem(
                    model=model_name,
                    answer=ans_text,
                    generation_latency_ms=round(gen_latency_ms, 2),
                    token_count_estimate=token_count_est,
                    tokens_per_second=tok_per_sec,
                )
            )

        return CompareResponse(
            question=request.question,
            sources=sources,
            source_chunks=source_chunks,
            retrieval_diagnostics=diagnostics,
            prompt_id=strategy.id,
            prompt_version=strategy.version,
            models=model_results,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Model comparison pipeline error: {exc}")
        raise HTTPException(
            status_code=500, detail="Internal model comparison pipeline error."
        )
