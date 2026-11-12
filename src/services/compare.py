import asyncio
import time
from fastapi import HTTPException
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from src.metrics import (
    CONTEXT_CHAR_COUNT,
    EVAL_PREEMPTIONS_TOTAL,
    LLM_ACTIVE_REQUESTS,
    LLM_GENERATION_DURATION_SECONDS,
    LLM_LAST_TOKENS_PER_SECOND,
    LLM_REQUESTS_TOTAL,
    LLM_TOKENS_GENERATED_TOTAL,
    LLM_TOKENS_PER_SECOND,
)
from src.models import (
    CompareRequest,
    CompareResponse,
    ModelComparisonItem,
)
from src.prompts.catalog import get_prompt_strategy
from src.services.evaluate import (
    cancel_current_evaluation,
    is_evaluation_running,
)
from src.services.query import (
    _detect_token_source,
    _extract_token_metrics,
    _get_vectorstore,
    get_generation_settings,
    get_llm,
    retrieve_context_chunks,
)
from src.utils.context import format_context_for_llm
from src.utils.env import MAX_CONTEXT_CHARS
from src.utils.log_manager import logger

COMPARE_MODELS = ["llama3.2", "mistral"]


async def handle_compare(request: CompareRequest) -> CompareResponse:
    """Execute sequential model comparison off the asyncio event loop on shared compacted context."""
    try:
        if is_evaluation_running():
            logger.info("Preempting ongoing judge evaluation for incoming model comparison...")
            EVAL_PREEMPTIONS_TOTAL.labels(trigger="compare").inc()
            cancel_current_evaluation()

        vectorstore = _get_vectorstore()
        sources, source_chunks, diagnostics, scope_label = await asyncio.to_thread(
            retrieve_context_chunks,
            vectorstore,
            request.question,
            request.top_k,
            request.document_id,
            request.document_name,
            request.use_mmr,
            request.mmr_lambda,
        )

        max_safe_context = min(MAX_CONTEXT_CHARS, 3500)
        source_labels = [
            f"{sc.document_name} p.{sc.page_number}" if sc.page_number else (sc.document_name or "")
            for sc in source_chunks
        ]
        context = format_context_for_llm(
            sources,
            max_chars=max_safe_context,
            source_labels=source_labels if source_chunks else None,
        )
        CONTEXT_CHAR_COUNT.observe(float(len(context)))

        strategy = get_prompt_strategy(request.prompt_id)
        compare_template = (
            strategy.template
            + "\nKeep response concise (2-3 sentences), direct, and strictly grounded."
        )
        prompt = PromptTemplate(
            input_variables=["contexto", "question"], template=compare_template
        )

        # Sequential Model Invocations off the event loop (strictly non-concurrent to preserve RAM/CPU)
        model_results: list[ModelComparisonItem] = []

        for model_name in COMPARE_MODELS:
            logger.info(f"Running sequential comparison inference on model: {model_name}")
            llm = get_llm(model_name)
            sequence = RunnableSequence(prompt | llm)

            t0 = time.perf_counter()
            LLM_ACTIVE_REQUESTS.labels(model=model_name).inc()
            try:
                response = await asyncio.to_thread(
                    sequence.invoke, {"contexto": context, "question": request.question}
                )
                gen_time_sec = max(0.0001, time.perf_counter() - t0)
            except Exception:
                gen_time_sec = max(0.0001, time.perf_counter() - t0)
                LLM_REQUESTS_TOTAL.labels(
                    model=model_name, endpoint="compare", status="error"
                ).inc()
                LLM_GENERATION_DURATION_SECONDS.labels(
                    model=model_name, endpoint="compare"
                ).observe(gen_time_sec)
                raise
            finally:
                LLM_ACTIVE_REQUESTS.labels(model=model_name).dec()

            gen_latency_ms = gen_time_sec * 1000.0

            ans_text = response.content
            token_count_est, tok_per_sec = _extract_token_metrics(
                response, ans_text, gen_time_sec
            )
            token_source = _detect_token_source(response)
            LLM_REQUESTS_TOTAL.labels(
                model=model_name, endpoint="compare", status="success"
            ).inc()
            LLM_GENERATION_DURATION_SECONDS.labels(
                model=model_name, endpoint="compare"
            ).observe(gen_time_sec)
            LLM_TOKENS_GENERATED_TOTAL.labels(
                model=model_name, token_source=token_source
            ).inc(token_count_est)
            if tok_per_sec is not None and tok_per_sec > 0:
                LLM_TOKENS_PER_SECOND.labels(model=model_name).observe(tok_per_sec)
                LLM_LAST_TOKENS_PER_SECOND.labels(model=model_name).set(tok_per_sec)

            model_results.append(
                ModelComparisonItem(
                    model=model_name,
                    answer=ans_text,
                    generation_latency_ms=round(gen_latency_ms, 2),
                    token_count_estimate=token_count_est,
                    tokens_per_second=tok_per_sec,
                )
            )

        base_settings = get_generation_settings(COMPARE_MODELS[0])
        return CompareResponse(
            question=request.question,
            sources=sources,
            source_chunks=source_chunks,
            retrieval_diagnostics=diagnostics,
            document_scope=scope_label,
            prompt_id=strategy.id,
            prompt_version=strategy.version,
            retrieval_latency_ms=diagnostics.retrieval_latency_ms,
            context_char_count=len(context),
            generation_settings=base_settings,
            models=model_results,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Model comparison pipeline error: {exc}")
        raise HTTPException(
            status_code=500, detail="Internal model comparison pipeline error."
        )
