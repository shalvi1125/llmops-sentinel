import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import difflib
import math
import os
import threading
import time
from fastapi import BackgroundTasks, HTTPException
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.metrics import (
    CHROMA_SEARCH_DURATION_SECONDS,
    CONTEXT_CHAR_COUNT,
    EVAL_PREEMPTIONS_TOTAL,
    LLM_ACTIVE_REQUESTS,
    LLM_GENERATION_DURATION_SECONDS,
    LLM_LAST_TOKENS_PER_SECOND,
    LLM_REQUESTS_TOTAL,
    LLM_TOKENS_GENERATED_TOTAL,
    LLM_TOKENS_PER_SECOND,
    MMR_RERANK_DURATION_SECONDS,
    QUERY_EMBEDDING_DURATION_SECONDS,
    RETRIEVAL_CANDIDATES_FETCHED,
    RETRIEVAL_DUPLICATES_REMOVED_TOTAL,
    RETRIEVAL_DURATION_SECONDS,
    RETRIEVAL_UNIQUE_CHUNKS,
)
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
    record_query_trace,
    run_judge_evaluations,
)
from src.utils.context import format_context_for_llm
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

# Hardware-tuned CPU inference parameters (13th Gen Intel i5-1340P, 12c/16t, 15.65 GB RAM, no VRAM)
_DEFAULT_NUM_CTX = 2048
_DEFAULT_NUM_PREDICT = 220
_DEFAULT_NUM_THREAD = 8

# Bounded, thread-safe TTL cache for query embeddings keyed by (embed_model, normalized_question)
_EMBED_CACHE_MAX_SIZE = 256
_EMBED_CACHE_TTL_SECONDS = 900.0
_embed_cache: OrderedDict[tuple[str, str], tuple[float, list[float]]] = OrderedDict()
_embed_cache_lock = threading.Lock()


@dataclass
class _RetrievedDoc:
    page_content: str
    metadata: dict


def clear_vectorstore_cache() -> None:
    """Clear the cached vectorstore instance so fresh data is loaded."""
    global _vectorstore
    _vectorstore = None


def clear_embedding_cache() -> None:
    """Clear the bounded TTL query-embedding cache."""
    with _embed_cache_lock:
        _embed_cache.clear()


def get_embedding_cache_stats() -> dict[str, int | float]:
    """Return current query-embedding cache size and configuration."""
    with _embed_cache_lock:
        return {
            "entries": len(_embed_cache),
            "max_size": _EMBED_CACHE_MAX_SIZE,
            "ttl_seconds": _EMBED_CACHE_TTL_SECONDS,
        }


def get_generation_settings(model: str = OLLAMA_MODEL) -> dict[str, int | float | str]:
    """Return effective Ollama generation parameters tuned to measured CPU & RAM constraints."""
    keep_alive = "2m" if "mistral" in model.lower() else "15m"
    return {
        "model": model,
        "temperature": 0,
        "num_ctx": _DEFAULT_NUM_CTX,
        "num_predict": _DEFAULT_NUM_PREDICT,
        "num_thread": _DEFAULT_NUM_THREAD,
        "keep_alive": keep_alive,
        "max_context_chars": min(MAX_CONTEXT_CHARS, 3500),
    }


def get_llm(model: str = OLLAMA_MODEL) -> ChatOllama:
    """Return a cached ChatOllama instance tuned for CPU memory/latency efficiency."""
    global _llm_cache
    if model not in _llm_cache:
        cfg = get_generation_settings(model)
        _llm_cache[model] = ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=cfg["temperature"],
            num_ctx=cfg["num_ctx"],
            num_predict=cfg["num_predict"],
            num_thread=cfg["num_thread"],
            keep_alive=cfg["keep_alive"],
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
                keep_alive=900,
            )
        _vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=_embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
    return _vectorstore


def _embed_query_cached(
    embed_fn,
    question: str,
    embed_model: str = OLLAMA_EMBED_MODEL,
) -> tuple[list[float] | None, float, bool]:
    """Compute or retrieve a query embedding from bounded TTL cache.

    Returns (vector_or_none, embedding_duration_ms, cache_hit).
    """
    norm_q = " ".join(question.strip().split())
    cache_key = (embed_model, norm_q)
    now = time.monotonic()

    t0 = time.perf_counter()
    with _embed_cache_lock:
        entry = _embed_cache.get(cache_key)
        if entry is not None:
            ts, vec = entry
            if (now - ts) <= _EMBED_CACHE_TTL_SECONDS:
                _embed_cache.move_to_end(cache_key)
                return list(vec), (time.perf_counter() - t0) * 1000.0, True
            _embed_cache.pop(cache_key, None)

    raw_vec = embed_fn.embed_query(question)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if isinstance(raw_vec, list) and raw_vec and isinstance(raw_vec[0], (int, float)):
        clean_vec = [float(x) for x in raw_vec]
        with _embed_cache_lock:
            _embed_cache[cache_key] = (time.monotonic(), clean_vec)
            _embed_cache.move_to_end(cache_key)
            while len(_embed_cache) > _EMBED_CACHE_MAX_SIZE:
                _embed_cache.popitem(last=False)
        return clean_vec, elapsed_ms, False

    return None, elapsed_ms, False


def _build_chroma_filter(
    document_id: str | None = None,
    document_name: str | None = None,
) -> tuple[dict | None, str]:
    """Build explicit Chroma metadata filter and human-readable document scope label."""
    if document_id:
        return {"document_id": document_id}, (document_name or document_id)
    if document_name:
        clean_name = (
            os.path.basename(document_name)
            if ("/" in document_name or "\\" in document_name)
            else document_name
        )
        return (
            {"$or": [{"document_name": clean_name}, {"filename": clean_name}]},
            clean_name,
        )
    return None, "all"


def _normalize_for_dedup(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _is_near_duplicate(candidate_norm: str, existing_norms: list[str]) -> bool:
    """Detect exact or near-identical text chunks (>= 90% similarity) without dropping overlapping neighbors."""
    if not candidate_norm:
        return True
    c_len = len(candidate_norm)
    for prev_norm in existing_norms:
        if candidate_norm == prev_norm:
            return True
        p_len = len(prev_norm)
        min_len, max_len = min(c_len, p_len), max(c_len, p_len)
        if max_len == 0:
            continue
        if min_len >= 20 and (candidate_norm in prev_norm or prev_norm in candidate_norm):
            if (min_len / max_len) >= 0.85:
                return True
        if (min_len / max_len) >= 0.60:
            ratio = difflib.SequenceMatcher(None, candidate_norm, prev_norm).ratio()
            if ratio >= 0.90:
                return True
    return False


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def _jaccard_similarity(norm_a: str, norm_b: str) -> float:
    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _fetch_candidate_embeddings(
    vectorstore: Chroma, chunk_ids: list[str]
) -> dict[str, list[float]]:
    """Fallback fetch of persisted chunk embeddings from Chroma when not included in initial search."""
    valid_ids = [cid for cid in chunk_ids if cid]
    if not valid_ids or not hasattr(vectorstore, "_collection"):
        return {}
    try:
        res = vectorstore._collection.get(ids=valid_ids, include=["embeddings"])
        if not isinstance(res, dict):
            return {}
        res_ids = res.get("ids") or []
        res_embs = res.get("embeddings")
        if res_embs is None or not isinstance(res_ids, list):
            return {}
        emb_map: dict[str, list[float]] = {}
        for cid, vec in zip(res_ids, res_embs):
            if vec is not None:
                emb_list = [float(x) for x in vec]
                if emb_list:
                    emb_map[str(cid)] = emb_list
        return emb_map
    except Exception:
        return {}


def _apply_mmr_rerank(
    candidates: list[tuple[object, float | None, str, str]],
    top_k: int,
    mmr_lambda: float,
    emb_map: dict[str, list[float]],
) -> list[tuple[object, float | None, str, str]]:
    """Greedy Maximal Marginal Relevance reranking balancing relevance and diversity."""
    if len(candidates) <= 1:
        return candidates[:top_k]

    scored_indices = list(range(len(candidates)))
    selected_indices: list[int] = [scored_indices[0]]
    remaining_indices = scored_indices[1:]

    while remaining_indices and len(selected_indices) < top_k:
        best_idx = remaining_indices[0]
        best_mmr = -1e9

        for idx in remaining_indices:
            _, dist_score, cid, norm_text = candidates[idx]
            relevance = max(0.0, 1.0 - float(dist_score)) if dist_score is not None else 0.5

            max_redundancy = 0.0
            cand_vec = emb_map.get(cid)
            for sel_idx in selected_indices:
                _, _, sel_cid, sel_norm = candidates[sel_idx]
                sel_vec = emb_map.get(sel_cid)
                if cand_vec is not None and sel_vec is not None:
                    sim = _cosine_similarity(cand_vec, sel_vec)
                else:
                    sim = _jaccard_similarity(norm_text, sel_norm)
                if sim > max_redundancy:
                    max_redundancy = sim

            mmr_score = (mmr_lambda * relevance) - ((1.0 - mmr_lambda) * max_redundancy)
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [candidates[i] for i in selected_indices]


def _extract_token_metrics(response: object, answer: str, gen_time_sec: float) -> tuple[int, float | None]:
    """Extract real token count from Ollama metadata when available, with safe char fallback."""
    token_count: int | None = None
    tok_per_sec: float | None = None

    resp_meta = getattr(response, "response_metadata", None)
    if isinstance(resp_meta, dict):
        raw_eval_count = resp_meta.get("eval_count")
        if isinstance(raw_eval_count, (int, float)) and raw_eval_count > 0:
            token_count = int(raw_eval_count)
        raw_eval_dur_ns = resp_meta.get("eval_duration")
        if token_count and isinstance(raw_eval_dur_ns, (int, float)) and raw_eval_dur_ns > 0:
            tok_per_sec = round(token_count / (float(raw_eval_dur_ns) / 1e9), 2)

    if token_count is None:
        usage_meta = getattr(response, "usage_metadata", None)
        if isinstance(usage_meta, dict):
            out_tokens = usage_meta.get("output_tokens")
            if isinstance(out_tokens, (int, float)) and out_tokens > 0:
                token_count = int(out_tokens)

    if token_count is None:
        token_count = max(1, len(answer) // 4)

    if tok_per_sec is None and gen_time_sec > 0:
        tok_per_sec = round(token_count / gen_time_sec, 2)

    return token_count, tok_per_sec


def _detect_token_source(response: object) -> str:
    """Return 'actual' if completion token count came from Ollama/LangChain metadata, else 'estimated'."""
    resp_meta = getattr(response, "response_metadata", None)
    if isinstance(resp_meta, dict):
        raw_eval_count = resp_meta.get("eval_count")
        if isinstance(raw_eval_count, (int, float)) and raw_eval_count > 0:
            return "actual"
    usage_meta = getattr(response, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        out_tokens = usage_meta.get("output_tokens")
        if isinstance(out_tokens, (int, float)) and out_tokens > 0:
            return "actual"
    return "estimated"


def _retrieve_with_timings(
    vectorstore: Chroma,
    question: str,
    top_k: int,
    document_id: str | None = None,
    document_name: str | None = None,
    use_mmr: bool = True,
    mmr_lambda: float = 0.72,
) -> tuple[list[str], list[SourceChunk], RetrievalDiagnostics, str, dict[str, float]]:
    """Retrieve deduplicated chunks with strict document scoping and stage-by-stage latency measurements."""
    try:
        collection_count = vectorstore._collection.count()
    except Exception:
        collection_count = 0

    if collection_count == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents found. Please ingest documents first.",
        )

    chroma_filter, scope_label = _build_chroma_filter(
        document_id=document_id, document_name=document_name
    )

    safe_top_k = min(top_k, MAX_TOP_K)
    candidate_k = min(max(safe_top_k * 3, safe_top_k + 4), MAX_TOP_K * 3)
    if isinstance(collection_count, int) and collection_count > 0:
        candidate_k = max(1, min(candidate_k, collection_count))

    t_retrieval_start = time.perf_counter()
    raw_docs: list[object] = []
    raw_scores: list[float | None] = []
    preloaded_emb_map: dict[str, list[float]] = {}
    used_native_query = False
    used_scored_search = False

    query_embedding_ms = 0.0
    chroma_search_ms = 0.0
    embed_cache_hit = False

    # Fast single-roundtrip path: cached/single-pass query embedding + native Chroma query with embeddings
    emb_fn = getattr(vectorstore, "_embedding_function", None)
    if (
        emb_fn is not None
        and callable(getattr(emb_fn, "embed_query", None))
        and hasattr(vectorstore, "_collection")
        and callable(getattr(vectorstore._collection, "query", None))
    ):
        try:
            q_vec, query_embedding_ms, embed_cache_hit = _embed_query_cached(
                emb_fn, question, embed_model=OLLAMA_EMBED_MODEL
            )
            if q_vec is not None:
                include_fields = (
                    ["documents", "metadatas", "distances", "embeddings"]
                    if use_mmr
                    else ["documents", "metadatas", "distances"]
                )
                query_kwargs: dict = {
                    "query_embeddings": [q_vec],
                    "n_results": candidate_k,
                    "include": include_fields,
                }
                if chroma_filter is not None:
                    query_kwargs["where"] = chroma_filter

                t_chroma0 = time.perf_counter()
                res = vectorstore._collection.query(**query_kwargs)
                chroma_search_ms = (time.perf_counter() - t_chroma0) * 1000.0

                if isinstance(res, dict) and isinstance(res.get("documents"), list):
                    docs_batch = res["documents"][0] if res["documents"] else []
                    metas_batch = (
                        res["metadatas"][0]
                        if isinstance(res.get("metadatas"), list) and res["metadatas"]
                        else [{} for _ in docs_batch]
                    )
                    dists_batch = (
                        res["distances"][0]
                        if isinstance(res.get("distances"), list) and res["distances"]
                        else [None for _ in docs_batch]
                    )
                    embs_batch = (
                        res["embeddings"][0]
                        if use_mmr and isinstance(res.get("embeddings"), list) and res["embeddings"]
                        else None
                    )
                    ids_batch = (
                        res["ids"][0]
                        if isinstance(res.get("ids"), list) and res["ids"]
                        else [None for _ in docs_batch]
                    )

                    # Validate empty filtered result only on rare 0-hit case (avoids extra pre-check roundtrip on hits)
                    if chroma_filter is not None and len(docs_batch) == 0:
                        scope_check = vectorstore._collection.get(where=chroma_filter, limit=1)
                        if (
                            isinstance(scope_check, dict)
                            and isinstance(scope_check.get("ids"), list)
                            and len(scope_check["ids"]) == 0
                        ):
                            raise HTTPException(
                                status_code=404,
                                detail=f"Requested document '{scope_label}' was not found in the knowledge base.",
                            )

                    used_native_query = True
                    used_scored_search = True
                    for idx_row, doc_text in enumerate(docs_batch):
                        meta_row = metas_batch[idx_row] if idx_row < len(metas_batch) and isinstance(metas_batch[idx_row], dict) else {}
                        row_id = ids_batch[idx_row] if idx_row < len(ids_batch) else None
                        if "chunk_id" not in meta_row and row_id:
                            meta_row = {**meta_row, "chunk_id": str(row_id)}
                        raw_docs.append(_RetrievedDoc(page_content=str(doc_text or ""), metadata=meta_row))

                        dist_val = dists_batch[idx_row] if idx_row < len(dists_batch) else None
                        try:
                            raw_scores.append(float(dist_val) if dist_val is not None else None)
                        except (ValueError, TypeError):
                            raw_scores.append(None)

                        if embs_batch is not None and idx_row < len(embs_batch) and embs_batch[idx_row] is not None:
                            cid_key = str(meta_row.get("chunk_id") or row_id or "")
                            if cid_key:
                                preloaded_emb_map[cid_key] = [float(x) for x in embs_batch[idx_row]]
        except HTTPException:
            raise
        except Exception as native_exc:
            logger.debug(f"Native Chroma query fallback triggered: {native_exc}")
            raw_docs = []
            raw_scores = []
            preloaded_emb_map = {}
            used_native_query = False

    # Fallback path (e.g. unit test mocks or stores without native collection query)
    if not used_native_query:
        if chroma_filter is not None:
            try:
                scope_check = vectorstore._collection.get(where=chroma_filter, limit=1)
                if (
                    isinstance(scope_check, dict)
                    and isinstance(scope_check.get("ids"), list)
                    and len(scope_check["ids"]) == 0
                ):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Requested document '{scope_label}' was not found in the knowledge base.",
                    )
            except HTTPException:
                raise
            except Exception as scope_exc:
                logger.debug(f"Scope pre-check skipped for mock/non-standard store: {scope_exc}")

        t_chroma0 = time.perf_counter()
        if hasattr(vectorstore, "similarity_search_with_score"):
            for try_k in (candidate_k, safe_top_k):
                try:
                    if chroma_filter is not None:
                        results = vectorstore.similarity_search_with_score(
                            question, k=try_k, filter=chroma_filter
                        )
                    else:
                        results = vectorstore.similarity_search_with_score(
                            question, k=try_k
                        )

                    if isinstance(results, list) and (
                        len(results) == 0 or isinstance(results[0], tuple)
                    ):
                        used_scored_search = True
                        for doc, score in results:
                            raw_docs.append(doc)
                            try:
                                raw_scores.append(float(score))
                            except (ValueError, TypeError):
                                raw_scores.append(None)
                        break
                except Exception as exc:
                    logger.debug(f"Scored search (k={try_k}) unavailable or failed: {exc}")
                    raw_docs = []
                    raw_scores = []

        if not used_scored_search:
            search_kwargs: dict = {"k": candidate_k}
            if chroma_filter is not None:
                search_kwargs["filter"] = chroma_filter
            retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
            raw_docs = retriever.invoke(question)
            raw_scores = [None] * len(raw_docs)
        chroma_search_ms = (time.perf_counter() - t_chroma0) * 1000.0

    # Stage 1: Exact chunk_id, exact normalized text, and near-duplicate removal
    t_dedup0 = time.perf_counter()
    deduped_candidates: list[tuple[object, float | None, str, str]] = []
    seen_chunk_ids: set[str] = set()
    kept_norms: list[str] = []
    duplicates_removed = 0

    for doc, score in zip(raw_docs, raw_scores):
        text_content = str(getattr(doc, "page_content", ""))
        norm_text = _normalize_for_dedup(text_content)
        meta = getattr(doc, "metadata", {})
        meta_cid = (
            str(meta.get("chunk_id"))
            if isinstance(meta, dict) and meta.get("chunk_id")
            else ""
        )

        if meta_cid and meta_cid in seen_chunk_ids:
            duplicates_removed += 1
            continue

        if _is_near_duplicate(norm_text, kept_norms):
            duplicates_removed += 1
            continue

        if meta_cid:
            seen_chunk_ids.add(meta_cid)
        kept_norms.append(norm_text)
        deduped_candidates.append((doc, score, meta_cid, norm_text))

    # Stage 2: Diversity-aware MMR reranking reusing pre-fetched candidate embeddings (zero extra DB call)
    strategy_used = "similarity_deduplicated"
    if use_mmr and len(deduped_candidates) > 1:
        emb_map = preloaded_emb_map
        if not emb_map:
            emb_map = _fetch_candidate_embeddings(
                vectorstore, [c[2] for c in deduped_candidates if c[2]]
            )
        final_candidates = _apply_mmr_rerank(
            deduped_candidates, top_k=safe_top_k, mmr_lambda=mmr_lambda, emb_map=emb_map
        )
        strategy_used = "mmr_deduplicated"
    else:
        final_candidates = deduped_candidates[:safe_top_k]

    dedup_mmr_ms = (time.perf_counter() - t_dedup0) * 1000.0
    retrieval_latency_ms = (time.perf_counter() - t_retrieval_start) * 1000.0

    sources = [str(doc.page_content) for doc, _, _, _ in final_candidates]
    source_chunks: list[SourceChunk] = []

    for idx, (doc, score, _, _) in enumerate(final_candidates):
        meta = getattr(doc, "metadata", {})
        clean_doc_name = "Document"
        doc_id_val: str | None = None
        chunk_id_val = f"chunk-{idx + 1}"
        page_num: int | None = None
        source_val: str | None = None
        ingested_at_val: str | None = None

        if isinstance(meta, dict):
            raw_name = (
                meta.get("document_name")
                or meta.get("filename")
                or meta.get("source")
            )
            if raw_name and isinstance(raw_name, str):
                clean_doc_name = (
                    os.path.basename(raw_name)
                    if ("/" in raw_name or "\\" in raw_name)
                    else raw_name
                )
                source_val = clean_doc_name

            if meta.get("document_id"):
                doc_id_val = str(meta["document_id"])
            if meta.get("chunk_id"):
                chunk_id_val = str(meta["chunk_id"])
            if meta.get("ingestion_timestamp"):
                ingested_at_val = str(meta["ingestion_timestamp"])

            if meta.get("page_number") is not None:
                try:
                    page_num = int(meta["page_number"])
                except (ValueError, TypeError):
                    page_num = None
            elif meta.get("page") is not None:
                try:
                    page_num = int(meta["page"]) + 1
                except (ValueError, TypeError):
                    page_num = None

        text_content = str(doc.page_content)
        token_est = max(1, len(text_content) // 4)

        source_chunks.append(
            SourceChunk(
                chunk_id=chunk_id_val,
                text=text_content,
                score=round(score, 4) if score is not None else None,
                document_id=doc_id_val,
                document_name=clean_doc_name,
                page_number=page_num,
                source=source_val,
                ingestion_timestamp=ingested_at_val,
                token_count_estimate=token_est,
            )
        )

    final_scores = [sc.score for sc in source_chunks if sc.score is not None]
    top_score = final_scores[0] if final_scores else None
    low_confidence = top_score is not None and top_score > 0.75

    insufficient_reason: str | None = None
    if len(source_chunks) < safe_top_k:
        if duplicates_removed > 0:
            insufficient_reason = (
                f"Requested K={safe_top_k}; returned {len(source_chunks)} unique relevant "
                f"chunk(s) after removing {duplicates_removed} duplicate/near-duplicate match(es)."
            )
        else:
            insufficient_reason = (
                f"Requested K={safe_top_k}; only {len(source_chunks)} unique relevant "
                f"chunk(s) exist in the selected document scope."
            )

    diagnostics = RetrievalDiagnostics(
        retrieval_latency_ms=round(retrieval_latency_ms, 2),
        top_score=round(top_score, 4) if top_score is not None else None,
        low_confidence=low_confidence,
        score_metric="cosine_distance",
        requested_k=safe_top_k,
        candidates_fetched=len(raw_docs),
        unique_chunks_returned=len(source_chunks),
        duplicates_removed=duplicates_removed,
        retrieval_strategy=strategy_used,
        insufficient_unique_reason=insufficient_reason,
    )

    retrieval_timings = {
        "query_embedding_ms": round(query_embedding_ms, 2),
        "embedding_cache_hit": 1.0 if embed_cache_hit else 0.0,
        "chroma_search_ms": round(chroma_search_ms, 2),
        "dedup_mmr_ms": round(dedup_mmr_ms, 2),
        "retrieval_total_ms": round(retrieval_latency_ms, 2),
    }

    RETRIEVAL_DURATION_SECONDS.labels(strategy=strategy_used).observe(
        max(0.0, retrieval_latency_ms / 1000.0)
    )
    QUERY_EMBEDDING_DURATION_SECONDS.labels(
        cache_hit="true" if embed_cache_hit else "false"
    ).observe(max(0.0, query_embedding_ms / 1000.0))
    CHROMA_SEARCH_DURATION_SECONDS.observe(max(0.0, chroma_search_ms / 1000.0))
    MMR_RERANK_DURATION_SECONDS.observe(max(0.0, dedup_mmr_ms / 1000.0))
    RETRIEVAL_CANDIDATES_FETCHED.observe(float(len(raw_docs)))
    RETRIEVAL_UNIQUE_CHUNKS.observe(float(len(source_chunks)))
    if duplicates_removed > 0:
        RETRIEVAL_DUPLICATES_REMOVED_TOTAL.inc(duplicates_removed)

    return sources, source_chunks, diagnostics, scope_label, retrieval_timings


def retrieve_context_chunks(
    vectorstore: Chroma,
    question: str,
    top_k: int,
    document_id: str | None = None,
    document_name: str | None = None,
    use_mmr: bool = True,
    mmr_lambda: float = 0.72,
) -> tuple[list[str], list[SourceChunk], RetrievalDiagnostics, str]:
    """Retrieve deduplicated chunks with strict document-level scoping, latency timing, and optional MMR."""
    sources, source_chunks, diagnostics, scope_label, _ = _retrieve_with_timings(
        vectorstore=vectorstore,
        question=question,
        top_k=top_k,
        document_id=document_id,
        document_name=document_name,
        use_mmr=use_mmr,
        mmr_lambda=mmr_lambda,
    )
    return sources, source_chunks, diagnostics, scope_label


async def handle_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks | None = None,
) -> QueryResponse:
    """Execute end-to-end RAG query off the async event loop with stage timings and MLflow telemetry."""
    t_req_start = time.perf_counter()
    try:
        if is_evaluation_running():
            logger.info("Preempting ongoing judge evaluation for incoming query...")
            EVAL_PREEMPTIONS_TOTAL.labels(trigger="query").inc()
            cancel_current_evaluation()
        t_val_ms = (time.perf_counter() - t_req_start) * 1000.0

        vectorstore = _get_vectorstore()
        sources, source_chunks, diagnostics, scope_label, retr_timings = await asyncio.to_thread(
            _retrieve_with_timings,
            vectorstore,
            request.question,
            request.top_k,
            request.document_id,
            request.document_name,
            request.use_mmr,
            request.mmr_lambda,
        )

        # Compact PDF extraction whitespace ONLY for LLM prompt construction (Chroma data stays untouched)
        t_ctx0 = time.perf_counter()
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
        strategy = get_prompt_strategy(request.prompt_id)
        selected_model = request.model or OLLAMA_MODEL
        llm = get_llm(model=selected_model)

        prompt = PromptTemplate(
            input_variables=["contexto", "question"], template=strategy.template
        )
        sequence = RunnableSequence(prompt | llm)
        context_build_ms = (time.perf_counter() - t_ctx0) * 1000.0
        CONTEXT_CHAR_COUNT.observe(float(len(context)))

        t_gen_start = time.perf_counter()
        LLM_ACTIVE_REQUESTS.labels(model=selected_model).inc()
        try:
            response = await asyncio.to_thread(
                sequence.invoke, {"contexto": context, "question": request.question}
            )
            gen_time_sec = max(0.0001, time.perf_counter() - t_gen_start)
        except Exception:
            gen_time_sec = max(0.0001, time.perf_counter() - t_gen_start)
            LLM_REQUESTS_TOTAL.labels(
                model=selected_model, endpoint="query", status="error"
            ).inc()
            LLM_GENERATION_DURATION_SECONDS.labels(
                model=selected_model, endpoint="query"
            ).observe(gen_time_sec)
            raise
        finally:
            LLM_ACTIVE_REQUESTS.labels(model=selected_model).dec()

        gen_latency_ms = gen_time_sec * 1000.0
        answer = response.content

        token_count_est, tok_per_sec = _extract_token_metrics(response, answer, gen_time_sec)
        token_source = _detect_token_source(response)
        LLM_REQUESTS_TOTAL.labels(
            model=selected_model, endpoint="query", status="success"
        ).inc()
        LLM_GENERATION_DURATION_SECONDS.labels(
            model=selected_model, endpoint="query"
        ).observe(gen_time_sec)
        LLM_TOKENS_GENERATED_TOTAL.labels(
            model=selected_model, token_source=token_source
        ).inc(token_count_est)
        if tok_per_sec is not None and tok_per_sec > 0:
            LLM_TOKENS_PER_SECOND.labels(model=selected_model).observe(tok_per_sec)
            LLM_LAST_TOKENS_PER_SECOND.labels(model=selected_model).set(tok_per_sec)
        t_prep0 = time.perf_counter()
        total_user_latency_ms = round((t_prep0 - t_req_start) * 1000.0, 2)

        trace_id, eval_job_id, eval_status = await asyncio.to_thread(
            record_query_trace,
            question=request.question,
            answer=answer,
            model=selected_model,
            prompt_id=strategy.id,
            prompt_version=strategy.version,
            document_scope=scope_label,
            retrieval_latency_ms=diagnostics.retrieval_latency_ms,
            generation_latency_ms=round(gen_latency_ms, 2),
            total_latency_ms=total_user_latency_ms,
            candidates_fetched=diagnostics.candidates_fetched,
            unique_chunks_returned=diagnostics.unique_chunks_returned,
            duplicates_removed=diagnostics.duplicates_removed,
            use_mmr=request.use_mmr,
            context_char_count=len(context),
            token_count_estimate=token_count_est,
            tokens_per_second=tok_per_sec,
            has_sources=bool(sources),
        )
        response_prep_ms = (time.perf_counter() - t_prep0) * 1000.0
        final_total_ms = round((time.perf_counter() - t_req_start) * 1000.0, 2)

        # Schedule background LLM-as-a-Judge evaluation (keeping 3-arg call compatibility)
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

        stage_timings = {
            "api_validation_ms": round(t_val_ms, 2),
            **retr_timings,
            "context_build_ms": round(context_build_ms, 2),
            "generation_ms": round(gen_latency_ms, 2),
            "trace_and_prep_ms": round(response_prep_ms, 2),
            "total_latency_ms": final_total_ms,
        }

        return QueryResponse(
            answer=answer,
            sources=sources,
            source_chunks=source_chunks,
            retrieval_diagnostics=diagnostics,
            document_scope=scope_label,
            prompt_id=strategy.id,
            prompt_version=strategy.version,
            model=selected_model,
            generation_latency_ms=round(gen_latency_ms, 2),
            total_latency_ms=final_total_ms,
            context_char_count=len(context),
            token_count_estimate=token_count_est,
            tokens_per_second=tok_per_sec,
            stage_timings=stage_timings,
            trace_id=trace_id,
            evaluation_job_id=eval_job_id,
            evaluation_status=eval_status,
            generation_settings=get_generation_settings(selected_model),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Query pipeline error: {exc}")
        raise HTTPException(status_code=500, detail="Internal query pipeline error.")
