from pathlib import Path

from fastapi import HTTPException

from src.models import (
    ChunkDetail,
    ChunkPreviewRequest,
    ChunkPreviewResponse,
    DocumentChunksResponse,
    RetrievalInspectRequest,
    RetrievalInspectResponse,
)
from src.services.ingest import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    SPLITTER_NAME,
    _extract_doc_name,
    _get_ingest_vectorstore,
    build_text_splitter,
)
from src.services.query import _get_vectorstore, retrieve_context_chunks
from src.utils.context import format_context_for_llm
from src.utils.env import MAX_CONTEXT_CHARS, OLLAMA_EMBED_MODEL

COSINE_DISTANCE_EXPLANATION = (
    "ChromaDB computes cosine distance d = 1 - cos(theta), NOT cosine similarity. "
    "Lower distance indicates closer semantic relevance (0.0 = identical; d < 0.40 = High Match; "
    "0.40-0.75 = Moderate Match; d > 0.75 = Low Match)."
)


def _find_overlap_len(prev_text: str, next_text: str, max_overlap: int = 600) -> int:
    """Return longest matching suffix(prev_text) == prefix(next_text) length >= 4 chars."""
    limit = min(len(prev_text), len(next_text), max_overlap)
    for overlap_len in range(limit, 3, -1):
        if prev_text.endswith(next_text[:overlap_len]):
            return overlap_len
    return 0


def _build_chunk_timeline(
    raw_items: list[dict],
    document_name: str,
    max_overlap: int = 600,
) -> tuple[str, list[ChunkDetail]]:
    """Build ordered ChunkDetail items and stitched original document text with accurate offsets."""
    reconstructed_parts: list[str] = []
    current_len = 0
    details: list[ChunkDetail] = []

    for idx, item in enumerate(raw_items):
        text = str(item["text"])
        meta = item.get("meta") or {}

        if idx == 0:
            start_char = 0
            reconstructed_parts.append(text)
            current_len = len(text)
            overlap_prev_len = 0
            overlap_prev_text = ""
        else:
            prev_text = str(raw_items[idx - 1]["text"])
            overlap_prev_len = _find_overlap_len(prev_text, text, max_overlap=max_overlap)
            if overlap_prev_len > 0:
                start_char = max(0, current_len - overlap_prev_len)
                tail = text[overlap_prev_len:]
                reconstructed_parts.append(tail)
                current_len += len(tail)
                overlap_prev_text = text[:overlap_prev_len]
                # Update previous chunk's forward overlap info cleanly
                details[idx - 1].overlap_next_chars = overlap_prev_len
                details[idx - 1].overlap_next_text = overlap_prev_text
            else:
                reconstructed_parts.append("\n\n")
                current_len += 2
                start_char = current_len
                reconstructed_parts.append(text)
                current_len += len(text)
                overlap_prev_len = 0
                overlap_prev_text = ""

        end_char = start_char + len(text)
        page_num: int | None = None
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

        chunk_id = str(meta.get("chunk_id") or item.get("id") or f"chunk-{idx + 1}")
        chunk_idx = idx
        if meta.get("chunk_index") is not None:
            try:
                chunk_idx = int(meta["chunk_index"])
            except (ValueError, TypeError):
                chunk_idx = idx

        details.append(
            ChunkDetail(
                chunk_id=chunk_id,
                chunk_index=chunk_idx,
                text=text,
                document_id=str(meta["document_id"]) if meta.get("document_id") else None,
                document_name=document_name,
                page_number=page_num,
                source=str(meta.get("source") or document_name),
                char_count=len(text),
                token_count_estimate=max(1, len(text) // 4),
                start_char=start_char,
                end_char=end_char,
                overlap_prev_chars=overlap_prev_len,
                overlap_next_chars=0,
                overlap_prev_text=overlap_prev_text,
                overlap_next_text="",
                ingestion_timestamp=(
                    str(meta["ingestion_timestamp"])
                    if meta.get("ingestion_timestamp")
                    else None
                ),
            )
        )

    return "".join(reconstructed_parts), details


def get_document_chunks(filename: str) -> DocumentChunksResponse:
    """Return ordered persisted chunks and document metadata for a specific indexed file."""
    vector_store = _get_ingest_vectorstore()
    try:
        res = vector_store._collection.get(include=["documents", "metadatas"])
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed reading chunks from ChromaDB: {exc}"
        )

    if not isinstance(res, dict):
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")

    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []

    matched: list[dict] = []
    for order_idx, (cid, text, meta) in enumerate(zip(ids, docs, metas)):
        if not isinstance(meta, dict):
            continue
        clean_name = _extract_doc_name(meta)
        doc_id = meta.get("document_id")
        if filename in {clean_name, doc_id, meta.get("filename"), meta.get("document_name")}:
            matched.append(
                {
                    "id": cid,
                    "text": text or "",
                    "meta": meta,
                    "order": order_idx,
                }
            )

    if not matched:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' was not found in ChromaDB.",
        )

    def _sort_key(item: dict) -> tuple[int, int, int, int]:
        m = item["meta"]
        has_chunk_idx = 0 if m.get("chunk_index") is not None else 1
        chunk_idx = int(m["chunk_index"]) if m.get("chunk_index") is not None else 0
        page_val = int(m.get("page_number") or (int(m["page"]) + 1 if m.get("page") is not None else 0))
        start_idx = int(m.get("start_index") or 0)
        return (has_chunk_idx, chunk_idx, page_val, start_idx)

    matched.sort(key=_sort_key)

    first_meta = matched[0]["meta"]
    resolved_name = _extract_doc_name(first_meta)
    resolved_doc_id = str(first_meta["document_id"]) if first_meta.get("document_id") else None
    ingested_at = (
        str(first_meta["ingestion_timestamp"])
        if first_meta.get("ingestion_timestamp")
        else None
    )

    suffix = Path(resolved_name).suffix.lstrip(".").lower()
    file_type = str(first_meta.get("file_type") or suffix or "txt")

    page_numbers = []
    for item in matched:
        m = item["meta"]
        if m.get("total_pages") is not None:
            try:
                page_numbers.append(int(m["total_pages"]))
            except (ValueError, TypeError):
                pass
        if m.get("page_number") is not None:
            try:
                page_numbers.append(int(m["page_number"]))
            except (ValueError, TypeError):
                pass
        elif m.get("page") is not None:
            try:
                page_numbers.append(int(m["page"]) + 1)
            except (ValueError, TypeError):
                pass

    total_pages = max(page_numbers) if page_numbers else (1 if file_type == "txt" else None)

    reconstructed_text, chunk_details = _build_chunk_timeline(
        matched,
        document_name=resolved_name,
        max_overlap=max(DEFAULT_CHUNK_OVERLAP * 3, 400),
    )

    return DocumentChunksResponse(
        document_name=resolved_name,
        document_id=resolved_doc_id,
        file_type=file_type,
        total_pages=total_pages,
        total_chunks=len(chunk_details),
        total_extracted_chars=len(reconstructed_text),
        total_estimated_tokens=max(1, len(reconstructed_text) // 4),
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        splitter_name=SPLITTER_NAME,
        embedding_model=OLLAMA_EMBED_MODEL,
        ingestion_timestamp=ingested_at,
        reconstructed_text=reconstructed_text,
        chunks=chunk_details,
    )


def preview_chunking(request: ChunkPreviewRequest) -> ChunkPreviewResponse:
    """Compute a non-persistent chunking preview in memory without touching ChromaDB."""
    if request.chunk_overlap >= request.chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be strictly smaller than chunk_size.",
        )

    source_text = (request.text or "").strip()
    doc_name = request.document_name

    if not source_text:
        if not doc_name:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'document_name' or non-empty 'text' to preview chunking.",
            )
        doc_data = get_document_chunks(doc_name)
        source_text = doc_data.reconstructed_text
        doc_name = doc_data.document_name

    if not source_text:
        raise HTTPException(status_code=400, detail="Source document has no extractable text.")

    splitter = build_text_splitter(
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    preview_docs = splitter.create_documents([source_text])

    raw_items = [
        {
            "id": f"preview-chunk-{idx + 1}",
            "text": doc.page_content,
            "meta": {
                "chunk_id": f"preview-chunk-{idx + 1:04d}",
                "chunk_index": idx,
                "source": doc_name or "preview_sandbox",
                "start_index": doc.metadata.get("start_index"),
            },
        }
        for idx, doc in enumerate(preview_docs)
    ]

    _, preview_chunks = _build_chunk_timeline(
        raw_items,
        document_name=doc_name or "preview_sandbox",
        max_overlap=max(request.chunk_overlap * 2, 400),
    )

    total_chars = len(source_text)
    total_chunks = len(preview_chunks)
    avg_chars = round(sum(c.char_count for c in preview_chunks) / max(1, total_chunks), 1)

    return ChunkPreviewResponse(
        is_preview=True,
        persisted=False,
        document_name=doc_name,
        configured_chunk_size=DEFAULT_CHUNK_SIZE,
        configured_chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        preview_chunk_size=request.chunk_size,
        preview_chunk_overlap=request.chunk_overlap,
        total_chars=total_chars,
        total_chunks=total_chunks,
        avg_chunk_chars=avg_chars,
        total_estimated_tokens=max(1, total_chars // 4),
        source_text=source_text,
        chunks=preview_chunks,
    )


def inspect_retrieval(request: RetrievalInspectRequest) -> RetrievalInspectResponse:
    """Inspect deduplicated Top-K / MMR retrieval without running LLM generation or judge eval."""
    vectorstore = _get_vectorstore()
    sources, source_chunks, diagnostics, scope_label = retrieve_context_chunks(
        vectorstore=vectorstore,
        question=request.question,
        top_k=request.top_k,
        document_id=request.document_id,
        document_name=request.document_name,
        use_mmr=request.use_mmr,
        mmr_lambda=request.mmr_lambda,
    )

    max_safe_context = min(MAX_CONTEXT_CHARS, 3500)
    source_labels = [
        f"{sc.document_name} p.{sc.page_number}" if sc.page_number else (sc.document_name or "")
        for sc in source_chunks
    ]
    context_preview = format_context_for_llm(
        sources,
        max_chars=max_safe_context,
        source_labels=source_labels if source_chunks else None,
    )

    insufficient_evidence = len(source_chunks) == 0 or diagnostics.low_confidence
    empty_reason: str | None = None
    if len(source_chunks) == 0:
        empty_reason = (
            f"No chunks matched within document scope '{scope_label}'. "
            "Upload relevant content or broaden scope."
        )
    elif diagnostics.low_confidence:
        empty_reason = (
            f"Weak semantic alignment: top cosine distance is {diagnostics.top_score} (> 0.75). "
            "Retrieved context may not support a grounded answer."
        )

    return RetrievalInspectResponse(
        question=request.question,
        document_scope=scope_label,
        embedding_model=OLLAMA_EMBED_MODEL,
        score_metric=diagnostics.score_metric,
        score_explanation=COSINE_DISTANCE_EXPLANATION,
        retrieval_diagnostics=diagnostics,
        retrieved_chunks=source_chunks,
        context_preview=context_preview,
        context_char_count=len(context_preview),
        insufficient_evidence=insufficient_evidence,
        empty_reason=empty_reason,
    )
