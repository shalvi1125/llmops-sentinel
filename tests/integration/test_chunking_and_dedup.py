import tempfile
from unittest.mock import patch

import pytest
from chromadb.api.client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.services.ingest import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, build_text_splitter


class KeywordHashEmbeddings(Embeddings):
    """Deterministic embeddings projecting texts onto clean semantic buckets."""

    TERMS = [
        "shalvi",
        "scheduler",
        "worker",
        "mongodb",
        "heartbeat",
        "flood",
        "lstm",
        "rainfall",
        "kubernetes",
        "cache",
    ]

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        vec = [float(lower.count(term) + 0.01) for term in self.TERMS]
        # Add length signal as subtle tie-breaker dimension
        vec.append(min(1.0, len(lower) / 2000.0))
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def isolated_phase2_chroma():
    """Isolated temporary ChromaDB store for Phase 2 chunking & dedup tests."""
    SharedSystemClient.clear_system_cache()
    tmp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    emb = KeywordHashEmbeddings()

    def _clear():
        import src.services.query as q_mod

        q_mod._vectorstore = None

    with patch("src.services.ingest.CHROMA_PERSIST_DIR", tmp_dir.name), \
         patch("src.services.ingest.CHROMA_COLLECTION_NAME", "test_phase2_col"), \
         patch("src.services.ingest.OllamaEmbeddings", return_value=emb), \
         patch("src.services.query.CHROMA_PERSIST_DIR", tmp_dir.name), \
         patch("src.services.query.CHROMA_COLLECTION_NAME", "test_phase2_col"), \
         patch("src.services.query._embeddings", emb), \
         patch("src.services.query._vectorstore", None), \
         patch("src.services.query.clear_vectorstore_cache", side_effect=_clear):
        yield tmp_dir.name, emb

    SharedSystemClient.clear_system_cache()
    tmp_dir.cleanup()


@pytest.mark.unit
def test_chunk_splitting_and_overlap_configuration():
    assert DEFAULT_CHUNK_SIZE == 1200
    assert DEFAULT_CHUNK_OVERLAP == 100

    # Build a multi-paragraph document > 2600 chars so splitting creates >= 3 overlapping chunks
    para_a = "Section One explains distributed job scheduling queues and worker heartbeats. " * 18
    para_b = "Section Two explains atomic MongoDB reservation transactions and bounded retries. " * 18
    long_text = para_a.strip() + "\n\n" + para_b.strip()

    splitter = build_text_splitter(DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    docs = splitter.create_documents([long_text])

    assert len(docs) >= 3
    for d in docs:
        assert len(d.page_content) <= DEFAULT_CHUNK_SIZE
        assert "start_index" in d.metadata

    # Verify consecutive chunks overlap cleanly (~up to chunk_overlap chars)
    first_tail = docs[0].page_content[-60:]
    assert first_tail.strip()[:25] in docs[1].page_content


@pytest.mark.integration
def test_unique_chunk_ids_and_intra_document_duplicate_insertion_prevention(
    client, auth_headers, isolated_phase2_chroma
):
    tmp_dir, emb = isolated_phase2_chroma

    # Simulate a PDF with repeated identical header/boilerplate pages plus distinct pages
    mock_pages = [
        Document(page_content="This is Shalvi.", metadata={"source": "/tmp/fake.pdf", "page": 0}),
        Document(page_content="This is Shalvi.", metadata={"source": "/tmp/fake.pdf", "page": 1}),
        Document(page_content="Scheduler assigns workers by CPU and RAM.", metadata={"source": "/tmp/fake.pdf", "page": 2}),
        Document(page_content="Heartbeat daemon restarts failed workers.", metadata={"source": "/tmp/fake.pdf", "page": 3}),
    ]

    with patch("src.services.ingest.PyPDFLoader") as mock_pdf:
        mock_pdf.return_value.load_and_split.return_value = mock_pages
        resp = client.post(
            "/ingest",
            headers=auth_headers,
            files={"file": ("shalvi_report.pdf", b"%PDF-1.4 dummy", "application/pdf")},
        )

    assert resp.status_code == 200
    data = resp.json()
    # Exact duplicate page ("This is Shalvi.") must be inserted once, leaving 3 unique chunks
    assert data["chunks_stored"] == 3

    # Verify unique chunk IDs and metadata in ChromaDB
    vs = Chroma(
        collection_name="test_phase2_col",
        embedding_function=emb,
        persist_directory=tmp_dir,
    )
    raw = vs._collection.get(include=["documents", "metadatas"])
    assert len(raw["ids"]) == 3
    assert len(set(raw["ids"])) == 3
    for cid, meta in zip(raw["ids"], raw["metadatas"]):
        assert cid == meta["chunk_id"]
        assert meta["chunk_id"].startswith(data["document_id"])
        assert meta["document_name"] == "shalvi_report.pdf"
        assert meta["page_number"] >= 1
        assert meta["char_count"] > 0
        assert meta["token_count_estimate"] >= 1


@pytest.mark.integration
def test_exact_and_near_duplicate_removal_and_insufficient_unique_k(
    client, auth_headers, isolated_phase2_chroma
):
    tmp_dir, emb = isolated_phase2_chroma

    # Directly seed Chroma with 4 exact/near duplicates of "This is Shalvi." across legacy-style entries
    # plus 1 distinct scheduler chunk (simulating duplicate corpus across multiple legacy files)
    vs = Chroma(
        collection_name="test_phase2_col",
        embedding_function=emb,
        persist_directory=tmp_dir,
    )
    vs.add_documents(
        documents=[
            Document(page_content="This is Shalvi.", metadata={"document_name": "dup_1.txt", "chunk_id": "dup-1", "page_number": 1}),
            Document(page_content="This is Shalvi.", metadata={"document_name": "dup_2.txt", "chunk_id": "dup-2", "page_number": 1}),
            Document(page_content="This is   Shalvi.", metadata={"document_name": "dup_3.txt", "chunk_id": "dup-3", "page_number": 1}),
            Document(page_content="this is shalvi.", metadata={"document_name": "dup_4.txt", "chunk_id": "dup-4", "page_number": 1}),
            Document(page_content="Scheduler manages CPU queue priority.", metadata={"document_name": "sched.txt", "chunk_id": "sched-1", "page_number": 1}),
        ],
        ids=["dup-1", "dup-2", "dup-3", "dup-4", "sched-1"],
    )

    # Request K=4 for "Who is Shalvi?" via Retrieval Inspector API
    inspect_resp = client.post(
        "/query/inspect",
        headers=auth_headers,
        json={"question": "Shalvi", "top_k": 4},
    )
    assert inspect_resp.status_code == 200
    body = inspect_resp.json()

    # Must NEVER pad or return 4 copies of "This is Shalvi."!
    # At most 2 unique chunks exist in the whole store ("This is Shalvi." + "Scheduler...")
    returned_texts = [c["text"].lower().strip() for c in body["retrieved_chunks"]]
    assert returned_texts.count("this is shalvi.") == 1
    assert len(body["retrieved_chunks"]) == 2
    assert body["retrieval_diagnostics"]["requested_k"] == 4
    assert body["retrieval_diagnostics"]["unique_chunks_returned"] == 2
    assert body["retrieval_diagnostics"]["duplicates_removed"] == 3
    assert body["retrieval_diagnostics"]["insufficient_unique_reason"] is not None
    assert body["score_metric"] == "cosine_distance"
    assert "cosine distance" in body["score_explanation"].lower()


@pytest.mark.integration
def test_multiple_distinct_chunks_from_same_document_retrieved_and_mmr_diversity(
    client, auth_headers, isolated_phase2_chroma
):
    tmp_dir, emb = isolated_phase2_chroma

    long_scheduler_doc = (
        "Chapter 1: The Distributed Job Scheduler accepts jobs with CPU and RAM budgets.\n"
        + ("Filler detail about network sockets and serialization. " * 25)
        + "\n\nChapter 2: Worker processes pull tasks using MongoDB atomic findAndModify locks.\n"
        + ("Filler detail about logging formats and CLI flags. " * 25)
        + "\n\nChapter 3: The Heartbeat monitor detects worker crashes and triggers retries.\n"
    ).encode("utf-8")

    r_up = client.post(
        "/ingest",
        headers=auth_headers,
        files={"file": ("multi_chapter_scheduler.txt", long_scheduler_doc, "text/plain")},
    )
    assert r_up.status_code == 200
    assert r_up.json()["chunks_stored"] >= 3

    # Query with K=3 scoped to that single document -> must return 3 distinct chunks
    insp = client.post(
        "/query/inspect",
        headers=auth_headers,
        json={
            "question": "How do scheduler worker processes use MongoDB heartbeat retries?",
            "top_k": 3,
            "document_name": "multi_chapter_scheduler.txt",
            "use_mmr": True,
        },
    )
    assert insp.status_code == 200
    insp_data = insp.json()
    assert insp_data["document_scope"] == "multi_chapter_scheduler.txt"
    chunks = insp_data["retrieved_chunks"]
    assert len(chunks) == 3
    assert len({c["chunk_id"] for c in chunks}) == 3
    assert insp_data["retrieval_diagnostics"]["retrieval_strategy"] == "mmr_deduplicated"
    assert all(c["document_name"] == "multi_chapter_scheduler.txt" for c in chunks)


@pytest.mark.integration
def test_chunk_explorer_and_preview_non_persistent(
    client, auth_headers, isolated_phase2_chroma
):
    tmp_dir, emb = isolated_phase2_chroma

    content = (
        b"Paragraph One: Operating Systems Job Scheduler assigns tasks by CPU cores.\n"
        + (b"Intermediate scheduling detail for chunk splitting verification. " * 24)
        + b"\nParagraph Two: Worker processes report heartbeats every five seconds.\n"
    )
    up = client.post(
        "/ingest",
        headers=auth_headers,
        files={"file": ("explorer_test.txt", content, "text/plain")},
    )
    assert up.status_code == 200
    stored_count = up.json()["chunks_stored"]
    assert stored_count >= 2

    # 1. GET /ingest/documents/{filename}/chunks returns exact metadata & overlaps
    chunks_resp = client.get("/ingest/documents/explorer_test.txt/chunks", headers=auth_headers)
    assert chunks_resp.status_code == 200
    cd = chunks_resp.json()
    assert cd["document_name"] == "explorer_test.txt"
    assert cd["file_type"] == "txt"
    assert cd["total_chunks"] == stored_count
    assert cd["chunk_size"] == 1200
    assert cd["chunk_overlap"] == 100
    assert "Paragraph One" in cd["reconstructed_text"]
    assert "Paragraph Two" in cd["reconstructed_text"]
    assert len(cd["chunks"]) == stored_count

    # Snapshot production Chroma state before preview
    vs = Chroma(
        collection_name="test_phase2_col",
        embedding_function=emb,
        persist_directory=tmp_dir,
    )
    before_ids = set(vs._collection.get()["ids"])

    # 2. POST /ingest/chunks/preview simulates custom chunk size (300/60) WITHOUT touching ChromaDB
    prev_resp = client.post(
        "/ingest/chunks/preview",
        headers=auth_headers,
        json={
            "document_name": "explorer_test.txt",
            "chunk_size": 300,
            "chunk_overlap": 60,
        },
    )
    assert prev_resp.status_code == 200
    pv = prev_resp.json()
    assert pv["is_preview"] is True
    assert pv["persisted"] is False
    assert pv["preview_chunk_size"] == 300
    assert pv["preview_chunk_overlap"] == 60
    assert pv["configured_chunk_size"] == 1200
    assert pv["total_chunks"] > stored_count
    # Check overlap metadata exists between consecutive chunks when text is continuous
    assert any(c["overlap_prev_chars"] > 0 for c in pv["chunks"][1:])

    # Invalid overlap >= chunk_size must return 400
    bad_prev = client.post(
        "/ingest/chunks/preview",
        headers=auth_headers,
        json={"document_name": "explorer_test.txt", "chunk_size": 300, "chunk_overlap": 300},
    )
    assert bad_prev.status_code == 400

    # Assert persistent ChromaDB collection is 100% unchanged after previews
    after_ids = set(vs._collection.get()["ids"])
    assert after_ids == before_ids
    assert vs._collection.count() == stored_count


@pytest.mark.integration
def test_explorer_endpoints_require_auth(client):
    assert client.get("/ingest/documents/any.txt/chunks").status_code == 401
    assert client.post("/ingest/chunks/preview", json={"text": "hello world"}).status_code == 401
    assert client.post("/query/inspect", json={"question": "hello"}).status_code == 401
