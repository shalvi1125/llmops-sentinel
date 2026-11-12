import tempfile
from unittest.mock import MagicMock, patch

import pytest
from chromadb.api.client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document


class DomainTestEmbeddings(Embeddings):
    """Deterministic embeddings projecting OS Job Scheduling vs Flood Prediction into distinct subspaces."""

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        os_terms = ["scheduler", "job", "cpu", "ram", "worker", "mongodb", "process", "queue", "heartbeat", "priority"]
        dl_terms = ["flood", "rainfall", "lstm", "water", "river", "neural", "prediction", "hydrology", "watershed", "sensor"]
        os_score = sum(1.0 for t in os_terms if t in lower)
        dl_score = sum(1.0 for t in dl_terms if t in lower)
        base = 0.05
        return [os_score + base, dl_score + base, 0.1]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


OS_DOCUMENT_CONTENT = (
    b"Operating Systems Project Report: Resource-Aware Distributed Job Scheduling and Execution System.\n"
    b"Version 1 Alpha.\n"
    b"Architecture: Clients submit tasks with CPU, RAM, and priority requirements to a FastAPI coordinator.\n"
    b"Independent worker processes poll for eligible jobs, and MongoDB transactions atomically reserve worker resources.\n"
    b"A heartbeat monitor detects dead workers and a recovery daemon retries timed-out tasks.\n"
)

FLOOD_DOCUMENT_CONTENT = (
    b"Deep Learning Technical Report: FloodVision River Basin Flood Prediction.\n"
    b"Architecture: Spatio-temporal LSTM and CNN ensemble trained on historical rainfall, river gauge, and watershed hydrology sensor data.\n"
    b"The FloodVision model predicts downstream water level surges 24 hours in advance using precipitation forecasts.\n"
)


class ContextGroundedMockLLM:
    """Simulates a context-only LLM that reflects prompt context rules without fabricating cross-domain answers."""

    def __call__(self, inputs: dict) -> MagicMock:
        prompt_text = inputs if isinstance(inputs, str) else str(inputs)
        resp = MagicMock()
        # Extract the Context block if RunnableSequence passes dict inputs directly
        if isinstance(inputs, dict):
            ctx = inputs.get("contexto", "")
            q = inputs.get("question", "").lower()
        else:
            ctx = prompt_text
            q = prompt_text.lower()

        ctx_lower = ctx.lower()
        if ("flood" in q or "lstm" in q or "rainfall" in q) and "flood" not in ctx_lower and "lstm" not in ctx_lower:
            resp.content = "The provided document context does not contain sufficient information to answer this question."
        elif ("scheduler" in q or "worker" in q or "mongodb" in q) and "scheduler" not in ctx_lower and "worker" not in ctx_lower:
            resp.content = "The provided document context does not contain sufficient information to answer this question."
        elif "scheduler" in ctx_lower or "worker" in ctx_lower:
            resp.content = "Based on the Operating Systems report, the Distributed Job Scheduler assigns tasks by CPU, RAM, and priority using FastAPI, MongoDB transactions, and worker heartbeats."
        elif "flood" in ctx_lower or "lstm" in ctx_lower:
            resp.content = "Based on the FloodVision report, the system uses a spatio-temporal LSTM and CNN ensemble on rainfall and watershed sensor data to predict flood surges."
        else:
            resp.content = "The provided document context does not contain sufficient information to answer this question."
        return resp


@pytest.fixture
def isolated_chroma_env():
    """Provide an isolated temporary ChromaDB persistence dir so tests never touch production .chroma."""
    SharedSystemClient.clear_system_cache()
    tmp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    emb = DomainTestEmbeddings()

    def _make_vs():
        return Chroma(
            collection_name="test_isolation_col",
            embedding_function=emb,
            persist_directory=tmp_dir.name,
        )

    with patch("src.services.ingest.CHROMA_PERSIST_DIR", tmp_dir.name), \
         patch("src.services.ingest.CHROMA_COLLECTION_NAME", "test_isolation_col"), \
         patch("src.services.ingest.OllamaEmbeddings", return_value=emb), \
         patch("src.services.query.CHROMA_PERSIST_DIR", tmp_dir.name), \
         patch("src.services.query.CHROMA_COLLECTION_NAME", "test_isolation_col"), \
         patch("src.services.query._embeddings", emb), \
         patch("src.services.query._vectorstore", None), \
         patch("src.services.query.clear_vectorstore_cache") as mock_clear_cache:

        # Ensure query cache clears cleanly between ingests/deletes
        def real_clear():
            import src.services.query as q_mod
            q_mod._vectorstore = None

        mock_clear_cache.side_effect = real_clear
        yield tmp_dir.name

    SharedSystemClient.clear_system_cache()
    tmp_dir.cleanup()


@pytest.mark.integration
def test_document_isolation_job_scheduler_vs_flood_prediction(client, auth_headers, isolated_chroma_env):
    # 1. Ingest Document B (Deep Learning - Flood Prediction) first (simulating previous upload)
    resp_flood = client.post(
        "/ingest",
        headers=auth_headers,
        files={"file": ("dl_flood_prediction.txt", FLOOD_DOCUMENT_CONTENT, "text/plain")},
    )
    assert resp_flood.status_code == 200
    flood_meta = resp_flood.json()
    assert flood_meta["filename"] == "dl_flood_prediction.txt"
    assert flood_meta["chunks_stored"] >= 1
    assert flood_meta["document_id"].startswith("doc-")
    assert flood_meta["ingestion_timestamp"]

    # 2. Ingest Document A (OS - Job Scheduler) without wiping Document B
    resp_os = client.post(
        "/ingest",
        headers=auth_headers,
        files={"file": ("os_job_scheduler.txt", OS_DOCUMENT_CONTENT, "text/plain")},
    )
    assert resp_os.status_code == 200
    os_meta = resp_os.json()
    assert os_meta["filename"] == "os_job_scheduler.txt"
    assert os_meta["document_id"] != flood_meta["document_id"]

    # 3. Verify GET /ingest/documents lists both distinct documents cleanly
    docs_resp = client.get("/ingest/documents", headers=auth_headers)
    assert docs_resp.status_code == 200
    docs_list = docs_resp.json()
    assert len(docs_list) == 2
    names = {d["filename"] for d in docs_list}
    assert names == {"dl_flood_prediction.txt", "os_job_scheduler.txt"}

    llm_handler = ContextGroundedMockLLM()

    with patch("src.services.query.get_llm", return_value=MagicMock()), \
         patch("src.services.query.RunnableSequence") as mock_seq, \
         patch("src.services.query.run_judge_evaluations"), \
         patch("src.services.compare.get_llm", return_value=MagicMock()), \
         patch("src.services.compare.RunnableSequence") as mock_cmp_seq:

        mock_seq.return_value.invoke.side_effect = llm_handler
        mock_cmp_seq.return_value.invoke.side_effect = llm_handler

        # 4. Ask about Job Scheduling scoped to os_job_scheduler.txt
        q_os = client.post(
            "/query",
            headers=auth_headers,
            json={
                "question": "How does the distributed job scheduler assign tasks to workers?",
                "top_k": 3,
                "document_name": "os_job_scheduler.txt",
            },
        )
        assert q_os.status_code == 200
        os_data = q_os.json()
        assert os_data["document_scope"] == "os_job_scheduler.txt"
        assert "Distributed Job Scheduler" in os_data["answer"]
        assert "FloodVision" not in os_data["answer"]
        assert "rainfall" not in os_data["answer"].lower()
        assert len(os_data["source_chunks"]) >= 1
        for chunk in os_data["source_chunks"]:
            assert chunk["document_name"] == "os_job_scheduler.txt"
            assert chunk["document_id"] == os_meta["document_id"]
            assert chunk["chunk_id"].startswith(os_meta["document_id"])
            assert chunk["source"] == "os_job_scheduler.txt"
            assert chunk["ingestion_timestamp"] == os_meta["ingestion_timestamp"]
            assert "tmp" not in (chunk["source"] or "").lower()

        # 5. Ask about Flood Prediction scoped to dl_flood_prediction.txt
        q_flood = client.post(
            "/query",
            headers=auth_headers,
            json={
                "question": "How does FloodVision predict flood water surges?",
                "top_k": 3,
                "document_name": "dl_flood_prediction.txt",
            },
        )
        assert q_flood.status_code == 200
        flood_data = q_flood.json()
        assert flood_data["document_scope"] == "dl_flood_prediction.txt"
        assert "LSTM" in flood_data["answer"]
        assert "Job Scheduler" not in flood_data["answer"]
        assert "MongoDB" not in flood_data["answer"]
        for chunk in flood_data["source_chunks"]:
            assert chunk["document_name"] == "dl_flood_prediction.txt"
            assert chunk["document_id"] == flood_meta["document_id"]

        # 6. Cross-document unrelated question while scoped to os_job_scheduler.txt
        q_unrelated = client.post(
            "/query",
            headers=auth_headers,
            json={
                "question": "What LSTM neural layers are used for flood prediction?",
                "top_k": 3,
                "document_name": "os_job_scheduler.txt",
            },
        )
        assert q_unrelated.status_code == 200
        unrelated_data = q_unrelated.json()
        # Retrieval must STILL strictly respect os_job_scheduler.txt and NEVER pull FloodVision chunks
        assert all(c["document_name"] == "os_job_scheduler.txt" for c in unrelated_data["source_chunks"])
        assert "does not contain sufficient information" in unrelated_data["answer"].lower()

        # 7. Non-existent document scope returns 404 (never falls back to other docs)
        q_missing = client.post(
            "/query",
            headers=auth_headers,
            json={
                "question": "What is this project about?",
                "top_k": 3,
                "document_name": "nonexistent_report.pdf",
            },
        )
        assert q_missing.status_code == 404

        # 8. Model Comparison respects document scope isolation
        cmp_resp = client.post(
            "/query/compare",
            headers=auth_headers,
            json={
                "question": "Explain the architecture",
                "top_k": 2,
                "document_name": "os_job_scheduler.txt",
            },
        )
        assert cmp_resp.status_code == 200
        cmp_data = cmp_resp.json()
        assert cmp_data["document_scope"] == "os_job_scheduler.txt"
        assert all(c["document_name"] == "os_job_scheduler.txt" for c in cmp_data["source_chunks"])


@pytest.mark.integration
def test_reupload_replaces_stale_chunks_without_corrupting_other_docs(client, auth_headers, isolated_chroma_env):
    # 1. Ingest Flood Prediction and OS Document v1
    r1 = client.post(
        "/ingest",
        headers=auth_headers,
        files={"file": ("dl_flood_prediction.txt", FLOOD_DOCUMENT_CONTENT, "text/plain")},
    )
    assert r1.status_code == 200

    r2 = client.post(
        "/ingest",
        headers=auth_headers,
        files={"file": ("os_job_scheduler.txt", OS_DOCUMENT_CONTENT, "text/plain")},
    )
    assert r2.status_code == 200
    v1_doc_id = r2.json()["document_id"]

    # 2. Re-upload os_job_scheduler.txt with updated v2 content (clear_existing=False)
    os_v2_content = (
        b"Operating Systems Project Report v2: Priority Round-Robin Job Scheduler.\n"
        b"Updated design uses Kubernetes worker pods and PostgreSQL advisory locks.\n"
    )
    r3 = client.post(
        "/ingest",
        headers=auth_headers,
        files={"file": ("os_job_scheduler.txt", os_v2_content, "text/plain")},
    )
    assert r3.status_code == 200
    v2_doc_id = r3.json()["document_id"]
    assert v2_doc_id != v1_doc_id

    # 3. Verify Flood Prediction doc remained intact, and os_job_scheduler only has v2 chunks
    docs_resp = client.get("/ingest/documents", headers=auth_headers).json()
    assert len(docs_resp) == 2
    os_entry = next(d for d in docs_resp if d["filename"] == "os_job_scheduler.txt")
    assert os_entry["document_id"] == v2_doc_id

    # Query OS document directly by document_id=v1_doc_id must return 404 (no stale chunks!)
    with patch("src.services.query.get_llm", return_value=MagicMock()), \
         patch("src.services.query.RunnableSequence") as mock_seq, \
         patch("src.services.query.run_judge_evaluations"):
        mock_seq.return_value.invoke.return_value = MagicMock(content="v2 uses PostgreSQL advisory locks.")
        stale_resp = client.post(
            "/query",
            headers=auth_headers,
            json={"question": "What database?", "document_id": v1_doc_id},
        )
        assert stale_resp.status_code == 404

        # Query by document_name must ONLY retrieve v2 text (no "Version 1 Alpha" or "MongoDB")
        fresh_resp = client.post(
            "/query",
            headers=auth_headers,
            json={"question": "What database does v2 scheduler use?", "document_name": "os_job_scheduler.txt"},
        )
        assert fresh_resp.status_code == 200
        fresh_data = fresh_resp.json()
        combined_sources = " ".join(fresh_data["sources"])
        assert "PostgreSQL" in combined_sources
        assert "Version 1 Alpha" not in combined_sources
        assert "MongoDB" not in combined_sources


@pytest.mark.integration
def test_pdf_metadata_retains_page_number_and_clean_source(client, auth_headers, isolated_chroma_env):
    mock_pages = [
        Document(page_content="Page one discusses OS process scheduling queues.", metadata={"source": "/tmp/fake123.pdf", "page": 0}),
        Document(page_content="Page two discusses CPU affinity and worker heartbeats.", metadata={"source": "/tmp/fake123.pdf", "page": 1}),
    ]

    with patch("src.services.ingest.PyPDFLoader") as mock_pdf_loader:
        mock_pdf_loader.return_value.load_and_split.return_value = mock_pages

        resp = client.post(
            "/ingest",
            headers=auth_headers,
            files={"file": ("os_scheduler_report.pdf", b"%PDF-1.4 dummy", "application/pdf")},
        )
    assert resp.status_code == 200
    doc_id = resp.json()["document_id"]

    with patch("src.services.query.get_llm", return_value=MagicMock()), \
         patch("src.services.query.RunnableSequence") as mock_seq, \
         patch("src.services.query.run_judge_evaluations"):
        mock_seq.return_value.invoke.return_value = MagicMock(content="Process scheduling queues and heartbeats.")

        q_resp = client.post(
            "/query",
            headers=auth_headers,
            json={"question": "What is on pages one and two?", "top_k": 2, "document_name": "os_scheduler_report.pdf"},
        )

    assert q_resp.status_code == 200
    chunks = q_resp.json()["source_chunks"]
    assert len(chunks) == 2
    pages_seen = sorted(c["page_number"] for c in chunks)
    assert pages_seen == [1, 2]
    for c in chunks:
        assert c["document_id"] == doc_id
        assert c["document_name"] == "os_scheduler_report.pdf"
        assert c["source"] == "os_scheduler_report.pdf"
        assert "fake123" not in c["source"]
        assert c["ingestion_timestamp"] is not None
