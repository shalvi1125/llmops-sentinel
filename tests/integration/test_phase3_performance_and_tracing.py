import sqlite3
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.services import audit as audit_svc
from src.services import evaluate as eval_svc
from src.services import query as query_svc
from src.utils.context import compact_context_text, format_context_for_llm


def _create_temp_mlflow_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE trace_info (
            request_id TEXT PRIMARY KEY,
            timestamp_ms INTEGER,
            execution_time_ms INTEGER,
            status TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE spans (
            trace_id TEXT,
            name TEXT,
            content TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE assessments (
            assessment_id TEXT PRIMARY KEY,
            trace_id TEXT,
            name TEXT,
            assessment_type TEXT,
            value TEXT,
            error TEXT,
            created_timestamp INTEGER,
            last_updated_timestamp INTEGER,
            source_type TEXT,
            source_id TEXT,
            rationale TEXT,
            valid INTEGER DEFAULT 1
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE trace_tags (
            request_id TEXT,
            key TEXT,
            value TEXT,
            PRIMARY KEY (request_id, key)
        )
        """
    )
    conn.commit()
    conn.close()


@pytest.mark.unit
def test_compact_context_text_normalizes_pdf_word_per_line_artifacts():
    raw_pdf = (
        "INTRODUCTION\n \nModern  computing  tasks  vary  in  their  CPU,\n "
        "\nmemory,\n \nand\n \nexecution-time\n \nrequirements.\n\n"
        "1.  Task  Scheduling\nThe  scheduler  assigns  priority  jobs  via  FIFO."
    )
    cleaned = compact_context_text(raw_pdf)
    assert "\n \n" not in cleaned
    assert "  " not in cleaned
    assert "Modern computing tasks vary in their CPU, memory, and execution-time requirements." in cleaned
    assert "INTRODUCTION\n\n" in cleaned
    assert "\n\n1. Task Scheduling " in cleaned

    formatted = format_context_for_llm(
        [raw_pdf, raw_pdf],
        max_chars=500,
        source_labels=["report.pdf p.1", "report.pdf p.1"],
    )
    assert formatted.startswith("[report.pdf p.1]")
    assert formatted.count("Modern computing tasks") == 1
    assert len(formatted) <= 500


@pytest.mark.unit
def test_query_embedding_cache_keys_ttl_and_eviction():
    query_svc.clear_embedding_cache()
    calls: list[str] = []

    class FakeEmb:
        def embed_query(self, text: str) -> list[float]:
            calls.append(text)
            return [0.1 * len(text), 0.25, 0.5]

    emb = FakeEmb()
    v1, _, hit1 = query_svc._embed_query_cached(emb, "What is priority scheduling?", "nomic-embed-text")
    v2, _, hit2 = query_svc._embed_query_cached(emb, "What is   priority   scheduling?", "nomic-embed-text")
    v3, _, hit3 = query_svc._embed_query_cached(emb, "What is priority scheduling?", "other-embed-model")

    assert hit1 is False
    assert hit2 is True
    assert hit3 is False
    assert v1 == v2
    assert len(calls) == 2

    # Force TTL expiration
    with query_svc._embed_cache_lock:
        for k in list(query_svc._embed_cache.keys()):
            _, vec = query_svc._embed_cache[k]
            query_svc._embed_cache[k] = (time.monotonic() - 9999.0, vec)

    _, _, hit_expired = query_svc._embed_query_cached(emb, "What is priority scheduling?", "nomic-embed-text")
    assert hit_expired is False
    assert len(calls) == 3

    query_svc.clear_embedding_cache()


@pytest.mark.unit
def test_native_chroma_query_single_roundtrip_preserves_scope_and_mmr():
    query_svc.clear_embedding_cache()

    fake_collection = MagicMock()
    fake_collection.count.return_value = 5
    fake_collection.query.return_value = {
        "ids": [["c1", "c2", "c3"]],
        "documents": [[
            "Task scheduler claims jobs atomically with MongoDB.",
            "Task scheduler claims jobs atomically with MongoDB.",
            "Heartbeat monitor detects unresponsive workers cleanly.",
        ]],
        "metadatas": [[
            {"chunk_id": "c1", "document_name": "os.pdf", "page_number": 2},
            {"chunk_id": "c2", "document_name": "os.pdf", "page_number": 2},
            {"chunk_id": "c3", "document_name": "os.pdf", "page_number": 4},
        ]],
        "distances": [[0.18, 0.19, 0.31]],
        "embeddings": [[
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
        ]],
    }

    fake_emb = MagicMock()
    fake_emb.embed_query.return_value = [0.8, 0.6, 0.0]

    fake_vs = MagicMock()
    fake_vs._collection = fake_collection
    fake_vs._embedding_function = fake_emb

    sources, chunks, diag, scope, timings = query_svc._retrieve_with_timings(
        vectorstore=fake_vs,
        question="How does the scheduler monitor workers?",
        top_k=3,
        document_name="os.pdf",
        use_mmr=True,
    )

    assert scope == "os.pdf"
    assert len(sources) == 2
    assert diag.duplicates_removed == 1
    assert diag.retrieval_strategy == "mmr_deduplicated"
    assert timings["embedding_cache_hit"] == 0.0
    fake_collection.query.assert_called_once()
    # Embeddings already came back from query(), so no extra _collection.get roundtrip
    fake_collection.get.assert_not_called()

    # Repeat call uses TTL embedding cache
    _, _, _, _, timings2 = query_svc._retrieve_with_timings(
        vectorstore=fake_vs,
        question="How does the scheduler monitor workers?",
        top_k=3,
        document_name="os.pdf",
        use_mmr=True,
    )
    assert timings2["embedding_cache_hit"] == 1.0
    fake_emb.embed_query.assert_called_once()
    query_svc.clear_embedding_cache()


@pytest.mark.unit
def test_generation_settings_and_thread_offload_compare_keeps_health_responsive(client, auth_headers):
    s_llama = query_svc.get_generation_settings("llama3.2")
    s_mistral = query_svc.get_generation_settings("mistral")
    assert s_llama["num_ctx"] == 2048
    assert s_llama["num_predict"] == 220
    assert s_llama["keep_alive"] == "15m"
    assert s_mistral["keep_alive"] == "2m"

    def slow_invoke(inputs):
        time.sleep(0.15)
        resp = MagicMock()
        resp.content = "Sequential answer from model."
        resp.response_metadata = {"eval_count": 24, "eval_duration": 150_000_000}
        return resp

    fake_chunks = [
        query_svc.SourceChunk(
            chunk_id="c1",
            text="Resource-aware scheduler assigns CPU and RAM via MongoDB transactions.",
            score=0.21,
            document_name="os.pdf",
            page_number=1,
            token_count_estimate=12,
        )
    ]
    fake_diag = query_svc.RetrievalDiagnostics(
        retrieval_latency_ms=14.5,
        top_score=0.21,
        low_confidence=False,
        requested_k=3,
        candidates_fetched=3,
        unique_chunks_returned=1,
        duplicates_removed=0,
        retrieval_strategy="mmr_deduplicated",
    )

    health_statuses: list[int] = []

    mock_vs = MagicMock()
    mock_health_http = MagicMock()
    mock_health_http.status_code = 200

    with patch("src.services.compare._get_vectorstore", return_value=mock_vs), \
         patch("src.services.compare.retrieve_context_chunks", return_value=(
             [fake_chunks[0].text], fake_chunks, fake_diag, "os.pdf"
         )), \
         patch("src.services.compare.get_llm", return_value=MagicMock()), \
         patch("src.services.compare.PromptTemplate"), \
         patch("src.services.compare.RunnableSequence") as mock_seq, \
         patch("src.services.health.httpx.AsyncClient") as mock_http_cls, \
         patch("src.services.health.OllamaEmbeddings"), \
         patch("src.services.health.Chroma") as mock_health_chroma:

        mock_seq.return_value.invoke.side_effect = slow_invoke
        mock_http_client = MagicMock()
        mock_http_client.get = MagicMock()
        mock_http_cls.return_value.__aenter__.return_value = mock_http_client
        mock_http_client.get.return_value = mock_health_http
        mock_health_chroma.return_value._collection.count.return_value = 5

        cmp_holder: dict = {}

        def run_compare_req():
            r = client.post(
                "/query/compare",
                headers=auth_headers,
                json={"question": "How does scheduling work?", "top_k": 3, "document_name": "os.pdf"},
            )
            cmp_holder["status"] = r.status_code
            cmp_holder["json"] = r.json()

        t = threading.Thread(target=run_compare_req)
        t.start()
        time.sleep(0.05)
        h_resp = client.get("/health")
        health_statuses.append(h_resp.status_code)
        t.join(timeout=5)

    assert health_statuses == [200]
    assert cmp_holder["status"] == 200
    body = cmp_holder["json"]
    assert body["retrieval_latency_ms"] == 14.5
    assert body["generation_settings"]["num_ctx"] == 2048
    assert [m["model"] for m in body["models"]] == ["llama3.2", "mistral"]
    assert body["models"][0]["token_count_estimate"] == 24


@pytest.mark.unit
def test_judge_cancellation_prevents_subsequent_scorer_calls(tmp_path):
    db_file = str(tmp_path / "mlflow_test.db")
    _create_temp_mlflow_db(db_file)

    scorer_calls: list[str] = []

    class FakeRelevancy:
        def __init__(self, model: str):
            self.model = model

        def __call__(self, **kwargs):
            scorer_calls.append("AnswerRelevancy")
            # Simulate cancellation mid-evaluation before second scorer runs
            eval_svc.cancel_current_evaluation()
            fb = MagicMock()
            fb.error = None
            fb.value = "yes"
            fb.rationale = "Relevant."
            return fb

    class FakeHallucination:
        def __init__(self, model: str):
            self.model = model

        def __call__(self, **kwargs):
            scorer_calls.append("Hallucination")
            return MagicMock(value="no", rationale="Grounded", error=None)

    class FakeSafety:
        def __init__(self, model: str):
            self.model = model

        def __call__(self, **kwargs):
            scorer_calls.append("Safety")
            return MagicMock(value="yes", rationale="Safe", error=None)

    conn = sqlite3.connect(db_file)
    conn.execute(
        "INSERT INTO trace_info (request_id, timestamp_ms, execution_time_ms, status) VALUES (?, ?, ?, ?)",
        ("tr-cancel-test", int(time.time() * 1000), 1234, "OK"),
    )
    conn.commit()
    conn.close()

    with patch.object(eval_svc, "MLFLOW_DB_PATH", db_file), \
         patch.object(audit_svc, "DB_PATH", db_file), \
         patch("src.services.evaluate._is_mlflow_server_reachable", return_value=False), \
         patch("src.services.evaluate.AnswerRelevancy", FakeRelevancy), \
         patch("src.services.evaluate.Hallucination", FakeHallucination), \
         patch("src.services.evaluate.Safety", FakeSafety), \
         patch("src.services.evaluate.mlflow.log_feedback"):

        job = eval_svc.EvaluationJob(
            job_id="eval-cancel123",
            trace_id="tr-cancel-test",
            question="Q cancel?",
            answer="A cancel",
            state="pending",
            created_at_ms=int(time.time() * 1000),
        )
        with eval_svc._state_lock:
            eval_svc._latest_queued_job_id = job.job_id
            eval_svc._pending_jobs_by_qa[("Q cancel?", "A cancel")] = job

        eval_svc.run_judge_evaluations(
            question="Q cancel?",
            answer="A cancel",
            context_chunks=["Scheduler context chunk"],
        )

        # Hallucination & Safety must NOT execute once cancelled after AnswerRelevancy
        assert scorer_calls == ["AnswerRelevancy"]
        traces = audit_svc.get_recent_traces(limit=5)
        assert len(traces) == 1
        assert traces[0].evaluation_status == "cancelled"
        assert traces[0].execution_time_ms == 1234.0


@pytest.mark.integration
def test_audit_and_performance_endpoint_reads_populated_trace_metadata(client, auth_headers, tmp_path):
    db_file = str(tmp_path / "mlflow_perf.db")
    _create_temp_mlflow_db(db_file)

    now_ms = int(time.time() * 1000)
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO trace_info (request_id, timestamp_ms, execution_time_ms, status) VALUES (?, ?, ?, ?)",
        [
            ("tr-1", now_ms - 2000, 4200, "OK"),
            ("tr-2", now_ms - 1000, 3100, "OK"),
            ("tr-legacy", now_ms - 5000, 1, "OK"),
        ],
    )
    cur.executemany(
        "INSERT INTO trace_tags (request_id, key, value) VALUES (?, ?, ?)",
        [
            ("tr-1", "model", "llama3.2"),
            ("tr-1", "prompt_id", "default_qa"),
            ("tr-1", "prompt_version", "1.1.0"),
            ("tr-1", "document_scope", "os.pdf"),
            ("tr-1", "retrieval_latency_ms", "85.4"),
            ("tr-1", "generation_latency_ms", "4100.2"),
            ("tr-1", "total_query_latency_ms", "4200.0"),
            ("tr-1", "tokens_per_second", "11.5"),
            ("tr-1", "evaluation_status", "completed"),
            ("tr-1", "evaluation_duration_ms", "18500.0"),
            ("tr-2", "model", "mistral"),
            ("tr-2", "prompt_id", "concise_facts"),
            ("tr-2", "prompt_version", "1.1.0"),
            ("tr-2", "document_scope", "os.pdf"),
            ("tr-2", "retrieval_latency_ms", "42.0"),
            ("tr-2", "generation_latency_ms", "3040.0"),
            ("tr-2", "total_query_latency_ms", "3100.0"),
            ("tr-2", "tokens_per_second", "4.8"),
            ("tr-2", "evaluation_status", "cancelled"),
            ("tr-2", "evaluation_duration_ms", "3200.0"),
        ],
    )
    conn.commit()
    conn.close()

    with patch.object(audit_svc, "DB_PATH", db_file):
        traces_resp = client.get("/eval/traces?limit=10", headers=auth_headers)
        assert traces_resp.status_code == 200
        traces_data = traces_resp.json()
        assert traces_data[0]["model"] == "mistral"
        assert traces_data[0]["prompt_id"] == "concise_facts"
        assert traces_data[0]["prompt_version"] == "1.1.0"
        assert traces_data[0]["evaluation_status"] == "cancelled"
        assert traces_data[1]["execution_time_ms"] == 4200.0

        perf_resp = client.get("/eval/performance?limit=20", headers=auth_headers)
        assert perf_resp.status_code == 200
        perf = perf_resp.json()
        assert perf["total_traces"] == 3
        assert perf["measured_query_traces"] == 2
        assert perf["avg_query_latency_ms"] == 3650.0
        assert perf["median_query_latency_ms"] == 3650.0
        assert perf["eval_counts"]["completed"] == 1
        assert perf["eval_counts"]["cancelled"] == 1
        assert perf["eval_completion_rate"] == 50.0
        assert len(perf["model_breakdown"]) == 2


@pytest.mark.unit
def test_judge_preserves_completed_scorers_when_preempted_after_first_and_second(tmp_path):
    """Completed scorer feedback must be persisted immediately and kept if preempted."""
    db_file = str(tmp_path / "mlflow_partial.db")
    _create_temp_mlflow_db(db_file)
    now_ms = int(time.time() * 1000)

    conn = sqlite3.connect(db_file)
    conn.executemany(
        "INSERT INTO trace_info (request_id, timestamp_ms, execution_time_ms, status) VALUES (?, ?, ?, ?)",
        [
            ("tr-partial-1", now_ms - 1000, 900, "OK"),
            ("tr-partial-2", now_ms, 950, "OK"),
        ],
    )
    conn.commit()
    conn.close()

    # Case 1: Preempted right as AnswerRelevancy finishes (before Hallucination runs)
    mid_stream_snapshots: list[dict[str, str]] = []

    class RelevancyTriggersCancel:
        def __init__(self, model: str):
            self.model = model

        def __call__(self, **kwargs):
            eval_svc.cancel_current_evaluation()
            return MagicMock(value="yes", rationale="Clear grounded answer", error=None)

    class UnusedScorer:
        called = False

        def __init__(self, model: str):
            self.model = model

        def __call__(self, **kwargs):
            UnusedScorer.called = True
            return MagicMock(value="yes", rationale="Should not run", error=None)

    with patch.object(eval_svc, "MLFLOW_DB_PATH", db_file), \
         patch.object(audit_svc, "DB_PATH", db_file), \
         patch("src.services.evaluate._is_mlflow_server_reachable", return_value=False), \
         patch("src.services.evaluate.AnswerRelevancy", RelevancyTriggersCancel), \
         patch("src.services.evaluate.Hallucination", UnusedScorer), \
         patch("src.services.evaluate.Safety", UnusedScorer), \
         patch("src.services.evaluate.mlflow.log_feedback"):
        job1 = eval_svc.EvaluationJob(
            job_id="eval-partial1",
            trace_id="tr-partial-1",
            question="Partial Q1?",
            answer="Partial A1",
            state="pending",
            created_at_ms=now_ms - 1000,
        )
        with eval_svc._state_lock:
            eval_svc._latest_queued_job_id = job1.job_id
            eval_svc._pending_jobs_by_qa[("Partial Q1?", "Partial A1")] = job1

        eval_svc.run_judge_evaluations("Partial Q1?", "Partial A1", ["Context 1"])

    assert UnusedScorer.called is False
    conn = sqlite3.connect(db_file)
    rows1 = conn.execute(
        "SELECT name, value FROM assessments WHERE trace_id = 'tr-partial-1' ORDER BY name"
    ).fetchall()
    conn.close()
    assert dict(rows1) == {
        "AnswerRelevancy": '"yes"',
        "Hallucination": '"cancelled"',
        "Safety": '"cancelled"',
    }
    assert len(rows1) == 3  # no duplicate rows

    # Case 2: AnswerRelevancy finishes (verify logged to DB BEFORE Hallucination finishes!),
    # then cancellation occurs right after Hallucination; Safety never runs.
    class RelevancyOk:
        def __init__(self, model: str):
            self.model = model

        def __call__(self, **kwargs):
            return MagicMock(value="yes", rationale="Relevant", error=None)

    class HallucinationChecksImmediateLogThenCancels:
        def __init__(self, model: str):
            self.model = model

        def __call__(self, **kwargs):
            conn_check = sqlite3.connect(db_file)
            logged_so_far = dict(
                conn_check.execute(
                    "SELECT name, value FROM assessments WHERE trace_id = 'tr-partial-2'"
                ).fetchall()
            )
            conn_check.close()
            mid_stream_snapshots.append(logged_so_far)
            eval_svc.cancel_current_evaluation()
            return MagicMock(value="yes", rationale="No hallucination", error=None)

    class SafetyShouldNotRun:
        called = False

        def __init__(self, model: str):
            self.model = model

        def __call__(self, **kwargs):
            SafetyShouldNotRun.called = True
            return MagicMock(value="yes", rationale="Safe", error=None)

    with patch.object(eval_svc, "MLFLOW_DB_PATH", db_file), \
         patch.object(audit_svc, "DB_PATH", db_file), \
         patch("src.services.evaluate._is_mlflow_server_reachable", return_value=False), \
         patch("src.services.evaluate.AnswerRelevancy", RelevancyOk), \
         patch("src.services.evaluate.Hallucination", HallucinationChecksImmediateLogThenCancels), \
         patch("src.services.evaluate.Safety", SafetyShouldNotRun), \
         patch("src.services.evaluate.mlflow.log_feedback"):
        job2 = eval_svc.EvaluationJob(
            job_id="eval-partial2",
            trace_id="tr-partial-2",
            question="Partial Q2?",
            answer="Partial A2",
            state="pending",
            created_at_ms=now_ms,
        )
        with eval_svc._state_lock:
            eval_svc._latest_queued_job_id = job2.job_id
            eval_svc._pending_jobs_by_qa[("Partial Q2?", "Partial A2")] = job2

        eval_svc.run_judge_evaluations("Partial Q2?", "Partial A2", ["Context 2"])

    assert SafetyShouldNotRun.called is False
    # AnswerRelevancy must already be committed in SQLite before Hallucination returns!
    assert mid_stream_snapshots == [{"AnswerRelevancy": '"yes"'}]

    conn = sqlite3.connect(db_file)
    rows2 = conn.execute(
        "SELECT name, value FROM assessments WHERE trace_id = 'tr-partial-2' ORDER BY name"
    ).fetchall()
    conn.close()
    assert dict(rows2) == {
        "AnswerRelevancy": '"yes"',
        "Hallucination": '"yes"',
        "Safety": '"cancelled"',
    }
    assert len(rows2) == 3

    with patch.object(audit_svc, "DB_PATH", db_file):
        t2 = [t for t in audit_svc.get_recent_traces(limit=5) if t.request_id == "tr-partial-2"][0]
        assert t2.evaluation_status == "cancelled"
        assert t2.assessments["AnswerRelevancy"].value == "yes"
        assert t2.assessments["Hallucination"].value == "yes"
        assert t2.assessments["Safety"].value == "cancelled"


@pytest.mark.unit
def test_shutdown_evaluation_cleanup_cancels_active_and_pending_jobs(tmp_path):
    """Server shutdown cleanup must unwind active workers, retain partial scores, and cancel pending jobs."""
    db_file = str(tmp_path / "mlflow_shutdown.db")
    _create_temp_mlflow_db(db_file)
    now_ms = int(time.time() * 1000)

    conn = sqlite3.connect(db_file)
    conn.executemany(
        "INSERT INTO trace_info (request_id, timestamp_ms, execution_time_ms, status) VALUES (?, ?, ?, ?)",
        [
            ("tr-active-sd", now_ms - 500, 400, "OK"),
            ("tr-pending-sd", now_ms, 300, "OK"),
        ],
    )
    conn.commit()
    conn.close()

    hallucination_entered = threading.Event()
    release_hallucination = threading.Event()

    class FastRelevancy:
        def __init__(self, model: str):
            self.model = model

        def __call__(self, **kwargs):
            return MagicMock(value="yes", rationale="Complete before shutdown", error=None)

    class BlockingHallucination:
        def __init__(self, model: str):
            self.model = model

        def __call__(self, **kwargs):
            hallucination_entered.set()
            release_hallucination.wait(timeout=3.0)
            raise RuntimeError("Interrupted by shutdown")

    class NeverSafety:
        def __init__(self, model: str):
            self.model = model

        def __call__(self, **kwargs):
            raise AssertionError("Safety must not start after shutdown cancellation")

    with patch.object(eval_svc, "MLFLOW_DB_PATH", db_file), \
         patch.object(audit_svc, "DB_PATH", db_file), \
         patch("src.services.evaluate._is_mlflow_server_reachable", return_value=False), \
         patch("src.services.evaluate.AnswerRelevancy", FastRelevancy), \
         patch("src.services.evaluate.Hallucination", BlockingHallucination), \
         patch("src.services.evaluate.Safety", NeverSafety), \
         patch("src.services.evaluate.mlflow.log_feedback"):

        active_job = eval_svc.EvaluationJob(
            job_id="eval-active-sd",
            trace_id="tr-active-sd",
            question="Active Q",
            answer="Active A",
            state="pending",
            created_at_ms=now_ms - 500,
        )
        pending_job = eval_svc.EvaluationJob(
            job_id="eval-pending-sd",
            trace_id="tr-pending-sd",
            question="Pending Q",
            answer="Pending A",
            state="pending",
            created_at_ms=now_ms,
        )
        with eval_svc._state_lock:
            eval_svc._eval_jobs.clear()
            eval_svc._eval_jobs[active_job.job_id] = active_job
            eval_svc._eval_jobs[pending_job.job_id] = pending_job
            eval_svc._pending_jobs_by_qa[("Active Q", "Active A")] = active_job
            # Pretend active_job owns latest slot until worker starts, while pending_job sits pending
            eval_svc._latest_queued_job_id = active_job.job_id

        worker = threading.Thread(
            target=eval_svc.run_judge_evaluations,
            args=("Active Q", "Active A", ["Chunk A"]),
        )
        worker.start()
        assert hallucination_entered.wait(timeout=3.0)

        # Unblock scorer slightly asynchronously as socket shutdown would, then call shutdown cleanup
        threading.Timer(0.05, release_hallucination.set).start()
        cleaned = eval_svc.shutdown_evaluation_cleanup(wait_timeout=3.0)
        worker.join(timeout=3.0)

        assert cleaned >= 1
        assert pending_job.state == "cancelled"
        assert active_job.state == "cancelled"

        traces = {t.request_id: t for t in audit_svc.get_recent_traces(limit=10)}
        assert traces["tr-active-sd"].evaluation_status == "cancelled"
        assert traces["tr-active-sd"].assessments["AnswerRelevancy"].value == "yes"
        assert traces["tr-active-sd"].assessments["Hallucination"].value == "cancelled"
        assert traces["tr-active-sd"].assessments["Safety"].value == "cancelled"

        assert traces["tr-pending-sd"].evaluation_status == "cancelled"
        assert traces["tr-pending-sd"].assessments["Hallucination"].value == "cancelled"

        eval_svc._eval_jobs.clear()


@pytest.mark.unit
def test_reconcile_orphaned_evaluations_recovers_dead_owners_and_preserves_history(tmp_path):
    """Startup reconciliation fixes crashed-process traces without altering live or legacy rows."""
    db_file = str(tmp_path / "mlflow_reconcile.db")
    _create_temp_mlflow_db(db_file)
    now_ms = int(time.time() * 1000)

    conn = sqlite3.connect(db_file)
    conn.executemany(
        "INSERT INTO trace_info (request_id, timestamp_ms, execution_time_ms, status) VALUES (?, ?, ?, ?)",
        [
            ("tr-orphan-dead", now_ms - 4000, 1200, "OK"),
            ("tr-orphan-pending", now_ms - 3000, 1100, "OK"),
            ("tr-live-owner", now_ms - 2000, 1000, "OK"),
            ("tr-legacy-no-owner", now_ms - 5000, 800, "OK"),
            ("tr-historical-done", now_ms - 6000, 700, "OK"),
        ],
    )
    live_token = eval_svc._get_process_owner_token()
    dead_token = "999999:1700000000:deadhost"
    conn.executemany(
        "INSERT INTO trace_tags (request_id, key, value) VALUES (?, ?, ?)",
        [
            ("tr-orphan-dead", "evaluation_status", "running"),
            ("tr-orphan-dead", "evaluation_owner", dead_token),
            ("tr-orphan-pending", "evaluation_status", "pending"),
            ("tr-orphan-pending", "evaluation_owner", dead_token),
            ("tr-live-owner", "evaluation_status", "running"),
            ("tr-live-owner", "evaluation_owner", live_token),
            ("tr-legacy-no-owner", "evaluation_status", "running"),
            ("tr-historical-done", "evaluation_status", "completed"),
        ],
    )
    # Seed partial completed assessment on tr-orphan-dead & historical legacy safety row
    conn.execute(
        """
        INSERT INTO assessments
        (assessment_id, trace_id, name, assessment_type, value, error, created_timestamp, last_updated_timestamp, source_type, source_id, rationale, valid)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("a-partial-1", "tr-orphan-dead", "AnswerRelevancy", "FEEDBACK", '"yes"', None, now_ms, now_ms, "LLM_JUDGE", "ollama:/mistral", "Preserved partial", 1),
    )
    conn.execute(
        """
        INSERT INTO assessments
        (assessment_id, trace_id, name, assessment_type, value, error, created_timestamp, last_updated_timestamp, source_type, source_id, rationale, valid)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("a-legacy-1", "tr-historical-done", "safety", "FEEDBACK", '"yes"', None, now_ms, now_ms, "LLM_JUDGE", "ollama:/mistral", "Legacy record", 1),
    )
    conn.commit()
    conn.close()

    with patch("src.services.evaluate._is_mlflow_server_reachable", return_value=False), \
         patch("src.services.evaluate.mlflow.log_feedback"):
        recovered = eval_svc.reconcile_orphaned_evaluations(db_path=db_file)
        assert recovered == 2
        # Calling again is strictly idempotent
        assert eval_svc.reconcile_orphaned_evaluations(db_path=db_file) == 0

    with patch.object(audit_svc, "DB_PATH", db_file):
        traces = {t.request_id: t for t in audit_svc.get_recent_traces(limit=10)}

    assert traces["tr-orphan-dead"].evaluation_status == "cancelled"
    assert traces["tr-orphan-dead"].assessments["AnswerRelevancy"].value == "yes"
    assert traces["tr-orphan-dead"].assessments["Hallucination"].value == "cancelled"
    assert traces["tr-orphan-dead"].assessments["Safety"].value == "cancelled"

    assert traces["tr-orphan-pending"].evaluation_status == "cancelled"
    assert traces["tr-orphan-pending"].assessments["AnswerRelevancy"].value == "cancelled"
    assert traces["tr-orphan-pending"].assessments["Hallucination"].value == "cancelled"
    assert traces["tr-orphan-pending"].assessments["Safety"].value == "cancelled"

    # Active owner trace, legacy trace without owner, and completed historical trace must remain untouched
    assert traces["tr-live-owner"].evaluation_status == "running"
    assert traces["tr-legacy-no-owner"].evaluation_status == "running"
    assert traces["tr-historical-done"].evaluation_status == "completed"
    assert traces["tr-historical-done"].assessments["Safety"].value == "yes"
