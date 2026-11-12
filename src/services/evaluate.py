from collections import OrderedDict
from dataclasses import dataclass
import json
import os
import re
import socket
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import mlflow
from mlflow.entities import SpanType
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import Safety
from mlflow.genai.scorers.deepeval import AnswerRelevancy, Hallucination
import mlflow.metrics.genai.model_utils as _model_utils
import psutil
import requests
import urllib3.connection

try:
    import mlflow.genai.scorers.deepeval.models as _deepeval_models
except ImportError:
    _deepeval_models = None

try:
    import litellm as _litellm
except ImportError:
    _litellm = None

from src.metrics import (
    EVAL_DURATION_SECONDS,
    EVAL_JOBS_ACTIVE,
    EVAL_JOBS_TOTAL,
    EVAL_SCORER_DURATION_SECONDS,
)
from src.utils.context import format_context_for_llm
from src.utils.env import MLFLOW_TRACKING_URI, OLLAMA_BASE_URL, OLLAMA_JUDGE_MODEL
from src.utils.log_manager import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB_PATH = str(PROJECT_ROOT / "mlflow.db")

os.environ.setdefault("CONFIDENT_METRIC_LOGGING_VERBOSE", "0")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", "1")
os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
os.environ.setdefault("MLFLOW_HTTP_TCP_KEEPALIVE", "false")
os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL

_PARSED_OLLAMA = urlparse(OLLAMA_BASE_URL)
_OLLAMA_PORT = _PARSED_OLLAMA.port or 11434
_PARSED_MLFLOW = urlparse(MLFLOW_TRACKING_URI)

# Bounded evaluation job tracking and safe preemption controls
_MAX_TRACKED_JOBS = 200
_EVAL_CONTEXT_CHAR_LIMIT = 1500
_JUDGE_CALL_TIMEOUT_SEC = 90

_eval_lock = threading.Lock()
_state_lock = threading.Lock()
_is_evaluating = False
_cancel_eval_event = threading.Event()
_eval_sockets: set[socket.socket] = set()
_eval_sockets_lock = threading.Lock()
_eval_thread_local = threading.local()

_latest_queued_job_id: str | None = None
_INSTANCE_ID = uuid.uuid4().hex[:8]
_ALL_EVAL_METRICS = ("AnswerRelevancy", "Hallucination", "Safety")


def _get_process_owner_token() -> str:
    """Return a collision-resistant '<pid>:<create_time_int>:<instance_id>' owner tag."""
    pid = os.getpid()
    try:
        create_ts = int(psutil.Process(pid).create_time())
    except Exception:
        create_ts = 0
    return f"{pid}:{create_ts}:{_INSTANCE_ID}"


def _is_owner_process_alive(owner_token: str | None) -> bool:
    """Return True if an evaluation owner token maps to a distinct live OS process."""
    if not owner_token or ":" not in owner_token:
        return True
    parts = owner_token.split(":", 2)
    if len(parts) != 3:
        return True
    try:
        pid = int(parts[0])
        recorded_create = int(parts[1])
    except ValueError:
        return True
    if pid <= 0 or not psutil.pid_exists(pid):
        return False
    try:
        proc = psutil.Process(pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False
        if recorded_create > 0 and abs(int(proc.create_time()) - recorded_create) > 2:
            return False
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


@dataclass
class EvaluationJob:
    job_id: str
    trace_id: str | None
    question: str
    answer: str
    state: str = "pending"  # pending | running | completed | skipped | cancelled | failed
    created_at_ms: int = 0
    duration_ms: float | None = None
    reason: str | None = None
    cancel_requested: bool = False


_pending_jobs_by_qa: OrderedDict[tuple[str, str], EvaluationJob] = OrderedDict()
_eval_jobs: OrderedDict[str, EvaluationJob] = OrderedDict()
_current_job: EvaluationJob | None = None


class EvaluationPreemptedError(Exception):
    """Raised when an ongoing evaluation is cancelled/preempted by an incoming query or compare."""


def is_evaluation_running() -> bool:
    """Return True if a background judge evaluation is actively executing."""
    return _is_evaluating


def get_evaluation_job(job_id: str) -> EvaluationJob | None:
    """Inspect a tracked evaluation job by ID."""
    with _state_lock:
        return _eval_jobs.get(job_id)


def _remember_job(job: EvaluationJob) -> None:
    with _state_lock:
        _eval_jobs[job.job_id] = job
        while len(_eval_jobs) > _MAX_TRACKED_JOBS:
            _eval_jobs.popitem(last=False)


# --- Socket-level preemption strictly for LiteLLM/HTTP judge calls to Ollama ---
_orig_socket_connect = socket.socket.connect


def _tracked_eval_socket_connect(self, address):
    res = _orig_socket_connect(self, address)
    try:
        if getattr(_eval_thread_local, "in_eval_llm_call", False):
            if isinstance(address, tuple) and len(address) >= 2 and int(address[1]) == _OLLAMA_PORT:
                with _eval_sockets_lock:
                    _eval_sockets.add(self)
    except Exception:
        pass
    return res


socket.socket.connect = _tracked_eval_socket_connect

_orig_http_connect = urllib3.connection.HTTPConnection.connect


def _patched_http_connect(self):
    _orig_http_connect(self)
    if getattr(self, "_is_eval_conn", False) and self.sock:
        with _eval_sockets_lock:
            _eval_sockets.add(self.sock)


urllib3.connection.HTTPConnection.connect = _patched_http_connect


def cancel_current_evaluation() -> None:
    """Signal active judge evaluation to stop spawning calls and close active Ollama judge sockets."""
    global _latest_queued_job_id
    _cancel_eval_event.set()
    with _state_lock:
        if _current_job is not None:
            _current_job.cancel_requested = True
            _current_job.reason = "Cancelled: Preempted by new incoming query or model comparison"
    with _eval_sockets_lock:
        sockets_to_close = list(_eval_sockets)
        _eval_sockets.clear()
    for sock in sockets_to_close:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass


def _clean_and_repair_json(text: str) -> str:
    """Strip markdown codeblocks and repair truncated or unclosed JSON from local LLMs."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    candidate = match.group(1).strip() if match else text

    try:
        json.loads(candidate)
        return candidate
    except Exception:
        pass

    repaired = re.sub(r",\s*([\}\]])", r"\1", candidate)
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")
    if open_brackets > 0:
        repaired += "]" * open_brackets
    if open_braces > 0:
        repaired += "}" * open_braces

    try:
        json.loads(repaired)
        return repaired
    except Exception:
        return candidate


if _deepeval_models is not None and hasattr(_deepeval_models, "_parse_json_output_with_schema"):
    _orig_parse_json = _deepeval_models._parse_json_output_with_schema

    def _robust_parse_json(output: str, schema):
        cleaned = _clean_and_repair_json(output)
        try:
            return _orig_parse_json(cleaned, schema)
        except Exception:
            return _orig_parse_json(output, schema)

    _deepeval_models._parse_json_output_with_schema = _robust_parse_json


def _install_litellm_cancellation_hook() -> None:
    """Wrap litellm.completion / acompletion to block new calls after cancellation and bound Ollama options."""
    if _litellm is None or getattr(_litellm, "_ragscope_hooked", False):
        return

    orig_completion = _litellm.completion

    def _guarded_completion(*args, **kwargs):
        in_eval = bool(getattr(_eval_thread_local, "in_eval_worker", False)) or _is_evaluating
        if in_eval and _cancel_eval_event.is_set():
            raise EvaluationPreemptedError("Evaluation cancelled prior to LLM scorer request.")

        if in_eval:
            kwargs.setdefault("num_retries", 0)
            kwargs.setdefault("timeout", _JUDGE_CALL_TIMEOUT_SEC)
            model_arg = str(kwargs.get("model") or (args[0] if args else ""))
            if model_arg.startswith("ollama"):
                extra = dict(kwargs.get("extra_body") or {})
                opts = dict(extra.get("options") or {})
                opts.setdefault("num_ctx", 2048)
                opts.setdefault("num_predict", 220)
                opts.setdefault("num_thread", 8)
                extra["options"] = opts
                extra.setdefault("keep_alive", "2m")
                kwargs["extra_body"] = extra

        prev_flag = getattr(_eval_thread_local, "in_eval_llm_call", False)
        if in_eval:
            _eval_thread_local.in_eval_llm_call = True
        try:
            res = orig_completion(*args, **kwargs)
            if in_eval and _cancel_eval_event.is_set():
                raise EvaluationPreemptedError("Evaluation cancelled during LLM scorer response.")
            return res
        except EvaluationPreemptedError:
            raise
        except Exception as exc:
            if in_eval and _cancel_eval_event.is_set():
                raise EvaluationPreemptedError("Evaluation socket closed on preemption.") from exc
            raise
        finally:
            _eval_thread_local.in_eval_llm_call = prev_flag

    _litellm.completion = _guarded_completion
    _litellm._ragscope_hooked = True


_install_litellm_cancellation_hook()


def _patched_send_request(endpoint, headers, payload):
    if _cancel_eval_event.is_set():
        raise EvaluationPreemptedError("Evaluation preempted by new incoming query.")

    if isinstance(endpoint, str):
        endpoint = endpoint.replace("://localhost:", "://127.0.0.1:").replace("://localhost/", "://127.0.0.1/")
    if isinstance(payload, dict):
        payload = dict(payload)
        payload.setdefault("max_tokens", 256)

    session = requests.Session()
    adapter = session.get_adapter("http://")
    orig_get_conn = adapter.poolmanager.connection_from_url

    def patched_conn(url, pool_kwargs=None):
        pool = orig_get_conn(url, pool_kwargs)
        orig_pool_conn = pool._get_conn

        def get_conn(timeout=None):
            conn = orig_pool_conn(timeout)
            conn._is_eval_conn = True
            return conn

        pool._get_conn = get_conn
        return pool

    adapter.poolmanager.connection_from_url = patched_conn

    prev_flag = getattr(_eval_thread_local, "in_eval_llm_call", False)
    _eval_thread_local.in_eval_llm_call = True
    try:
        response = session.post(
            url=endpoint,
            headers=headers,
            json=payload,
            timeout=_JUDGE_CALL_TIMEOUT_SEC,
        )
        response.raise_for_status()
        if _cancel_eval_event.is_set():
            raise EvaluationPreemptedError("Evaluation preempted by new incoming query.")
        return response.json()
    except EvaluationPreemptedError:
        raise
    except requests.exceptions.RequestException as exc:
        if _cancel_eval_event.is_set():
            raise EvaluationPreemptedError("Evaluation preempted by new incoming query.") from exc
        body = getattr(getattr(exc, "response", None), "text", "")
        raise MlflowException(
            f"Failed to call LLM endpoint at {endpoint}.\n- Error: {exc}\n"
            f"- Response body: {body}\n- Input payload: {payload}."
        ) from exc
    finally:
        _eval_thread_local.in_eval_llm_call = prev_flag
        session.close()


if hasattr(_model_utils, "_send_request"):
    _model_utils._send_request = _patched_send_request


def _is_mlflow_server_reachable() -> bool:
    """Quickly verify whether the configured MLflow server is reachable without blocking unit tests."""
    if not MLFLOW_TRACKING_URI:
        return False
    if MLFLOW_TRACKING_URI.startswith("sqlite:") or MLFLOW_TRACKING_URI.startswith("file:"):
        return True
    host = _PARSED_MLFLOW.hostname
    port = _PARSED_MLFLOW.port or (443 if _PARSED_MLFLOW.scheme == "https" else 80)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=0.15):
            return True
    except OSError:
        return False


_configured_mlflow_uri: str | None = None


def _ensure_mlflow_experiment() -> bool:
    """Configure MLflow tracking URI and experiment once per tracking endpoint."""
    global _configured_mlflow_uri
    if not _is_mlflow_server_reachable():
        return False
    if _configured_mlflow_uri != MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("ragscope")
        _configured_mlflow_uri = MLFLOW_TRACKING_URI
    return True


def _update_trace_sqlite(
    trace_id: str | None,
    execution_time_ms: float | None = None,
    tags: dict[str, str] | None = None,
    db_path: str | None = None,
) -> None:
    """Persist authentic total query latency and telemetry tags directly onto the trace row in SQLite."""
    target_db = db_path or MLFLOW_DB_PATH
    if not trace_id or not os.path.exists(target_db):
        return
    try:
        conn = sqlite3.connect(target_db, timeout=5)
        cur = conn.cursor()
        if execution_time_ms is not None:
            cur.execute(
                "UPDATE trace_info SET execution_time_ms = ? WHERE request_id = ?",
                (max(1, int(round(execution_time_ms))), trace_id),
            )
        if tags:
            for k, v in tags.items():
                if v is not None:
                    cur.execute(
                        "INSERT OR REPLACE INTO trace_tags (key, value, request_id) VALUES (?, ?, ?)",
                        (str(k), str(v), trace_id),
                    )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.debug(f"Could not update trace metadata in SQLite for {trace_id}: {exc}")


def _get_existing_assessment_names(trace_id: str | None, db_path: str | None = None) -> set[str]:
    """Return lowercase assessment metric names already recorded on a trace."""
    target_db = db_path or MLFLOW_DB_PATH
    if not trace_id or not os.path.exists(target_db):
        return set()
    try:
        conn = sqlite3.connect(target_db, timeout=5)
        rows = conn.execute(
            "SELECT DISTINCT LOWER(name) FROM assessments WHERE trace_id = ? AND LOWER(name) != 'context'",
            (trace_id,),
        ).fetchall()
        conn.close()
        return {str(r[0]) for r in rows if r and r[0]}
    except Exception:
        return set()


def _insert_assessment_sqlite(
    trace_id: str,
    name: str,
    value: str,
    rationale: str,
    db_path: str | None = None,
) -> bool:
    """Insert a fallback MLflow feedback assessment row directly into SQLite when server API is unavailable."""
    target_db = db_path or MLFLOW_DB_PATH
    if not trace_id or not os.path.exists(target_db):
        return False
    now_ms = int(time.time() * 1000)
    assess_id = f"a-{uuid.uuid4().hex[:16]}"
    try:
        conn = sqlite3.connect(target_db, timeout=5)
        conn.execute(
            """
            INSERT INTO assessments (
                assessment_id, trace_id, name, assessment_type, value,
                error, created_timestamp, last_updated_timestamp,
                source_type, source_id, rationale, valid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                assess_id,
                trace_id,
                name,
                "FEEDBACK",
                json.dumps(value),
                None,
                now_ms,
                now_ms,
                "CODE",
                "ragscope.evaluate",
                rationale,
                1,
            ),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as exc:
        logger.debug(f"Could not insert fallback assessment {name} for {trace_id}: {exc}")
        return False


def _wait_for_trace_in_db(trace_id: str, max_attempts: int = 8) -> bool:
    if not trace_id or not os.path.exists(MLFLOW_DB_PATH):
        return False
    for _ in range(max_attempts):
        try:
            conn = sqlite3.connect(MLFLOW_DB_PATH, timeout=2)
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM trace_info WHERE request_id = ?", (trace_id,))
            exists = cur.fetchone() is not None
            conn.close()
            if exists:
                return True
        except Exception:
            pass
        time.sleep(0.15)
    return False


def record_query_trace(
    *,
    question: str,
    answer: str,
    model: str,
    prompt_id: str,
    prompt_version: str,
    document_scope: str,
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    total_latency_ms: float,
    candidates_fetched: int | None,
    unique_chunks_returned: int | None,
    duplicates_removed: int,
    use_mmr: bool,
    context_char_count: int,
    token_count_estimate: int,
    tokens_per_second: float | None,
    has_sources: bool,
) -> tuple[str | None, str, str]:
    """Create a single authoritative MLflow trace with real query execution time and telemetry tags.

    Returns (trace_id, evaluation_job_id, initial_evaluation_status).
    """
    global _latest_queued_job_id
    job_id = f"eval-{uuid.uuid4().hex[:12]}"
    initial_state = "pending" if has_sources else "skipped"
    trace_id: str | None = None

    tags = {
        "model": model,
        "prompt_id": prompt_id,
        "prompt_version": prompt_version,
        "document_scope": document_scope,
        "retrieval_latency_ms": f"{round(retrieval_latency_ms, 2)}",
        "generation_latency_ms": f"{round(generation_latency_ms, 2)}",
        "total_query_latency_ms": f"{round(total_latency_ms, 2)}",
        "candidates_fetched": str(candidates_fetched if candidates_fetched is not None else 0),
        "unique_chunks_returned": str(unique_chunks_returned if unique_chunks_returned is not None else 0),
        "duplicates_removed": str(duplicates_removed),
        "use_mmr": "true" if use_mmr else "false",
        "context_char_count": str(context_char_count),
        "token_count_estimate": str(token_count_estimate),
        "tokens_per_second": f"{round(tokens_per_second, 2)}" if tokens_per_second is not None else "",
        "evaluation_job_id": job_id,
        "evaluation_status": initial_state,
        "evaluation_duration_ms": "0",
        "evaluation_owner": _get_process_owner_token(),
    }

    if _ensure_mlflow_experiment():
        try:
            with mlflow.start_span(name="root_span", span_type=SpanType.CHAIN) as span:
                span.set_inputs({"question": question})
                span.set_outputs({"outputs": answer})
                span.set_attributes(
                    {
                        "model": model,
                        "prompt_id": prompt_id,
                        "prompt_version": prompt_version,
                        "document_scope": document_scope,
                    }
                )
                trace_id = span.request_id

            try:
                mlflow.flush_trace_async_logging()
            except Exception:
                pass

            if trace_id and _wait_for_trace_in_db(trace_id):
                _update_trace_sqlite(
                    trace_id=trace_id,
                    execution_time_ms=total_latency_ms,
                    tags=tags,
                )
                if not has_sources:
                    for metric_name in ("AnswerRelevancy", "Hallucination", "Safety"):
                        try:
                            mlflow.log_feedback(
                                trace_id=trace_id,
                                name=metric_name,
                                value="skipped",
                                rationale="Skipped: No source context retrieved",
                            )
                        except Exception:
                            pass
        except Exception as exc:
            logger.warning(f"Could not log MLflow query trace: {exc}")

    job = EvaluationJob(
        job_id=job_id,
        trace_id=trace_id,
        question=question,
        answer=answer,
        state=initial_state,
        created_at_ms=int(time.time() * 1000),
        reason=None if has_sources else "Skipped: No source context retrieved",
    )
    _remember_job(job)
    EVAL_JOBS_TOTAL.labels(state=initial_state).inc()

    if has_sources:
        with _state_lock:
            _latest_queued_job_id = job_id
            _pending_jobs_by_qa[(question, answer)] = job
            while len(_pending_jobs_by_qa) > _MAX_TRACKED_JOBS:
                _pending_jobs_by_qa.popitem(last=False)

    return trace_id, job_id, initial_state


def _pop_or_create_job(question: str, answer: str) -> EvaluationJob:
    with _state_lock:
        job = _pending_jobs_by_qa.pop((question, answer), None)
    if job is not None:
        return job
    job = EvaluationJob(
        job_id=f"eval-{uuid.uuid4().hex[:12]}",
        trace_id=None,
        question=question,
        answer=answer,
        state="pending",
        created_at_ms=int(time.time() * 1000),
    )
    _remember_job(job)
    return job


def _log_metric_feedback(
    trace_id: str | None,
    name: str,
    value: str,
    rationale: str,
    db_path: str | None = None,
) -> None:
    if not trace_id:
        return
    target_db = db_path or MLFLOW_DB_PATH
    try:
        # Use MLflow API only when not targeting an alternate SQLite DB and server is reachable
        if (db_path is None or os.path.abspath(db_path) == os.path.abspath(MLFLOW_DB_PATH)) and _is_mlflow_server_reachable():
            mlflow.log_feedback(trace_id=trace_id, name=name, value=value, rationale=rationale)
            return
    except Exception as exc:
        logger.debug(f"Could not log feedback {name} via MLflow client for {trace_id}: {exc}")
    _insert_assessment_sqlite(trace_id, name, value, rationale, db_path=target_db)


def _record_terminal_trace_feedback(
    trace_id: str | None,
    question: str,
    answer: str,
    completed_metrics: dict[str, tuple[str, str]],
    terminal_state: str,
    reason: str,
    already_logged: set[str] | None = None,
    db_path: str | None = None,
) -> None:
    """Attach finished and unfilled scorer statuses onto the query trace without losing completed ones."""
    if not trace_id and _is_mlflow_server_reachable():
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("ragscope")
            with mlflow.start_span(name="root_span", span_type=SpanType.CHAIN) as span:
                span.set_inputs({"question": question})
                span.set_outputs({"outputs": answer})
                trace_id = span.request_id
            mlflow.flush_trace_async_logging()
            _wait_for_trace_in_db(trace_id)
        except Exception:
            trace_id = None

    if not trace_id:
        return

    logged_lower = {m.lower() for m in (already_logged or set())}
    logged_lower |= _get_existing_assessment_names(trace_id, db_path=db_path)

    fallback_value = (
        "cancelled"
        if terminal_state == "cancelled"
        else ("error" if terminal_state == "failed" else "skipped")
    )
    for metric in _ALL_EVAL_METRICS:
        if metric.lower() in logged_lower:
            continue
        if metric in completed_metrics:
            val, rat = completed_metrics[metric]
            _log_metric_feedback(trace_id, metric, val, rat, db_path=db_path)
        else:
            _log_metric_feedback(trace_id, metric, fallback_value, reason, db_path=db_path)


def run_judge_evaluations(
    question: str,
    answer: str,
    context_chunks: list[str],
    skip_reason: str | None = None,
) -> None:
    """Run background LLM-as-a-Judge evaluation cleanly linked to the originating query trace."""
    global _is_evaluating, _current_job

    job = _pop_or_create_job(question, answer)

    if skip_reason or not context_chunks:
        reason = skip_reason or "Skipped: No source context retrieved"
        job.state = "skipped"
        job.reason = reason
        job.duration_ms = 0.0
        EVAL_JOBS_TOTAL.labels(state="skipped").inc()
        _record_terminal_trace_feedback(job.trace_id, question, answer, {}, "skipped", reason)
        _update_trace_sqlite(
            trace_id=job.trace_id,
            tags={"evaluation_status": "skipped", "evaluation_duration_ms": "0"},
        )
        return

    # Skip stale queued job immediately if a newer interactive query has superseded it
    with _state_lock:
        superseded = _latest_queued_job_id is not None and _latest_queued_job_id != job.job_id
    if superseded:
        reason = "Skipped: Superseded by newer incoming query"
        job.state = "skipped"
        job.reason = reason
        job.duration_ms = 0.0
        EVAL_JOBS_TOTAL.labels(state="skipped").inc()
        _record_terminal_trace_feedback(job.trace_id, question, answer, {}, "skipped", reason)
        _update_trace_sqlite(
            trace_id=job.trace_id,
            tags={"evaluation_status": "skipped", "evaluation_duration_ms": "0"},
        )
        return

    # Wait briefly for an unwinding preempted worker; never pile up long waits
    if not _eval_lock.acquire(blocking=True, timeout=8):
        reason = "Skipped: Concurrent evaluation still unwinding after preemption"
        logger.info(f"{reason} ('{question[:50]}')")
        job.state = "skipped"
        job.reason = reason
        job.duration_ms = 0.0
        EVAL_JOBS_TOTAL.labels(state="skipped").inc()
        _record_terminal_trace_feedback(job.trace_id, question, answer, {}, "skipped", reason)
        _update_trace_sqlite(
            trace_id=job.trace_id,
            tags={"evaluation_status": "skipped", "evaluation_duration_ms": "0"},
        )
        return

    # Re-check whether superseded while waiting for lock
    with _state_lock:
        superseded_while_waiting = (
            _latest_queued_job_id is not None and _latest_queued_job_id != job.job_id
        )
    if superseded_while_waiting:
        _eval_lock.release()
        reason = "Skipped: Superseded by newer incoming query"
        job.state = "skipped"
        job.reason = reason
        job.duration_ms = 0.0
        EVAL_JOBS_TOTAL.labels(state="skipped").inc()
        _record_terminal_trace_feedback(job.trace_id, question, answer, {}, "skipped", reason)
        _update_trace_sqlite(
            trace_id=job.trace_id,
            tags={"evaluation_status": "skipped", "evaluation_duration_ms": "0"},
        )
        return

    _cancel_eval_event.clear()
    _is_evaluating = True
    EVAL_JOBS_ACTIVE.set(1.0)
    EVAL_JOBS_TOTAL.labels(state="running").inc()
    with _state_lock:
        _current_job = job
    job.state = "running"
    _eval_thread_local.in_eval_worker = True

    _update_trace_sqlite(
        trace_id=job.trace_id,
        tags={
            "evaluation_job_id": job.job_id,
            "evaluation_status": "running",
            "evaluation_owner": _get_process_owner_token(),
        },
    )

    t_eval_start = time.perf_counter()
    completed_metrics: dict[str, tuple[str, str]] = {}
    streamed_metrics: set[str] = set()
    preempted = False
    error_count = 0
    last_error_msg = ""

    try:
        _ensure_mlflow_experiment()

        judge_llm = f"ollama:/{OLLAMA_JUDGE_MODEL}"
        eval_context = format_context_for_llm(context_chunks, max_chars=_EVAL_CONTEXT_CHAR_LIMIT)

        scorers = [
            ("AnswerRelevancy", lambda: AnswerRelevancy(model=judge_llm)(
                inputs={"question": question},
                outputs={"outputs": answer},
                expectations={"context": eval_context},
            )),
            ("Hallucination", lambda: Hallucination(model=judge_llm)(
                inputs={"question": question},
                outputs={"outputs": answer},
                expectations={"context": eval_context},
            )),
            ("Safety", lambda: Safety(model=judge_llm)(outputs=answer)),
        ]

        logger.info(f"Running background judge evaluation [{job.job_id}] for: '{question[:60]}'")

        for metric_name, invoke_fn in scorers:
            if _cancel_eval_event.is_set() or job.cancel_requested:
                raise EvaluationPreemptedError(f"Cancelled before {metric_name} scorer.")
            t_scorer0 = time.perf_counter()
            scorer_outcome = "completed"
            try:
                feedback = invoke_fn()
                has_valid_feedback = (
                    not getattr(feedback, "error", None)
                    and getattr(feedback, "value", None) is not None
                )
                was_cancelled = _cancel_eval_event.is_set() or job.cancel_requested

                if has_valid_feedback:
                    val_str = str(feedback.value).strip('"')
                    rat_str = str(getattr(feedback, "rationale", None) or "")
                    completed_metrics[metric_name] = (val_str, rat_str)
                    _log_metric_feedback(job.trace_id, metric_name, val_str, rat_str)
                    streamed_metrics.add(metric_name)
                    if was_cancelled:
                        raise EvaluationPreemptedError(
                            f"Cancelled after {metric_name} scorer completed."
                        )
                elif was_cancelled:
                    scorer_outcome = "cancelled"
                    raise EvaluationPreemptedError(f"Cancelled during {metric_name} scorer.")
                else:
                    err_detail = str(getattr(feedback, "error", None) or "Scorer returned no value")
                    scorer_outcome = "error"
                    error_count += 1
                    last_error_msg = err_detail
                    rat_err = f"Evaluation error: {err_detail}"
                    completed_metrics[metric_name] = ("error", rat_err)
                    _log_metric_feedback(job.trace_id, metric_name, "error", rat_err)
                    streamed_metrics.add(metric_name)
            except EvaluationPreemptedError:
                if metric_name not in completed_metrics:
                    scorer_outcome = "cancelled"
                raise
            except Exception as scorer_exc:
                if _cancel_eval_event.is_set() or job.cancel_requested:
                    scorer_outcome = "cancelled"
                    raise EvaluationPreemptedError(f"Cancelled during {metric_name}: {scorer_exc}") from scorer_exc
                scorer_outcome = "error"
                error_count += 1
                last_error_msg = str(scorer_exc)
                logger.warning(f"Judge scorer {metric_name} failed: {scorer_exc}")
                rat_err = f"Evaluation failed: {scorer_exc}"
                completed_metrics[metric_name] = ("error", rat_err)
                _log_metric_feedback(job.trace_id, metric_name, "error", rat_err)
                streamed_metrics.add(metric_name)
            finally:
                EVAL_SCORER_DURATION_SECONDS.labels(
                    metric=metric_name, outcome=scorer_outcome
                ).observe(max(0.0, time.perf_counter() - t_scorer0))

    except EvaluationPreemptedError as exc:
        preempted = True
        logger.info(f"Background judge evaluation [{job.job_id}] cancelled cleanly: {exc}")
    except Exception as exc:
        error_count += 1
        last_error_msg = str(exc)
        logger.error(f"Background judge evaluation [{job.job_id}] failed: {exc}", exc_info=True)
    finally:
        eval_duration_ms = round((time.perf_counter() - t_eval_start) * 1000.0, 2)
        job.duration_ms = eval_duration_ms
        preempted = preempted or _cancel_eval_event.is_set() or job.cancel_requested

        if preempted:
            final_state = "cancelled"
            terminal_reason = job.reason or "Cancelled: Evaluation preempted by new incoming query"
        elif error_count >= 3 or (not completed_metrics and last_error_msg):
            final_state = "failed"
            terminal_reason = f"Failed: {last_error_msg or 'Evaluation error'}"
        else:
            final_state = "completed"
            terminal_reason = "Completed"

        job.state = final_state
        EVAL_JOBS_TOTAL.labels(state=final_state).inc()
        EVAL_DURATION_SECONDS.labels(outcome=final_state).observe(
            max(0.0, eval_duration_ms / 1000.0)
        )
        _record_terminal_trace_feedback(
            trace_id=job.trace_id,
            question=question,
            answer=answer,
            completed_metrics=completed_metrics,
            terminal_state=final_state,
            reason=terminal_reason,
            already_logged=streamed_metrics,
        )
        _update_trace_sqlite(
            trace_id=job.trace_id,
            tags={
                "evaluation_job_id": job.job_id,
                "evaluation_status": final_state,
                "evaluation_duration_ms": f"{eval_duration_ms}",
            },
        )
        logger.info(
            f"Background judge evaluation [{job.job_id}] finished: state={final_state}, duration={eval_duration_ms}ms"
        )

        _eval_thread_local.in_eval_worker = False
        _eval_thread_local.in_eval_llm_call = False
        with _eval_sockets_lock:
            _eval_sockets.clear()
        with _state_lock:
            if _current_job is job:
                _current_job = None
        _is_evaluating = False
        EVAL_JOBS_ACTIVE.set(0.0)
        _cancel_eval_event.clear()
        _eval_lock.release()


def shutdown_evaluation_cleanup(wait_timeout: float = 3.0) -> int:
    """Cancel active evaluation and mark remaining pending/running in-process jobs as cancelled."""
    global _latest_queued_job_id
    reason = "Cancelled: Backend server shut down before background evaluation completed"
    cancelled_count = 0

    with _state_lock:
        if _current_job is not None:
            _current_job.cancel_requested = True
            _current_job.reason = reason
    cancel_current_evaluation()

    # Wait briefly for an active worker's finally block to unwind and flush SQLite state
    acquired = _eval_lock.acquire(blocking=True, timeout=max(0.0, wait_timeout))
    if acquired:
        _eval_lock.release()

    with _state_lock:
        _latest_queued_job_id = None
        _pending_jobs_by_qa.clear()
        orphaned_jobs = [
            j for j in _eval_jobs.values() if j.state in ("pending", "running")
        ]

    for job in orphaned_jobs:
        job.state = "cancelled"
        job.reason = reason
        if job.duration_ms is None:
            job.duration_ms = 0.0
        EVAL_JOBS_TOTAL.labels(state="cancelled").inc()
        if job.trace_id:
            _record_terminal_trace_feedback(
                trace_id=job.trace_id,
                question=job.question,
                answer=job.answer,
                completed_metrics={},
                terminal_state="cancelled",
                reason=reason,
            )
            _update_trace_sqlite(
                trace_id=job.trace_id,
                tags={
                    "evaluation_status": "cancelled",
                    "evaluation_duration_ms": f"{job.duration_ms}",
                },
            )
        cancelled_count += 1

    if cancelled_count:
        logger.info(f"Shutdown evaluation cleanup transitioned {cancelled_count} pending/running jobs to cancelled.")
    return cancelled_count


def reconcile_orphaned_evaluations(db_path: str | None = None) -> int:
    """Mark interrupted running/pending evaluations owned by dead processes as cancelled on startup.

    Traces without an `evaluation_owner` tag (legacy/historical rows) or owned by another still-live
    backend process are left strictly untouched.
    """
    target_db = db_path or MLFLOW_DB_PATH
    if not os.path.exists(target_db):
        return 0

    reason = "Cancelled: Previous backend process exited before evaluation completed"
    reconciled = 0
    try:
        conn = sqlite3.connect(target_db, timeout=5)
        rows = conn.execute(
            """
            SELECT st.request_id, st.value, own.value
            FROM trace_tags st
            JOIN trace_tags own
              ON st.request_id = own.request_id AND own.key = 'evaluation_owner'
            WHERE st.key = 'evaluation_status'
              AND st.value IN ('running', 'pending')
            """
        ).fetchall()
        conn.close()
    except Exception as exc:
        logger.warning(f"Could not inspect orphaned evaluations in {target_db}: {exc}")
        return 0

    for trace_id, _status_val, owner_token in rows:
        if _is_owner_process_alive(owner_token):
            continue
        existing = _get_existing_assessment_names(trace_id, db_path=target_db)
        for metric in _ALL_EVAL_METRICS:
            if metric.lower() not in existing:
                _log_metric_feedback(trace_id, metric, "cancelled", reason, db_path=target_db)
        _update_trace_sqlite(
            trace_id=trace_id,
            tags={"evaluation_status": "cancelled"},
            db_path=target_db,
        )
        reconciled += 1

    if reconciled:
        logger.info(f"Startup recovery marked {reconciled} orphaned evaluation(s) as cancelled.")
    return reconciled
