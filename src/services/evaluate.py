import json
import os
import re
import sqlite3
import threading
import time

import requests
import mlflow
from mlflow.entities import SpanType
from mlflow.exceptions import MlflowException
from mlflow.genai import evaluate
from mlflow.genai.scorers import Safety
from mlflow.genai.scorers.deepeval import AnswerRelevancy, Hallucination
import mlflow.metrics.genai.model_utils as _model_utils

try:
    import mlflow.genai.scorers.deepeval.models as _deepeval_models
except ImportError:
    _deepeval_models = None

import urllib3.connection

from src.utils.env import MLFLOW_TRACKING_URI, OLLAMA_JUDGE_MODEL
from src.utils.log_manager import logger

os.environ.setdefault("CONFIDENT_METRIC_LOGGING_VERBOSE", "0")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", "1")

# Preemption and lifecycle management
_eval_lock = threading.Lock()
_is_evaluating = False
_cancel_eval_event = threading.Event()
_eval_sockets: set = set()
_orig_http_connect = urllib3.connection.HTTPConnection.connect

def _patched_http_connect(self):
    _orig_http_connect(self)
    if getattr(self, "_is_eval_conn", False) and self.sock:
        _eval_sockets.add(self.sock)

urllib3.connection.HTTPConnection.connect = _patched_http_connect


class EvaluationPreemptedError(Exception):
    """Raised when an ongoing evaluation is preempted by a new user query."""
    pass


def is_evaluation_running() -> bool:
    """Return True if a judge evaluation is currently executing."""
    return _is_evaluating


def cancel_current_evaluation() -> None:
    """Signal current evaluation to abort and shut down active sockets to free Ollama instantly."""
    _cancel_eval_event.set()
    for s in list(_eval_sockets):
        try:
            s.shutdown(2)
            s.close()
        except Exception:
            pass
    _eval_sockets.clear()


def _clean_and_repair_json(text: str) -> str:
    """Strip markdown codeblocks and repair truncated or unclosed JSON from local LLMs."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
    else:
        candidate = text

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


def _patched_send_request(endpoint, headers, payload):
    if _cancel_eval_event.is_set():
        raise EvaluationPreemptedError("Evaluation preempted by new incoming query.")

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

    try:
        response = session.post(
            url=endpoint,
            headers=headers,
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if _cancel_eval_event.is_set():
            raise EvaluationPreemptedError("Evaluation preempted by new incoming query.") from e
        body = getattr(getattr(e, "response", None), "text", "")
        raise MlflowException(
            f"Failed to call LLM endpoint at {endpoint}.\n- Error: {e}\n"
            f"- Response body: {body}\n- Input payload: {payload}."
        ) from e
    finally:
        _eval_sockets.clear()
        session.close()


if hasattr(_model_utils, "_send_request"):
    _model_utils._send_request = _patched_send_request


def _record_skipped_evaluation(question: str, answer: str, reason: str) -> None:
    """Record an explicit skipped evaluation trace in MLflow with reason."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("ragscope")
        with mlflow.start_span(name="root_span", span_type=SpanType.CHAIN) as span:
            span.set_inputs({"question": question})
            span.set_outputs({"outputs": answer})
            trace_id = span.request_id

        # Allow async export to register trace in SQLite store before adding feedback
        db_path = "D:/mlops cp github/ragscope/mlflow.db"
        for _ in range(10):
            time.sleep(0.5)
            try:
                if os.path.exists(db_path):
                    conn = sqlite3.connect(db_path, timeout=2)
                    c = conn.cursor()
                    c.execute("SELECT 1 FROM trace_info WHERE request_id = ?", (trace_id,))
                    exists = c.fetchone() is not None
                    conn.close()
                    if exists:
                        for metric_name in ["AnswerRelevancy", "Hallucination", "Safety"]:
                            mlflow.log_feedback(
                                trace_id=trace_id,
                                name=metric_name,
                                value="skipped",
                                rationale=reason,
                            )
                        logger.info(f"Recorded skipped evaluation trace {trace_id}: {reason}")
                        return
            except Exception:
                pass
    except Exception as exc:
        logger.warning(f"Could not record skipped evaluation in MLflow: {exc}")


def _normalize_eval_assessments(preempted: bool) -> None:
    """Ensure no trace assessments are left as null by converting skips and errors to explicit feedback."""
    try:
        db_path = "D:/mlops cp github/ragscope/mlflow.db"
        if not os.path.exists(db_path):
            return
        conn = sqlite3.connect(db_path, timeout=5)
        c = conn.cursor()
        if preempted:
            c.execute("""
                UPDATE assessments
                SET value = '"skipped"',
                    rationale = 'Skipped: Evaluation preempted by new incoming query'
                WHERE value IS NULL OR value = 'null'
            """)
        else:
            c.execute("""
                UPDATE assessments
                SET value = '"error"',
                    rationale = CASE 
                        WHEN rationale IS NOT NULL AND rationale != '' THEN rationale 
                        ELSE 'Evaluation failed during scorer execution' 
                    END
                WHERE value IS NULL OR value = 'null'
            """)
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning(f"Could not normalize assessments in SQLite: {exc}")


def run_judge_evaluations(
    question: str,
    answer: str,
    context_chunks: list[str],
    skip_reason: str | None = None,
) -> None:
    global _is_evaluating

    if skip_reason or not context_chunks:
        reason = skip_reason or "Skipped: No source context retrieved"
        _record_skipped_evaluation(question, answer, reason)
        return

    # Wait up to 30 seconds for any preempted evaluation to release the lock
    if not _eval_lock.acquire(blocking=True, timeout=30):
        logger.info(f"Another judge evaluation is currently running. Skipping evaluation for '{question[:50]}'")
        _record_skipped_evaluation(question, answer, "Skipped: Preempted by concurrent evaluation task")
        return

    _is_evaluating = True
    _cancel_eval_event.clear()
    preempted = False

    try:
        judge_llm = f"ollama:/{OLLAMA_JUDGE_MODEL}"
        scorers = [
            AnswerRelevancy(model=judge_llm),
            Hallucination(model=judge_llm),
            Safety(model=judge_llm),
        ]

        # Bounding context to ~1500 chars keeps prompt tokens small (~350 tokens) on CPU
        eval_context = "\n\n".join(context_chunks)
        if len(eval_context) > 1500:
            eval_context = eval_context[:1500]

        eval_data = [
            {
                "inputs": {"question": question},
                "outputs": {"outputs": answer},
                "expectations": {"context": eval_context},
            },
        ]

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("ragscope")
        logger.info(f"Running background judge evaluations for: '{question[:60]}'")
        evaluate(data=eval_data, scorers=scorers)
        logger.info("Background judge evaluations completed successfully.")
    except EvaluationPreemptedError as exc:
        preempted = True
        logger.info(f"Judge evaluation successfully preempted by subsequent query: {exc}")
    except Exception as exc:
        logger.error(f"Judge evaluation failed with error: {exc}", exc_info=True)
        try:
            _record_skipped_evaluation(
                question,
                answer,
                f"Failed: Evaluation error - {exc}",
            )
        except Exception:
            pass
    finally:
        preempted = preempted or _cancel_eval_event.is_set()
        time.sleep(0.5)
        _normalize_eval_assessments(preempted=preempted)
        _is_evaluating = False
        _cancel_eval_event.clear()
        _eval_lock.release()
