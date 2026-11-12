import json
import os
import sqlite3
from pathlib import Path

from src.models import (
    AssessmentDetail,
    LatencyTrendItem,
    ModelPerformanceItem,
    PerformanceSummaryResponse,
    TraceAuditItem,
)
from src.utils.log_manager import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = str(PROJECT_ROOT / "mlflow.db")

_TRACE_TAG_KEYS = (
    "prompt_id",
    "prompt_version",
    "model",
    "document_scope",
    "retrieval_latency_ms",
    "generation_latency_ms",
    "total_query_latency_ms",
    "candidates_fetched",
    "unique_chunks_returned",
    "duplicates_removed",
    "use_mmr",
    "context_char_count",
    "token_count_estimate",
    "tokens_per_second",
    "evaluation_job_id",
    "evaluation_status",
    "evaluation_duration_ms",
)


def _to_float(val: str | None) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _to_int(val: str | None) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _to_bool(val: str | None) -> bool | None:
    if val is None or val == "":
        return None
    low = val.strip().lower()
    if low in ("true", "1", "yes"):
        return True
    if low in ("false", "0", "no"):
        return False
    return None


def _infer_eval_status(
    explicit_status: str | None, assessments: dict[str, AssessmentDetail]
) -> str | None:
    if explicit_status:
        return explicit_status
    if not assessments:
        return "pending"
    vals = [a.value.lower() for a in assessments.values()]
    if any(v == "cancelled" for v in vals):
        return "cancelled"
    if all(v == "skipped" for v in vals):
        return "skipped"
    if any(v == "error" for v in vals) and not any(v in ("yes", "no") for v in vals):
        return "failed"
    if any(v in ("yes", "no") for v in vals):
        return "completed"
    return "skipped"


def get_recent_traces(limit: int = 25) -> list[TraceAuditItem]:
    """Retrieve the latest MLflow traces with assessments, timings, and telemetry metadata from SQLite."""
    if not os.path.exists(DB_PATH):
        logger.warning(f"MLflow database not found at {DB_PATH}")
        return []

    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        c = conn.cursor()

        c.execute(
            """
            SELECT request_id, timestamp_ms, execution_time_ms, status
            FROM trace_info
            ORDER BY timestamp_ms DESC
            LIMIT ?
            """,
            (limit,),
        )
        trace_rows = c.fetchall()
        if not trace_rows:
            conn.close()
            return []

        trace_ids = [r[0] for r in trace_rows]
        placeholders = ",".join("?" for _ in trace_ids)

        c.execute(
            f"""
            SELECT trace_id, content
            FROM spans
            WHERE trace_id IN ({placeholders}) AND name = 'root_span'
            """,
            trace_ids,
        )
        span_rows = c.fetchall()
        io_map: dict[str, tuple[str | None, str | None]] = {}
        for tid, content_str in span_rows:
            q_text, a_text = None, None
            try:
                content_obj = json.loads(content_str)
                attrs = content_obj.get("attributes", {})
                raw_inputs = attrs.get("mlflow.spanInputs")
                if raw_inputs:
                    inp_obj = json.loads(raw_inputs) if isinstance(raw_inputs, str) else raw_inputs
                    q_text = inp_obj.get("question")
                raw_outputs = attrs.get("mlflow.spanOutputs")
                if raw_outputs:
                    out_obj = json.loads(raw_outputs) if isinstance(raw_outputs, str) else raw_outputs
                    a_text = out_obj.get("outputs") or out_obj.get("answer")
            except Exception:
                pass
            io_map[tid] = (q_text, a_text)

        c.execute(
            f"""
            SELECT trace_id, name, value, rationale
            FROM assessments
            WHERE trace_id IN ({placeholders}) AND LOWER(name) != 'context'
            """,
            trace_ids,
        )
        assess_rows = c.fetchall()
        assessments_map: dict[str, dict[str, AssessmentDetail]] = {}
        for tid, name, val, rat in assess_rows:
            if tid not in assessments_map:
                assessments_map[tid] = {}
            clean_val = str(val).strip('"') if val is not None else "null"
            clean_rat = str(rat) if rat is not None else None
            display_name = "Safety" if name.lower() == "safety" else name
            assessments_map[tid][display_name] = AssessmentDetail(
                value=clean_val,
                rationale=clean_rat,
            )

        tag_placeholders = ",".join("?" for _ in _TRACE_TAG_KEYS)
        tags_map: dict[str, dict[str, str]] = {}
        try:
            c.execute(
                f"""
                SELECT request_id, key, value
                FROM trace_tags
                WHERE request_id IN ({placeholders})
                  AND key IN ({tag_placeholders})
                """,
                [*trace_ids, *_TRACE_TAG_KEYS],
            )
            for req_id, key, val in c.fetchall():
                tags_map.setdefault(req_id, {})[key] = val
        except Exception:
            pass

        conn.close()

        results: list[TraceAuditItem] = []
        for req_id, ts, exec_ms, status in trace_rows:
            q_text, a_text = io_map.get(req_id, (None, None))
            assess_dict = assessments_map.get(req_id, {})
            trace_tags = tags_map.get(req_id, {})

            tagged_total_ms = _to_float(trace_tags.get("total_query_latency_ms"))
            effective_exec_ms = (
                tagged_total_ms
                if tagged_total_ms is not None and tagged_total_ms > 0
                else float(exec_ms or 0.0)
            )
            eval_status = _infer_eval_status(
                trace_tags.get("evaluation_status"), assess_dict
            )

            results.append(
                TraceAuditItem(
                    request_id=req_id,
                    timestamp_ms=ts,
                    execution_time_ms=round(effective_exec_ms, 2),
                    status=status or "UNKNOWN",
                    question=q_text,
                    answer=a_text,
                    assessments=assess_dict,
                    prompt_id=trace_tags.get("prompt_id"),
                    prompt_version=trace_tags.get("prompt_version"),
                    model=trace_tags.get("model"),
                    document_scope=trace_tags.get("document_scope"),
                    retrieval_latency_ms=_to_float(trace_tags.get("retrieval_latency_ms")),
                    generation_latency_ms=_to_float(trace_tags.get("generation_latency_ms")),
                    candidates_fetched=_to_int(trace_tags.get("candidates_fetched")),
                    unique_chunks_returned=_to_int(trace_tags.get("unique_chunks_returned")),
                    duplicates_removed=_to_int(trace_tags.get("duplicates_removed")),
                    use_mmr=_to_bool(trace_tags.get("use_mmr")),
                    context_char_count=_to_int(trace_tags.get("context_char_count")),
                    token_count_estimate=_to_int(trace_tags.get("token_count_estimate")),
                    tokens_per_second=_to_float(trace_tags.get("tokens_per_second")),
                    evaluation_job_id=trace_tags.get("evaluation_job_id"),
                    evaluation_status=eval_status,
                    evaluation_duration_ms=_to_float(trace_tags.get("evaluation_duration_ms")),
                )
            )

        return results

    except Exception as exc:
        logger.error(f"Error querying MLflow trace audit log: {exc}")
        return []


def _percentile(sorted_vals: list[float], pct: float) -> float | None:
    """Compute Type-7 (R-7 / NumPy default) linear-interpolated percentile on ascending values."""
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    if n == 1:
        return round(sorted_vals[0], 2)
    pos = (max(0.0, min(100.0, pct)) / 100.0) * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    val = sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])
    return round(val, 2)


def get_performance_summary(limit: int = 100) -> PerformanceSummaryResponse:
    """Compute real aggregate query, retrieval, generation, throughput, and judge evaluation metrics."""
    traces = get_recent_traces(limit=limit)

    eval_counts: dict[str, int] = {
        "pending": 0,
        "running": 0,
        "completed": 0,
        "skipped": 0,
        "cancelled": 0,
        "failed": 0,
    }
    for t in traces:
        st = (t.evaluation_status or "pending").lower()
        eval_counts[st] = eval_counts.get(st, 0) + 1

    # Exclude legacy Phase 1-2 MLflow wrapper rows (0-2 ms with no generation telemetry) from latency percentiles
    measured = [
        t
        for t in traces
        if t.generation_latency_ms is not None or t.execution_time_ms > 10.0
    ]

    total_latencies = sorted(t.execution_time_ms for t in measured if t.execution_time_ms > 0)
    retr_latencies = [
        t.retrieval_latency_ms for t in measured if t.retrieval_latency_ms is not None
    ]
    gen_latencies = [
        t.generation_latency_ms for t in measured if t.generation_latency_ms is not None
    ]
    eval_durations = [
        t.evaluation_duration_ms
        for t in traces
        if t.evaluation_duration_ms is not None and t.evaluation_duration_ms > 0
    ]
    tps_vals = [
        t.tokens_per_second
        for t in measured
        if t.tokens_per_second is not None and t.tokens_per_second > 0
    ]

    avg_query = round(sum(total_latencies) / len(total_latencies), 2) if total_latencies else None
    min_query = round(total_latencies[0], 2) if total_latencies else None
    med_query = _percentile(total_latencies, 50.0)
    p90_query = _percentile(total_latencies, 90.0)
    p95_query = _percentile(total_latencies, 95.0)
    max_query = round(total_latencies[-1], 2) if total_latencies else None
    avg_retr = round(sum(retr_latencies) / len(retr_latencies), 2) if retr_latencies else None
    avg_gen = round(sum(gen_latencies) / len(gen_latencies), 2) if gen_latencies else None
    avg_eval = round(sum(eval_durations) / len(eval_durations), 2) if eval_durations else None
    avg_tps = round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else None

    finished_evals = (
        eval_counts["completed"]
        + eval_counts["skipped"]
        + eval_counts["cancelled"]
        + eval_counts["failed"]
    )
    eval_completion_rate = (
        round((eval_counts["completed"] / finished_evals) * 100.0, 1)
        if finished_evals > 0
        else None
    )

    model_groups: dict[str, list[TraceAuditItem]] = {}
    for t in measured:
        m_name = t.model or "llama3.2"
        model_groups.setdefault(m_name, []).append(t)

    model_breakdown: list[ModelPerformanceItem] = []
    for m_name, items in sorted(model_groups.items()):
        gens = [i.generation_latency_ms for i in items if i.generation_latency_ms is not None]
        totals = [i.execution_time_ms for i in items if i.execution_time_ms > 0]
        tpss = [i.tokens_per_second for i in items if i.tokens_per_second is not None]
        model_breakdown.append(
            ModelPerformanceItem(
                model=m_name,
                query_count=len(items),
                avg_generation_latency_ms=round(sum(gens) / len(gens), 2) if gens else None,
                avg_total_latency_ms=round(sum(totals) / len(totals), 2) if totals else None,
                avg_tokens_per_second=round(sum(tpss) / len(tpss), 2) if tpss else None,
            )
        )

    chronological = sorted(measured, key=lambda x: x.timestamp_ms)[-30:]
    latency_trend = [
        LatencyTrendItem(
            request_id=t.request_id,
            timestamp_ms=t.timestamp_ms,
            total_latency_ms=t.execution_time_ms,
            retrieval_latency_ms=t.retrieval_latency_ms,
            generation_latency_ms=t.generation_latency_ms,
            evaluation_duration_ms=t.evaluation_duration_ms,
            tokens_per_second=t.tokens_per_second,
            model=t.model,
            evaluation_status=t.evaluation_status,
            question_preview=(t.question[:70] if t.question else None),
        )
        for t in chronological
    ]

    return PerformanceSummaryResponse(
        total_traces=len(traces),
        measured_query_traces=len(measured),
        avg_query_latency_ms=avg_query,
        min_query_latency_ms=min_query,
        median_query_latency_ms=med_query,
        p90_query_latency_ms=p90_query,
        p95_query_latency_ms=p95_query,
        max_query_latency_ms=max_query,
        percentile_method="linear_interpolation",
        sample_sufficient_for_p95=len(total_latencies) >= 20,
        measured_latencies_ms=[round(v, 2) for v in total_latencies],
        avg_retrieval_latency_ms=avg_retr,
        avg_generation_latency_ms=avg_gen,
        avg_evaluation_duration_ms=avg_eval,
        avg_tokens_per_second=avg_tps,
        eval_counts=eval_counts,
        eval_completion_rate=eval_completion_rate,
        model_breakdown=model_breakdown,
        latency_trend=latency_trend,
    )
