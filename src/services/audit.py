import json
import os
import sqlite3
from src.models import AssessmentDetail, TraceAuditItem
from src.utils.log_manager import logger

DB_PATH = "D:/mlops cp github/ragscope/mlflow.db"


def get_recent_traces(limit: int = 25) -> list[TraceAuditItem]:
    """Retrieve the latest MLflow traces with their assessment scores and rationales from SQLite."""
    if not os.path.exists(DB_PATH):
        logger.warning(f"MLflow database not found at {DB_PATH}")
        return []

    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        c = conn.cursor()

        # 1. Fetch recent trace_info records
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

        # 2. Fetch root_span content to extract question and answer
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

        # 3. Fetch assessments for these traces
        c.execute(
            f"""
            SELECT trace_id, name, value, rationale
            FROM assessments
            WHERE trace_id IN ({placeholders})
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
            # Normalize display name
            display_name = name
            if name.lower() == "safety":
                display_name = "Safety"
            assessments_map[tid][display_name] = AssessmentDetail(
                value=clean_val,
                rationale=clean_rat,
            )

        conn.close()

        # 4. Assemble final TraceAuditItems
        results: list[TraceAuditItem] = []
        for req_id, ts, exec_ms, status in trace_rows:
            q_text, a_text = io_map.get(req_id, (None, None))
            assess_dict = assessments_map.get(req_id, {})
            results.append(
                TraceAuditItem(
                    request_id=req_id,
                    timestamp_ms=ts,
                    execution_time_ms=float(exec_ms or 0.0),
                    status=status or "UNKNOWN",
                    question=q_text,
                    answer=a_text,
                    assessments=assess_dict,
                )
            )

        return results

    except Exception as exc:
        logger.error(f"Error querying MLflow trace audit log: {exc}")
        return []
