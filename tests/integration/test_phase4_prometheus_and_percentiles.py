import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from prometheus_client.parser import text_string_to_metric_families

from src.metrics import list_registered_metric_names
from src.services.audit import _percentile, get_performance_summary

REPO_ROOT = Path(__file__).resolve().parents[2]
MONITORING_DIR = REPO_ROOT / "infra" / "monitoring"


@pytest.mark.unit
def test_percentile_linear_interpolation_even_and_odd():
    # Odd list [10, 20, 30, 40, 50] -> P50 = 30.0
    assert _percentile([10.0, 20.0, 30.0, 40.0, 50.0], 50.0) == 30.0
    # Even list [100, 200, 300, 400] -> R-7 P50 is 250.0, P95 is 300 + 0.85 * 100 = 385.0
    assert _percentile([100.0, 200.0, 300.0, 400.0], 50.0) == 250.0
    assert _percentile([100.0, 200.0, 300.0, 400.0], 95.0) == 385.0
    # Edge cases
    assert _percentile([], 50.0) is None
    assert _percentile([123.45], 95.0) == 123.45


@pytest.mark.unit
def test_performance_summary_reports_linear_interpolation_and_sample_sufficiency(tmp_path):
    # Verify summary fields cleanly without touching real mlflow.db
    with patch("src.services.audit.get_recent_traces", return_value=[]):
        empty_summary = get_performance_summary(limit=50)
    assert empty_summary.percentile_method == "linear_interpolation"
    assert empty_summary.sample_sufficient_for_p95 is False
    assert empty_summary.measured_latencies_ms == []


@pytest.mark.integration
def test_metrics_endpoint_exposes_bounded_cardinality_prometheus_payload(client, auth_headers):
    # Exercise a route with a path parameter & a secret question query
    secret_question = "TOP_SECRET_USER_PROMPT_NEVER_IN_LABELS"
    secret_filename = "confidential_contract_2026.pdf"

    client.get(f"/ingest/documents/{secret_filename}/chunks", headers=auth_headers)
    with patch("src.services.query._get_vectorstore") as mock_get_vs:
        mock_vs = MagicMock()
        mock_vs._collection.count.return_value = 0
        mock_get_vs.return_value = mock_vs
        client.post(
            "/query",
            json={"question": secret_question, "top_k": 2},
            headers=auth_headers,
        )

    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text

    # Must parse cleanly as valid Prometheus text exposition format
    # Note: text_string_to_metric_families strips the '_total' suffix from Counter family names
    families = {f.name: f for f in text_string_to_metric_families(body)}
    assert "llmops_sentinel_http_requests" in families
    assert "llmops_sentinel_http_request_duration_seconds" in families
    assert "llmops_sentinel_process_memory_rss_bytes" in families
    assert "llmops_sentinel_service_up" in families

    # High-cardinality security check
    assert secret_question not in body
    assert secret_filename not in body
    assert "test-api-key" not in body

    # Path parameter must normalize to FastAPI route template
    http_routes = {
        s.labels.get("route")
        for s in families["llmops_sentinel_http_requests"].samples
        if s.name == "llmops_sentinel_http_requests_total"
    }
    assert "/query" in http_routes
    assert "/ingest/documents/{filename}/chunks" in http_routes
    assert f"/ingest/documents/{secret_filename}/chunks" not in http_routes


@pytest.mark.integration
def test_query_actual_vs_estimated_tokens_and_observability_summary(client, auth_headers):
    mock_vs = MagicMock()
    mock_vs._collection.count.return_value = 2
    mock_doc = MagicMock()
    mock_doc.page_content = "Phase 4 adds Prometheus and Grafana while preserving MLflow."
    mock_doc.metadata = {
        "document_id": "doc-p4",
        "document_name": "phase4.txt",
        "chunk_id": "doc-p4-chunk-0000",
        "page_number": 1,
    }
    mock_vs._collection.query.return_value = {
        "documents": [[mock_doc.page_content]],
        "metadatas": [[mock_doc.metadata]],
        "distances": [[0.15]],
        "embeddings": [[[0.1, 0.2, 0.3]]],
    }
    mock_vs._embedding_function.embed_query.return_value = [0.1, 0.2, 0.3]

    mock_resp = MagicMock()
    mock_resp.content = "MLflow handles traces while Prometheus tracks operational metrics."
    mock_resp.response_metadata = {"eval_count": 14, "eval_duration": 1_000_000_000}
    mock_resp.usage_metadata = None

    with patch("src.services.query._get_vectorstore", return_value=mock_vs), \
         patch("src.services.query.get_llm", return_value=MagicMock()), \
         patch("src.services.query.PromptTemplate"), \
         patch("src.services.query.RunnableSequence") as mock_seq, \
         patch("src.services.query.record_query_trace", return_value=("tr-p4", "eval-p4", "pending")), \
         patch("src.services.query.run_judge_evaluations"):
        mock_seq.return_value.invoke.return_value = mock_resp
        q_res = client.post(
            "/query",
            json={"question": "How do MLflow and Prometheus work together?", "top_k": 2},
            headers=auth_headers,
        )
    assert q_res.status_code == 200

    metrics_text = client.get("/metrics").text
    families = {f.name: f for f in text_string_to_metric_families(metrics_text)}
    token_samples = [
        s
        for s in families["llmops_sentinel_llm_tokens_generated"].samples
        if s.name == "llmops_sentinel_llm_tokens_generated_total"
    ]
    assert any(
        s.labels.get("token_source") == "actual" and s.value >= 14.0
        for s in token_samples
    )

    # Observability summary requires API key
    unauth = client.get("/observability/summary")
    assert unauth.status_code in (401, 403)

    with patch("src.services.health.check_health") as mock_health:
        from src.models import HealthResponse

        async def _ok_health():
            return HealthResponse(status="ok", chromadb="ok", ollama="ok")

        mock_health.side_effect = _ok_health
        with patch("src.api.router.check_health", side_effect=_ok_health), \
             patch("src.api.router.probe_mlflow_status", return_value=True):
            obs_res = client.get("/observability/summary", headers=auth_headers)

    assert obs_res.status_code == 200
    payload = obs_res.json()
    assert payload["metrics_endpoint"] == "/metrics"
    assert payload["dependencies"]["mlflow"] is True
    assert payload["process_memory_rss_mb"] > 0
    assert "llmops_sentinel_http_requests" in payload["registered_metric_families"]


@pytest.mark.unit
def test_monitoring_configs_and_five_grafana_dashboards_are_valid_and_aligned():
    assert (MONITORING_DIR / "docker-compose.yml").exists()
    assert (MONITORING_DIR / ".env.example").exists()

    compose = yaml.safe_load((MONITORING_DIR / "docker-compose.yml").read_text(encoding="utf-8"))
    assert set(compose["services"].keys()) == {"prometheus", "grafana"}
    assert "latest" not in compose["services"]["prometheus"]["image"]
    assert "latest" not in compose["services"]["grafana"]["image"]

    prom_cfg = yaml.safe_load((MONITORING_DIR / "prometheus" / "prometheus.yml").read_text(encoding="utf-8"))
    assert prom_cfg["scrape_configs"][0]["metrics_path"] == "/metrics"

    alerts_cfg = yaml.safe_load((MONITORING_DIR / "prometheus" / "alerts.yml").read_text(encoding="utf-8"))
    rules = alerts_cfg["groups"][0]["rules"]
    assert len(rules) >= 6

    ds_cfg = yaml.safe_load(
        (MONITORING_DIR / "grafana" / "provisioning" / "datasources" / "prometheus.yml").read_text(
            encoding="utf-8"
        )
    )
    assert ds_cfg["datasources"][0]["uid"] == "prometheus"

    registered = set(list_registered_metric_names())
    dashboard_files = sorted((MONITORING_DIR / "grafana" / "dashboards").glob("*.json"))
    assert len(dashboard_files) == 5

    import re

    for df in dashboard_files:
        dash = json.loads(df.read_text(encoding="utf-8"))
        assert dash["panels"], f"{df.name} has no panels"
        for panel in dash["panels"]:
            assert panel["datasource"]["uid"] == "prometheus"
            for target in panel.get("targets", []):
                expr = target.get("expr", "")
                referenced = set(re.findall(r"llmops_sentinel_[a-z0-9_]+", expr))
                for metric in referenced:
                    base = re.sub(r"_(bucket|sum|count|total)$", "", metric)
                    # Counter families already end in _total; histograms expose _bucket/_sum/_count
                    assert (
                        metric in registered
                        or base in registered
                        or re.sub(r"_(bucket|sum|count)$", "", metric) in registered
                    ), f"Unregistered metric '{metric}' in {df.name}"
