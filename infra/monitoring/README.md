# LLMOps Sentinel — Prometheus & Grafana Monitoring Stack

This directory packages the Phase 4 operational observability stack while preserving **MLflow** (`http://127.0.0.1:5000`, `mlflow.db`) as the source of truth for LLM traces, prompts, and LLM-as-a-Judge evaluations.

## Services
- **Prometheus (`9090`)**: Scrapes `http://host.docker.internal:8000/metrics` every 10s and evaluates `prometheus/alerts.yml`.
- **Grafana (`3000`)**: Auto-provisions Prometheus datasource (`uid: prometheus`) and loads 5 dashboards from `grafana/dashboards/`.

## Host Networking Requirement (Windows -> Docker)
Prometheus inside Docker reaches the Windows FastAPI process via `host.docker.internal:8000`.
Start FastAPI bound to `0.0.0.0` (never `127.0.0.1` loopback alone) while keeping API key auth (`X-API-Key`) and Windows Defender Firewall enabled:

```powershell
.venv\Scripts\python.exe -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Usage
```powershell
cd infra/monitoring
Copy-Item .env.example .env   # if .env does not already exist
# Set a strong GF_SECURITY_ADMIN_PASSWORD in .env
docker compose up -d
```

- Prometheus UI & Targets: http://localhost:9090/targets
- Prometheus Alerts: http://localhost:9090/alerts
- Grafana Dashboards: http://localhost:3000
