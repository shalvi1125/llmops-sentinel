# LLMOps Sentinel

<div align="center">

**Local RAG Diagnostics, Model Arena, Evaluation & LLMOps Control Centre**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![MLflow](https://img.shields.io/badge/MLflow-3.12-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_Inference-white.svg?logo=ollama&logoColor=black)](https://ollama.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*A fully offline, portfolio-grade LLMOps control centre and RAG observatory. Upload documents, inspect vector chunks, execute grounded queries with low-latency diagnostics, benchmark local models side-by-side, and track production quality scores with automated LLM-as-a-judge evaluations.*

</div>

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LLMOps Sentinel Control Centre                        │
│                                                                                 │
│   ┌────────────────────────────────┐       ┌────────────────────────────────┐   │
│   │   React 18 SPA (Vite + TW)     │       │     MLflow Tracking Server     │   │
│   │   Port :5173                   │       │     Port :5000 (SQLite Store)  │   │
│   │   • Modern Vertical Sidebar    │       │     • Traces & Metric History  │   │
│   │   • Live Chat & Inspector      │       │     • GenAI Evaluation Runs    │   │
│   │   • Local Model Arena          │       └──────────────▲─────────────────┘   │
│   │   • Benchmark Lab & Telemetry  │                      │                     │
│   └───────────────┬────────────────┘                      │ Autolog & Traces    │
│                   │ REST API / JSON                       │                     │
│                   ▼                                       │                     │
│   ┌───────────────────────────────────────────────────────┴─────────────────┐   │
│   │                  FastAPI Backend Server (:8000)                         │   │
│   │   • Vector Ingestion Engine (1200-char splitters, cosine metric)        │   │
│   │   • Prompt Catalog (`default_qa`, `strict_grounding`, `concise_facts`)  │   │
│   │   • Sequential Model Arena Runner (Zero RAM thrashing)                  │   │
│   │   • Non-blocking Background Evaluation Pipeline                         │   │
│   │   • Embedded ChromaDB Vector Store (/tmp/chroma or chroma_data)        │   │
│   └───────────────────────┬─────────────────────────────────────────────────┘   │
│                           │                                                     │
│                           ▼                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                     Ollama Engine (:11434)                              │   │
│   │   • Generation LLM: llama3.2 (warm 15m keep-alive, 8 threads)           │   │
│   │   • Judge LLM: mistral (Evaluates Relevancy, Hallucination, Safety)     │   │
│   │   • Embeddings: nomic-embed-text (Cosine distance indexing)             │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Capabilities

### 1. Live Chat & Retrieval Diagnostics
- **Grounded Q&A:** Fast question answering grounded strictly in uploaded documents.
- **Sub-chunk Attribution:** Expandable source cards displaying document origin, page numbers, chunk IDs, and estimated token counts.
- **Retrieval Diagnostics:** Displays retrieval latency in milliseconds, top cosine distance scores, and automated low-confidence threshold indicators.
- **Prompt Catalog & Versioning:** Dynamic prompt selection across `default_qa` (v1.0.0), `strict_grounding` (v1.0.0), and `concise_facts` (v1.0.0) logged directly in trace metadata.

### 2. Local Model Arena
- **Side-by-Side Comparison:** Compare generation between local models (e.g. `llama3.2` vs. `mistral`) using identical question context.
- **Resource-Aware Sequential Execution:** Runs models sequentially to prevent CPU/RAM saturation on local hardware.
- **Latency & Token Metrics:** Exact generation latency (ms), output token estimates, and word counts.

### 3. Continuous Evaluation & Trace Audit
- **LLM-as-Judge Scoring:** Non-blocking background evaluations powered by DeepEval and MLflow GenAI scorers:
  - `answer_relevancy`: Evaluates whether the answer directly addresses the user's inquiry.
  - `hallucination`: Measures faithfulness and flags unsupported assertions.
  - `safety`: Assesses safety and compliance.
- **Trace Inspector:** Inspect recent query executions, generation latencies, retrieval scores, and evaluation verdict tags (`Pass` / `Flagged`).

### 4. RAG Evaluation Laboratory (Benchmark Lab)
- **Standardized Multi-Question Benchmarks:** Run automated evaluation suites against active knowledge bases.
- **Aggregated Telemetry:** Measures overall pass rate, average answer relevancy, average hallucination score, and median latency.
- **Historical Comparison:** Track benchmark runs over time to detect regressions across prompt or model upgrades.

### 5. Dynamic Knowledge Base Management
- **ChromaDB Control:** Live inspection of indexed documents and vector chunk counts.
- **Per-Document Deletion:** Granular deletion of individual documents to avoid cross-contamination.
- **Replace on Ingestion:** Toggleable option to wipe existing documents before embedding new files.

---

## Quickstart Guide

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- [Ollama](https://ollama.com/) installed locally
- Git

---

### Step 1: Clone Repository & Setup Environment

```bash
git clone https://github.com/shalvi1125/llmops-sentinel.git
cd llmops-sentinel

# Copy example environment configuration
cp .env.example .env
```

---

### Step 2: Prepare Ollama Models

Make sure Ollama is running (`ollama serve`), then pull the required local models:

```bash
ollama pull llama3.2
ollama pull mistral
ollama pull nomic-embed-text
```

---

### Step 3: Setup Backend

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 4: Setup Frontend

```bash
cd frontend
npm install
cd ..
```

---

### Step 5: Start the Platform Services

Open separate terminal windows or run as background processes:

**1. MLflow Tracking Server:**
```bash
# Windows:
.venv\Scripts\python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 127.0.0.1 --port 5000

# Linux / macOS:
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 127.0.0.1 --port 5000
```

**2. FastAPI Backend:**
```bash
# Windows:
.venv\Scripts\python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload

# Linux / macOS:
uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

**3. Frontend Dev Server:**
```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

---

### Access the Applications

- **LLMOps Sentinel UI:** [http://localhost:5173](http://localhost:5173)
- **FastAPI Interactive Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow Observability Dashboard:** [http://localhost:5000](http://localhost:5000)

---

## Docker Compose Setup

For a containerized deployment:

```bash
# 1. Set hardware profile in .env (cpu | gpu-nvidia | gpu-amd)
COMPOSE_PROFILES=cpu

# 2. Start stack
docker compose up
```

---

## API Reference

All protected endpoints require the header `X-API-Key: <your-key>` (default: `ragscope-local-dev-key`).

### Ingestion & Knowledge Management
- `POST /ingest`: Upload PDF or TXT document (`?clear_existing=true` to replace).
- `GET /ingest/documents`: List currently indexed documents and chunk counts.
- `DELETE /ingest/documents`: Wipe all documents from ChromaDB.
- `DELETE /ingest/documents/{filename}`: Delete a specific document and its chunks.

### Query & Comparison
- `POST /query`: Grounded RAG query with retrieval diagnostics and background judge scoring.
- `POST /query/compare`: Sequential model comparison (`llama3.2` vs `mistral`).
- `GET /prompts`: List available prompt strategies (`default_qa`, `strict_grounding`, `concise_facts`).

### Evaluation & Benchmarking
- `GET /eval/traces`: Retrieve recent execution traces with LLM-as-a-judge scores.
- `POST /eval/benchmark`: Trigger a benchmark evaluation suite.
- `GET /eval/benchmarks`: List completed benchmark runs and summaries.
- `GET /eval/benchmark/{id}`: Detailed telemetry for a specific benchmark run.

---

## Testing

Run the automated test suite covering unit, integration, and end-to-end flows:

```bash
pytest -v
```

Frontend build check:
```bash
cd frontend
npm run build
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
