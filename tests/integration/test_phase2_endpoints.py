import pytest
from unittest.mock import MagicMock, patch

from src.models import CompareResponse, QueryResponse


@pytest.mark.integration
def test_get_prompts_requires_auth(client):
    resp = client.get("/prompts")
    assert resp.status_code == 401


@pytest.mark.integration
def test_get_prompts_returns_catalog(client, auth_headers):
    resp = client.get("/prompts", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 3
    ids = [p["id"] for p in data]
    assert "default_qa" in ids
    assert "strict_grounding" in ids
    assert "concise_facts" in ids


@pytest.mark.integration
def test_get_eval_traces_requires_auth(client):
    resp = client.get("/eval/traces")
    assert resp.status_code == 401


@pytest.mark.integration
def test_get_eval_traces_returns_list(client, auth_headers):
    resp = client.get("/eval/traces", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.integration
def test_query_compare_requires_auth(client):
    resp = client.post("/query/compare", json={"question": "What is RAG?"})
    assert resp.status_code == 401


@pytest.mark.integration
def test_query_compare_empty_collection_returns_404(client, auth_headers):
    mock_vs = MagicMock()
    mock_vs._collection.count.return_value = 0

    with patch("src.services.compare._get_vectorstore", return_value=mock_vs):
        resp = client.post(
            "/query/compare",
            headers=auth_headers,
            json={"question": "What is RAG?", "top_k": 3},
        )
    assert resp.status_code == 404


@pytest.mark.integration
def test_query_compare_success_returns_both_models(client, auth_headers):
    mock_vs = MagicMock()
    mock_vs._collection.count.return_value = 2

    mock_doc = MagicMock()
    mock_doc.page_content = "RAG combines retrieval with generation."
    mock_doc.metadata = {"filename": "rag.pdf", "page": 1}
    mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.15)]

    mock_llm_llama = MagicMock()
    mock_llm_llama.content = "Llama answer: RAG is retrieval augmented generation."
    mock_llm_mistral = MagicMock()
    mock_llm_mistral.content = "Mistral answer: RAG augments LLMs with knowledge."

    with patch("src.services.compare._get_vectorstore", return_value=mock_vs), \
         patch("src.services.compare.get_llm") as mock_get_llm, \
         patch("src.services.compare.PromptTemplate"), \
         patch("src.services.compare.RunnableSequence") as mock_seq:

        mock_seq.return_value.invoke.side_effect = [mock_llm_llama, mock_llm_mistral]

        resp = client.post(
            "/query/compare",
            headers=auth_headers,
            json={"question": "What is RAG?", "top_k": 2},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) == 2
    model_names = [m["model"] for m in data["models"]]
    assert "llama3.2" in model_names
    assert "mistral" in model_names
    assert "retrieval_diagnostics" in data
    assert "source_chunks" in data
    assert len(data["source_chunks"]) == 1


@pytest.mark.integration
def test_eval_benchmark_endpoints(client, auth_headers):
    # Test list
    list_resp = client.get("/eval/benchmarks", headers=auth_headers)
    assert list_resp.status_code == 200
    assert isinstance(list_resp.json(), list)

    # Test trigger
    trigger_resp = client.post("/eval/benchmark", headers=auth_headers)
    assert trigger_resp.status_code == 200
    data = trigger_resp.json()
    assert "benchmark_id" in data
    bench_id = data["benchmark_id"]
    assert data["status"] in ["running", "completed"]

    # Test get by ID
    get_resp = client.get(f"/eval/benchmark/{bench_id}", headers=auth_headers)
    assert get_resp.status_code == 200
    get_data = get_resp.json()
    assert get_data["benchmark_id"] == bench_id
    assert "cases" in get_data
