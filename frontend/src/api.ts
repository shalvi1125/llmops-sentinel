import {
  BenchmarkRunResponse,
  BenchmarkSummary,
  CompareResponse,
  HealthResponse,
  IndexedDocItem,
  IngestResponse,
  PromptStrategy,
  QueryRequest,
  QueryResponse,
  TraceAuditItem,
} from './types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = `Request failed with status ${res.status}`;
    try {
      const data = await res.json();
      if (data && data.detail) {
        if (typeof data.detail === 'string') {
          errorDetail = data.detail;
        } else if (Array.isArray(data.detail)) {
          errorDetail = data.detail
            .map((err: { msg?: string }) => err.msg || JSON.stringify(err))
            .join(', ');
        }
      }
    } catch {
      // Body not JSON
    }
    throw new Error(errorDetail);
  }
  return res.json() as Promise<T>;
}

export async function checkHealth(customBase = API_BASE): Promise<HealthResponse> {
  const url = `${customBase}/health`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10000);
  try {
    const res = await fetch(url, {
      method: 'GET',
      signal: controller.signal,
    });
    return await handleResponse<HealthResponse>(res);
  } catch (err: unknown) {
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error('Health check request timed out after 10 seconds.');
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

export async function ingestFile(
  file: File,
  apiKey: string,
  clearExisting = false,
  customBase = API_BASE,
  timeoutMs = 120000
): Promise<IngestResponse> {
  const url = `${customBase}/ingest?clear_existing=${clearExisting}`;
  const formData = new FormData();
  formData.append('file', file);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'X-API-Key': apiKey,
      },
      body: formData,
      signal: controller.signal,
    });
    return await handleResponse<IngestResponse>(res);
  } catch (err: unknown) {
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error(`Document ingestion timed out after ${timeoutMs / 1000} seconds.`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

export async function getIndexedDocuments(
  apiKey: string,
  customBase = API_BASE
): Promise<IndexedDocItem[]> {
  const url = `${customBase}/ingest/documents`;
  const res = await fetch(url, {
    method: 'GET',
    headers: { 'X-API-Key': apiKey },
  });
  return handleResponse<IndexedDocItem[]>(res);
}

export async function clearAllDocuments(
  apiKey: string,
  customBase = API_BASE
): Promise<{ status: string; deleted_chunks: number }> {
  const url = `${customBase}/ingest/documents`;
  const res = await fetch(url, {
    method: 'DELETE',
    headers: { 'X-API-Key': apiKey },
  });
  return handleResponse<{ status: string; deleted_chunks: number }>(res);
}

export async function deleteDocument(
  filename: string,
  apiKey: string,
  customBase = API_BASE
): Promise<{ status: string; deleted_chunks: number }> {
  const url = `${customBase}/ingest/documents/${encodeURIComponent(filename)}`;
  const res = await fetch(url, {
    method: 'DELETE',
    headers: { 'X-API-Key': apiKey },
  });
  return handleResponse<{ status: string; deleted_chunks: number }>(res);
}

export async function queryRAG(
  question: string,
  topK = 3,
  apiKey: string,
  promptId = 'default_qa',
  model?: string,
  customBase = API_BASE,
  timeoutMs = 90000
): Promise<QueryResponse> {
  const url = `${customBase}/query`;
  const payload: QueryRequest = {
    question,
    top_k: topK,
    prompt_id: promptId,
    model,
  };

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': apiKey,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    return await handleResponse<QueryResponse>(res);
  } catch (err: unknown) {
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error(`Query request timed out after ${timeoutMs / 1000} seconds.`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

export async function getPrompts(
  apiKey: string,
  customBase = API_BASE
): Promise<PromptStrategy[]> {
  const url = `${customBase}/prompts`;
  const res = await fetch(url, {
    method: 'GET',
    headers: { 'X-API-Key': apiKey },
  });
  return handleResponse<PromptStrategy[]>(res);
}

export async function compareModels(
  question: string,
  topK = 3,
  promptId = 'default_qa',
  apiKey: string,
  customBase = API_BASE,
  timeoutMs = 90000
): Promise<CompareResponse> {
  const url = `${customBase}/query/compare`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': apiKey,
      },
      body: JSON.stringify({ question, top_k: topK, prompt_id: promptId }),
      signal: controller.signal,
    });
    return await handleResponse<CompareResponse>(res);
  } catch (err: unknown) {
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error(`Model comparison timed out after ${timeoutMs / 1000} seconds.`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

export async function getEvaluationTraces(
  apiKey: string,
  limit = 25,
  customBase = API_BASE
): Promise<TraceAuditItem[]> {
  const url = `${customBase}/eval/traces?limit=${limit}`;
  const res = await fetch(url, {
    method: 'GET',
    headers: { 'X-API-Key': apiKey },
  });
  return handleResponse<TraceAuditItem[]>(res);
}

export async function triggerBenchmark(
  apiKey: string,
  customBase = API_BASE
): Promise<BenchmarkRunResponse> {
  const url = `${customBase}/eval/benchmark`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'X-API-Key': apiKey },
  });
  return handleResponse<BenchmarkRunResponse>(res);
}

export async function getBenchmark(
  benchmarkId: string,
  apiKey: string,
  customBase = API_BASE
): Promise<BenchmarkRunResponse> {
  const url = `${customBase}/eval/benchmark/${encodeURIComponent(benchmarkId)}`;
  const res = await fetch(url, {
    method: 'GET',
    headers: { 'X-API-Key': apiKey },
  });
  return handleResponse<BenchmarkRunResponse>(res);
}

export async function listBenchmarks(
  apiKey: string,
  customBase = API_BASE
): Promise<BenchmarkSummary[]> {
  const url = `${customBase}/eval/benchmarks`;
  const res = await fetch(url, {
    method: 'GET',
    headers: { 'X-API-Key': apiKey },
  });
  return handleResponse<BenchmarkSummary[]>(res);
}
