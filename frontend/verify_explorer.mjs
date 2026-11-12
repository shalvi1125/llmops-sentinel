import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const mockDocData = {
  document_name: 'os_job_scheduler.pdf',
  document_id: 'doc-123456789abc',
  file_type: 'pdf',
  total_pages: 2,
  total_chunks: 2,
  total_extracted_chars: 185,
  total_estimated_tokens: 46,
  chunk_size: 1200,
  chunk_overlap: 100,
  splitter_name: 'RecursiveCharacterTextSplitter',
  embedding_model: 'nomic-embed-text',
  ingestion_timestamp: '2026-11-12T10:00:00Z',
  reconstructed_text:
    'Chapter 1 introduces Resource-Aware Job Scheduling with CPU and RAM budgets. Overlap section bridges worker registration and MongoDB atomic transactions for worker heartbeats.',
  chunks: [
    {
      chunk_id: 'doc-123456789abc-chunk-0000',
      chunk_index: 0,
      text: 'Chapter 1 introduces Resource-Aware Job Scheduling with CPU and RAM budgets. Overlap section bridges worker registration',
      document_id: 'doc-123456789abc',
      document_name: 'os_job_scheduler.pdf',
      page_number: 1,
      source: 'os_job_scheduler.pdf',
      char_count: 118,
      token_count_estimate: 29,
      start_char: 0,
      end_char: 118,
      overlap_prev_chars: 0,
      overlap_next_chars: 35,
      overlap_prev_text: '',
      overlap_next_text: 'Overlap section bridges worker registration',
      ingestion_timestamp: '2026-11-12T10:00:00Z',
    },
    {
      chunk_id: 'doc-123456789abc-chunk-0001',
      chunk_index: 1,
      text: 'Overlap section bridges worker registration and MongoDB atomic transactions for worker heartbeats.',
      document_id: 'doc-123456789abc',
      document_name: 'os_job_scheduler.pdf',
      page_number: 2,
      source: 'os_job_scheduler.pdf',
      char_count: 98,
      token_count_estimate: 24,
      start_char: 77,
      end_char: 175,
      overlap_prev_chars: 43,
      overlap_next_chars: 0,
      overlap_prev_text: 'Overlap section bridges worker registration',
      overlap_next_text: '',
      ingestion_timestamp: '2026-11-12T10:00:00Z',
    },
  ],
};

const mockInspectData = {
  question: 'How does the job scheduler assign workers?',
  document_scope: 'os_job_scheduler.pdf',
  embedding_model: 'nomic-embed-text',
  score_metric: 'cosine_distance',
  score_explanation:
    'ChromaDB computes cosine distance d = 1 - cos(theta), NOT cosine similarity.',
  retrieval_diagnostics: {
    retrieval_latency_ms: 14.2,
    top_score: 0.2145,
    low_confidence: false,
    score_metric: 'cosine_distance',
    requested_k: 4,
    candidates_fetched: 5,
    unique_chunks_returned: 2,
    duplicates_removed: 2,
    retrieval_strategy: 'mmr_deduplicated',
    insufficient_unique_reason:
      'Requested K=4; returned 2 unique relevant chunk(s) after removing 2 duplicate/near-duplicate match(es).',
  },
  retrieved_chunks: [
    {
      chunk_id: 'doc-123456789abc-chunk-0000',
      text: mockDocData.chunks[0].text,
      score: 0.2145,
      document_id: 'doc-123456789abc',
      document_name: 'os_job_scheduler.pdf',
      page_number: 1,
      source: 'os_job_scheduler.pdf',
      ingestion_timestamp: '2026-11-12T10:00:00Z',
      token_count_estimate: 29,
    },
  ],
  context_preview: mockDocData.chunks[0].text,
  context_char_count: 118,
  insufficient_evidence: false,
  empty_reason: null,
};

async function run() {
  const vite = await createServer({
    server: { middlewareMode: true },
    appType: 'custom',
    logLevel: 'silent',
  });

  try {
    const { ChunkingViewer } = await vite.ssrLoadModule('/src/components/explorer/ChunkingViewer.tsx');
    const { RetrievalInspectorPanel } = await vite.ssrLoadModule(
      '/src/components/explorer/RetrievalInspectorPanel.tsx'
    );
    const { PipelineVisualizer } = await vite.ssrLoadModule(
      '/src/components/explorer/PipelineVisualizer.tsx'
    );
    const { RagExplorerView } = await vite.ssrLoadModule('/src/components/RagExplorerView.tsx');

    // 1. Render ChunkingViewer with active chunk & overlap metadata
    const chunkHtml = renderToStaticMarkup(
      React.createElement(ChunkingViewer, {
        docData: mockDocData,
        previewData: null,
        loadingDoc: false,
        loadingPreview: false,
        previewError: null,
        activeChunkIndex: 1,
        onSelectChunkIndex: () => {},
        previewChunkSize: 1200,
        previewChunkOverlap: 100,
        onPreviewChunkSizeChange: () => {},
        onPreviewChunkOverlapChange: () => {},
        onRunPreview: () => {},
        onResetPreview: () => {},
        highlightedRetrievedChunkIds: ['doc-123456789abc-chunk-0000'],
      })
    );
    assert.ok(chunkHtml.includes('Original Extracted Document Text'));
    assert.ok(chunkHtml.includes('doc-123456789abc-chunk-0001'));
    assert.ok(chunkHtml.includes('Overlap with Preceding Chunk (43 chars)'));
    assert.ok(chunkHtml.includes('os_job_scheduler.pdf'));

    // 2. Render RetrievalInspectorPanel with Cosine Distance + Deduplication stats
    const inspectHtml = renderToStaticMarkup(
      React.createElement(RetrievalInspectorPanel, {
        documents: [{ filename: 'os_job_scheduler.pdf', chunks: 2 }],
        question: 'How does the job scheduler assign workers?',
        onQuestionChange: () => {},
        scopeDoc: 'os_job_scheduler.pdf',
        onScopeDocChange: () => {},
        topK: 4,
        onTopKChange: () => {},
        useMmr: true,
        onUseMmrChange: () => {},
        isInspecting: false,
        error: null,
        result: mockInspectData,
        onInspect: () => {},
        onLocateChunk: () => {},
      })
    );
    assert.ok(inspectHtml.includes('Cosine Distance'));
    assert.ok(inspectHtml.includes('Cosine Dist: 0.2145'));
    assert.ok(inspectHtml.includes('Duplicates Filtered: 2'));
    assert.ok(inspectHtml.includes('mmr_deduplicated'));
    assert.ok(inspectHtml.includes('Locate in Chunk Viewer'));

    // 3. Render 10-Stage PipelineVisualizer & Embeddings explanation
    const pipelineHtml = renderToStaticMarkup(
      React.createElement(PipelineVisualizer, {
        docData: mockDocData,
        inspectData: mockInspectData,
        liveQueryResult: null,
        isRunningLiveQuery: false,
      })
    );
    assert.ok(pipelineHtml.includes('Interactive End-to-End RAG Pipeline'));
    assert.ok(pipelineHtml.includes('Text Chunking'));
    assert.ok(pipelineHtml.includes('Document Upload'));
    assert.ok(pipelineHtml.includes('ChromaDB Storage'));
    assert.ok(pipelineHtml.includes('LLM Answer Generation'));
    assert.ok(pipelineHtml.includes('Conceptual Diagram Only'));

    // 4. Render top-level RagExplorerView cleanly
    const viewHtml = renderToStaticMarkup(
      React.createElement(RagExplorerView, {
        apiKey: 'test-key',
        selectedDocument: 'os_job_scheduler.pdf',
        onSelectDocument: () => {},
      })
    );
    assert.ok(viewHtml.includes('RAG Pipeline Explorer'));

    console.log('ALL FRONTEND EXPLORER RENDER ASSERTIONS PASSED');
  } finally {
    await vite.close();
  }
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
