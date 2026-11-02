import React, { useState } from 'react';
import {
  Swords,
  Loader2,
  Cpu,
  Clock,
  SlidersHorizontal,
  AlertCircle,
  Zap,
  Copy,
  Check,
  Info,
  Layers,
  ArrowRight,
  TrendingDown,
} from 'lucide-react';
import { compareModels } from '../api';
import { CompareResponse, PromptStrategy } from '../types';
import { SourceCitation } from './SourceCitation';

interface ModelArenaProps {
  apiKey: string;
  prompts: PromptStrategy[];
}

export const ModelArena: React.FC<ModelArenaProps> = ({ apiKey, prompts }) => {
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(3);
  const [promptId, setPromptId] = useState('default_qa');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState<'idle' | 'retrieving' | 'llama' | 'mistral'>('idle');
  const [comparisonResult, setComparisonResult] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copiedModel, setCopiedModel] = useState<string | null>(null);

  const handleCompare = async () => {
    const q = question.trim();
    if (!q || isLoading) return;

    setError(null);
    setIsLoading(true);
    setLoadingStep('retrieving');

    // Sequential step transitions for honest UX
    const step1Timer = setTimeout(() => setLoadingStep('llama'), 1000);
    const step2Timer = setTimeout(() => setLoadingStep('mistral'), 8000);

    try {
      const data = await compareModels(q, topK, promptId, apiKey);
      clearTimeout(step1Timer);
      clearTimeout(step2Timer);
      setComparisonResult(data);
    } catch (err: unknown) {
      clearTimeout(step1Timer);
      clearTimeout(step2Timer);
      const msg = err instanceof Error ? err.message : 'Model comparison failed.';
      setError(msg);
    } finally {
      setIsLoading(false);
      setLoadingStep('idle');
    }
  };

  const handleCopy = (text: string, modelName: string) => {
    navigator.clipboard.writeText(text);
    setCopiedModel(modelName);
    setTimeout(() => setCopiedModel(null), 2000);
  };

  const sampleQuestions = [
    'What is LLMOps Sentinel and what database does it use?',
    'Which local models are used for generation, judge evaluation, and embeddings?',
    'How does zero-latency preemption prevent query timeouts during long judge evaluations?',
  ];

  // Helper to extract Llama and Mistral results
  const llamaResult = comparisonResult?.models.find((m) => m.model.toLowerCase().includes('llama'));
  const mistralResult = comparisonResult?.models.find((m) =>
    m.model.toLowerCase().includes('mistral')
  );

  // Compute differentials
  let fasterModel: string | null = null;
  let latencyDiffMs = 0;
  let latencyDiffPct = 0;

  if (llamaResult && mistralResult) {
    if (llamaResult.generation_latency_ms < mistralResult.generation_latency_ms) {
      fasterModel = llamaResult.model;
      latencyDiffMs = mistralResult.generation_latency_ms - llamaResult.generation_latency_ms;
      latencyDiffPct = Math.round((latencyDiffMs / mistralResult.generation_latency_ms) * 100);
    } else if (mistralResult.generation_latency_ms < llamaResult.generation_latency_ms) {
      fasterModel = mistralResult.model;
      latencyDiffMs = llamaResult.generation_latency_ms - mistralResult.generation_latency_ms;
      latencyDiffPct = Math.round((latencyDiffMs / llamaResult.generation_latency_ms) * 100);
    }
  }

  const maxLatency = Math.max(
    llamaResult?.generation_latency_ms || 1,
    mistralResult?.generation_latency_ms || 1
  );

  const maxTokens = Math.max(
    llamaResult?.token_count_estimate || 1,
    mistralResult?.token_count_estimate || 1
  );

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl flex flex-col h-full overflow-hidden">
      {/* Top Header & Strategy Config */}
      <div className="px-5 py-3.5 border-b border-slate-800 bg-slate-950/40 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center space-x-2">
          <Swords className="w-4 h-4 text-sky-400" />
          <span className="font-semibold text-slate-200">Local Model Arena</span>
          <span className="text-slate-400">· Side-by-Side Dual Inference</span>
        </div>

        <div className="flex items-center space-x-3">
          {/* Prompt strategy */}
          <div className="flex items-center space-x-1.5">
            <span className="text-[11px] text-slate-400 font-mono">Prompt:</span>
            <select
              value={promptId}
              onChange={(e) => setPromptId(e.target.value)}
              className="bg-slate-800/90 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1 focus:outline-none focus:border-sky-500 font-mono"
            >
              {prompts.length > 0 ? (
                prompts.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))
              ) : (
                <option value="default_qa">Balanced QA (Default)</option>
              )}
            </select>
          </div>

          {/* Top_K */}
          <div className="flex items-center space-x-1.5">
            <SlidersHorizontal className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-[11px] text-slate-400 font-mono">k =</span>
            <select
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value, 10))}
              className="bg-slate-800/90 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1 focus:outline-none focus:border-sky-500 font-mono"
            >
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="4">4</option>
              <option value="6">6</option>
            </select>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto p-5 space-y-5">
        {/* Hardware Resource Guidance Banner */}
        <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3 flex items-start space-x-3 text-xs">
          <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
          <div className="flex-1 leading-relaxed text-slate-400">
            <strong className="text-slate-200 font-mono">Hardware-Safe Sequential Execution: </strong>
            To prevent system memory paging thrashing and GPU VRAM exhaustion, Sentinel runs
            retrieval once in ChromaDB, then executes <span className="text-sky-300 font-mono">Llama 3.2 (3B)</span>,
            followed sequentially by <span className="text-amber-300 font-mono">Mistral (7B)</span> on identical retrieved context.
          </div>
        </div>

        {/* Input Bar */}
        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 shadow-sm">
          <label className="block text-xs font-semibold text-slate-300 mb-2">
            Ask a question to compare side-by-side on identical retrieved context:
          </label>
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleCompare()}
              placeholder="e.g. What is LLMOps Sentinel and what database does it use?"
              disabled={isLoading}
              className="flex-1 bg-slate-900 border border-slate-700/80 rounded-lg px-3.5 py-2.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500 font-sans"
            />
            <button
              onClick={handleCompare}
              disabled={!question.trim() || isLoading}
              className="px-4 py-2.5 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:bg-slate-800 disabled:text-slate-600 text-white text-xs font-semibold flex items-center space-x-2 transition shrink-0 shadow-md shadow-sky-600/20"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Evaluating Models...</span>
                </>
              ) : (
                <>
                  <Swords className="w-4 h-4" />
                  <span>Compare Models</span>
                </>
              )}
            </button>
          </div>

          {/* Quick starter chips */}
          {!comparisonResult && !isLoading && (
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <span className="text-[11px] text-slate-500 font-mono">Try:</span>
              {sampleQuestions.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => setQuestion(q)}
                  className="text-[11px] bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-white px-2.5 py-1 rounded-md border border-slate-800 transition"
                >
                  "{q}"
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Error notification */}
        {error && (
          <div className="p-3 bg-rose-950/40 border border-rose-800/60 rounded-xl text-xs text-rose-300 flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}

        {/* Loading State Pipeline */}
        {isLoading && (
          <div className="py-10 bg-slate-950/60 border border-slate-800 rounded-xl p-6 flex flex-col items-center justify-center text-center space-y-4">
            <Loader2 className="w-8 h-8 text-sky-400 animate-spin" />
            <div>
              <h4 className="text-sm font-semibold text-slate-200">
                Executing Sequential Comparison Pipeline
              </h4>
              <p className="text-xs text-slate-400 mt-1 max-w-md">
                Measuring raw generation latency, token volume, and response grounding.
              </p>
            </div>

            {/* Sequential Steps Indicator */}
            <div className="flex items-center space-x-2 text-xs font-mono pt-2">
              <span
                className={`px-2.5 py-1 rounded-lg border ${
                  loadingStep === 'retrieving'
                    ? 'bg-sky-500/20 text-sky-300 border-sky-500/40 font-bold'
                    : 'bg-slate-900 text-slate-500 border-slate-800'
                }`}
              >
                1. ChromaDB Retrieval
              </span>
              <ArrowRight className="w-3.5 h-3.5 text-slate-600" />
              <span
                className={`px-2.5 py-1 rounded-lg border ${
                  loadingStep === 'llama'
                    ? 'bg-sky-500/20 text-sky-300 border-sky-500/40 font-bold'
                    : 'bg-slate-900 text-slate-500 border-slate-800'
                }`}
              >
                2. Llama 3.2 (3B)
              </span>
              <ArrowRight className="w-3.5 h-3.5 text-slate-600" />
              <span
                className={`px-2.5 py-1 rounded-lg border ${
                  loadingStep === 'mistral'
                    ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 font-bold'
                    : 'bg-slate-900 text-slate-500 border-slate-800'
                }`}
              >
                3. Mistral (7B)
              </span>
            </div>
          </div>
        )}

        {/* Comparison Result Display */}
        {comparisonResult && !isLoading && (
          <div className="space-y-4">
            {/* Visual Differential KPI Summary */}
            {fasterModel && (
              <div className="bg-slate-950/90 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center space-x-2">
                  <TrendingDown className="w-4 h-4 text-emerald-400" />
                  <span className="text-xs font-semibold text-slate-200">
                    Latency Differential:
                  </span>
                  <span className="text-xs font-bold text-emerald-400 font-mono">
                    {fasterModel} is {latencyDiffMs} ms faster ({latencyDiffPct}% speedup)
                  </span>
                </div>

                <div className="text-[11px] text-slate-400 font-mono">
                  Context Retrieval Latency:{' '}
                  <span className="text-sky-300">
                    {comparisonResult.retrieval_diagnostics.retrieval_latency_ms} ms
                  </span>
                </div>
              </div>
            )}

            {/* Interactive Comparative Latency & Token Bars */}
            {llamaResult && mistralResult && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Latency Comparison Card */}
                <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-300 flex items-center space-x-1.5">
                      <Clock className="w-3.5 h-3.5 text-sky-400" />
                      <span>Generation Latency (Lower is faster)</span>
                    </span>
                    <span className="text-[10px] font-mono text-slate-500">Unit: ms</span>
                  </div>

                  <div className="space-y-2">
                    {/* Llama Bar */}
                    <div>
                      <div className="flex justify-between text-[11px] font-mono mb-1">
                        <span className="text-sky-400 font-bold">{llamaResult.model}</span>
                        <span className="text-slate-200">{llamaResult.generation_latency_ms} ms</span>
                      </div>
                      <div className="h-2.5 w-full bg-slate-900 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-sky-500 rounded-full transition-all duration-500"
                          style={{
                            width: `${(llamaResult.generation_latency_ms / maxLatency) * 100}%`,
                          }}
                        />
                      </div>
                    </div>

                    {/* Mistral Bar */}
                    <div>
                      <div className="flex justify-between text-[11px] font-mono mb-1">
                        <span className="text-amber-400 font-bold">{mistralResult.model}</span>
                        <span className="text-slate-200">{mistralResult.generation_latency_ms} ms</span>
                      </div>
                      <div className="h-2.5 w-full bg-slate-900 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-amber-500 rounded-full transition-all duration-500"
                          style={{
                            width: `${(mistralResult.generation_latency_ms / maxLatency) * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Token Volume Comparison Card */}
                <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-300 flex items-center space-x-1.5">
                      <Layers className="w-3.5 h-3.5 text-indigo-400" />
                      <span>Answer Length & Density</span>
                    </span>
                    <span className="text-[10px] font-mono text-slate-500">Est. Tokens</span>
                  </div>

                  <div className="space-y-2">
                    {/* Llama Bar */}
                    <div>
                      <div className="flex justify-between text-[11px] font-mono mb-1">
                        <span className="text-sky-400 font-bold">{llamaResult.model}</span>
                        <span className="text-slate-200">~{llamaResult.token_count_estimate} tok</span>
                      </div>
                      <div className="h-2.5 w-full bg-slate-900 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-sky-500/80 rounded-full transition-all duration-500"
                          style={{
                            width: `${(llamaResult.token_count_estimate / maxTokens) * 100}%`,
                          }}
                        />
                      </div>
                    </div>

                    {/* Mistral Bar */}
                    <div>
                      <div className="flex justify-between text-[11px] font-mono mb-1">
                        <span className="text-amber-400 font-bold">{mistralResult.model}</span>
                        <span className="text-slate-200">~{mistralResult.token_count_estimate} tok</span>
                      </div>
                      <div className="h-2.5 w-full bg-slate-900 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-amber-500/80 rounded-full transition-all duration-500"
                          style={{
                            width: `${(mistralResult.token_count_estimate / maxTokens) * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Shared Retrieved Context Panel */}
            <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
              <div className="flex items-center justify-between text-xs text-slate-300 mb-2">
                <span className="font-semibold text-sky-400">Identical ChromaDB Grounding Context</span>
                <span className="text-[11px] text-slate-500 font-mono">
                  {comparisonResult.source_chunks.length} Chunks Retrieved
                </span>
              </div>
              <SourceCitation
                sources={comparisonResult.sources}
                source_chunks={comparisonResult.source_chunks}
                retrieval_diagnostics={comparisonResult.retrieval_diagnostics}
              />
            </div>

            {/* Side-by-side Answer Panels */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {comparisonResult.models.map((modelItem) => {
                const isLlama = modelItem.model.includes('llama');
                const isFaster = fasterModel === modelItem.model;

                return (
                  <div
                    key={modelItem.model}
                    className={`bg-slate-950/90 border rounded-xl p-5 flex flex-col justify-between transition hover:border-slate-700 ${
                      isLlama ? 'border-sky-500/30' : 'border-amber-500/30'
                    }`}
                  >
                    <div>
                      {/* Card Header */}
                      <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-3">
                        <div className="flex items-center space-x-2.5">
                          <div
                            className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                              isLlama
                                ? 'bg-sky-500/20 text-sky-400 border border-sky-500/30'
                                : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                            }`}
                          >
                            <Cpu className="w-4 h-4" />
                          </div>
                          <div>
                            <div className="flex items-center space-x-2">
                              <h4 className="text-xs font-bold text-slate-100 uppercase tracking-wide font-mono">
                                {modelItem.model}
                              </h4>
                              {isFaster && (
                                <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 font-semibold">
                                  Faster
                                </span>
                              )}
                            </div>
                            <span className="text-[10px] text-slate-400 font-mono">
                              {isLlama ? 'Meta · 3.2B parameters' : 'Mistral AI · 7.3B parameters'}
                            </span>
                          </div>
                        </div>

                        {/* Actions */}
                        <div className="flex items-center space-x-2">
                          {modelItem.tokens_per_second && (
                            <div className="flex items-center space-x-1 text-emerald-400 text-xs font-mono font-medium bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                              <Zap className="w-3 h-3" />
                              <span>{modelItem.tokens_per_second} tok/s</span>
                            </div>
                          )}

                          <button
                            onClick={() => handleCopy(modelItem.answer, modelItem.model)}
                            className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition"
                            title="Copy model answer"
                          >
                            {copiedModel === modelItem.model ? (
                              <Check className="w-3.5 h-3.5 text-emerald-400" />
                            ) : (
                              <Copy className="w-3.5 h-3.5" />
                            )}
                          </button>
                        </div>
                      </div>

                      {/* Model Answer Body */}
                      <p className="text-xs text-slate-200 leading-relaxed whitespace-pre-wrap select-text mb-4">
                        {modelItem.answer}
                      </p>
                    </div>

                    {/* Metrics Footer */}
                    <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400 font-mono">
                      <span className="flex items-center space-x-1">
                        <Clock className="w-3 h-3 text-slate-500" />
                        <span>Latency: {modelItem.generation_latency_ms} ms</span>
                      </span>
                      <span>~{modelItem.token_count_estimate} tokens</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
