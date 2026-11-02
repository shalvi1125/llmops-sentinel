import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Bot,
  User,
  Loader2,
  Sparkles,
  SlidersHorizontal,
  Trash2,
  AlertCircle,
  Cpu,
  Clock,
  Layers,
  ShieldCheck,
} from 'lucide-react';
import { getPrompts, queryRAG } from '../api';
import { ChatMessage, PromptStrategy } from '../types';
import { SourceCitation } from './SourceCitation';

interface ChatInterfaceProps {
  apiKey: string;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ apiKey }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState<'idle' | 'generating' | 'evaluating'>('idle');
  const [topK, setTopK] = useState(3);
  const [promptId, setPromptId] = useState('default_qa');
  const [availablePrompts, setAvailablePrompts] = useState<PromptStrategy[]>([]);
  const [showSettings, setShowSettings] = useState(false);
  const [generalError, setGeneralError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  useEffect(() => {
    if (apiKey) {
      getPrompts(apiKey)
        .then((prompts) => setAvailablePrompts(prompts))
        .catch(() => {});
    }
  }, [apiKey]);

  const handleSend = async (questionText?: string) => {
    const textToSend = (questionText || inputValue).trim();
    if (!textToSend || isLoading) return;

    setGeneralError(null);
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: textToSend,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setLoadingStage('generating');

    const evalTimer = setTimeout(() => {
      setLoadingStage('evaluating');
    }, 10000);

    try {
      const response = await queryRAG(textToSend, topK, apiKey, promptId);
      clearTimeout(evalTimer);

      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
        source_chunks: response.source_chunks,
        retrieval_diagnostics: response.retrieval_diagnostics,
        prompt_id: response.prompt_id,
        prompt_version: response.prompt_version,
        model: response.model,
        generation_latency_ms: response.generation_latency_ms,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: unknown) {
      clearTimeout(evalTimer);
      const errMsg = err instanceof Error ? err.message : 'Query execution failed.';
      const errorMessage: ChatMessage = {
        id: `err-${Date.now()}`,
        role: 'assistant',
        content: `Execution Notice: ${errMsg}`,
        timestamp: new Date(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
      setGeneralError(errMsg);
    } finally {
      setIsLoading(false);
      setLoadingStage('idle');
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const sampleQuestions = [
    'What is LLMOps Sentinel and what database does it use?',
    'Which local models are used for generation, judge evaluation, and embeddings?',
    'How does zero-latency preemption prevent query timeouts during long judge evaluations?',
  ];

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl flex flex-col h-full overflow-hidden">
      {/* Top Bar / Inspector Controls */}
      <div className="px-5 py-3 border-b border-slate-800 bg-slate-950/40 flex items-center justify-between text-xs">
        <div className="flex items-center space-x-2">
          <Bot className="w-4 h-4 text-sky-400" />
          <span className="font-semibold text-slate-200">Live Chat & Inspector</span>
          <span className="text-slate-400 hidden sm:inline">· Primary Model: Llama 3.2 (3B)</span>
        </div>

        <div className="flex items-center space-x-2 sm:space-x-3">
          {/* Prompt Strategy Selector */}
          <div className="flex items-center space-x-1.5">
            <span className="text-[11px] text-slate-400 hidden sm:inline">Prompt:</span>
            <select
              value={promptId}
              onChange={(e) => setPromptId(e.target.value)}
              className="bg-slate-800/90 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1 focus:outline-none focus:border-sky-500 font-mono"
              title="Select Prompt Strategy"
            >
              {availablePrompts.length > 0 ? (
                availablePrompts.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} (v{p.version})
                  </option>
                ))
              ) : (
                <>
                  <option value="default_qa">Balanced QA (v1.0)</option>
                  <option value="strict_grounding">Strict Grounding (v1.0)</option>
                  <option value="concise_facts">Concise Facts (v1.0)</option>
                </>
              )}
            </select>
          </div>

          {/* Top_K toggle */}
          <div className="relative">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-lg border text-xs font-mono transition ${
                showSettings
                  ? 'bg-sky-500/10 border-sky-500/40 text-sky-300'
                  : 'bg-slate-800/80 border-slate-700 text-slate-300 hover:text-white'
              }`}
              title="Configure Retrieval Top-K"
            >
              <SlidersHorizontal className="w-3.5 h-3.5 text-sky-400" />
              <span>k = {topK}</span>
            </button>

            {showSettings && (
              <div className="absolute right-0 mt-2 w-64 bg-slate-900 border border-slate-700 rounded-xl p-4 shadow-2xl z-30">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-slate-200 text-xs">Top-K Context Chunks</span>
                  <span className="font-mono text-sky-400 font-bold text-xs">{topK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="20"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                  className="w-full accent-sky-500 cursor-pointer"
                />
                <p className="text-[11px] text-slate-400 mt-2">
                  Number of most relevant vector chunks retrieved from ChromaDB (1–20).
                </p>
              </div>
            )}
          </div>

          {/* Clear Messages */}
          {messages.length > 0 && (
            <button
              onClick={() => setMessages([])}
              className="p-1.5 text-slate-400 hover:text-rose-400 hover:bg-slate-800 rounded-lg transition"
              title="Clear conversation"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Message Thread */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 my-auto">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-sky-500/20 to-indigo-500/20 border border-sky-500/30 flex items-center justify-center mb-4 shadow-lg shadow-sky-500/10">
              <Sparkles className="w-6 h-6 text-sky-400" />
            </div>
            <h3 className="font-semibold text-base text-slate-100 mb-1">
              Ask Anything About Your Ingested Documents
            </h3>
            <p className="text-xs text-slate-400 max-w-md mb-6 leading-relaxed">
              Questions are embedded via <code className="text-sky-300 font-mono">nomic-embed-text</code>, retrieved from ChromaDB with distance scoring, synthesized via <code className="text-sky-300 font-mono">Llama 3.2</code>, and audited asynchronously by MLflow GenAI judges.
            </p>

            {/* Starter Chips */}
            <div className="space-y-2 w-full max-w-md">
              <span className="text-[11px] uppercase tracking-wider text-slate-500 font-semibold block mb-2 font-mono">
                Suggested Telemetry Queries
              </span>
              {sampleQuestions.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(q)}
                  className="w-full text-left text-xs bg-slate-950/60 hover:bg-slate-800/80 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white px-3.5 py-2.5 rounded-lg transition"
                >
                  "{q}"
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex items-start space-x-3 ${
                msg.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {msg.role === 'assistant' && (
                <div
                  className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5 ${
                    msg.isError
                      ? 'bg-rose-950/80 text-rose-400 border border-rose-800'
                      : 'bg-sky-950/80 text-sky-400 border border-sky-800'
                  }`}
                >
                  <Bot className="w-4 h-4" />
                </div>
              )}

              <div
                className={`max-w-[88%] rounded-2xl px-4 py-3 text-xs leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-sky-600 text-white rounded-tr-none shadow-md shadow-sky-600/20'
                    : msg.isError
                    ? 'bg-rose-950/40 border border-rose-800/80 text-rose-200 rounded-tl-none'
                    : 'bg-slate-950/90 border border-slate-800 text-slate-100 rounded-tl-none shadow-sm'
                }`}
              >
                <div className="whitespace-pre-wrap">{msg.content}</div>

                {/* Structured Source Citations & Diagnostics */}
                {((msg.source_chunks && msg.source_chunks.length > 0) ||
                  (msg.sources && msg.sources.length > 0)) && (
                  <SourceCitation
                    sources={msg.sources}
                    source_chunks={msg.source_chunks}
                    retrieval_diagnostics={msg.retrieval_diagnostics}
                  />
                )}

                {/* Execution Inspector Footer */}
                {msg.role === 'assistant' && !msg.isError && (
                  <div className="mt-2.5 pt-2 border-t border-slate-800 flex flex-wrap items-center justify-between text-[10px] text-slate-400 gap-2 font-mono">
                    <div className="flex flex-wrap items-center gap-1.5">
                      {msg.model && (
                        <span className="flex items-center space-x-1 bg-slate-900 border border-slate-800 px-1.5 py-0.5 rounded text-sky-300">
                          <Cpu className="w-3 h-3 text-sky-400" />
                          <span>{msg.model}</span>
                        </span>
                      )}

                      {msg.prompt_id && (
                        <span className="flex items-center space-x-1 bg-slate-900 border border-slate-800 px-1.5 py-0.5 rounded text-indigo-300">
                          <Layers className="w-3 h-3 text-indigo-400" />
                          <span>{msg.prompt_id}</span>
                        </span>
                      )}

                      {msg.generation_latency_ms && (
                        <span className="flex items-center space-x-1 bg-slate-900 border border-slate-800 px-1.5 py-0.5 rounded text-slate-300">
                          <Clock className="w-3 h-3 text-slate-400" />
                          <span>Gen: {msg.generation_latency_ms} ms</span>
                        </span>
                      )}

                      <span className="text-emerald-400 font-medium flex items-center space-x-1">
                        <ShieldCheck className="w-3 h-3" />
                        <span>MLflow Trace Synced</span>
                      </span>
                    </div>

                    <span>
                      {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                )}

                {msg.role === 'user' && (
                  <div className="text-[10px] text-sky-200/70 text-right mt-1 font-mono">
                    {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                )}
              </div>

              {msg.role === 'user' && (
                <div className="w-7 h-7 rounded-lg bg-indigo-950/80 border border-indigo-800 flex items-center justify-center shrink-0 mt-0.5 text-indigo-400">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          ))
        )}

        {/* Loading Indicator */}
        {isLoading && (
          <div className="flex items-start space-x-3">
            <div className="w-7 h-7 rounded-lg bg-sky-950/80 border border-sky-800 flex items-center justify-center shrink-0 mt-0.5 text-sky-400">
              <Bot className="w-4 h-4" />
            </div>
            <div className="bg-slate-950/80 border border-slate-800 rounded-2xl rounded-tl-none px-4 py-3 text-xs text-slate-300">
              <div className="flex items-center space-x-2.5">
                <Loader2 className="w-4 h-4 text-sky-400 animate-spin" />
                <span className="font-semibold text-slate-200">
                  {loadingStage === 'evaluating'
                    ? 'Synthesizing response with Llama 3.2...'
                    : 'Retrieving context from ChromaDB & generating answer...'}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1 pl-6.5 font-mono">
                {loadingStage === 'evaluating'
                  ? 'Context retrieved · MLflow judge evaluation scheduled in background'
                  : 'Searching top-k vector embeddings'}
              </p>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Error Notice */}
      {generalError && (
        <div className="mx-5 mb-2 px-3 py-2 bg-rose-950/30 border border-rose-800/40 rounded-lg text-xs text-rose-300 flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
          <span className="truncate">{generalError}</span>
        </div>
      )}

      {/* Input Box */}
      <div className="p-4 border-t border-slate-800 bg-slate-950/60">
        <div className="relative flex items-end bg-slate-900 border border-slate-700/80 rounded-xl focus-within:border-sky-500 focus-within:ring-1 focus-within:ring-sky-500 transition shadow-inner">
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about the document... (Press Enter to send)"
            rows={1}
            disabled={isLoading}
            className="w-full bg-transparent px-4 py-3 text-xs text-slate-100 placeholder-slate-500 focus:outline-none resize-none max-h-32 min-h-[44px]"
          />
          <button
            onClick={() => handleSend()}
            disabled={!inputValue.trim() || isLoading}
            className="m-2 p-2 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:bg-slate-800 disabled:text-slate-600 text-white transition flex items-center justify-center shrink-0 shadow-md shadow-sky-600/20"
            title="Send Question"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
        <div className="flex items-center justify-between text-[11px] text-slate-400 mt-2 px-1 font-mono">
          <span>Enter to submit · Shift+Enter for new line</span>
          <span>Zero-Latency Preemption · Offline RAG</span>
        </div>
      </div>
    </div>
  );
};
