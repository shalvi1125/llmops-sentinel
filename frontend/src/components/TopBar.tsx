import React from 'react';
import { Menu, Sparkles } from 'lucide-react';
import { ActiveTab } from '../types';

interface TopBarProps {
  activeTab: ActiveTab;
  onMobileMenuToggle: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({ activeTab, onMobileMenuToggle }) => {
  const titles: Record<ActiveTab, { title: string; subtitle: string; tag: string }> = {
    overview: {
      title: 'Sentinel Overview',
      subtitle: 'System telemetry, latency distribution, and evaluation reliability hub',
      tag: 'Executive Telemetry',
    },
    chat: {
      title: 'Live Chat & Inspector',
      subtitle: 'Knowledge ingestion, vector retrieval diagnostics, and answer synthesis',
      tag: 'Offline RAG',
    },
    arena: {
      title: 'Model Arena',
      subtitle: 'Sequential side-by-side inference benchmark: Llama 3.2 (3B) vs Mistral (7B)',
      tag: 'Dual Inference',
    },
    eval: {
      title: 'Evaluation & Traces',
      subtitle: 'Real-time MLflow trace audit log with Mistral judge rationale breakdowns',
      tag: 'Judge Telemetry',
    },
    benchmarks: {
      title: 'Benchmark Lab',
      subtitle: 'Curated golden test dataset regression suite with automated pass metrics',
      tag: 'Regression Lab',
    },
  };

  const current = titles[activeTab] || titles.overview;

  return (
    <header className="h-14 border-b border-slate-800 bg-slate-900/60 backdrop-blur px-4 sm:px-6 flex items-center justify-between shrink-0">
      <div className="flex items-center space-x-3">
        {/* Mobile Hamburger Menu Toggle */}
        <button
          onClick={onMobileMenuToggle}
          className="lg:hidden p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          title="Toggle Navigation Menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-sm sm:text-base font-bold text-slate-100 tracking-tight font-sans">
              {current.title}
            </h2>
            <span className="text-[9px] font-mono font-semibold px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20 uppercase">
              {current.tag}
            </span>
          </div>
          <p className="text-[11px] text-slate-400 hidden sm:block truncate max-w-xl">
            {current.subtitle}
          </p>
        </div>
      </div>

      <div className="flex items-center space-x-2 text-xs font-mono text-slate-500">
        <span className="hidden md:flex items-center text-sky-400/80 text-[11px]">
          <Sparkles className="w-3 h-3 mr-1" />
          Active Workspace
        </span>
      </div>
    </header>
  );
};
