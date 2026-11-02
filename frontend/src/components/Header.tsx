import React, { useState } from 'react';
import {
  Activity,
  Key,
  ExternalLink,
  ShieldCheck,
  Database,
  Cpu,
  CheckCircle2,
  AlertCircle,
  MessageSquare,
  Swords,
  ScrollText,
  FlaskConical,
  Gauge,
} from 'lucide-react';
import { ActiveTab, HealthResponse } from '../types';
import { SentinelLogo } from './common/SentinelLogo';

interface HeaderProps {
  health: HealthResponse | null;
  healthLoading: boolean;
  healthError: string | null;
  apiKey: string;
  onApiKeyChange: (key: string) => void;
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  healthLoading,
  healthError,
  apiKey,
  onApiKeyChange,
  activeTab,
  onTabChange,
}) => {
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [tempKey, setTempKey] = useState(apiKey);

  const isHealthy =
    health?.status === 'ok' &&
    health?.chromadb === 'ok' &&
    health?.ollama === 'ok';

  const navItems: {
    id: ActiveTab;
    label: string;
    compactLabel?: React.ReactNode;
    icon: React.ReactNode;
  }[] = [
    {
      id: 'overview',
      label: 'Overview',
      icon: <Gauge className="w-3.5 h-3.5" />,
    },
    {
      id: 'chat',
      label: 'Live Chat & Inspector',
      compactLabel: (
        <>
          Live Chat <span className="hidden xl:inline">& Inspector</span>
        </>
      ),
      icon: <MessageSquare className="w-3.5 h-3.5" />,
    },
    {
      id: 'arena',
      label: 'Model Arena',
      icon: <Swords className="w-3.5 h-3.5" />,
    },
    {
      id: 'eval',
      label: 'Evaluation & Traces',
      compactLabel: (
        <>
          Evaluation <span className="hidden xl:inline">& Traces</span>
        </>
      ),
      icon: <ScrollText className="w-3.5 h-3.5" />,
    },
    {
      id: 'benchmarks',
      label: 'Benchmark Lab',
      icon: <FlaskConical className="w-3.5 h-3.5" />,
    },
  ];

  return (
    <header className="border-b border-slate-800 bg-slate-900/95 backdrop-blur sticky top-0 z-30">
      <div className="max-w-7xl mx-auto px-3 sm:px-5 lg:px-8">
        <div className="h-16 flex items-center justify-between gap-2 sm:gap-4">
          {/* Left: Brand area + Navigation tabs tightly grouped */}
          <div className="flex items-center space-x-3 lg:space-x-4 min-w-0">
            {/* 1. Compact Brand Area (Far Left) */}
            <div
              className="flex items-center space-x-2.5 cursor-pointer shrink-0"
              onClick={() => onTabChange('overview')}
              title="LLMOps Sentinel Overview"
            >
              <SentinelLogo size={30} />
              <div className="flex flex-col justify-center">
                <div className="flex items-center space-x-1.5 leading-none">
                  <span className="font-bold text-sm sm:text-base text-slate-100 tracking-tight whitespace-nowrap font-sans">
                    LLMOps Sentinel
                  </span>
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20 font-semibold tracking-wider uppercase font-mono">
                    v2.0
                  </span>
                </div>
                <span className="text-[10px] text-slate-400 leading-tight hidden sm:block whitespace-nowrap pt-0.5">
                  AI Reliability & Observability Engine
                </span>
              </div>
            </div>

            {/* Subtle Divider between Brand and Nav */}
            <div className="hidden md:block h-5 w-[1px] bg-slate-800 shrink-0" />

            {/* 2. Navigation Tabs (Immediately following branding on desktop/laptop) */}
            <nav className="hidden md:flex items-center space-x-1 bg-slate-950/70 p-1 rounded-xl border border-slate-800/80">
              {navItems.map((item) => {
                const active = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => onTabChange(item.id)}
                    className={`flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition duration-150 ${
                      active
                        ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30 shadow-sm'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                    }`}
                  >
                    <span className={active ? 'text-sky-400' : 'text-slate-400'}>
                      {item.icon}
                    </span>
                    <span>{item.compactLabel || item.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>

          {/* Right: Compact Utility Controls */}
          <div className="flex items-center space-x-2 shrink-0">
            {/* Health Indicator Badge */}
            <div className="flex items-center space-x-1.5 bg-slate-800/80 border border-slate-700/60 rounded-full px-2.5 py-1 text-xs">
              {healthLoading ? (
                <span className="flex items-center text-slate-400 text-[11px] whitespace-nowrap">
                  <Activity className="w-3 h-3 animate-spin mr-1 text-sky-400" />
                  Connecting...
                </span>
              ) : isHealthy ? (
                <span className="flex items-center text-emerald-400 font-medium text-[11px] whitespace-nowrap">
                  <CheckCircle2 className="w-3 h-3 mr-1" />
                  Healthy
                </span>
              ) : (
                <span className="flex items-center text-amber-400 font-medium text-[11px] whitespace-nowrap">
                  <AlertCircle className="w-3 h-3 mr-1" />
                  {healthError ? 'Offline' : 'Degraded'}
                </span>
              )}

              <div className="hidden xl:flex items-center space-x-1.5 pl-1.5 border-l border-slate-700 text-[10px] text-slate-400 font-mono">
                <span
                  className="flex items-center whitespace-nowrap"
                  title={`ChromaDB: ${health?.chromadb || 'unknown'}`}
                >
                  <Database className="w-2.5 h-2.5 mr-1 text-sky-400" />
                  {health?.chromadb === 'ok' ? 'Chroma' : 'Chroma!'}
                </span>
                <span
                  className="flex items-center whitespace-nowrap"
                  title={`Ollama: ${health?.ollama || 'unknown'}`}
                >
                  <Cpu className="w-2.5 h-2.5 mr-1 text-indigo-400" />
                  {health?.ollama === 'ok' ? 'Ollama' : 'Ollama!'}
                </span>
              </div>
            </div>

            {/* External MLflow Link */}
            <a
              href="http://localhost:5000"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden sm:flex items-center space-x-1 text-xs text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700/80 border border-slate-700 px-2.5 py-1 rounded-lg transition whitespace-nowrap"
              title="Open MLflow UI in new tab"
            >
              <span>MLflow UI</span>
              <ExternalLink className="w-3 h-3 text-slate-400" />
            </a>

            {/* API Key Auth Button */}
            <button
              onClick={() => {
                setTempKey(apiKey);
                setShowKeyModal(true);
              }}
              className="flex items-center space-x-1 text-xs text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700/80 border border-slate-700 px-2.5 py-1 rounded-lg transition whitespace-nowrap"
              title="Configure API Key"
            >
              <Key className="w-3 h-3 text-sky-400" />
              <span>Auth</span>
            </button>
          </div>
        </div>

        {/* Mobile Navigation Bar (Clean collapse below md breakpoint) */}
        <div className="flex md:hidden items-center justify-around py-1.5 border-t border-slate-800/80 text-xs overflow-x-auto">
          {navItems.map((item) => {
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={`flex items-center space-x-1 px-2 py-1 rounded-md transition whitespace-nowrap ${
                  active
                    ? 'bg-sky-500/20 text-sky-300 font-medium'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {item.icon}
                <span className="text-[11px]">{item.label.split(' ')[0]}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* API Key Modal */}
      {showKeyModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-xl max-w-md w-full p-6 shadow-2xl">
            <div className="flex items-center space-x-2 text-sky-400 mb-2">
              <ShieldCheck className="w-5 h-5" />
              <h3 className="font-semibold text-slate-100">LLMOps Sentinel Authentication</h3>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Enter your backend <code className="bg-slate-800 px-1 py-0.5 rounded text-slate-300">API_KEY</code>.
              Requests pass this in the <code className="text-sky-300">X-API-Key</code> header.
            </p>
            <input
              type="text"
              value={tempKey}
              onChange={(e) => setTempKey(e.target.value)}
              placeholder="e.g. ragscope-local-dev-key"
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500 font-mono mb-4"
            />
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowKeyModal(false)}
                className="px-4 py-2 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  onApiKeyChange(tempKey);
                  setShowKeyModal(false);
                }}
                className="px-4 py-2 rounded-lg text-xs font-medium bg-sky-500 hover:bg-sky-400 text-white font-semibold transition"
              >
                Save Key
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
};
