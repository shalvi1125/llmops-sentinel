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
  X,
} from 'lucide-react';
import { ActiveTab, HealthResponse } from '../types';
import { SentinelLogo } from './common/SentinelLogo';

interface SidebarProps {
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
  health: HealthResponse | null;
  healthLoading: boolean;
  healthError: string | null;
  apiKey: string;
  onApiKeyChange: (key: string) => void;
  mobileOpen: boolean;
  onMobileClose: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onTabChange,
  health,
  healthLoading,
  healthError,
  apiKey,
  onApiKeyChange,
  mobileOpen,
  onMobileClose,
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
    icon: React.ReactNode;
  }[] = [
    {
      id: 'overview',
      label: 'Overview',
      icon: <Gauge className="w-4 h-4" />,
    },
    {
      id: 'chat',
      label: 'Live Chat & Inspector',
      icon: <MessageSquare className="w-4 h-4" />,
    },
    {
      id: 'arena',
      label: 'Model Arena',
      icon: <Swords className="w-4 h-4" />,
    },
    {
      id: 'eval',
      label: 'Evaluation & Traces',
      icon: <ScrollText className="w-4 h-4" />,
    },
    {
      id: 'benchmarks',
      label: 'Benchmark Lab',
      icon: <FlaskConical className="w-4 h-4" />,
    },
  ];

  return (
    <>
      {/* Mobile Backdrop Overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40 lg:hidden"
          onClick={onMobileClose}
        />
      )}

      {/* Vertical Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-64 bg-slate-900/95 border-r border-slate-800 flex flex-col justify-between transition-transform duration-200 ease-in-out lg:static lg:translate-x-0 ${
          mobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Top: Branding Area */}
        <div className="p-5 border-b border-slate-800/80">
          <div className="flex items-center justify-between">
            <div
              className="flex items-center space-x-3 cursor-pointer group"
              onClick={() => {
                onTabChange('overview');
                onMobileClose();
              }}
              title="LLMOps Sentinel Overview"
            >
              <SentinelLogo size={32} />
              <div className="flex flex-col">
                <div className="flex items-center space-x-2">
                  <h1 className="font-bold text-base text-slate-100 tracking-tight font-sans group-hover:text-sky-300 transition">
                    LLMOps Sentinel
                  </h1>
                </div>
                <p className="text-[10px] text-slate-400 tracking-wide font-sans leading-tight mt-0.5">
                  AI Reliability & Observability Engine
                </p>
              </div>
            </div>

            {/* Mobile Close Button */}
            <button
              onClick={onMobileClose}
              className="lg:hidden p-1 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Middle: Vertically Stacked Navigation Links */}
        <div className="flex-1 py-4 px-3 overflow-y-auto">
          <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider px-3 mb-2 font-mono">
            Platform Workspaces
          </div>
          <nav className="space-y-1">
            {navItems.map((item) => {
              const active = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    onTabChange(item.id);
                    onMobileClose();
                  }}
                  className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-xs font-medium transition duration-150 ${
                    active
                      ? 'bg-sky-500/15 text-sky-300 border-l-2 border-sky-400 font-semibold shadow-sm shadow-sky-500/10'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border-l-2 border-transparent'
                  }`}
                >
                  <span className={active ? 'text-sky-400' : 'text-slate-400'}>
                    {item.icon}
                  </span>
                  <span className="truncate">{item.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Bottom / Sidebar Footer: Telemetry & Utility Controls */}
        <div className="p-3.5 border-t border-slate-800/80 bg-slate-950/60 space-y-3">
          {/* System Health Status Badge */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-2.5 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-[11px] font-semibold text-slate-400">System Telemetry</span>
              {healthLoading ? (
                <span className="flex items-center text-slate-400 text-[10px] font-mono">
                  <Activity className="w-3 h-3 animate-spin mr-1 text-sky-400" />
                  Connecting
                </span>
              ) : isHealthy ? (
                <span className="flex items-center text-emerald-400 font-medium text-[10px] font-mono">
                  <CheckCircle2 className="w-3 h-3 mr-1" />
                  Healthy
                </span>
              ) : (
                <span className="flex items-center text-amber-400 font-medium text-[10px] font-mono">
                  <AlertCircle className="w-3 h-3 mr-1" />
                  {healthError ? 'Offline' : 'Degraded'}
                </span>
              )}
            </div>

            {/* ChromaDB and Ollama status chips */}
            <div className="grid grid-cols-2 gap-1.5 text-[10px] font-mono">
              <div
                className="bg-slate-950 px-2 py-1 rounded border border-slate-800/80 flex items-center justify-between text-slate-400"
                title={`ChromaDB: ${health?.chromadb || 'unknown'}`}
              >
                <span className="flex items-center truncate">
                  <Database className="w-2.5 h-2.5 mr-1 text-sky-400 shrink-0" />
                  Chroma
                </span>
                <span
                  className={
                    health?.chromadb === 'ok' ? 'text-emerald-400 font-bold' : 'text-rose-400'
                  }
                >
                  {health?.chromadb === 'ok' ? 'OK' : '!'}
                </span>
              </div>

              <div
                className="bg-slate-950 px-2 py-1 rounded border border-slate-800/80 flex items-center justify-between text-slate-400"
                title={`Ollama: ${health?.ollama || 'unknown'}`}
              >
                <span className="flex items-center truncate">
                  <Cpu className="w-2.5 h-2.5 mr-1 text-indigo-400 shrink-0" />
                  Ollama
                </span>
                <span
                  className={
                    health?.ollama === 'ok' ? 'text-emerald-400 font-bold' : 'text-rose-400'
                  }
                >
                  {health?.ollama === 'ok' ? 'OK' : '!'}
                </span>
              </div>
            </div>
          </div>

          {/* Utility Actions: MLflow UI + Auth Modal Trigger */}
          <div className="flex items-center space-x-1.5">
            <a
              href="http://localhost:5000"
              target="_blank"
              rel="noopener noreferrer"
              className="flex-1 flex items-center justify-center space-x-1.5 text-xs text-slate-300 hover:text-white bg-slate-900 hover:bg-slate-800 border border-slate-800 py-1.5 rounded-lg transition"
              title="Open MLflow UI in new tab"
            >
              <span>MLflow UI</span>
              <ExternalLink className="w-3 h-3 text-slate-400" />
            </a>

            <button
              onClick={() => {
                setTempKey(apiKey);
                setShowKeyModal(true);
              }}
              className="flex-1 flex items-center justify-center space-x-1.5 text-xs text-slate-300 hover:text-white bg-slate-900 hover:bg-slate-800 border border-slate-800 py-1.5 rounded-lg transition"
              title="Configure API Key"
            >
              <Key className="w-3 h-3 text-sky-400" />
              <span>Auth</span>
            </button>
          </div>

          {/* Engine V2.0 Badge Strip */}
          <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono px-1">
            <span>LLMOps Core</span>
            <span className="px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20 font-semibold uppercase">
              Engine v2.0
            </span>
          </div>
        </div>
      </aside>

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
    </>
  );
};
