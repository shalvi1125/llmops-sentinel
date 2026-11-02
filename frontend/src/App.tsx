import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { OverviewView } from './components/OverviewView';
import { DocumentUpload } from './components/DocumentUpload';
import { ChatInterface } from './components/ChatInterface';
import { ModelArena } from './components/ModelArena';
import { AuditLogView } from './components/AuditLogView';
import { BenchmarkLabView } from './components/BenchmarkLabView';
import { checkHealth, getPrompts } from './api';
import { ActiveTab, HealthResponse, PromptStrategy, UploadedDoc } from './types';

export function App() {
  const [apiKey, setApiKey] = useState('ragscope-local-dev-key');
  const [activeTab, setActiveTab] = useState<ActiveTab>('overview');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [uploadedDocs, setUploadedDocs] = useState<UploadedDoc[]>([]);
  const [prompts, setPrompts] = useState<PromptStrategy[]>([]);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  const pollHealth = async () => {
    try {
      const data = await checkHealth();
      setHealth(data);
      setHealthError(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Cannot reach backend';
      setHealthError(msg);
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }
  };

  useEffect(() => {
    pollHealth();
    const interval = setInterval(pollHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (apiKey) {
      getPrompts(apiKey)
        .then((data) => setPrompts(data))
        .catch(() => {});
    }
  }, [apiKey]);

  const handleDocumentIngested = (doc: UploadedDoc) => {
    setUploadedDocs((prev) => [doc, ...prev]);
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 font-sans text-slate-100 selection:bg-sky-500/30 selection:text-sky-200">
      {/* 1. Left Vertical Sidebar (Fixed Width ~256px, Full Viewport Height) */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        health={health}
        healthLoading={healthLoading}
        healthError={healthError}
        apiKey={apiKey}
        onApiKeyChange={setApiKey}
        mobileOpen={mobileSidebarOpen}
        onMobileClose={() => setMobileSidebarOpen(false)}
      />

      {/* 2. Right Main Content Area (Fills Remaining Width) */}
      <div id="main-stage" className="flex-1 flex flex-col h-screen min-w-0 overflow-hidden bg-slate-950">
        {/* Minimal Top Bar in Main Content */}
        <TopBar
          activeTab={activeTab}
          onMobileMenuToggle={() => setMobileSidebarOpen(true)}
        />

        {/* Dynamic Workspace Container */}
        <main className="flex-1 overflow-y-auto p-4 sm:p-5 lg:p-6 flex flex-col min-h-0">
          {activeTab === 'overview' && (
            <OverviewView
              apiKey={apiKey}
              health={health}
              healthLoading={healthLoading}
              onNavigateTab={setActiveTab}
            />
          )}

          {activeTab === 'chat' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 flex-1 min-h-0 h-full">
              {/* Left Column: Knowledge Ingestion */}
              <div className="lg:col-span-4 h-full min-h-[380px] lg:min-h-0 flex flex-col">
                <DocumentUpload
                  apiKey={apiKey}
                  uploadedDocs={uploadedDocs}
                  onDocumentIngested={handleDocumentIngested}
                />
              </div>

              {/* Right Column: Live Chat & Inspector */}
              <div className="lg:col-span-8 h-full min-h-[480px] lg:min-h-0 flex flex-col">
                <ChatInterface apiKey={apiKey} />
              </div>
            </div>
          )}

          {activeTab === 'arena' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 flex-1 min-h-0 h-full">
              {/* Left Column: Knowledge Ingestion */}
              <div className="lg:col-span-4 h-full min-h-[380px] lg:min-h-0 flex flex-col">
                <DocumentUpload
                  apiKey={apiKey}
                  uploadedDocs={uploadedDocs}
                  onDocumentIngested={handleDocumentIngested}
                />
              </div>

              {/* Right Column: Model Arena */}
              <div className="lg:col-span-8 h-full min-h-[480px] lg:min-h-0 flex flex-col">
                <ModelArena apiKey={apiKey} prompts={prompts} />
              </div>
            </div>
          )}

          {activeTab === 'eval' && (
            <div className="flex-1 min-h-0 h-full flex flex-col">
              <AuditLogView apiKey={apiKey} />
            </div>
          )}

          {activeTab === 'benchmarks' && (
            <div className="flex-1 min-h-0 h-full flex flex-col">
              <BenchmarkLabView apiKey={apiKey} />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
