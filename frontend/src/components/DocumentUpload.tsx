import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Upload,
  FileText,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  Sparkles,
  Files,
  Database,
  Trash2,
  RefreshCw,
} from 'lucide-react';
import { ingestFile, getIndexedDocuments, clearAllDocuments, deleteDocument } from '../api';
import { UploadedDoc, IndexedDocItem } from '../types';

interface DocumentUploadProps {
  apiKey: string;
  uploadedDocs: UploadedDoc[];
  onDocumentIngested: (doc: UploadedDoc) => void;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({
  apiKey,
  onDocumentIngested,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [clearExistingOnUpload, setClearExistingOnUpload] = useState(false);
  const [serverDocs, setServerDocs] = useState<IndexedDocItem[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [deletingFilename, setDeletingFilename] = useState<string | null>(null);
  const [showConfirmClear, setShowConfirmClear] = useState(false);
  const [isClearingAll, setIsClearingAll] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const refreshDocuments = useCallback(async () => {
    if (!apiKey) return;
    setLoadingDocs(true);
    try {
      const docs = await getIndexedDocuments(apiKey);
      setServerDocs(docs);
    } catch {
      // Backend might be warming up
    } finally {
      setLoadingDocs(false);
    }
  }, [apiKey]);

  useEffect(() => {
    refreshDocuments();
  }, [refreshDocuments]);

  const handleUpload = async (file: File) => {
    const lowerName = file.name.toLowerCase();
    if (!lowerName.endsWith('.pdf') && !lowerName.endsWith('.txt')) {
      setUploadStatus({
        type: 'error',
        message: `Unsupported file '${file.name}'. Only .pdf and .txt are supported.`,
      });
      return;
    }

    if (file.size > 15 * 1024 * 1024) {
      setUploadStatus({
        type: 'error',
        message: 'File size exceeds maximum limit of 15 MB.',
      });
      return;
    }

    setIsUploading(true);
    setUploadStatus(null);

    try {
      const res = await ingestFile(file, apiKey, clearExistingOnUpload);
      const newDoc: UploadedDoc = {
        filename: res.filename,
        chunksStored: res.chunks_stored,
        uploadedAt: new Date(),
      };
      onDocumentIngested(newDoc);
      setUploadStatus({
        type: 'success',
        message: `${clearExistingOnUpload ? 'Replaced knowledge base with' : 'Indexed'} "${res.filename}" (${res.chunks_stored} chunk${res.chunks_stored > 1 ? 's' : ''}).`,
      });
      await refreshDocuments();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Upload failed. Check backend connection.';
      setUploadStatus({
        type: 'error',
        message: msg,
      });
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleDeleteSingle = async (filename: string) => {
    if (deletingFilename) return;
    setDeletingFilename(filename);
    try {
      await deleteDocument(filename, apiKey);
      setUploadStatus({
        type: 'success',
        message: `Removed "${filename}" from ChromaDB.`,
      });
      await refreshDocuments();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to delete document.';
      setUploadStatus({
        type: 'error',
        message: msg,
      });
    } finally {
      setDeletingFilename(null);
    }
  };

  const handleClearAll = async () => {
    setIsClearingAll(true);
    setShowConfirmClear(false);
    try {
      const res = await clearAllDocuments(apiKey);
      setUploadStatus({
        type: 'success',
        message: `ChromaDB vector store cleared (${res.deleted_chunks} chunks removed).`,
      });
      await refreshDocuments();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to clear documents.';
      setUploadStatus({
        type: 'error',
        message: msg,
      });
    } finally {
      setIsClearingAll(false);
    }
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 flex flex-col h-full">
      {/* Panel Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <Database className="w-4 h-4 text-sky-400" />
          <h2 className="font-semibold text-sm text-slate-200">Knowledge Ingestion</h2>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20">
            ChromaDB
          </span>
          <button
            onClick={() => refreshDocuments()}
            disabled={loadingDocs}
            title="Refresh Knowledge Base list"
            className="text-slate-400 hover:text-sky-400 transition p-1"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loadingDocs ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      <p className="text-xs text-slate-400 mb-3 leading-relaxed">
        Upload PDF or TXT documents. Text is split into 1,200-char chunks, embedded via{' '}
        <code className="text-sky-300 font-mono text-[11px]">nomic-embed-text</code>, and stored in
        ChromaDB.
      </p>

      {/* Option: Clear Existing Before Upload */}
      <div className="mb-3 flex items-center justify-between bg-slate-950/40 border border-slate-800/80 rounded-lg px-3 py-2 text-xs">
        <label className="flex items-center space-x-2 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={clearExistingOnUpload}
            onChange={(e) => setClearExistingOnUpload(e.target.checked)}
            className="rounded border-slate-700 bg-slate-900 text-sky-500 focus:ring-sky-500/20 h-3.5 w-3.5"
          />
          <span className="text-slate-300 font-medium">Replace existing knowledge</span>
        </label>
        <span className="text-[10px] text-slate-400">
          {clearExistingOnUpload ? 'Clears old docs on upload' : 'Appends to KB'}
        </span>
      </div>

      {/* Drag & Drop Zone */}
      <div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => !isUploading && fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-5 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 ${
          isDragging
            ? 'border-sky-500 bg-sky-500/10 shadow-lg shadow-sky-500/10 scale-[1.01]'
            : isUploading
            ? 'border-slate-700 bg-slate-950/60 cursor-not-allowed opacity-80'
            : 'border-slate-700/80 hover:border-slate-600 hover:bg-slate-800/40 bg-slate-950/40'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleUpload(e.target.files[0]);
            }
          }}
          disabled={isUploading}
        />

        {isUploading ? (
          <div className="flex flex-col items-center py-2">
            <Loader2 className="w-8 h-8 text-sky-400 animate-spin mb-2" />
            <span className="text-xs font-semibold text-slate-200">
              Chunking & Embedding Document...
            </span>
            <span className="text-[11px] text-slate-400 font-mono mt-1">
              Generating vectors via Ollama
            </span>
          </div>
        ) : (
          <>
            <div className="w-9 h-9 rounded-xl bg-slate-800 flex items-center justify-center mb-2.5 text-slate-400 group-hover:text-sky-400 transition">
              <Upload className="w-4 h-4 text-sky-400" />
            </div>
            <span className="text-xs font-medium text-slate-200">
              Click to browse or drop file here
            </span>
            <span className="text-[11px] text-slate-400 mt-0.5">Supports .pdf and .txt (max 15 MB)</span>
          </>
        )}
      </div>

      {/* Upload Feedback Alert */}
      {uploadStatus && (
        <div
          className={`mt-3 p-2.5 rounded-lg text-xs flex items-start space-x-2.5 transition ${
            uploadStatus.type === 'success'
              ? 'bg-emerald-950/40 border border-emerald-800/60 text-emerald-300'
              : 'bg-rose-950/40 border border-rose-800/60 text-rose-300'
          }`}
        >
          {uploadStatus.type === 'success' ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
          ) : (
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          )}
          <span className="leading-snug">{uploadStatus.message}</span>
        </div>
      )}

      {/* Active Knowledge Base Section */}
      <div className="mt-4 flex-1 flex flex-col min-h-0">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-1.5 text-xs font-semibold text-slate-300">
            <Files className="w-3.5 h-3.5 text-slate-400" />
            <span>Active Knowledge Base</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-[10px] text-slate-400 font-mono bg-slate-800 px-1.5 py-0.5 rounded border border-slate-700">
              {serverDocs.length} File{serverDocs.length === 1 ? '' : 's'}
            </span>
            {serverDocs.length > 0 && !showConfirmClear && (
              <button
                onClick={() => setShowConfirmClear(true)}
                disabled={isClearingAll}
                className="text-[10px] text-rose-400 hover:text-rose-300 transition flex items-center space-x-1 bg-rose-950/30 border border-rose-800/40 px-1.5 py-0.5 rounded"
                title="Wipe all documents from vector store"
              >
                <Trash2 className="w-2.5 h-2.5 mr-0.5" />
                Clear All
              </button>
            )}
          </div>
        </div>

        {/* Clear Confirmation Prompt */}
        {showConfirmClear && (
          <div className="mb-2 p-2 bg-rose-950/60 border border-rose-800 rounded-lg text-xs flex items-center justify-between">
            <span className="text-rose-200 text-[11px]">Wipe all {serverDocs.length} document(s)?</span>
            <div className="flex items-center space-x-1.5">
              <button
                onClick={handleClearAll}
                disabled={isClearingAll}
                className="px-2 py-0.5 rounded bg-rose-600 hover:bg-rose-500 text-white text-[11px] font-medium"
              >
                {isClearingAll ? 'Clearing...' : 'Yes, Wipe'}
              </button>
              <button
                onClick={() => setShowConfirmClear(false)}
                className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px]"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {serverDocs.length === 0 ? (
          <div className="bg-slate-950/40 border border-slate-800/60 rounded-xl p-4 text-center my-auto">
            <FileText className="w-7 h-7 text-slate-600 mx-auto mb-2" />
            <p className="text-xs text-slate-400 font-medium">No documents currently in ChromaDB.</p>
            <p className="text-[11px] text-slate-500 mt-1">
              Upload your document above to ground query answers in verified context.
            </p>
          </div>
        ) : (
          <div className="overflow-y-auto space-y-2 pr-1 flex-1">
            {serverDocs.map((doc) => (
              <div
                key={doc.filename}
                className="bg-slate-950/70 border border-slate-800/90 rounded-lg p-2.5 text-xs hover:border-slate-700 transition"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2 truncate">
                    <FileText className="w-3.5 h-3.5 text-sky-400 shrink-0" />
                    <span className="font-medium text-slate-200 truncate" title={doc.filename}>
                      {doc.filename}
                    </span>
                  </div>
                  <div className="flex items-center space-x-1.5 shrink-0 ml-2">
                    <span className="text-[10px] text-emerald-400 bg-emerald-950/50 border border-emerald-800/40 px-1.5 py-0.5 rounded font-mono">
                      {doc.chunks} chunk{doc.chunks > 1 ? 's' : ''}
                    </span>
                    <button
                      onClick={() => handleDeleteSingle(doc.filename)}
                      disabled={deletingFilename === doc.filename}
                      title={`Remove "${doc.filename}" from ChromaDB`}
                      className="text-slate-400 hover:text-rose-400 transition p-1"
                    >
                      {deletingFilename === doc.filename ? (
                        <Loader2 className="w-3 h-3 animate-spin text-rose-400" />
                      ) : (
                        <Trash2 className="w-3 h-3" />
                      )}
                    </button>
                  </div>
                </div>
                <div className="text-[10px] text-slate-400 mt-1.5 flex items-center justify-between font-mono">
                  <span className="text-slate-500 text-[10px]">Indexed in ChromaDB</span>
                  <span className="flex items-center text-sky-400">
                    <Sparkles className="w-2.5 h-2.5 mr-1" />
                    Active Context
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
