import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, Loader2, X } from 'lucide-react';
import PageHeader from '@/components/PageHeader';
import { uploadContract } from '@/lib/api';
import { formatBytes, cn } from '@/lib/utils';
import type { Jurisdiction } from '@/types';

export default function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [jurisdiction, setJurisdiction] = useState<Jurisdiction>('US');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted.length > 0) {
      setFile(accepted[0]);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024, // 50MB
  });

  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const contract = await uploadContract(file, jurisdiction);
      navigate(`/analysis/${contract.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed');
      setUploading(false);
    }
  }

  return (
    <>
      <PageHeader
        title="Upload Contract"
        description="Drop a contract file to begin AI-powered clause extraction and risk analysis"
      />

      <div className="p-8 max-w-3xl">
        {/* Dropzone */}
        {!file ? (
          <div
            {...getRootProps()}
            className={cn(
              'border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors',
              isDragActive
                ? 'border-brand-500 bg-brand-50'
                : 'border-slate-300 hover:border-brand-400 hover:bg-slate-50'
            )}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto text-slate-400 mb-4" size={48} />
            <p className="text-lg font-medium text-slate-700 mb-1">
              {isDragActive ? 'Drop the file here' : 'Drag & drop a contract'}
            </p>
            <p className="text-sm text-slate-500 mb-4">or click to browse</p>
            <p className="text-xs text-slate-400">PDF, DOCX, or TXT · up to 50 MB</p>
          </div>
        ) : (
          <div className="card p-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg bg-brand-50 flex items-center justify-center flex-shrink-0">
                <FileText size={22} className="text-brand-600" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="font-medium text-slate-900 truncate">{file.name}</div>
                <div className="text-sm text-slate-500">{formatBytes(file.size)}</div>
              </div>
              <button
                onClick={() => setFile(null)}
                disabled={uploading}
                className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-md transition-colors disabled:opacity-50"
              >
                <X size={18} />
              </button>
            </div>

            {/* Jurisdiction selector */}
            <div className="mt-6 pt-6 border-t border-slate-100">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Jurisdiction
              </label>
              <p className="text-xs text-slate-500 mb-3">
                Selects the appropriate LoRA adapter for legal domain context.
              </p>
              <div className="flex gap-2">
                {(['US', 'EU', 'India'] as Jurisdiction[]).map((j) => (
                  <button
                    key={j}
                    onClick={() => setJurisdiction(j)}
                    disabled={uploading}
                    className={cn(
                      'px-4 py-2 rounded-lg text-sm font-medium border transition-colors',
                      jurisdiction === j
                        ? 'bg-brand-600 text-white border-brand-600'
                        : 'bg-white text-slate-700 border-slate-200 hover:border-slate-300'
                    )}
                  >
                    {j}
                  </button>
                ))}
              </div>
            </div>

            {error && (
              <div className="mt-4 p-3 rounded-lg bg-risk-high/10 text-risk-high text-sm border border-risk-high/30">
                {error}
              </div>
            )}

            <div className="mt-6 flex gap-2 justify-end">
              <button
                onClick={() => setFile(null)}
                disabled={uploading}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={handleUpload}
                disabled={uploading}
                className="btn-primary inline-flex items-center gap-2"
              >
                {uploading ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  'Analyze Contract'
                )}
              </button>
            </div>
          </div>
        )}

        {/* What happens next */}
        <div className="mt-8 card p-6">
          <h3 className="font-semibold text-slate-900 mb-4">What happens after upload</h3>
          <ol className="space-y-3 text-sm text-slate-600">
            {[
              'Document is parsed (PyMuPDF + OCR if needed) and chunked into semantic blocks.',
              'Chunks are embedded with e5-large-v2 and indexed in Qdrant for retrieval.',
              'The fine-tuned Qwen2.5-3B extracts 41 clause categories with per-clause risk scoring.',
              'Every response is validated through NLI entailment and logged with SHA-256 audit trail.',
            ].map((step, i) => (
              <li key={i} className="flex gap-3">
                <span className="w-5 h-5 rounded-full bg-brand-50 text-brand-700 text-xs font-semibold flex items-center justify-center flex-shrink-0 mt-0.5">
                  {i + 1}
                </span>
                <span>{step}</span>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </>
  );
}
