import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { ShieldCheck, AlertCircle, Info, AlertTriangle, Download, Copy, Check } from 'lucide-react';
import PageHeader from '@/components/PageHeader';
import { getContract, getComplianceReport, listContracts } from '@/lib/api';
import { cn } from '@/lib/utils';
import type { Contract, ComplianceReport } from '@/types';

const severityStyles = {
  info:     { icon: Info,          color: 'text-blue-600',    bg: 'bg-blue-50',   border: 'border-blue-200' },
  warning:  { icon: AlertTriangle, color: 'text-amber-600',   bg: 'bg-amber-50',  border: 'border-amber-200' },
  critical: { icon: AlertCircle,   color: 'text-risk-high',   bg: 'bg-red-50',    border: 'border-red-200' },
};

export default function CompliancePage() {
  const { contractId } = useParams();
  const [contract, setContract] = useState<Contract | null>(null);
  const [report, setReport] = useState<ComplianceReport | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const id = contractId;
    if (id) {
      getContract(id).then(setContract);
      getComplianceReport(id).then(setReport);
    } else {
      listContracts().then((cs) => {
        if (cs.length > 0) {
          getContract(cs[0].id).then(setContract);
          getComplianceReport(cs[0].id).then(setReport);
        }
      });
    }
  }, [contractId]);

  function copyHash() {
    if (report) {
      navigator.clipboard.writeText(report.auditHash);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }

  if (!contract || !report) {
    return (
      <>
        <PageHeader title="Compliance Report" />
        <div className="p-8 text-slate-400 text-sm">Loading...</div>
      </>
    );
  }

  const maxShap = Math.max(...report.shapExplanations.map((e) => Math.abs(e.contribution)));

  return (
    <>
      <PageHeader
        title="Compliance Report"
        description={contract.filename}
      >
        <button className="btn-secondary inline-flex items-center gap-2">
          <Download size={16} />
          Export PDF
        </button>
      </PageHeader>

      <div className="p-8 space-y-6 max-w-5xl">
        {/* Summary scores */}
        <div className="grid grid-cols-3 gap-4">
          <ScoreCard
            label="Bias Score"
            value={report.biasScore}
            threshold={0.15}
            description="Lower is better"
          />
          <ScoreCard
            label="Toxicity"
            value={report.toxicityScore}
            threshold={0.1}
            description="toxic-bert classifier"
          />
          <div className="card p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-500">Audit Status</span>
              <ShieldCheck size={18} className="text-risk-low" />
            </div>
            <div className="text-lg font-semibold text-slate-900">Verified</div>
            <div className="text-xs text-slate-400 mt-1">
              {new Date(report.generatedAt).toLocaleString()}
            </div>
          </div>
        </div>

        {/* Tamper-proof hash */}
        <div className="card p-5">
          <h3 className="font-semibold text-slate-900 mb-1">Tamper-Proof Audit Record</h3>
          <p className="text-xs text-slate-500 mb-3">
            SHA-256 hash of the full inference record. Store alongside your contract for legal/audit trail.
          </p>
          <div className="flex items-center gap-2">
            <code className="flex-1 px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg font-mono text-xs text-slate-700 overflow-x-auto">
              {report.auditHash}
            </code>
            <button
              onClick={copyHash}
              className="btn-secondary inline-flex items-center gap-2 whitespace-nowrap"
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
        </div>

        {/* Flagged issues */}
        <div className="card">
          <div className="px-5 py-4 border-b border-slate-100">
            <h3 className="font-semibold text-slate-900">Flagged Issues</h3>
            <p className="text-sm text-slate-500 mt-0.5">
              Potential GDPR / AI Act compliance concerns detected during analysis
            </p>
          </div>
          <div className="divide-y divide-slate-100">
            {report.flaggedIssues.map((issue, i) => {
              const s = severityStyles[issue.severity];
              const Icon = s.icon;
              return (
                <div key={i} className="px-5 py-4 flex items-start gap-3">
                  <div className={cn('w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0', s.bg, s.border, 'border')}>
                    <Icon size={16} className={s.color} />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs uppercase tracking-wide font-semibold text-slate-500">
                        {issue.category.replace('_', ' ')}
                      </span>
                      <span className={cn('text-xs font-medium', s.color)}>
                        {issue.severity}
                      </span>
                    </div>
                    <p className="text-sm text-slate-700">{issue.description}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* SHAP explainability */}
        <div className="card p-5">
          <h3 className="font-semibold text-slate-900 mb-1">SHAP Token Explainability</h3>
          <p className="text-sm text-slate-500 mb-4">
            Most influential tokens in the model's risk scoring decision. Positive = pushed risk up, negative = pushed risk down.
          </p>
          <div className="space-y-2">
            {report.shapExplanations.map((exp) => {
              const pct = Math.abs(exp.contribution) / maxShap;
              const isPositive = exp.contribution > 0;
              return (
                <div key={exp.token} className="flex items-center gap-3">
                  <div className="w-28 text-sm font-mono text-slate-700 text-right">
                    {exp.token}
                  </div>
                  <div className="flex-1 h-6 bg-slate-100 rounded relative overflow-hidden">
                    <div
                      className={cn(
                        'absolute top-0 bottom-0 transition-all',
                        isPositive ? 'bg-risk-high/60 left-1/2' : 'bg-risk-low/60 right-1/2'
                      )}
                      style={{ width: `${pct * 50}%` }}
                    />
                    <div className="absolute top-0 bottom-0 left-1/2 w-px bg-slate-300" />
                  </div>
                  <div className={cn(
                    'w-16 text-right text-xs font-mono font-semibold',
                    isPositive ? 'text-risk-high' : 'text-risk-low'
                  )}>
                    {exp.contribution > 0 ? '+' : ''}{exp.contribution.toFixed(2)}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </>
  );
}

function ScoreCard({
  label, value, threshold, description,
}: { label: string; value: number; threshold: number; description: string }) {
  const isGood = value < threshold;
  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-slate-500">{label}</span>
        <span className={cn(
          'text-xs font-medium px-2 py-0.5 rounded',
          isGood ? 'bg-risk-low/10 text-risk-low' : 'bg-risk-medium/10 text-risk-medium'
        )}>
          {isGood ? 'Pass' : 'Review'}
        </span>
      </div>
      <div className="text-2xl font-semibold text-slate-900">{value.toFixed(3)}</div>
      <div className="text-xs text-slate-400 mt-1">{description}</div>
    </div>
  );
}
