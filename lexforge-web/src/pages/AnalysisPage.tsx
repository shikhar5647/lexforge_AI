import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { FileText, MessageSquare, ShieldCheck, ChevronRight } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend } from 'recharts';
import PageHeader from '@/components/PageHeader';
import RiskBadge from '@/components/RiskBadge';
import { getContract, listContracts } from '@/lib/api';
import { formatBytes, formatDate, cn } from '@/lib/utils';
import type { Contract, Clause, RiskLevel } from '@/types';

export default function AnalysisPage() {
  const { contractId } = useParams();
  const [contract, setContract] = useState<Contract | null>(null);
  const [allContracts, setAllContracts] = useState<Contract[]>([]);
  const [selectedClause, setSelectedClause] = useState<Clause | null>(null);
  const [filterRisk, setFilterRisk] = useState<RiskLevel | 'all'>('all');

  useEffect(() => {
    listContracts().then(setAllContracts);
  }, []);

  useEffect(() => {
    if (contractId) {
      getContract(contractId).then(setContract);
    } else {
      // If no contract selected, default to the most recent one
      listContracts().then((cs) => {
        if (cs.length > 0) getContract(cs[0].id).then(setContract);
      });
    }
  }, [contractId]);

  if (!contract) {
    return (
      <>
        <PageHeader title="Analysis" />
        <div className="p-8 text-slate-400 text-sm">Loading...</div>
      </>
    );
  }

  const clauses = contract.clauses ?? [];
  const filteredClauses = filterRisk === 'all'
    ? clauses
    : clauses.filter((c) => c.risk === filterRisk);

  const riskCounts = {
    low:    clauses.filter((c) => c.risk === 'low').length,
    medium: clauses.filter((c) => c.risk === 'medium').length,
    high:   clauses.filter((c) => c.risk === 'high').length,
  };

  const chartData = [
    { name: 'Low Risk',    value: riskCounts.low,    color: '#10b981' },
    { name: 'Medium Risk', value: riskCounts.medium, color: '#f59e0b' },
    { name: 'High Risk',   value: riskCounts.high,   color: '#ef4444' },
  ].filter((d) => d.value > 0);

  return (
    <>
      <PageHeader title={contract.filename}>
        <Link to={`/chat/${contract.id}`} className="btn-secondary inline-flex items-center gap-2">
          <MessageSquare size={16} />
          Ask Questions
        </Link>
        <Link to={`/compliance/${contract.id}`} className="btn-primary inline-flex items-center gap-2">
          <ShieldCheck size={16} />
          Compliance Report
        </Link>
      </PageHeader>

      <div className="p-8">
        {/* Contract switcher (only shows when multiple exist) */}
        {allContracts.length > 1 && (
          <div className="mb-6 flex items-center gap-2 text-sm">
            <span className="text-slate-500">Switch contract:</span>
            <div className="flex flex-wrap gap-2">
              {allContracts.map((c) => (
                <Link
                  key={c.id}
                  to={`/analysis/${c.id}`}
                  className={cn(
                    'px-3 py-1.5 rounded-md border text-xs transition-colors',
                    c.id === contract.id
                      ? 'bg-brand-600 text-white border-brand-600'
                      : 'bg-white text-slate-600 border-slate-200 hover:border-slate-300'
                  )}
                >
                  {c.filename}
                </Link>
              ))}
            </div>
          </div>
        )}

        {/* Summary row */}
        <div className="grid grid-cols-4 gap-4 mb-8">
          <div className="card p-5">
            <div className="text-sm text-slate-500 mb-2">Overall Risk</div>
            {contract.overallRisk && <RiskBadge level={contract.overallRisk} size="md" />}
            <div className="mt-3 text-2xl font-semibold">{contract.riskScore}/100</div>
          </div>
          <div className="card p-5">
            <div className="text-sm text-slate-500 mb-2">Clauses Extracted</div>
            <div className="text-2xl font-semibold">{clauses.length}</div>
            <div className="text-xs text-slate-400 mt-1">of 41 categories</div>
          </div>
          <div className="card p-5">
            <div className="text-sm text-slate-500 mb-2">Jurisdiction</div>
            <div className="text-2xl font-semibold">{contract.jurisdiction}</div>
            <div className="text-xs text-slate-400 mt-1">LoRA adapter applied</div>
          </div>
          <div className="card p-5">
            <div className="text-sm text-slate-500 mb-2">Document</div>
            <div className="text-sm font-medium">{contract.pages} pages</div>
            <div className="text-xs text-slate-400 mt-1">
              {formatBytes(contract.sizeBytes)} · {formatDate(contract.uploadedAt)}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-6">
          {/* Chart */}
          <div className="card p-5">
            <h3 className="font-semibold text-slate-900 mb-4">Risk Distribution</h3>
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%" cy="50%"
                    innerRadius={50} outerRadius={80}
                    paddingAngle={2} dataKey="value"
                  >
                    {chartData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Pie>
                  <Legend verticalAlign="bottom" height={36} iconType="circle" />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Clause list */}
          <div className="card col-span-2">
            <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
              <h3 className="font-semibold text-slate-900">Extracted Clauses</h3>
              <div className="flex gap-1 text-xs">
                {(['all', 'high', 'medium', 'low'] as const).map((r) => (
                  <button
                    key={r}
                    onClick={() => setFilterRisk(r)}
                    className={cn(
                      'px-2.5 py-1 rounded-md font-medium transition-colors',
                      filterRisk === r
                        ? 'bg-slate-900 text-white'
                        : 'text-slate-500 hover:bg-slate-100'
                    )}
                  >
                    {r === 'all' ? 'All' : r.charAt(0).toUpperCase() + r.slice(1)}
                  </button>
                ))}
              </div>
            </div>
            <div className="divide-y divide-slate-100 max-h-[500px] overflow-y-auto">
              {filteredClauses.map((clause) => (
                <button
                  key={clause.id}
                  onClick={() => setSelectedClause(clause)}
                  className="w-full text-left px-5 py-4 hover:bg-slate-50 transition-colors flex items-start gap-3"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-medium text-slate-900 text-sm">{clause.type}</span>
                      <RiskBadge level={clause.risk} />
                    </div>
                    <p className="text-sm text-slate-600 line-clamp-2">{clause.text}</p>
                    <div className="mt-2 flex items-center gap-3 text-xs text-slate-400">
                      <span>Page {clause.pageNumber}</span>
                      <span>·</span>
                      <span>{(clause.confidence * 100).toFixed(0)}% confidence</span>
                    </div>
                  </div>
                  <ChevronRight size={18} className="text-slate-300 mt-1" />
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Clause detail drawer */}
      {selectedClause && (
        <div
          className="fixed inset-0 bg-slate-900/30 z-50 flex justify-end"
          onClick={() => setSelectedClause(null)}
        >
          <div
            className="w-full max-w-xl bg-white h-full overflow-y-auto shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6 border-b border-slate-200">
              <div className="flex items-start justify-between gap-3 mb-3">
                <h2 className="text-xl font-semibold text-slate-900">{selectedClause.type}</h2>
                <button
                  onClick={() => setSelectedClause(null)}
                  className="text-slate-400 hover:text-slate-600 text-sm"
                >
                  Close
                </button>
              </div>
              <div className="flex items-center gap-2">
                <RiskBadge level={selectedClause.risk} size="md" />
                <span className="text-xs text-slate-500">
                  {(selectedClause.confidence * 100).toFixed(1)}% confidence · Page {selectedClause.pageNumber}
                </span>
              </div>
            </div>
            <div className="p-6 space-y-6">
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
                  Clause Text
                </h3>
                <div className="p-4 bg-slate-50 rounded-lg text-sm text-slate-700 leading-relaxed">
                  {selectedClause.text}
                </div>
              </div>
              {selectedClause.riskRationale && (
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
                    Risk Rationale
                  </h3>
                  <p className="text-sm text-slate-700">{selectedClause.riskRationale}</p>
                </div>
              )}
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
                  Metadata
                </h3>
                <dl className="text-sm space-y-2">
                  <div className="flex justify-between py-2 border-b border-slate-100">
                    <dt className="text-slate-500">Category</dt>
                    <dd className="text-slate-900 font-medium">{selectedClause.category}</dd>
                  </div>
                  <div className="flex justify-between py-2 border-b border-slate-100">
                    <dt className="text-slate-500">Clause ID</dt>
                    <dd className="text-slate-900 font-mono text-xs">{selectedClause.id}</dd>
                  </div>
                </dl>
              </div>
              <Link
                to={`/chat/${contract.id}?clause=${selectedClause.id}`}
                className="btn-primary w-full inline-flex items-center justify-center gap-2"
              >
                <MessageSquare size={16} />
                Ask about this clause
              </Link>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
