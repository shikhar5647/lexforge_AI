import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { FileText, TrendingUp, AlertTriangle, CheckCircle2, ArrowRight } from 'lucide-react';
import PageHeader from '@/components/PageHeader';
import RiskBadge from '@/components/RiskBadge';
import { listContracts } from '@/lib/api';
import { formatDate, formatBytes } from '@/lib/utils';
import type { Contract } from '@/types';

interface Stat {
  label: string;
  value: string;
  change?: string;
  icon: React.ElementType;
  iconColor: string;
}

export default function DashboardPage() {
  const [contracts, setContracts] = useState<Contract[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listContracts().then((cs) => {
      setContracts(cs);
      setLoading(false);
    });
  }, []);

  const totalContracts = contracts.length;
  const highRisk = contracts.filter((c) => c.overallRisk === 'high').length;
  const readyCount = contracts.filter((c) => c.status === 'ready').length;

  const stats: Stat[] = [
    { label: 'Total Contracts',  value: String(totalContracts), icon: FileText,       iconColor: 'text-brand-600' },
    { label: 'Analyzed',         value: String(readyCount),     icon: CheckCircle2,   iconColor: 'text-risk-low' },
    { label: 'High Risk Flags',  value: String(highRisk),       icon: AlertTriangle,  iconColor: 'text-risk-high' },
    { label: 'Avg. Confidence',  value: '92.4%',                icon: TrendingUp,     iconColor: 'text-brand-600' },
  ];

  return (
    <>
      <PageHeader
        title="Dashboard"
        description="Overview of your contract analysis activity"
      />

      <div className="p-8 space-y-8">
        {/* Stat grid */}
        <div className="grid grid-cols-4 gap-4">
          {stats.map((stat) => (
            <div key={stat.label} className="card p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm text-slate-500">{stat.label}</span>
                <stat.icon size={18} className={stat.iconColor} />
              </div>
              <div className="text-2xl font-semibold text-slate-900">{stat.value}</div>
            </div>
          ))}
        </div>

        {/* Recent contracts */}
        <div className="card">
          <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between">
            <h2 className="font-semibold text-slate-900">Recent Contracts</h2>
            <Link to="/upload" className="text-sm text-brand-600 hover:text-brand-700 font-medium">
              Upload new →
            </Link>
          </div>

          {loading ? (
            <div className="p-12 text-center text-slate-400 text-sm">Loading...</div>
          ) : contracts.length === 0 ? (
            <div className="p-12 text-center">
              <FileText className="mx-auto text-slate-300 mb-3" size={40} />
              <p className="text-slate-500 mb-4">No contracts yet</p>
              <Link to="/upload" className="btn-primary inline-flex">Upload your first contract</Link>
            </div>
          ) : (
            <div className="divide-y divide-slate-100">
              {contracts.map((contract) => (
                <Link
                  key={contract.id}
                  to={`/analysis/${contract.id}`}
                  className="flex items-center gap-4 px-6 py-4 hover:bg-slate-50 transition-colors group"
                >
                  <div className="w-10 h-10 rounded-lg bg-brand-50 flex items-center justify-center flex-shrink-0">
                    <FileText size={18} className="text-brand-600" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-slate-900 truncate">{contract.filename}</div>
                    <div className="text-sm text-slate-500 flex items-center gap-3">
                      <span>{contract.pages} pages</span>
                      <span>·</span>
                      <span>{formatBytes(contract.sizeBytes)}</span>
                      <span>·</span>
                      <span>{formatDate(contract.uploadedAt)}</span>
                      <span>·</span>
                      <span>{contract.jurisdiction}</span>
                    </div>
                  </div>
                  {contract.overallRisk && <RiskBadge level={contract.overallRisk} />}
                  <ArrowRight size={16} className="text-slate-300 group-hover:text-slate-600 transition-colors" />
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
