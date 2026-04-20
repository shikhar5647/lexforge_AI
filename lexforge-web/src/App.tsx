import { Routes, Route, NavLink } from 'react-router-dom';
import { Scale, Upload, FileSearch, MessageSquare, ShieldCheck, Activity } from 'lucide-react';
import { cn } from '@/lib/utils';
import UploadPage from '@/pages/UploadPage';
import AnalysisPage from '@/pages/AnalysisPage';
import ChatPage from '@/pages/ChatPage';
import CompliancePage from '@/pages/CompliancePage';
import DashboardPage from '@/pages/DashboardPage';

const navItems = [
  { to: '/',            label: 'Dashboard',  icon: Activity },
  { to: '/upload',      label: 'Upload',     icon: Upload },
  { to: '/analysis',    label: 'Analysis',   icon: FileSearch },
  { to: '/chat',        label: 'Chat',       icon: MessageSquare },
  { to: '/compliance',  label: 'Compliance', icon: ShieldCheck },
];

function Sidebar() {
  return (
    <aside className="w-64 bg-brand-900 text-white flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-brand-500 flex items-center justify-center">
            <Scale size={20} />
          </div>
          <div>
            <div className="font-semibold text-lg leading-tight">LexForge AI</div>
            <div className="text-xs text-white/60">Legal Intelligence</div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'bg-brand-500/20 text-white'
                  : 'text-white/70 hover:bg-white/5 hover:text-white'
              )
            }
          >
            <item.icon size={18} />
            {item.label}
          </NavLink>
        ))}
      </nav>

      {/* Model status footer */}
      <div className="p-4 border-t border-white/10 text-xs">
        <div className="flex items-center gap-2 text-white/60 mb-1">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          Model online
        </div>
        <div className="text-white/40 font-mono">qwen2.5-3b-cuad · v0.1</div>
      </div>
    </aside>
  );
}

export default function App() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/"           element={<DashboardPage />} />
          <Route path="/upload"     element={<UploadPage />} />
          <Route path="/analysis"   element={<AnalysisPage />} />
          <Route path="/analysis/:contractId" element={<AnalysisPage />} />
          <Route path="/chat"       element={<ChatPage />} />
          <Route path="/chat/:contractId"     element={<ChatPage />} />
          <Route path="/compliance" element={<CompliancePage />} />
          <Route path="/compliance/:contractId" element={<CompliancePage />} />
        </Routes>
      </main>
    </div>
  );
}
