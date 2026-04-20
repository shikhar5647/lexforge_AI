import type { RiskLevel } from '@/types';

export function cn(...classes: (string | undefined | false | null)[]) {
  return classes.filter(Boolean).join(' ');
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDate(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  const hours = diff / (1000 * 60 * 60);
  if (hours < 1) return `${Math.floor(diff / 60000)}m ago`;
  if (hours < 24) return `${Math.floor(hours)}h ago`;
  return d.toLocaleDateString();
}

export const riskStyles: Record<RiskLevel, { bg: string; text: string; border: string; label: string }> = {
  low:    { bg: 'bg-risk-low/10',    text: 'text-risk-low',    border: 'border-risk-low/30',    label: 'Low Risk'    },
  medium: { bg: 'bg-risk-medium/10', text: 'text-risk-medium', border: 'border-risk-medium/30', label: 'Medium Risk' },
  high:   { bg: 'bg-risk-high/10',   text: 'text-risk-high',   border: 'border-risk-high/30',   label: 'High Risk'   },
};
