import type { RiskLevel } from '@/types';
import { cn, riskStyles } from '@/lib/utils';

interface RiskBadgeProps {
  level: RiskLevel;
  size?: 'sm' | 'md';
}

export default function RiskBadge({ level, size = 'sm' }: RiskBadgeProps) {
  const s = riskStyles[level];
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full font-medium border',
        s.bg, s.text, s.border,
        size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm'
      )}
    >
      <span className={cn('w-1.5 h-1.5 rounded-full',
        level === 'low' ? 'bg-risk-low' :
        level === 'medium' ? 'bg-risk-medium' : 'bg-risk-high'
      )} />
      {s.label}
    </span>
  );
}
