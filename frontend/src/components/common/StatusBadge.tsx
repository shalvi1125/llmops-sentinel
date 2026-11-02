import React from 'react';

export type StatusVariant =
  | 'healthy'
  | 'degraded'
  | 'offline'
  | 'running'
  | 'completed'
  | 'failed'
  | 'skipped'
  | 'neutral';

interface StatusBadgeProps {
  status: string;
  variant?: StatusVariant;
  pulse?: boolean;
  className?: string;
  size?: 'sm' | 'md';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  variant = 'neutral',
  pulse = false,
  className = '',
  size = 'md',
}) => {
  const styles: Record<StatusVariant, { bg: string; text: string; border: string; dot: string }> = {
    healthy: {
      bg: 'bg-emerald-500/15',
      text: 'text-emerald-400',
      border: 'border-emerald-500/30',
      dot: 'bg-emerald-400',
    },
    degraded: {
      bg: 'bg-amber-500/15',
      text: 'text-amber-300',
      border: 'border-amber-500/30',
      dot: 'bg-amber-400',
    },
    offline: {
      bg: 'bg-rose-500/15',
      text: 'text-rose-400',
      border: 'border-rose-500/30',
      dot: 'bg-rose-400',
    },
    running: {
      bg: 'bg-sky-500/15',
      text: 'text-sky-300',
      border: 'border-sky-500/30',
      dot: 'bg-sky-400',
    },
    completed: {
      bg: 'bg-emerald-500/15',
      text: 'text-emerald-300',
      border: 'border-emerald-500/30',
      dot: 'bg-emerald-400',
    },
    failed: {
      bg: 'bg-rose-500/15',
      text: 'text-rose-300',
      border: 'border-rose-500/30',
      dot: 'bg-rose-400',
    },
    skipped: {
      bg: 'bg-slate-800',
      text: 'text-slate-400',
      border: 'border-slate-700',
      dot: 'bg-slate-400',
    },
    neutral: {
      bg: 'bg-slate-800/80',
      text: 'text-slate-300',
      border: 'border-slate-700/80',
      dot: 'bg-slate-400',
    },
  };

  const current = styles[variant] || styles.neutral;
  const padding = size === 'sm' ? 'px-2 py-0.5 text-[10px]' : 'px-2.5 py-1 text-xs';

  return (
    <span
      className={`inline-flex items-center space-x-1.5 rounded-full font-medium border font-mono tracking-wide ${current.bg} ${current.text} ${current.border} ${padding} ${className}`}
    >
      <span className="relative flex h-2 w-2">
        {pulse && (
          <span
            className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${current.dot}`}
          />
        )}
        <span className={`relative inline-flex rounded-full h-2 w-2 ${current.dot}`} />
      </span>
      <span className="capitalize">{status}</span>
    </span>
  );
};
