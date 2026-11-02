import React from 'react';

interface KpiCardProps {
  title: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  icon: React.ReactNode;
  accent?: 'sky' | 'emerald' | 'indigo' | 'amber' | 'rose' | 'slate';
  badge?: {
    text: string;
    variant: 'success' | 'warning' | 'info' | 'neutral' | 'error';
  };
  onClick?: () => void;
}

export const KpiCard: React.FC<KpiCardProps> = ({
  title,
  value,
  unit,
  subtitle,
  icon,
  accent = 'sky',
  badge,
  onClick,
}) => {
  const accentBorder = {
    sky: 'border-slate-800/90 hover:border-sky-500/50 hover:shadow-sky-500/10',
    emerald: 'border-slate-800/90 hover:border-emerald-500/50 hover:shadow-emerald-500/10',
    indigo: 'border-slate-800/90 hover:border-indigo-500/50 hover:shadow-indigo-500/10',
    amber: 'border-slate-800/90 hover:border-amber-500/50 hover:shadow-amber-500/10',
    rose: 'border-slate-800/90 hover:border-rose-500/50 hover:shadow-rose-500/10',
    slate: 'border-slate-800/90 hover:border-slate-700',
  }[accent];

  const iconBg = {
    sky: 'bg-sky-500/15 text-sky-400 border border-sky-500/30',
    emerald: 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30',
    indigo: 'bg-indigo-500/15 text-indigo-400 border border-indigo-500/30',
    amber: 'bg-amber-500/15 text-amber-400 border border-amber-500/30',
    rose: 'bg-rose-500/15 text-rose-400 border border-rose-500/30',
    slate: 'bg-slate-800 text-slate-400 border border-slate-700',
  }[accent];

  const badgeClasses = badge
    ? {
        success: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
        warning: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
        info: 'bg-sky-500/15 text-sky-300 border-sky-500/30',
        neutral: 'bg-slate-800 text-slate-300 border-slate-700',
        error: 'bg-rose-500/15 text-rose-300 border-rose-500/30',
      }[badge.variant]
    : '';

  return (
    <div
      onClick={onClick}
      className={`relative overflow-hidden bg-slate-900/70 backdrop-blur border rounded-xl p-4 transition-all duration-200 hover:shadow-lg ${accentBorder} ${
        onClick ? 'cursor-pointer' : ''
      }`}
    >
      <div className="flex items-start justify-between mb-2">
        <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
          {title}
        </span>
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${iconBg}`}>
          {icon}
        </div>
      </div>

      <div className="flex items-baseline space-x-1.5 my-1">
        <span className="text-2xl font-bold tracking-tight text-slate-100 font-mono">
          {value}
        </span>
        {unit && <span className="text-xs font-medium text-slate-400 font-mono">{unit}</span>}
      </div>

      <div className="flex items-center justify-between text-[11px] text-slate-500 mt-2">
        <span className="truncate">{subtitle}</span>
        {badge && (
          <span
            className={`px-1.5 py-0.5 rounded text-[10px] font-medium border font-mono shrink-0 ml-2 ${badgeClasses}`}
          >
            {badge.text}
          </span>
        )}
      </div>
    </div>
  );
};
