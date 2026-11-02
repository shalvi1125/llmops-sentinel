import React, { useState } from 'react';

export interface BarDataPoint {
  label: string;
  value: number;
  color?: string;
  subtext?: string;
}

interface InteractiveBarChartProps {
  data: BarDataPoint[];
  title?: string;
  unit?: string;
  height?: number;
  emptyMessage?: string;
  onBarClick?: (item: BarDataPoint, index: number) => void;
}

export const InteractiveBarChart: React.FC<InteractiveBarChartProps> = ({
  data,
  title,
  unit = '',
  height = 180,
  emptyMessage = 'No comparison data available',
  onBarClick,
}) => {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!data || data.length === 0) {
    return (
      <div
        className="w-full bg-slate-950/40 border border-slate-800/80 rounded-xl flex items-center justify-center p-6 text-slate-500 text-xs italic"
        style={{ height }}
      >
        {emptyMessage}
      </div>
    );
  }

  const maxVal = Math.max(...data.map((d) => d.value), 1);

  return (
    <div className="relative w-full bg-slate-950/60 border border-slate-800/90 rounded-xl p-4 flex flex-col">
      {title && (
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold text-slate-300">{title}</span>
          {unit && <span className="text-[10px] font-mono text-slate-500 uppercase">{unit}</span>}
        </div>
      )}

      <div className="flex items-end justify-between gap-3 flex-1 pt-4 pb-2 px-2" style={{ minHeight: height - 60 }}>
        {data.map((bar, idx) => {
          const heightPct = Math.max((bar.value / maxVal) * 100, 4);
          const isHovered = hoveredIdx === idx;
          const barColor = bar.color || '#38bdf8';

          return (
            <div
              key={idx}
              className="flex-1 flex flex-col items-center h-full justify-end cursor-pointer group"
              onMouseEnter={() => setHoveredIdx(idx)}
              onMouseLeave={() => setHoveredIdx(null)}
              onClick={() => onBarClick && onBarClick(bar, idx)}
            >
              {/* Tooltip on hover */}
              <div
                className={`text-[10px] font-mono font-bold transition-all duration-150 mb-1 ${
                  isHovered ? 'text-white scale-110' : 'text-slate-400'
                }`}
              >
                {bar.value}
                {unit ? ` ${unit}` : ''}
              </div>

              {/* Bar Rect */}
              <div className="w-full max-w-[48px] bg-slate-900 rounded-t-md overflow-hidden flex items-end h-[100px]">
                <div
                  className="w-full rounded-t-md transition-all duration-300 relative group-hover:brightness-125"
                  style={{
                    height: `${heightPct}%`,
                    backgroundColor: barColor,
                    boxShadow: isHovered ? `0 0 12px ${barColor}66` : 'none',
                  }}
                />
              </div>

              {/* Category Label */}
              <span className="text-[10px] font-mono text-slate-400 text-center truncate w-full mt-2 group-hover:text-slate-200">
                {bar.label}
              </span>
              {bar.subtext && (
                <span className="text-[9px] text-slate-500 text-center truncate w-full">
                  {bar.subtext}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
