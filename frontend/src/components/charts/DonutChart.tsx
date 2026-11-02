import React, { useState } from 'react';

export interface DonutSegment {
  label: string;
  value: number;
  color: string;
}

interface DonutChartProps {
  data: DonutSegment[];
  title?: string;
  centerLabel?: string;
  centerValue?: string | number;
  size?: number;
  emptyMessage?: string;
  onSegmentClick?: (segment: DonutSegment) => void;
}

export const DonutChart: React.FC<DonutChartProps> = ({
  data,
  title,
  centerLabel,
  centerValue,
  size = 170,
  emptyMessage = 'No outcome data available',
  onSegmentClick,
}) => {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const total = data.reduce((acc, curr) => acc + curr.value, 0);

  if (total === 0) {
    return (
      <div className="w-full bg-slate-950/60 border border-slate-800/90 rounded-xl p-4 flex flex-col items-center justify-center min-h-[180px]">
        {title && <span className="text-xs font-semibold text-slate-300 self-start mb-2">{title}</span>}
        <div className="text-slate-500 text-xs italic my-auto">{emptyMessage}</div>
      </div>
    );
  }

  // Polar to Cartesian conversion
  const radius = size / 2;
  const strokeWidth = 24;
  const normalizedRadius = radius - strokeWidth / 2;
  const circumference = normalizedRadius * 2 * Math.PI;

  let accumulatedPercent = 0;

  return (
    <div className="w-full bg-slate-950/60 border border-slate-800/90 rounded-xl p-4 flex flex-col">
      {title && <span className="text-xs font-semibold text-slate-300 mb-2">{title}</span>}

      <div className="flex flex-col sm:flex-row items-center justify-center gap-6 my-auto py-2">
        {/* SVG Donut */}
        <div className="relative shrink-0" style={{ width: size, height: size }}>
          <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="rotate-[-90deg]">
            {/* Background track circle */}
            <circle
              cx={radius}
              cy={radius}
              r={normalizedRadius}
              fill="transparent"
              stroke="#1e293b"
              strokeWidth={strokeWidth}
            />

            {/* Slices */}
            {data.map((seg, idx) => {
              if (seg.value === 0) return null;
              const percent = (seg.value / total) * 100;
              const strokeDasharray = `${(percent / 100) * circumference} ${circumference}`;
              const strokeDashoffset = -((accumulatedPercent / 100) * circumference);
              accumulatedPercent += percent;

              const isHovered = hoveredIdx === idx;

              return (
                <circle
                  key={idx}
                  cx={radius}
                  cy={radius}
                  r={normalizedRadius}
                  fill="transparent"
                  stroke={seg.color}
                  strokeWidth={isHovered ? strokeWidth + 4 : strokeWidth}
                  strokeDasharray={strokeDasharray}
                  strokeDashoffset={strokeDashoffset}
                  className="transition-all duration-200 cursor-pointer"
                  onMouseEnter={() => setHoveredIdx(idx)}
                  onMouseLeave={() => setHoveredIdx(null)}
                  onClick={() => onSegmentClick && onSegmentClick(seg)}
                />
              );
            })}
          </svg>

          {/* Center text in donut */}
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
            <span className="text-xl font-bold font-mono text-slate-100">
              {hoveredIdx !== null ? data[hoveredIdx].value : centerValue ?? total}
            </span>
            <span className="text-[10px] text-slate-400 font-mono uppercase tracking-wider">
              {hoveredIdx !== null ? data[hoveredIdx].label : centerLabel ?? 'Total'}
            </span>
          </div>
        </div>

        {/* Legend */}
        <div className="flex flex-col space-y-2 text-xs w-full max-w-[200px]">
          {data.map((seg, idx) => {
            const pct = Math.round((seg.value / total) * 100);
            const isHovered = hoveredIdx === idx;

            return (
              <div
                key={idx}
                onMouseEnter={() => setHoveredIdx(idx)}
                onMouseLeave={() => setHoveredIdx(null)}
                onClick={() => onSegmentClick && onSegmentClick(seg)}
                className={`flex items-center justify-between p-1.5 rounded-lg cursor-pointer transition ${
                  isHovered ? 'bg-slate-800/80' : 'hover:bg-slate-900'
                }`}
              >
                <div className="flex items-center space-x-2 truncate">
                  <span
                    className="w-2.5 h-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: seg.color }}
                  />
                  <span className="text-slate-300 truncate text-[11px]">{seg.label}</span>
                </div>
                <div className="flex items-center space-x-1 font-mono text-[11px]">
                  <span className="text-slate-400 font-bold">{seg.value}</span>
                  <span className="text-slate-500">({pct}%)</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
