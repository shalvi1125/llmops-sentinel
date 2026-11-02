import React, { useState } from 'react';

export interface LineDataPoint {
  label: string;
  value: number;
  secondaryText?: string;
}

interface InteractiveLineChartProps {
  data: LineDataPoint[];
  title?: string;
  yAxisLabel?: string;
  unit?: string;
  height?: number;
  accentColor?: string;
  emptyMessage?: string;
  onPointClick?: (point: LineDataPoint, index: number) => void;
}

export const InteractiveLineChart: React.FC<InteractiveLineChartProps> = ({
  data,
  title,
  yAxisLabel = 'Latency (ms)',
  unit = 'ms',
  height = 200,
  accentColor = '#38bdf8', // sky-400
  emptyMessage = 'No telemetry data points available',
  onPointClick,
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

  // Calculate scales
  const values = data.map((d) => d.value);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values, 10);
  const paddingLeft = 50;
  const paddingRight = 20;
  const paddingTop = 20;
  const paddingBottom = 30;

  const chartWidth = 560; // internal coordinate space
  const chartHeight = height;

  const plotWidth = chartWidth - paddingLeft - paddingRight;
  const plotHeight = chartHeight - paddingTop - paddingBottom;

  const yRange = maxVal - Math.min(0, minVal);
  const safeRange = yRange === 0 ? 1 : yRange;

  const points = data.map((d, i) => {
    const x =
      data.length === 1
        ? paddingLeft + plotWidth / 2
        : paddingLeft + (i / (data.length - 1)) * plotWidth;
    const y = paddingTop + plotHeight - ((d.value - Math.min(0, minVal)) / safeRange) * plotHeight;
    return { x, y, ...d };
  });

  // Construct SVG path for line
  const linePath = points.reduce((acc, pt, i) => {
    return i === 0 ? `M ${pt.x} ${pt.y}` : `${acc} L ${pt.x} ${pt.y}`;
  }, '');

  // Construct area path
  const areaPath =
    points.length > 0
      ? `${linePath} L ${points[points.length - 1].x} ${paddingTop + plotHeight} L ${points[0].x} ${
          paddingTop + plotHeight
        } Z`
      : '';

  // Y-axis grid ticks (4 ticks)
  const ticks = [0, 0.33, 0.66, 1].map((pct) => {
    const val = Math.round(Math.min(0, minVal) + safeRange * pct);
    const y = paddingTop + plotHeight - pct * plotHeight;
    return { val, y };
  });

  const activePoint = hoveredIdx !== null ? points[hoveredIdx] : null;

  return (
    <div className="relative w-full bg-slate-950/60 border border-slate-800/90 rounded-xl p-4 flex flex-col">
      {title && (
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-slate-300">{title}</span>
          <span className="text-[10px] font-mono text-slate-500 uppercase">{yAxisLabel}</span>
        </div>
      )}

      <div className="relative w-full overflow-hidden">
        <svg
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
          className="w-full h-auto overflow-visible select-none"
        >
          <defs>
            <linearGradient id="lineAreaGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={accentColor} stopOpacity="0.3" />
              <stop offset="100%" stopColor={accentColor} stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Grid lines & Y-axis labels */}
          {ticks.map((t, idx) => (
            <g key={idx}>
              <line
                x1={paddingLeft}
                y1={t.y}
                x2={chartWidth - paddingRight}
                y2={t.y}
                stroke="#334155"
                strokeWidth="1"
                strokeDasharray="3 3"
                opacity="0.4"
              />
              <text
                x={paddingLeft - 8}
                y={t.y + 3}
                fill="#64748b"
                fontSize="9"
                fontFamily="JetBrains Mono, monospace"
                textAnchor="end"
              >
                {t.val}
              </text>
            </g>
          ))}

          {/* Area fill */}
          {areaPath && <path d={areaPath} fill="url(#lineAreaGrad)" />}

          {/* Line stroke */}
          {linePath && (
            <path
              d={linePath}
              fill="none"
              stroke={accentColor}
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {/* Active Hover Guide line */}
          {activePoint && (
            <line
              x1={activePoint.x}
              y1={paddingTop}
              x2={activePoint.x}
              y2={paddingTop + plotHeight}
              stroke="#94a3b8"
              strokeWidth="1"
              strokeDasharray="2 2"
            />
          )}

          {/* Interactive Data Points */}
          {points.map((pt, idx) => {
            const isHovered = hoveredIdx === idx;
            return (
              <g
                key={idx}
                onMouseEnter={() => setHoveredIdx(idx)}
                onMouseLeave={() => setHoveredIdx(null)}
                onClick={() => onPointClick && onPointClick(data[idx], idx)}
                className="cursor-pointer"
              >
                {/* Invisible larger hit target */}
                <circle cx={pt.x} cy={pt.y} r="14" fill="transparent" />

                {/* Point circle */}
                <circle
                  cx={pt.x}
                  cy={pt.y}
                  r={isHovered ? 5.5 : 3.5}
                  fill={isHovered ? '#ffffff' : accentColor}
                  stroke="#0f172a"
                  strokeWidth="2"
                  className="transition-all duration-150"
                />
              </g>
            );
          })}

          {/* X-axis baseline */}
          <line
            x1={paddingLeft}
            y1={paddingTop + plotHeight}
            x2={chartWidth - paddingRight}
            y2={paddingTop + plotHeight}
            stroke="#475569"
            strokeWidth="1"
          />
        </svg>

        {/* Hover Tooltip Overlay */}
        {activePoint && (
          <div
            className="absolute pointer-events-none bg-slate-900 border border-slate-700/80 rounded-lg p-2 shadow-2xl text-[11px] z-20 font-mono transition-opacity"
            style={{
              left: `${Math.min(Math.max((activePoint.x / chartWidth) * 100, 15), 85)}%`,
              top: '8px',
              transform: 'translateX(-50%)',
            }}
          >
            <div className="font-semibold text-sky-400">
              {activePoint.value} {unit}
            </div>
            <div className="text-slate-400 text-[10px]">{activePoint.label}</div>
            {activePoint.secondaryText && (
              <div className="text-slate-300 text-[10px] max-w-[200px] truncate mt-0.5">
                {activePoint.secondaryText}
              </div>
            )}
          </div>
        )}
      </div>

      {/* X-Axis bottom labels: Start and End */}
      <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono mt-1 px-2">
        <span>{data[0]?.label || ''}</span>
        <span>{data[data.length - 1]?.label || ''}</span>
      </div>
    </div>
  );
};
