import React from 'react';

interface SentinelLogoProps {
  className?: string;
  size?: number;
}

export const SentinelLogo: React.FC<SentinelLogoProps> = ({ className = '', size = 32 }) => {
  return (
    <div
      className={`relative flex items-center justify-center shrink-0 ${className}`}
      style={{ width: size, height: size }}
    >
      <svg
        viewBox="0 0 40 40"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full drop-shadow-[0_0_12px_rgba(14,165,233,0.35)]"
      >
        <defs>
          <linearGradient id="sentinelGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#38bdf8" />
            <stop offset="50%" stopColor="#0284c7" />
            <stop offset="100%" stopColor="#6366f1" />
          </linearGradient>
          <linearGradient id="shieldFill" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#0f172a" />
            <stop offset="100%" stopColor="#020617" />
          </linearGradient>
          <linearGradient id="pulseGrad" x1="0%" y1="50%" x2="100%" y2="50%">
            <stop offset="0%" stopColor="#38bdf8" />
            <stop offset="100%" stopColor="#22d3ee" />
          </linearGradient>
        </defs>

        {/* Shield Outer Path */}
        <path
          d="M20 3L6 8.5V19.2C6 28.1 12 34.6 20 37C28 34.6 34 28.1 34 19.2V8.5L20 3Z"
          fill="url(#shieldFill)"
          stroke="url(#sentinelGrad)"
          strokeWidth="2"
          strokeLinejoin="round"
        />

        {/* Inner Sentinel Pulse / Radar Line */}
        <path
          d="M12 20.5H16L18.5 14.5L21.5 26.5L24 20.5H28"
          stroke="url(#pulseGrad)"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Core Sentinel Node */}
        <circle cx="20" cy="8" r="1.5" fill="#38bdf8" />
      </svg>
    </div>
  );
};
