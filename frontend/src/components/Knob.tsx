import { useRef, useEffect, useCallback } from 'react';

interface Props {
  value: number;
  min?: number;
  max?: number;
  label: string;
  unit?: string;
  onChange: (v: number) => void;
  size?: number;
  color?: string;
  decimals?: number;
}

/**
 * SVG rotary knob — drag up/down to change value.
 * Full range = 280° sweep (-140° to +140°).
 */
export default function Knob({
  value,
  min = 0,
  max = 1,
  label,
  unit = '',
  onChange,
  size = 54,
  color = '#00ff88',
  decimals = 2,
}: Props) {
  const isDragging = useRef(false);
  const startY = useRef(0);
  const startVal = useRef(0);

  const norm = Math.max(0, Math.min(1, (value - min) / (max - min)));
  const deg = -140 + norm * 280;
  const rad = (deg * Math.PI) / 180;
  const cx = size / 2;
  const cy = size / 2;
  const r = size * 0.37;

  // Arc helper
  const arc = (startDeg: number, endDeg: number): string => {
    const s = (startDeg * Math.PI) / 180;
    const e = (endDeg * Math.PI) / 180;
    const x1 = cx + r * Math.cos(s - Math.PI / 2);
    const y1 = cy + r * Math.sin(s - Math.PI / 2);
    const x2 = cx + r * Math.cos(e - Math.PI / 2);
    const y2 = cy + r * Math.sin(e - Math.PI / 2);
    const large = endDeg - startDeg > 180 ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
  };

  // Pointer tip
  const px = cx + (r - 5) * Math.cos(rad - Math.PI / 2);
  const py = cy + (r - 5) * Math.sin(rad - Math.PI / 2);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    isDragging.current = true;
    startY.current = e.clientY;
    startVal.current = value;
    e.preventDefault();
  }, [value]);

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const step = (max - min) / 100;
    const next = Math.max(min, Math.min(max, value - Math.sign(e.deltaY) * step));
    onChange(next);
  }, [value, min, max, onChange]);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      const delta = (startY.current - e.clientY) / 160;
      const next = Math.max(min, Math.min(max, startVal.current + delta * (max - min)));
      onChange(next);
    };
    const onUp = () => { isDragging.current = false; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [min, max, onChange]);

  const display = Number.isInteger(value) ? value : value.toFixed(decimals);

  return (
    <div
      className="flex flex-col items-center gap-0.5 select-none"
      style={{ cursor: 'ns-resize' }}
      onMouseDown={onMouseDown}
      onWheel={onWheel}
    >
      <svg
        width={size} height={size}
        style={{ filter: `drop-shadow(0 0 3px ${color}55)` }}
      >
        {/* Track background */}
        <path d={arc(-140, 140)} fill="none" stroke="#2a2a3a" strokeWidth="3.5" strokeLinecap="round" />
        {/* Active arc */}
        {norm > 0.002 && (
          <path d={arc(-140, deg)} fill="none" stroke={color} strokeWidth="3.5" strokeLinecap="round" />
        )}
        {/* Body */}
        <circle cx={cx} cy={cy} r={r * 0.48} fill="#141420" stroke="#2a2a3a" strokeWidth="1" />
        {/* Indicator */}
        <circle cx={px} cy={py} r={2.5} fill={color} />
      </svg>
      <div style={{ textAlign: 'center', lineHeight: 1.3 }}>
        <div style={{ fontSize: 9, color: '#505070', letterSpacing: 1, textTransform: 'uppercase' }}>
          {label}
        </div>
        <div style={{ fontSize: 10, color, fontWeight: 700 }}>
          {display}{unit}
        </div>
      </div>
    </div>
  );
}
