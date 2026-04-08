import { useRef, useEffect } from 'react';
import type { AudioEngine } from '../engine/AudioEngine';

interface Props {
  engine: AudioEngine;
  height?: number;
  color?: string;
}

/** Real-time oscilloscope via Web Audio AnalyserNode. */
export default function WaveformViewer({ engine, height = 80, color = '#00ff88' }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const bufLen = engine.waveformAnalyser.fftSize;
    const data = new Float32Array(bufLen);

    const draw = () => {
      engine.getWaveformData(data);

      const w = canvas.width;
      const h = canvas.height;

      ctx.fillStyle = '#08080d';
      ctx.fillRect(0, 0, w, h);

      // Zero line
      ctx.strokeStyle = '#1e1e2a';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();

      // Waveform
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.shadowColor = color;
      ctx.shadowBlur = 5;

      const sliceW = w / bufLen;
      for (let i = 0; i < bufLen; i++) {
        const x = i * sliceW;
        const y = (1 - (data[i] * 0.8 + 0.5)) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      rafRef.current = requestAnimationFrame(draw);
    };

    const ro = new ResizeObserver(() => {
      canvas.width = canvas.offsetWidth;
      canvas.height = height;
    });
    ro.observe(canvas);
    canvas.width = canvas.offsetWidth;
    canvas.height = height;

    rafRef.current = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(rafRef.current); ro.disconnect(); };
  }, [engine, height, color]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: `${height}px`, borderRadius: 4 }}
    />
  );
}
