import { useRef, useEffect } from 'react';
import type { AudioEngine } from '../engine/AudioEngine';

interface Props {
  engine: AudioEngine;
  height?: number;
}

/**
 * Real-time FFT spectrum analyser — renders 1024 frequency bins
 * using Canvas 2D with a logarithmic frequency axis.
 */
export default function SpectrumAnalyzer({ engine, height = 130 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const dataRef = useRef<Float32Array | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const binCount = engine.spectrumAnalyser.frequencyBinCount;
    const data = new Float32Array(binCount);
    dataRef.current = data;

    const ctx = canvas.getContext('2d')!;
    const nyquist = engine.ctx.sampleRate / 2;

    // Frequency labels to draw
    const freqLabels = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

    const draw = () => {
      engine.getSpectrumData(data);

      const w = canvas.width;
      const h = canvas.height;

      // Background
      ctx.fillStyle = '#08080d';
      ctx.fillRect(0, 0, w, h);

      // Grid lines (dB)
      ctx.strokeStyle = '#1e1e2a';
      ctx.lineWidth = 1;
      ctx.font = '9px monospace';
      ctx.fillStyle = '#404058';
      for (let db = -100; db <= 0; db += 20) {
        const y = h - ((db + 100) / 100) * h;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
        ctx.fillText(`${db}`, 2, y - 2);
      }

      // Frequency bars — log scale
      const logMin = Math.log10(20);
      const logMax = Math.log10(nyquist);
      const logRange = logMax - logMin;

      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let i = 1; i < binCount; i++) {
        const freq = (i / binCount) * nyquist;
        const logFreq = Math.log10(Math.max(20, freq));
        const x = ((logFreq - logMin) / logRange) * w;
        const db = data[i];
        const normalised = Math.max(0, (db + 100) / 100);
        const y = h - normalised * h;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(w, h);
      ctx.closePath();

      // Gradient fill
      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, '#00ff8888');
      grad.addColorStop(0.5, '#00cc6644');
      grad.addColorStop(1, '#00ff8811');
      ctx.fillStyle = grad;
      ctx.fill();

      // Line on top
      ctx.beginPath();
      ctx.strokeStyle = '#00ff88cc';
      ctx.lineWidth = 1.5;
      for (let i = 1; i < binCount; i++) {
        const freq = (i / binCount) * nyquist;
        const logFreq = Math.log10(Math.max(20, freq));
        const x = ((logFreq - logMin) / logRange) * w;
        const db = data[i];
        const normalised = Math.max(0, (db + 100) / 100);
        const y = h - normalised * h;
        if (i === 1) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Frequency label ticks
      ctx.fillStyle = '#505070';
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      freqLabels.forEach(hz => {
        if (hz > nyquist) return;
        const logF = Math.log10(hz);
        const x = ((logF - logMin) / logRange) * w;
        ctx.fillStyle = '#303048';
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.strokeStyle = '#202030';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = '#505070';
        ctx.fillText(hz >= 1000 ? `${hz / 1000}k` : `${hz}`, x, h - 2);
      });
      ctx.textAlign = 'left';

      rafRef.current = requestAnimationFrame(draw);
    };

    // Resize observer
    const ro = new ResizeObserver(() => {
      canvas.width = canvas.offsetWidth;
      canvas.height = height;
    });
    ro.observe(canvas);
    canvas.width = canvas.offsetWidth;
    canvas.height = height;

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
    };
  }, [engine, height]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: `${height}px`, borderRadius: 4 }}
    />
  );
}
