import { useEffect, useRef } from 'react';

export interface Critique {
  passed: boolean;
  message: string;
  metric: string;
  severity: "low" | "medium" | "high";
  delta: number;
}

interface GeometricFractureProps {
  critiques: Critique[];
}

export function GeometricFracture({ critiques }: GeometricFractureProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  
  // Physics State
  const phys = useRef({
    rx: 0, ry: 0, 
    shatterForce: 0, // 0 = cohesive, 1+ = fractured
    fatigue: 0,      // 0 = vibrant, 1 = grey/sluggish
    driftAngle: 0    // Adds a shear distortion based on pitch pitch
  });

  // Track target physics state based on incoming critiques
  useEffect(() => {
    let targetShatter = 0;
    let targetFatigue = 0;
    let targetDrift = 0;

    for (const c of critiques) {
       if (c.metric === 'onsetStrength') {
          targetShatter += c.delta * 20; // Shatter on transient failure
       }
       if (c.metric === 'pitchDrift') {
          targetDrift += c.delta * 0.05; // Shear on intonation failure
       }
       if (c.metric === 'timbralFatigue') {
          targetFatigue = Math.min(1.0, c.delta * 2); // Desaturate on spectral loss
       }
    }
    
    // Smooth easing state toward targets
    phys.current.shatterForce = targetShatter > 0 ? targetShatter : phys.current.shatterForce * 0.95;
    phys.current.driftAngle += (targetDrift - phys.current.driftAngle) * 0.1;
    phys.current.fatigue += (targetFatigue - phys.current.fatigue) * 0.05;

  }, [critiques]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Responsive container
    const resize = () => {
      if (canvas.parentElement) {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = canvas.parentElement.clientHeight;
      }
    };
    const ro = new ResizeObserver(resize);
    if (canvas.parentElement) ro.observe(canvas.parentElement);
    resize();

    // 3D Geometry vertices (Icosahedron basic mapping)
    const numNodes = 60;
    const nodes = Array.from({ length: numNodes }).map(() => ({
       x: (Math.random() - 0.5) * 2,
       y: (Math.random() - 0.5) * 2,
       z: (Math.random() - 0.5) * 2,
       // normalize into a sphere
    }));
    
    // Normalize and scale
    nodes.forEach(n => {
       const d = Math.sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
       n.x /= d; n.y /= d; n.z /= d;
    });

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const { width, height } = canvas;
      const cx = width / 2;
      const cy = height / 2;
      const baseRadius = Math.min(width, height) * 0.35;

      const p = phys.current;
      
      // Sluggish rotation based on fatigue
      const rotSpeed = 0.005 * (1 - p.fatigue * 0.8);
      p.ry += rotSpeed;
      p.rx += rotSpeed * 0.4;

      const cosY = Math.cos(p.ry); const sinY = Math.sin(p.ry);
      const cosX = Math.cos(p.rx); const sinX = Math.sin(p.rx);

      // Desaturate and cool down Colors based on fatigue
      const glowStr = Math.max(0, 0.8 - p.fatigue);
      const hue = 150 - p.fatigue * 150; // green to blue/grey
      const sat = 100 - p.fatigue * 100;

      ctx.save();
      // Apply pitch drift shearing (Affine Matrix transform)
      if (p.driftAngle > 0.01) {
         ctx.translate(cx, cy);
         ctx.transform(1, p.driftAngle, -p.driftAngle, 1, 0, 0);
         ctx.translate(-cx, -cy);
      }

      ctx.lineWidth = 1;

      // Project and draw
      const projected = nodes.map((n, i) => {
         // Rotate Y
         let x1 = n.x * cosY - n.z * sinY;
         let z1 = n.z * cosY + n.x * sinY;
         // Rotate X
         let y2 = n.y * cosX - z1 * sinX;
         let z2 = z1 * cosX + n.y * sinX;

         // Apply Shatter force (explode outwards based on node index hash)
         const shatter = p.shatterForce * 20 * (i % 2 === 0 ? 1 : -0.5);
         x1 += (x1 * shatter) / baseRadius;
         y2 += (y2 * shatter) / baseRadius;

         const zScale = 2 / (2 - z2);
         const px = cx + x1 * baseRadius * zScale;
         const py = cy + y2 * baseRadius * zScale;

         return { px, py, z: z2 };
      });

      // Draw Edges
      ctx.strokeStyle = `hsla(${hue}, ${sat}%, 50%, ${glowStr * 0.5 + 0.1})`;
      for (let i = 0; i < projected.length; i++) {
         for (let j = i + 1; j < projected.length; j++) {
            const dx = projected[i].px - projected[j].px;
            const dy = projected[i].py - projected[j].py;
            const distSq = dx*dx + dy*dy;
            
            // Shatter logic: break lines if shatter is high
            const maxDist = (baseRadius * 0.6) + (p.shatterForce * 80);
            
            if (distSq < maxDist * maxDist) {
               ctx.beginPath();
               ctx.moveTo(projected[i].px, projected[i].py);
               ctx.lineTo(projected[j].px, projected[j].py);
               ctx.stroke();
            }
         }
      }

      // Draw Nodes
      ctx.fillStyle = `hsla(${hue}, ${sat}%, 70%, ${glowStr + 0.2})`;
      projected.forEach(p => {
         const size = Math.max(0.5, (p.z + 1.5) * 2);
         ctx.beginPath();
         ctx.arc(p.px, p.py, size, 0, Math.PI * 2);
         ctx.fill();
      });

      ctx.restore();

      // UI Overlay Text (Brutal Honesty)
      if (critiques.length > 0) {
         ctx.fillStyle = '#ff4444';
         ctx.font = 'bold 24px Inter, sans-serif';
         ctx.textAlign = 'center';
         ctx.shadowBlur = 10;
         ctx.shadowColor = 'rgba(255, 68, 68, 0.8)';
         
         critiques.forEach((c, idx) => {
            const p = cy + baseRadius + 60 + (idx * 30);
            ctx.fillText(c.message, cx, p);
         });
         ctx.shadowBlur = 0; // reset
      }

      animRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      ro.disconnect();
      cancelAnimationFrame(animRef.current);
    };
  }, [critiques]);

  return (
    <div style={{
      position: 'relative',
      width: '100%',
      height: '100%',
      backgroundColor: 'rgba(20, 25, 30, 0.4)',
      borderRadius: '12px',
      border: '1px solid rgba(255, 255, 255, 0.05)',
      overflow: 'hidden',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      boxShadow: 'inset 0 0 20px rgba(0,0,0,0.5)',
    }}>
      <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
    </div>
  );
}
