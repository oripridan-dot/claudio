/**
 * HeadTracker.ts — Claudio 6DoF Head Tracking
 *
 * Converts device orientation sensors into a continuous Quaternion6DoF stream
 * for SpatialEngine.updateOrientation() (or the C++ HolographicBinauralNode
 * via the WASM bridge).
 *
 * Source priority (best → fallback):
 *   1. WebXR Device API    — VR/AR headsets (6DoF with translation)
 *   2. DeviceOrientationEvent — Mobile/tablet gyro+compass (3DoF)
 *   3. Mouse right-drag    — Desktop pointer fallback (2DoF yaw+pitch)
 *
 * Output: ~60 Hz Quaternion6DoF events via the onOrientation callback.
 */
import type { Quaternion6DoF } from './SpatialEngine';

export type OrientationCallback = (q: Quaternion6DoF) => void;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Convert ZYX Euler angles (degrees) to a unit quaternion. */
function eulerToQuat(
  alpha: number,  // Yaw   — compass heading (Z rotation)
  beta:  number,  // Pitch — front/back tilt (X rotation)
  gamma: number   // Roll  — left/right tilt  (Y rotation)
): Pick<Quaternion6DoF, 'w' | 'x' | 'y' | 'z'> {
  const r = Math.PI / 180;
  const a = alpha * r * 0.5, b = beta * r * 0.5, g = gamma * r * 0.5;
  const ca = Math.cos(a), sa = Math.sin(a);
  const cb = Math.cos(b), sb = Math.sin(b);
  const cg = Math.cos(g), sg = Math.sin(g);
  return {
    w:  ca * cb * cg + sa * sb * sg,
    x:  ca * sb * cg + sa * cb * sg,
    y:  sa * cb * cg - ca * sb * sg,
    z:  ca * cb * sg - sa * sb * cg,
  };
}

// ─── HeadTracker ─────────────────────────────────────────────────────────────

export class HeadTracker {
  private callback:     OrientationCallback | null = null;
  private mode:         'webxr' | 'device' | 'mouse' | 'idle' = 'idle';
  private mouseYaw    = 0;
  private mousePitch  = 0;
  private mouseDown   = false;
  private lastMX      = 0;
  private lastMY      = 0;
  private xrSession:  XRSession | null = null;
  private rafId:      number | null    = null;

  /** Detected tracking mode (read after start()). */
  get trackingMode(): string { return this.mode; }

  /**
   * Start head tracking.  Automatically selects the best sensor source
   * and invokes cb at ~60 Hz.
   *
   * WebXR requires user gesture to request immersive-vr session.
   * DeviceOrientation on iOS 13+ requires explicit permission.
   * Desktop always falls back to mouse right-drag.
   */
  async start(cb: OrientationCallback): Promise<void> {
    this.callback = cb;

    // 1. WebXR (VR/AR headset — highest quality, 6DoF)
    if ('xr' in navigator) {
      try {
        const supported = await (navigator as any).xr
          .isSessionSupported('immersive-vr');
        if (supported) { await this._startWebXR(); return; }
      } catch { /* silently fall through */ }
    }

    // 2. DeviceOrientation (mobile / gyro-equipped tablet)
    if (typeof DeviceOrientationEvent !== 'undefined') {
      if (typeof (DeviceOrientationEvent as any).requestPermission === 'function') {
        try {
          const result = await (DeviceOrientationEvent as any).requestPermission();
          if (result === 'granted') { this._startDeviceOrientation(); return; }
        } catch { /* no permission */ }
      } else {
        this._startDeviceOrientation();
        return;
      }
    }

    // 3. Mouse drag fallback
    this._startMouseFallback();
  }

  stop(): void {
    if (this.mode === 'device')
      window.removeEventListener('deviceorientation', this._onDeviceOrientation);
    if (this.mode === 'mouse') {
      window.removeEventListener('mousemove',  this._onMouseMove);
      window.removeEventListener('mousedown',  this._onMouseDown);
      window.removeEventListener('mouseup',    this._onMouseUp);
      window.removeEventListener('contextmenu', this._onContextMenu);
    }
    if (this.xrSession) {
      this.xrSession.end().catch(() => {});
      this.xrSession = null;
    }
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    this.mode     = 'idle';
    this.callback = null;
  }

  // ── WebXR ─────────────────────────────────────────────────────────────────

  private async _startWebXR(): Promise<void> {
    this.mode = 'webxr';
    const session: XRSession = await (navigator as any).xr
      .requestSession('immersive-vr');
    this.xrSession = session;
    const refSpace = await session.requestReferenceSpace('local');

    const tick = (_t: number, frame: XRFrame) => {
      const pose = frame.getViewerPose(refSpace);
      if (pose && this.callback) {
        const { x, y, z, w } = pose.transform.orientation;
        const { x: tx, y: ty, z: tz } = pose.transform.position;
        this.callback({ w, x, y, z, tx, ty, tz });
      }
      this.rafId = session.requestAnimationFrame(tick);
    };
    this.rafId = session.requestAnimationFrame(tick);
  }

  // ── DeviceOrientation ─────────────────────────────────────────────────────

  private _startDeviceOrientation(): void {
    this.mode = 'device';
    window.addEventListener('deviceorientation',
      this._onDeviceOrientation, { passive: true });
  }

  private _onDeviceOrientation = (e: DeviceOrientationEvent): void => {
    if (!this.callback) return;
    const q = eulerToQuat(e.alpha ?? 0, e.beta ?? 0, e.gamma ?? 0);
    this.callback({ ...q, tx: 0, ty: 0, tz: -1 });
  };

  // ── Mouse drag fallback ───────────────────────────────────────────────────

  private _startMouseFallback(): void {
    this.mode = 'mouse';
    window.addEventListener('mousemove',   this._onMouseMove,   { passive: true });
    window.addEventListener('mousedown',   this._onMouseDown);
    window.addEventListener('mouseup',     this._onMouseUp);
    window.addEventListener('contextmenu', this._onContextMenu);
    // Emit initial identity orientation immediately.
    this._emitMouse();
  }

  private _onContextMenu = (e: Event): void => { e.preventDefault(); };
  private _onMouseDown   = (e: MouseEvent): void => {
    if (e.button !== 2) return;  // Right-click drag = head rotation
    this.mouseDown = true;
    this.lastMX = e.clientX;
    this.lastMY = e.clientY;
  };
  private _onMouseUp     = ():             void  => { this.mouseDown = false; };
  private _onMouseMove   = (e: MouseEvent): void => {
    if (!this.mouseDown) return;
    const dx = e.clientX - this.lastMX;
    const dy = e.clientY - this.lastMY;
    this.lastMX = e.clientX;
    this.lastMY = e.clientY;
    // 0.3° per pixel — comfortable sensitivity for yaw/pitch simulation.
    this.mouseYaw   = (this.mouseYaw   + dx * 0.3) % 360;
    this.mousePitch = Math.max(-80, Math.min(80, this.mousePitch + dy * 0.3));
    this._emitMouse();
  };

  private _emitMouse(): void {
    if (!this.callback) return;
    const q = eulerToQuat(this.mouseYaw, this.mousePitch, 0);
    this.callback({ ...q, tx: 0, ty: 0, tz: -1 });
  }
}
