/**
 * SpatialEngine.ts — Claudio Holographic Spatial Audio (Web Audio API layer)
 *
 * Browser integration for the Holographic Binaural Rendering Node.
 *
 * Architecture:
 *   [Mono AudioNode] → PannerNode (HRTF) → [Destination]
 *
 * The PannerNode uses the browser's built-in HRTF database for real-time
 * binaural rendering via the Web Audio API.  The C++ HolographicBinauralNode
 * handles the full 192 kHz / 32-bit path in production via the
 * claudio-core.wasm bridge module (WebAssembly, future milestone).
 *
 * Head tracking: call updateOrientation() with any Quaternion6DoF.
 * Position updates are applied as non-blocking AudioParam ramps (~5 ms)
 * to prevent clicks — mirroring the C++ ring-buffer drain semantic.
 */

export interface Quaternion6DoF {
  /** Unit quaternion — rotation component */
  w: number; x: number; y: number; z: number;
  /** Translation in metres — source or listener position */
  tx: number; ty: number; tz: number;
}

export interface SpatialSourceConfig {
  distance?: number;
  rolloffModel?: 'inverse' | 'linear' | 'exponential';
  maxDistance?: number;
  refDistance?: number;
}

/**
 * SpatialEngine
 *
 * Wraps a mono AudioNode source in a binaural 3D spatial field using HRTF.
 * Interface is intentionally parallel to the C++ HolographicBinauralNode.
 */
export class SpatialEngine {
  private readonly ctx: AudioContext;
  private readonly panner: PannerNode;
  private _azimuth   = 0;
  private _elevation = 0;
  private _distance  = 1;

  constructor(ctx: AudioContext, config: SpatialSourceConfig = {}) {
    this.ctx    = ctx;
    this.panner = ctx.createPanner();

    // HRTF panning: browser performs binaural convolution internally.
    this.panner.panningModel  = 'HRTF';
    this.panner.distanceModel = config.rolloffModel === 'linear'
      ? 'linear'
      : config.rolloffModel === 'exponential'
        ? 'exponential'
        : 'inverse';

    this.panner.refDistance   = config.refDistance ?? 1;
    this.panner.maxDistance   = config.maxDistance ?? 10000;
    this.panner.rolloffFactor = 1;

    // Full sphere — no directional cone on the source.
    this.panner.coneInnerAngle = 360;
    this.panner.coneOuterAngle = 360;
    this.panner.coneOuterGain  = 0;

    // Default position: 1 m directly in front of the listener.
    this.panner.positionX.value = 0;
    this.panner.positionY.value = 0;
    this.panner.positionZ.value = -1;
  }

  /** Connect source into the spatial field and route to destination. */
  attach(source: AudioNode,
         destination: AudioNode = this.ctx.destination): void {
    source.connect(this.panner);
    this.panner.connect(destination);
  }

  detach(): void {
    this.panner.disconnect();
  }

  /**
   * Update source orientation from a 6DoF quaternion.
   *
   * Converts quaternion rotation to Cartesian source position relative to
   * the listener's head.  Position is ramped over 5 ms via AudioParam
   * automation (non-blocking — no mutex, no thread coordination required).
   *
   * This mirrors the lock-free ring-buffer drain in the C++ engine.
   */
  updateOrientation(q: Quaternion6DoF): void {
    // Extract yaw (azimuth) and pitch (elevation) from the unit quaternion.
    const sinYaw   = 2 * (q.w * q.y + q.z * q.x);
    const cosYaw   = 1 - 2 * (q.x * q.x + q.y * q.y);
    const sinPitch = 2 * (q.w * q.x - q.y * q.z);
    const cosPitch = 1 - 2 * (q.x * q.x + q.z * q.z);

    this._azimuth   = Math.atan2(sinYaw, cosYaw);
    this._elevation = Math.atan2(sinPitch, cosPitch);
    this._distance  = Math.sqrt(q.tx ** 2 + q.ty ** 2 + q.tz ** 2) || 1;

    // Spherical → Cartesian: Web Audio convention X=right, Y=up, Z=towards listener
    const cosEl = Math.cos(this._elevation);
    const x =  this._distance * Math.sin(this._azimuth)  * cosEl;
    const y =  this._distance * Math.sin(this._elevation);
    const z = -this._distance * Math.cos(this._azimuth)  * cosEl;

    // 5 ms ramp prevents audible discontinuities (mirrors C++ 0.5° threshold).
    const ramp = this.ctx.currentTime + 0.005;
    this.panner.positionX.linearRampToValueAtTime(x, ramp);
    this.panner.positionY.linearRampToValueAtTime(y, ramp);
    this.panner.positionZ.linearRampToValueAtTime(z, ramp);
  }

  /** Set source Cartesian position directly (metres). */
  setPosition(x: number, y: number, z: number): void {
    const tc = this.ctx.currentTime;
    this.panner.positionX.setTargetAtTime(x, tc, 0.005);
    this.panner.positionY.setTargetAtTime(y, tc, 0.005);
    this.panner.positionZ.setTargetAtTime(z, tc, 0.005);
    this._distance = Math.sqrt(x ** 2 + y ** 2 + z ** 2) || 1;
  }

  get currentAzimuth():   number     { return this._azimuth; }
  get currentElevation(): number     { return this._elevation; }
  get currentDistance():  number     { return this._distance; }
  get pannerNode():        PannerNode { return this.panner; }
}
