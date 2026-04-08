## Holographic Binaural Rendering Node

> **Wave 52 — TooLoo Governor Mandate · Approved 2026-03-13**

The `dsp/spatial/` module delivers the complete holographic audio experience:
192 kHz binaural HRTF rendering, lock-free 6DoF head tracking, and dynamic
acoustic raytracing — all wired together in the `HolographicBinauralNode`.

### Signal Path

```
Mono DDSP Output
  │
  ▼
AcousticRaytracer ─────── Inverse-square law (1/d distance rolloff)
  │                        Proximity effect (LF shelf boost inside 30 cm)
  │                        3 early reflections: floor + left wall + right wall
  │                        (image-source method, absorption-attenuated)
  ▼
HRTFConvolutionEngine ─── Overlap-save FFT convolution (kFFTBlockSize = 512)
  │                        Spherical-head model (Woodworth ITD + Rayleigh ILD)
  │                        Elevation pinna notch (comb filter)
  │                        Internal oversampling: ×1 / ×2 / ×4  (192 kHz max)
  │                        QuaternionRingBuffer drain — lock-free, per block
  ▼
Stereo Binaural Output (L + R, phase-matched, ITD-correct)
```

### Head Tracking — Lock-Free 6DoF

`QuaternionRingBuffer<128>` is the **sole** cross-thread interface.

| Constraint | Enforcement |
|---|---|
| No `std::mutex` in audio path | SpatialLatencyGate Phase 0 (source scan) |
| No `std::lock_guard` | Same |
| HRTF update < **1.5 ms** after 90° head turn | SpatialLatencyGate Phase 2 |
| ILD delta ≥ **3 dB** at 90° azimuth | SpatialLatencyGate Phase 3 |

### Build & Test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --target claudio_spatial SpatialLatencyGate
ctest --test-dir build -R SpatialLatencyGate --verbose
```

### Frontend Integration

```typescript
import { SpatialEngine } from './spatial/SpatialEngine';
import { HeadTracker }   from './spatial/HeadTracker';

const engine  = new SpatialEngine(audioCtx);
engine.attach(sourceNode);                 // route mono → binaural

const tracker = new HeadTracker();
await tracker.start(q => engine.updateOrientation(q));
// Tracking selects: WebXR → DeviceOrientation → Mouse drag
```

### Sample Rate vs. ITD Resolution

| Rate | Time/sample | ITD precision |
|------|-------------|---------------|
| 48 kHz  (×1) | 20.8 µs | Casual listening |
| 96 kHz  (×2) | 10.4 µs | Studio monitoring |
| 192 kHz (×4) |  5.2 µs | Mastering / spatial research |

At 192 kHz the engine resolves ~10 µs ITD differences — sufficient to
localise virtual instruments within **±1°** of azimuth.
