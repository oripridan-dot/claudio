# Claudio — Real-Time AI Musician Collaboration Platform

> **Ultra-low latency semantic audio generation for global networked music performance.**

Claudio transforms live acoustic performances into semantic intents — capturing pitch, timbre, loudness, and physical playing nuance — compresses them into ultra-low bitrate packets, and regenerates pristine audio at the destination using DDSP and GPU-accelerated generative AI.

## Quick Start

### C++ DSP Engine
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Python AI Engine
```bash
uv sync
uv run pytest tests/ -v
```

### Frontend (DSP Studio)
```bash
cd frontend && npm install && npm run dev
```

### DDSP Training
```bash
uv run python scripts/train.py --data ./data/violin --epochs 100
uv run python scripts/export_onnx.py --checkpoint ./checkpoints/best.pt --output ./dist/synth.onnx
```

## License

MIT
