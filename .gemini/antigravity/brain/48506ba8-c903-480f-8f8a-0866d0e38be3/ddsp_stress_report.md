# Neural Vocoder (DDSP ONNX) Stress Simulation Report

Tested on Pure Intent Architecture (v3.0) via `onnxruntime` CPU provider mimicking WebNN constraints.

| Instrument | Avg Latency | P99 Latency | Spectral Corr | SNR (approx) | RTF |
|------------|-------------|-------------|---------------|--------------|-----|
| Acoustic Guitar | 0.023ms | 0.054ms | 0.544 | -0.8 dB | 1.8x |
| Bass Guitar | 0.023ms | 0.056ms | 0.624 | -6.4 dB | 1.8x |
| Drum Kit | 0.023ms | 0.057ms | 0.272 | -2.7 dB | 1.8x |
| Electric Guitar | 0.022ms | 0.043ms | 0.228 | -1.1 dB | 1.8x |
| Female Vocal | 0.024ms | 0.061ms | 0.229 | -1.5 dB | 1.8x |
| Male Vocal | 0.023ms | 0.055ms | 0.383 | -4.7 dB | 1.8x |
| Piano | 0.055ms | 0.317ms | 0.651 | -5.4 dB | 1.4x |
| Saxophone | 0.027ms | 0.075ms | 0.587 | -3.2 dB | 1.8x |
| Trumpet | 0.031ms | 0.093ms | 0.509 | -3.4 dB | 1.7x |
