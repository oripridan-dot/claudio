#pragma once
/**
 * HRTFConvolutionEngine.hpp — Claudio Spatial Audio Engine
 *
 * Frequency-domain (overlap-save) HRTF convolution engine.
 *
 * Synthesises binaural filters from the analytical spherical-head model:
 *   - ITD: Woodworth-Schlosberg formula (1962) — physically accurate for
 *          azimuth-dependent Interaural Time Difference
 *   - ILD: Rayleigh/Brown-Duda frequency-dependent head shadow approximation
 *   - Elevation: Simplified pinna spectral notch via comb filter
 * Supports internal oversampling × 1/2/4/8 (48 kHz → 96 → 192 → 384 kHz).
 * At 384 kHz, ITD resolution = 2.6 µs/sample — sufficient to localise
 * a source within ~0.5° of azimuth.
 *
 * THREAD / MUTEX POLICY
 *   All methods are audio-thread-only.
 *   Quaternion data enters exclusively via QuaternionRingBuffer (std::atomic).
 *   No locking overhead. No blocking synchronisation. Audio thread never sleeps.
 */
#ifndef M_PI
#  define M_PI    3.14159265358979323846
#endif
#ifndef M_PI_2
#  define M_PI_2  1.57079632679489661923
#endif

#include <complex>
#include <vector>
#include <cmath>
#include <cstdint>
#include "QuaternionRingBuffer.hpp"

namespace claudio::spatial {

// ─── Constants ────────────────────────────────────────────────────────────────
/// HRTF filter impulse-response length (samples at the effective sample rate).
/// Increased to 512 to support the larger sample delay at 384kHz (which requires ~253 samples for max ITD).
static constexpr int kHRTFFilterLen = 512;

/// Overlap-save FFT block size (power of 2, must be >= 2 * kHRTFFilterLen).
static constexpr int kFFTBlockSize = 2048;

// ─── Utility ──────────────────────────────────────────────────────────────────

/// Azimuth + elevation pair (radians).
struct AzEl {
    float azimuth{0.f};    ///< Horizontal angle; +right = +π/2
    float elevation{0.f};  ///< Vertical angle;   +up   = +π/2
};

/// Extract azimuth (yaw) and elevation (pitch) from a unit quaternion.
inline AzEl quaternion_to_azel(float w, float x, float y, float z) noexcept {
    const float sinyaw   = 2.f * (w * y + z * x);
    const float cosyaw   = 1.f - 2.f * (x * x + y * y);
    const float sinpitch = 2.f * (w * x - y * z);
    const float cospitch = 1.f - 2.f * (x * x + z * z);
    return { std::atan2(sinyaw, cosyaw), std::atan2(sinpitch, cospitch) };
}

/**
 * Woodworth-Schlosberg spherical-head ITD (seconds).
 *
 * Reference: Woodworth & Schlosberg (1962) "Experimental Psychology".
 * For a rigid sphere of radius a: ITD(θ) = (a/c)(sin θ + θ)  for |θ| ≤ π/2
 *                                  ITD(θ) = (a/c)(1 + π/2)   for |θ| > π/2
 *
 * @param azimuth_rad   Azimuth angle in radians (positive = right).
 * @param head_radius_m Head radius metres (KEMAR standard: 0.0875 m).
 */
inline float compute_itd_seconds(float azimuth_rad,
                                  float head_radius_m = 0.0875f) noexcept {
    constexpr float kSoundSpeed = 343.0f;
    const float a   = std::abs(azimuth_rad);
    const float itd = (head_radius_m / kSoundSpeed) *
                      (std::sin(a) + std::min(a, static_cast<float>(M_PI_2)));
    return std::copysign(itd, azimuth_rad);
}

// ─── Engine ───────────────────────────────────────────────────────────────────

/**
 * HRTFConvolutionEngine
 *
 * Per-block processing pipeline:
 *   1. Drain QuaternionRingBuffer → update current azimuth / elevation
 *   2. If orientation changed by > 0.5°, rebuild HRTF filters (lazy)
 *   3. Overlap-save FFT convolution → left + right binaural output
 */
class HRTFConvolutionEngine {
  public:
    /**
     * @param sample_rate  Nominal sample rate Hz (e.g. 48000, 96000, 192000).
     *                     If oversampling > 1, pass the BASE rate; the engine
     *                     scales internally.
     * @param oversampling Internal oversampling factor: 1, 2, 4, or 8.
     */
    explicit HRTFConvolutionEngine(float sample_rate = 48000.f,
                                   int   oversampling = 1);

    /// Attach the shared ring buffer (owned by HolographicBinauralNode).
    void set_tracking_buffer(QuaternionRingBuffer<128>* buf) noexcept {
        tracking_buf_ = buf;
    }

    /**
     * Process one block of mono audio → stereo binaural.
     *
     * @param in         Mono input  [block_size samples]
     * @param out_l      Left  output [block_size samples]
     * @param out_r      Right output [block_size samples]
     * @param block_size Frames per block (must be <= kFFTBlockSize/2 = 256)
     */
    void process_block(const float* in,
                       float*       out_l,
                       float*       out_r,
                       int          block_size) noexcept;

    // ── Accessors ──────────────────────────────────────────────────────────
    float effective_sample_rate()     const noexcept { return sample_rate_; }
    float current_azimuth()           const noexcept { return az_; }
    float current_elevation()         const noexcept { return el_; }
    bool  hrtf_updated_last_block()   const noexcept { return hrtf_updated_; }

  private:
    void drain_tracking_buffer() noexcept;
    void rebuild_hrtf_filters()  noexcept;

    static void fft_ct(std::vector<std::complex<float>>& buf,
                       bool inverse) noexcept;

    float  sample_rate_{48000.f};
    float  az_{0.f};
    float  el_{0.f};
    float  last_az_{999.f};   // sentinel → triggers first build
    float  last_el_{999.f};
    bool   hrtf_updated_{false};

    QuaternionRingBuffer<128>* tracking_buf_{nullptr};

    std::vector<std::complex<float>> hrtf_l_;   // frequency domain, length = kFFTBlockSize
    std::vector<std::complex<float>> hrtf_r_;

    std::vector<float> overlap_in_;             // last (kHRTFFilterLen-1) input samples
    std::vector<float> tail_l_;
    std::vector<float> tail_r_;
};

}  // namespace claudio::spatial
