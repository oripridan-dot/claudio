/**
 * HRTFConvolutionEngine.cpp — Claudio Spatial Audio Engine
 *
 * Spherical-head HRTF filter synthesis + overlap-save convolution.
 *
 * References:
 *   [1] Woodworth & Schlosberg (1962) — Experimental Psychology. ITD formula.
 *   [2] Brown & Duda (1998) — "A structural model for binaural sound synthesis."
 *       IEEE Trans. Speech & Audio Processing 6(5):476-488. ILD shelf model.
 *   [3] Zölzer (2011) — DAFX: Digital Audio Effects, 2nd Ed. Ch. 3 (convolution).
 */
#ifndef M_PI
#  define M_PI    3.14159265358979323846
#endif
#ifndef M_PI_2
#  define M_PI_2  1.57079632679489661923
#endif

#include "HRTFConvolutionEngine.hpp"

#include <algorithm>
#include <cstring>

namespace claudio::spatial {
namespace {

// ─── Cooley-Tukey in-place FFT ────────────────────────────────────────────────

void fft_ct_impl(std::vector<std::complex<float>>& x, bool inverse) noexcept {
    const int N = static_cast<int>(x.size());
    if (N <= 1) return;

    // Bit-reversal permutation
    for (int i = 1, j = 0; i < N; ++i) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }

    // Butterfly stages (Decimation-In-Time)
    for (int len = 2; len <= N; len <<= 1) {
        const float ang =
            static_cast<float>(M_PI * 2.0 / len) * (inverse ? 1.f : -1.f);
        const std::complex<float> wlen{std::cos(ang), std::sin(ang)};
        for (int i = 0; i < N; i += len) {
            std::complex<float> w{1.f, 0.f};
            for (int j = 0; j < len / 2; ++j) {
                const auto u = x[i + j];
                const auto v = x[i + j + len / 2] * w;
                x[i + j]           = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        const float scale = 1.f / static_cast<float>(N);
        for (auto& c : x) c *= scale;
    }
}

// ─── Analytical HRTF filter synthesis ────────────────────────────────────────

/**
 * Synthesise a mono HRTF impulse response (IR) for one ear using the
 * Brown-Duda spherical-head model.
 *
 * The filter encodes:
 *   ① Fractional sample delay (Woodworth ITD) via windowed-sinc kernel
 *   ② Head shadow (first-order low-pass shelf) — Rayleigh ILD approximation
 *   ③ Elevation-dependent pinna spectral notch via comb filtering
 *
 * @param ir          Output buffer [len] — will be overwritten.
 * @param len         IR length in samples (= kHRTFFilterLen).
 * @param az_shadow   Azimuth used for head-shadow magnitude (0 = no shadow).
 *                    Pass abs(azimuth) for contralateral ear, 0 for ipsilateral.
 * @param elevation   Elevation angle in radians (used for pinna notch).
 * @param itd_samp    Fractional ITD delay in samples (may be 0 for the
 *                    near-side ear).
 * @param sample_rate Effective rendering sample rate.
 */
static void synthesise_hrtf_ir(float* ir, int len,
                                 float az_shadow, float elevation,
                                 float itd_samp,  float sample_rate) noexcept {
    // ── Head shadow: Brown-Duda one-pole low-pass shelf ──
    // Cutoff frequency of the shadow: f_s = c / (2π·a·|sin az|)
    constexpr float kSoundSpeed = 343.f;
    constexpr float kHeadRadius = 0.0875f;

    const float sin_az = std::abs(std::sin(az_shadow));
    const float f_cut  = (sin_az < 0.01f)
        ? sample_rate * 0.45f                       // no shadow → transparent
        : kSoundSpeed / (2.f * static_cast<float>(M_PI) * kHeadRadius * sin_az);

    const float wc = 2.f * static_cast<float>(M_PI) *
                     std::min(f_cut, sample_rate * 0.45f) / sample_rate;

    // IIR one-pole: H(z) = b0 / (1 - a1·z^-1)
    const float a1 = std::exp(-wc);
    const float b0 = 1.f - a1;

    // ── Windowed-sinc fractional delay ──
    for (int n = 0; n < len; ++n) {
        // Causal shadow filter impulse response: b0 · a1^n
        const float h_shadow = b0 * std::pow(a1, static_cast<float>(n));

        // Windowed sinc for fractional delay
        const float d = static_cast<float>(n) - itd_samp;
        const float sinc_val = (std::abs(d) < 1e-6f)
            ? 1.f
            : std::sin(static_cast<float>(M_PI) * d) /
              (static_cast<float>(M_PI) * d);

        // Hann window — suppresses Gibbs ringing
        const float win = 0.5f * (1.f - std::cos(
            2.f * static_cast<float>(M_PI) * static_cast<float>(n) /
            static_cast<float>(len - 1)));

        ir[n] = h_shadow * sinc_val * win;
    }

    // Normalise to unit DC gain
    float dc = 0.f;
    for (int n = 0; n < len; ++n) dc += ir[n];
    if (std::abs(dc) > 1e-9f)
        for (int n = 0; n < len; ++n) ir[n] /= dc;

    // ── Elevation pinna notch (comb filter) ──
    // Notch frequency rises with positive elevation (sound above the head).
    if (std::abs(elevation) > 0.1f) {
        const float f_notch = 8000.f +
            4000.f * (elevation / static_cast<float>(M_PI));
        const int d_int = static_cast<int>(sample_rate / f_notch);
        const float depth = 0.25f * std::abs(std::sin(elevation));
        if (d_int > 0 && d_int < len) {
            for (int n = d_int; n < len; ++n)
                ir[n] -= depth * ir[n - d_int];
        }
    }
}

}  // anonymous namespace

// ─── HRTFConvolutionEngine implementation ────────────────────────────────────

HRTFConvolutionEngine::HRTFConvolutionEngine(float sample_rate, int oversampling)
    : sample_rate_(sample_rate * static_cast<float>(oversampling)) {
    hrtf_l_.assign(kFFTBlockSize, {0.f, 0.f});
    hrtf_r_.assign(kFFTBlockSize, {0.f, 0.f});
    overlap_in_.assign(kHRTFFilterLen - 1, 0.f);
    tail_l_.assign(kHRTFFilterLen - 1, 0.f);
    tail_r_.assign(kHRTFFilterLen - 1, 0.f);
    // Build initial frontal (0°/0°) HRTF filters at construction time.
    rebuild_hrtf_filters();
}

void HRTFConvolutionEngine::drain_tracking_buffer() noexcept {
    if (!tracking_buf_) return;
    Quaternion6DoF q;
    if (!tracking_buf_->pop_latest(q)) return;
    const auto azel = quaternion_to_azel(q.w, q.x, q.y, q.z);
    az_ = azel.azimuth;
    el_ = azel.elevation;
}

void HRTFConvolutionEngine::rebuild_hrtf_filters() noexcept {
    const float itd_samp = compute_itd_seconds(az_) * sample_rate_;

    std::vector<float> ir_l(kHRTFFilterLen, 0.f);
    std::vector<float> ir_r(kHRTFFilterLen, 0.f);

    // Ipsilateral ear = little/no shadow, near delay.
    // Contralateral ear = full shadow, far delay (delay = ITD, shadow = az_shadow).
    const float delay_l = (az_ >= 0.f) ? 0.f        : -itd_samp;
    const float delay_r = (az_ >= 0.f) ? itd_samp   : 0.f;

    // Head-shadow magnitude parameter: abs(az) for the far-side ear, ~0 for near.
    const float shadow_l = (az_ < 0.f)  ? -az_ : 0.f;
    const float shadow_r = (az_ >= 0.f) ?  az_ : 0.f;

    synthesise_hrtf_ir(ir_l.data(), kHRTFFilterLen,
                        shadow_l, el_, delay_l, sample_rate_);
    synthesise_hrtf_ir(ir_r.data(), kHRTFFilterLen,
                        shadow_r, el_, delay_r, sample_rate_);

    // Zero-pad and convert to frequency domain (overlap-save).
    auto to_spectrum = [&](const std::vector<float>& ir,
                           std::vector<std::complex<float>>& spec) {
        spec.assign(kFFTBlockSize, {0.f, 0.f});
        for (int i = 0; i < kHRTFFilterLen; ++i) spec[i] = {ir[i], 0.f};
        fft_ct_impl(spec, false);
    };

    to_spectrum(ir_l, hrtf_l_);
    to_spectrum(ir_r, hrtf_r_);

    last_az_ = az_;
    last_el_ = el_;
    hrtf_updated_ = true;
}

void HRTFConvolutionEngine::fft_ct(std::vector<std::complex<float>>& buf,
                                    bool inverse) noexcept {
    fft_ct_impl(buf, inverse);
}

void HRTFConvolutionEngine::process_block(const float* in,
                                           float*       out_l,
                                           float*       out_r,
                                           int          block_size) noexcept {
    hrtf_updated_ = false;

    // 1. Consume the latest head orientation (drains ring buffer).
    drain_tracking_buffer();

    // 2. Rebuild HRTF if orientation changed by > 0.5° (≈ 0.0087 rad).
    if (std::abs(az_ - last_az_) > 0.0087f ||
        std::abs(el_ - last_el_) > 0.0087f) {
        rebuild_hrtf_filters();
    }

    // 3. Overlap-save convolution.
    //    Valid hop size: kFFTBlockSize - (kHRTFFilterLen - 1)
    const int tail_len = kHRTFFilterLen - 1;
    const int hop      = std::min(block_size,
                                  kFFTBlockSize - tail_len);

    // Build the overlap-save input frame: [overlap | new samples]
    std::vector<std::complex<float>> frame(kFFTBlockSize, {0.f, 0.f});
    for (int i = 0; i < tail_len; ++i)
        frame[i] = {overlap_in_[i], 0.f};
    for (int i = 0; i < hop; ++i)
        frame[tail_len + i] = {in[i], 0.f};

    // Slide the overlap buffer forward by 'hop' samples.
    const int remaining = tail_len - hop;
    if (remaining > 0) {
        for (int i = 0; i < remaining; ++i)
            overlap_in_[i] = overlap_in_[i + hop];
        for (int i = 0; i < hop && i < tail_len; ++i)
            overlap_in_[remaining + i] = in[i];
    } else {
        // hop >= tail_len: fill completely from new input
        for (int i = 0; i < tail_len; ++i)
            overlap_in_[i] = in[hop - tail_len + i];
    }

    // Forward FFT of the input frame.
    fft_ct_impl(frame, false);

    // Multiply by HRTF filter spectra.
    std::vector<std::complex<float>> spec_l(kFFTBlockSize),
                                     spec_r(kFFTBlockSize);
    for (int k = 0; k < kFFTBlockSize; ++k) {
        spec_l[k] = frame[k] * hrtf_l_[k];
        spec_r[k] = frame[k] * hrtf_r_[k];
    }

    // Inverse FFT.
    fft_ct_impl(spec_l, true);
    fft_ct_impl(spec_r, true);

    // Overlap-save: valid output starts at index tail_len.
    for (int i = 0; i < hop; ++i) {
        out_l[i] = spec_l[tail_len + i].real();
        out_r[i] = spec_r[tail_len + i].real();
    }
}

}  // namespace claudio::spatial
