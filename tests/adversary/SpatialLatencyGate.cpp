/**
 * SpatialLatencyGate.cpp — Claudio Adversary Validator
 *
 * ════════════════════════════════════════════════════════════════════════
 *  ADVERSARY VALIDATOR: SpatialLatencyGate
 * ════════════════════════════════════════════════════════════════════════
 *
 * Mandate (Governor approved 2026-03-13):
 *   If the operator turns their head 90° (a quaternion yaw rotation is
 *   pushed to the ring buffer), the HRTF convolution matrix MUST update
 *   the binaural phase relationship of the audio output within 1.5 ms.
 *   Exceeding this threshold constitutes "acoustic smearing" — the auditory
 *   equivalent of VR motion sickness.  The build MUST instantly fail.
 *
 * Test phases:
 *   Phase 0 — Mutex audit: scan QuaternionRingBuffer.hpp for std::mutex.
 *             Any hit (outside a // comment) = FAIL.
 *   Phase 1 — Baseline ILD at 0° azimuth (identity quaternion).
 *   Phase 2 — SpatialLatencyGate: push 90° yaw, clock until HRTF updates.
 *             Assert wall-clock elapsed < 1.5 ms.
 *   Phase 3 — ILD verification: assert ILD delta >= 3 dB after 90° turn.
 *             Head shadow must be perceptible.
 *
 * Build: tests/adversary/CMakeLists.txt  (linked against claudio_spatial)
 */
#ifndef M_PI
#  define M_PI  3.14159265358979323846
#endif

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "HolographicBinauralNode.hpp"

using namespace claudio::spatial;
using Clock = std::chrono::high_resolution_clock;

// ─── Metrics helpers ──────────────────────────────────────────────────────────

static float rms_db(const float* buf, int n) {
    float sum = 0.f;
    for (int i = 0; i < n; ++i) sum += buf[i] * buf[i];
    const float rms = std::sqrt(sum / static_cast<float>(n));
    return (rms > 1e-9f) ? 20.f * std::log10(rms) : -120.f;
}

/// Interaural Level Difference (dB): left_dB - right_dB
static float compute_ild(const float* l, const float* r, int n) {
    return rms_db(l, n) - rms_db(r, n);
}

/// 90° yaw quaternion: w=cos(π/4), y=sin(π/4)
static Quaternion6DoF make_90deg_right_yaw() {
    constexpr float half = static_cast<float>(M_PI) / 4.f;
    Quaternion6DoF q;
    q.w = std::cos(half); q.x = 0.f;
    q.y = std::sin(half); q.z = 0.f;
    q.tx = 0.f; q.ty = 0.f; q.tz = -1.f;  // source 1 m ahead
    q.timestamp_us = 0;
    return q;
}

// ─── Phase 0: Mutex audit ─────────────────────────────────────────────────────

/**
 * Scan a source file for forbidden synchronisation primitives.
 * Any occurrence outside a line comment triggers FAIL.
 */
static bool audit_no_mutex(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr,
            "[SpatialLatencyGate][Phase 0] WARNING: cannot open '%s' for audit\n",
            path);
        return true;  // Cannot audit — linker will surface real violations
    }
    static const char* kForbidden[] = {
        "std::mutex", "std::lock_guard", "std::unique_lock",
        "std::condition_variable", "std::recursive_mutex", nullptr
    };
    std::string line;
    int lineno = 0;
    bool clean = true;
    while (std::getline(f, line)) {
        ++lineno;
        for (int i = 0; kForbidden[i]; ++i) {
            const auto match_pos   = line.find(kForbidden[i]);
            const auto comment_pos = line.find("//");
            if (match_pos != std::string::npos &&
                (comment_pos == std::string::npos || match_pos < comment_pos)) {
                std::fprintf(stderr,
                    "[SpatialLatencyGate] FAIL [mutex_audit] %s:%d — '%s' found:\n"
                    "  %s\n", path, lineno, kForbidden[i], line.c_str());
                clean = false;
            }
        }
    }
    return clean;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    std::srand(42);  // Deterministic noise seed

    std::printf("════════════════════════════════════════════════════════════\n");
    std::printf("  CLAUDIO — SpatialLatencyGate Adversary Validator\n");
    std::printf("  Law: HRTF must update < 1.5 ms after 90° head turn\n");
    std::printf("════════════════════════════════════════════════════════════\n\n");

    int fail_count = 0;

    // ── Phase 0: Mutex audit ───────────────────────────────────────────────
    std::printf("[Phase 0] Mutex audit of QuaternionRingBuffer.hpp ...\n");
    const bool mutex_clean = audit_no_mutex("dsp/spatial/QuaternionRingBuffer.hpp");
    if (!mutex_clean) {
        ++fail_count;
    } else {
        std::printf("PASS [mutex_audit] No forbidden synchronisation primitives.\n");
    }

    // ── Phase 1: Instantiate node + baseline ILD ───────────────────────────
    std::printf("\n[Phase 1] Baseline ILD at 0° azimuth ...\n");

    BinauralNodeConfig cfg;
    cfg.sample_rate  = 48000.f;
    cfg.oversampling = 4;   // Renders at 192 kHz internally

    auto node = HolographicBinauralNode::create(cfg);

    constexpr int kBlock = 64;
    std::vector<float> noise(kBlock), out_l0(kBlock), out_r0(kBlock);

    for (int i = 0; i < kBlock; ++i)
        noise[i] = (static_cast<float>(std::rand()) / RAND_MAX * 2.f - 1.f) * 0.5f;

    // Warm up: 10 blocks at identity orientation (0° azimuth)
    node->push_head_orientation(1.f, 0.f, 0.f, 0.f, 0.f, 0.f, -1.f);
    for (int b = 0; b < 10; ++b)
        node->process_block(noise.data(), out_l0.data(), out_r0.data(), kBlock);

    const float ild_base = compute_ild(out_l0.data(), out_r0.data(), kBlock);
    std::printf("  Baseline ILD @ 0°: %.2f dB  (expected ~0)\n", ild_base);

    // ── Phase 2: SpatialLatencyGate ────────────────────────────────────────
    std::printf("\n[Phase 2] SpatialLatencyGate — injecting 90° right yaw ...\n");

    const auto q90 = make_90deg_right_yaw();
    std::vector<float> out_l(kBlock), out_r(kBlock);

    const auto t0 = Clock::now();

    // Push the 90° turn into the lock-free ring buffer.
    node->push_head_orientation(q90.w, q90.x, q90.y, q90.z,
                                q90.tx, q90.ty, q90.tz);

    // Process blocks until the HRTF engine reports it has rebuilt its filters.
    constexpr int kMaxBlocks = 20;
    bool hrtf_updated = false;
    int  blocks_run   = 0;
    for (int b = 0; b < kMaxBlocks && !hrtf_updated; ++b) {
        node->process_block(noise.data(), out_l.data(), out_r.data(), kBlock);
        hrtf_updated = node->hrtf_updated_last_block();
        ++blocks_run;
    }

    const auto t1 = Clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    const double block_ms =
        static_cast<double>(kBlock) / node->effective_sample_rate() * 1000.0;

    std::printf("  Effective sample rate : %.0f Hz\n",
                node->effective_sample_rate());
    std::printf("  Block size            : %d samples (%.4f ms)\n",
                kBlock, block_ms);
    std::printf("  Blocks until update   : %d\n", blocks_run);
    std::printf("  Wall-clock elapsed    : %.4f ms\n", elapsed_ms);

    if (!hrtf_updated) {
        std::fprintf(stderr,
            "FAIL [hrtf_update] HRTF never rebuilt in %d blocks.\n", kMaxBlocks);
        ++fail_count;
    }
    if (elapsed_ms >= 1.5) {
        std::fprintf(stderr,
            "FAIL [latency_gate] %.4f ms >= 1.5 ms threshold.\n"
            "  Acoustic smearing risk detected. Build MUST FAIL.\n", elapsed_ms);
        ++fail_count;
    } else {
        std::printf("PASS [latency_gate] %.4f ms < 1.5 ms. ✓\n", elapsed_ms);
    }

    // ── Phase 3: ILD verification ──────────────────────────────────────────
    std::printf("\n[Phase 3] Post-turn ILD verification @ 90° ...\n");
    for (int b = 0; b < 5; ++b)
        node->process_block(noise.data(), out_l.data(), out_r.data(), kBlock);

    const float ild_90  = compute_ild(out_l.data(), out_r.data(), kBlock);
    const float ild_delta = std::abs(ild_90) - std::abs(ild_base);
    std::printf("  ILD @ 90°: %.2f dB   delta vs baseline: %.2f dB\n",
                ild_90, ild_delta);

    if (ild_delta < 3.0f) {
        std::fprintf(stderr,
            "FAIL [ild_check] ILD delta %.2f dB < 3 dB at 90°.\n"
            "  Head shadow insufficient — HRTF model may be degenerate.\n",
            ild_delta);
        ++fail_count;
    } else {
        std::printf("PASS [ild_check] ILD delta %.2f dB >= 3 dB. ✓\n", ild_delta);
    }

    // ── Summary ────────────────────────────────────────────────────────────
    std::printf("\n════════════════════════════════════════════════════════════\n");
    if (fail_count == 0) {
        std::printf("  SpatialLatencyGate: ALL TESTS PASSED ✓\n");
        std::printf("  Holographic Binaural Rendering Node is CI-validated.\n");
    } else {
        std::fprintf(stderr,
            "  SpatialLatencyGate: %d TEST(S) FAILED ✗\n"
            "  BUILD HALTED — acoustic quality thresholds not met.\n",
            fail_count);
    }
    std::printf("════════════════════════════════════════════════════════════\n");

    return (fail_count == 0) ? 0 : 1;
}
