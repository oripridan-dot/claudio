#pragma once
/**
 * HolographicBinauralNode.hpp — Claudio Spatial Audio Engine
 *
 * Governor Mandate: Wave 52 — Holographic Binaural Rendering Node
 * Approved: 2026-03-13
 *
 * Top-level DSP node integrating:
 *   ① 6DoF head-tracking via lock-free QuaternionRingBuffer
 *   ② Acoustic distance + proximity + early reflections (AcousticRaytracer)
 *   ③ HRTF binaural convolution at up to 192 kHz (HRTFConvolutionEngine)
 *
 * Signal path:
 *   Mono DDSP input
 *     → AcousticRaytracer (distance law + proximity + early reflections)
 *     → HRTFConvolutionEngine (binaural HRTF, drains QuaternionRingBuffer)
 *     → Stereo binaural output  [L | R]
 *
 * ╔══════════════════════════════════════════════════════════════════════╗
 * ║  MUTEX POLICY                                                       ║
 * ║  The complete audio path (process_block) is mutex-free.            ║
 * ║  External threads communicate ONLY via the QuaternionRingBuffer.    ║
 * ║  push_head_orientation() is the ONLY cross-thread API.             ║
 * ╚══════════════════════════════════════════════════════════════════════╝
 */
#include <memory>
#include "QuaternionRingBuffer.hpp"
#include "HRTFConvolutionEngine.hpp"
#include "AcousticRaytracer.hpp"

namespace claudio::spatial {

struct BinauralNodeConfig {
    float          sample_rate{48000.f};
    int            oversampling{1};      ///< 1=48kHz, 2=96kHz, 4=192kHz
    RaytracerParams room{};
};

/**
 * HolographicBinauralNode
 *
 * Public factory: HolographicBinauralNode::create(config)
 * Audio thread:   process_block()
 * Any thread:     push_head_orientation()
 */
class HolographicBinauralNode {
  public:
    static std::unique_ptr<HolographicBinauralNode> create(
        const BinauralNodeConfig& config = {});

    /**
     * Push a 6DoF head orientation update.
     * THREAD-SAFE (any thread).  Lock-free.  Never blocks.
     *
     * @param w,x,y,z   Unit quaternion encoding the head rotation.
     * @param tx,ty,tz  Listener or source translation in metres.
     */
    void push_head_orientation(float w, float x, float y, float z,
                               float tx = 0.f, float ty = 0.f,
                               float tz = 0.f) noexcept;

    /**
     * Process one block: mono input → stereo binaural output.
     * AUDIO THREAD ONLY.  Lock-free.
     *
     * @param in         Mono dry input  [block_size samples]
     * @param out_l      Left  binaural output [block_size samples]
     * @param out_r      Right binaural output [block_size samples]
     * @param block_size Frames per block (max 256 at default config)
     */
    void process_block(const float* in,
                       float*       out_l,
                       float*       out_r,
                       int          block_size) noexcept;

    // ── Diagnostics ────────────────────────────────────────────────────────
    float current_azimuth()          const noexcept;
    float current_elevation()        const noexcept;
    float current_distance()         const noexcept;
    float effective_sample_rate()    const noexcept;
    bool  hrtf_updated_last_block()  const noexcept;

    /// Direct access to the tracking buffer (test injection only).
    QuaternionRingBuffer<128>& tracking_buffer() noexcept {
        return *tracking_buf_;
    }

  private:
    explicit HolographicBinauralNode(const BinauralNodeConfig& config);

    std::unique_ptr<QuaternionRingBuffer<128>> tracking_buf_;
    std::unique_ptr<HRTFConvolutionEngine>     hrtf_engine_;
    std::unique_ptr<AcousticRaytracer>         raytracer_;
};

}  // namespace claudio::spatial
