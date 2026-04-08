#pragma once
/**
 * AcousticRaytracer.hpp — Claudio Spatial Audio Engine
 *
 * Lightweight real-time acoustic physics simulation:
 *   - Inverse-square-law amplitude attenuation (distance rolloff)
 *   - Proximity effect: low-frequency shelf boost < 30 cm (close-mic effect)
 *   - Three early room reflections: floor + left wall + right wall
 *     (image-source method, attenuated by wall absorption coefficient)
 *
 * Called once per audio block by HolographicBinauralNode before HRTF.
 * All filtering is done with biquad IIR sections (no per-sample heap alloc).
 */
#ifndef M_PI
#  define M_PI  3.14159265358979323846
#endif

#include <array>
#include <cmath>

namespace claudio::spatial {

struct RaytracerParams {
    float room_width_m{5.f};    ///< Room width  (X axis, left→right)
    float room_depth_m{7.f};    ///< Room depth  (Z axis, front→back)
    float room_height_m{3.f};   ///< Room height (Y axis, floor→ceiling)
    float wall_absorption{0.3f}; ///< 0 = fully reflective, 1 = fully absorptive
};

/**
 * AcousticRaytracer
 *
 * Designed for per-block parameter updates (≈ 1–5 ms blocks).
 * No heap allocation in the audio path after construction.
 */
class AcousticRaytracer {
  public:
    explicit AcousticRaytracer(float         sample_rate = 48000.f,
                               RaytracerParams params     = {});

    /// Set virtual source position relative to the listener (metres, Cartesian).
    void set_source_position(float x, float y, float z) noexcept;

    /**
     * Apply inverse-square law attenuation + proximity LF boost in-place.
     *
     * @param buf        Mono audio buffer [block_size].  Modified in place.
     * @param block_size Number of samples.
     */
    void apply_distance_law(float* buf, int block_size) noexcept;

    /**
     * Accumulate three early reflections into the stereo output buffers.
     *
     * Each image source is delayed by its extra travel distance, attenuated by
     * the wall's energy absorption coefficient, and panned left/right by its
     * lateral displacement within the room.
     *
     * @param dry_mono   Direct-path mono signal (already distance-scaled).
     * @param out_l      Left  accumulator [block_size].  Added to (not replaced).
     * @param out_r      Right accumulator [block_size].  Added to (not replaced).
     * @param block_size Number of samples.
     */
    void mix_early_reflections(const float* dry_mono,
                               float*       out_l,
                               float*       out_r,
                               int          block_size) noexcept;

    float direct_gain() const noexcept { return direct_gain_; }
    float distance()    const noexcept { return distance_m_; }

  private:
    struct ImageSource {
        float x, y, z;      ///< Listener-relative position (metres)
        float gain;         ///< Combined amplitude scale
        int   delay_samps;  ///< Extra delay samples vs. direct path
    };

    void recompute_image_sources() noexcept;
    void update_proximity_filter()  noexcept;

    float           sample_rate_;
    RaytracerParams params_;
    float           src_x_{0.f}, src_y_{0.f}, src_z_{-1.f};
    float           distance_m_{1.f};
    float           direct_gain_{1.f};

    // Stereo delay lines for early reflections (85 ms @ 48 kHz = 4096 samps).
    static constexpr int kDelayLen = 8192;  // 170 ms @ 48 kHz / 85 ms @ 96 kHz
    std::array<float, kDelayLen> delay_l_{}, delay_r_{};
    int write_ptr_{0};

    std::array<ImageSource, 3> images_{};

    // Proximity biquad filter state (first-order low-shelf boost).
    float prox_x1_{0.f}, prox_y1_{0.f};
    float prox_b0_{1.f}, prox_b1_{0.f};
    float prox_a1_{0.f};
};

}  // namespace claudio::spatial
