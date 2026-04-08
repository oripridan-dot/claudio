/**
 * AcousticRaytracer.cpp — Claudio Spatial Audio Engine
 */
#ifndef M_PI
#  define M_PI  3.14159265358979323846
#endif

#include "AcousticRaytracer.hpp"

#include <algorithm>
#include <cstring>

namespace claudio::spatial {

AcousticRaytracer::AcousticRaytracer(float sample_rate, RaytracerParams params)
    : sample_rate_(sample_rate), params_(params) {
    delay_l_.fill(0.f);
    delay_r_.fill(0.f);
    set_source_position(0.f, 0.f, -1.f);
}

void AcousticRaytracer::set_source_position(float x, float y, float z) noexcept {
    src_x_ = x;  src_y_ = y;  src_z_ = z;
    distance_m_ = std::sqrt(x * x + y * y + z * z);
    if (distance_m_ < 0.01f) distance_m_ = 0.01f;

    // Inverse-square-law: gain = 1/d (reference distance = 1 m)
    direct_gain_ = 1.f / distance_m_;

    recompute_image_sources();
    update_proximity_filter();
}

void AcousticRaytracer::recompute_image_sources() noexcept {
    constexpr float kSoundSpeed = 343.f;
    const float refl = 1.f - params_.wall_absorption;  // reflected amplitude

    // ── Floor reflection (mirror Y → -Y) ──
    {
        const float ix = src_x_, iy = -src_y_, iz = src_z_;
        const float d  = std::sqrt(ix*ix + iy*iy + iz*iz);
        const int   ds = static_cast<int>(
            std::max(0.f, (d - distance_m_) / kSoundSpeed * sample_rate_));
        images_[0] = { ix, iy, iz, refl / std::max(d, 0.1f),
                       std::min(ds, kDelayLen - 1) };
    }

    // ── Left-wall reflection (mirror X → -(W + X)) ──
    {
        const float ix = -(params_.room_width_m + src_x_);
        const float iy = src_y_, iz = src_z_;
        const float d  = std::sqrt(ix*ix + iy*iy + iz*iz);
        const int   ds = static_cast<int>(
            std::max(0.f, (d - distance_m_) / kSoundSpeed * sample_rate_));
        images_[1] = { ix, iy, iz, refl / std::max(d, 0.1f),
                       std::min(ds, kDelayLen - 1) };
    }

    // ── Right-wall reflection (mirror X → (W - X)) ──
    {
        const float ix = params_.room_width_m - src_x_;
        const float iy = src_y_, iz = src_z_;
        const float d  = std::sqrt(ix*ix + iy*iy + iz*iz);
        const int   ds = static_cast<int>(
            std::max(0.f, (d - distance_m_) / kSoundSpeed * sample_rate_));
        images_[2] = { ix, iy, iz, refl / std::max(d, 0.1f),
                       std::min(ds, kDelayLen - 1) };
    }
}

void AcousticRaytracer::update_proximity_filter() noexcept {
    // Proximity / close-mic effect: +dB LF boost when distance < 30 cm.
    // Modelled as a first-order low-shelf (bilinear-transformed) filter.
    const float boost_db = (distance_m_ < 0.3f)
        ? 6.f * (1.f - distance_m_ / 0.3f)  // up to +6 dB at 0 m
        : 0.f;
    const float A = std::pow(10.f, boost_db / 20.f);

    // Shelf frequency moves with distance: lower shelf → less effect when far.
    const float f_shelf = std::min(300.f / distance_m_, sample_rate_ * 0.4f);
    const float k = std::tan(
        static_cast<float>(M_PI) * f_shelf / sample_rate_);

    // First-order shelf H(z) = (A + k) / (1 + k) * (1 + (-1) * z^-1 * (A-k)/(A+k))
    // Simplified one-pole implementation:
    prox_b0_ = (A + k) / (1.f + k);
    prox_b1_ = (A - k) / (1.f + k);
    prox_a1_ = (k - 1.f) / (1.f + k);
}

void AcousticRaytracer::apply_distance_law(float* buf, int block_size) noexcept {
    const float g = direct_gain_;
    for (int n = 0; n < block_size; ++n) {
        const float x = buf[n] * g;
        const float y = prox_b0_ * x + prox_b1_ * prox_x1_ - prox_a1_ * prox_y1_;
        prox_x1_ = x;
        prox_y1_ = y;
        buf[n] = y;
    }
}

void AcousticRaytracer::mix_early_reflections(const float* dry_mono,
                                               float*       out_l,
                                               float*       out_r,
                                               int          block_size) noexcept {
    const float half_w = params_.room_width_m * 0.5f;

    for (int n = 0; n < block_size; ++n) {
        delay_l_[write_ptr_] = dry_mono[n];
        delay_r_[write_ptr_] = dry_mono[n];

        for (const auto& img : images_) {
            if (img.delay_samps <= 0) continue;
            const int rd  = (write_ptr_ - img.delay_samps + kDelayLen) % kDelayLen;
            // Pan by lateral image position.
            const float pan   = std::max(-1.f, std::min(1.f, img.x / half_w));
            const float pan_l = std::sqrt(0.5f * (1.f - pan));
            const float pan_r = std::sqrt(0.5f * (1.f + pan));
            out_l[n] += img.gain * pan_l * delay_l_[rd];
            out_r[n] += img.gain * pan_r * delay_r_[rd];
        }

        write_ptr_ = (write_ptr_ + 1) % kDelayLen;
    }
}

}  // namespace claudio::spatial
