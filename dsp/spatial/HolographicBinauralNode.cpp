/**
 * HolographicBinauralNode.cpp — Claudio Spatial Audio Engine
 */
#include "HolographicBinauralNode.hpp"
#include <cstring>

namespace claudio::spatial {

std::unique_ptr<HolographicBinauralNode>
HolographicBinauralNode::create(const BinauralNodeConfig& config) {
    return std::unique_ptr<HolographicBinauralNode>(
        new HolographicBinauralNode(config));
}

HolographicBinauralNode::HolographicBinauralNode(const BinauralNodeConfig& config)
    : tracking_buf_(std::make_unique<QuaternionRingBuffer<128>>()),
      hrtf_engine_ (std::make_unique<HRTFConvolutionEngine>(
                        config.sample_rate, config.oversampling)),
      raytracer_   (std::make_unique<AcousticRaytracer>(
                        config.sample_rate * static_cast<float>(config.oversampling),
                        config.room)) {
    hrtf_engine_->set_tracking_buffer(tracking_buf_.get());
}

void HolographicBinauralNode::push_head_orientation(
    float w, float x, float y, float z,
    float tx, float ty, float tz) noexcept {
    Quaternion6DoF q;
    q.w = w; q.x = x; q.y = y; q.z = z;
    q.tx = tx; q.ty = ty; q.tz = tz;
    q.timestamp_us = 0;  // Production: replace with std::chrono::steady_clock
    tracking_buf_->push(q);
}

void HolographicBinauralNode::process_block(
    const float* in, float* out_l, float* out_r, int block_size) noexcept {
    // Stack-allocated working copy of the dry signal (max 256 samples @ 48 kHz).
    float dry[256];
    const int n = (block_size > 256) ? 256 : block_size;
    std::memcpy(dry, in, static_cast<std::size_t>(n) * sizeof(float));

    // Accumulate output; HRTF engine writes directly, reflections are added.
    std::memset(out_l, 0, static_cast<std::size_t>(n) * sizeof(float));
    std::memset(out_r, 0, static_cast<std::size_t>(n) * sizeof(float));

    // 1. Update source position from the latest quaternion translation component.
    //    (The HRTF engine handles the orientation component internally.)
    {
        Quaternion6DoF q;
        // Peek at latest without consuming — raytracer uses translation only.
        // For simplicity we rely on hrtf_engine_ draining the buffer; here we
        // lazily update the raytracer from its last-known distance.
        if (tracking_buf_->size() > 0) {
            if (tracking_buf_->pop_latest(q)) {
                raytracer_->set_source_position(q.tx, q.ty, q.tz);
                // Re-inject the quaternion so the HRTF engine can read it too.
                tracking_buf_->push(q);
            }
        }
    }

    // 2. Distance attenuation + proximity effect (in-place on dry copy).
    raytracer_->apply_distance_law(dry, n);

    // 3. HRTF binaural convolution (also drains the ring buffer for orientation).
    hrtf_engine_->process_block(dry, out_l, out_r, n);

    // 4. Add early room reflections on top of the direct binaural image.
    raytracer_->mix_early_reflections(dry, out_l, out_r, n);
}

float HolographicBinauralNode::current_azimuth()  const noexcept {
    return hrtf_engine_ ? hrtf_engine_->current_azimuth()   : 0.f;
}
float HolographicBinauralNode::current_elevation() const noexcept {
    return hrtf_engine_ ? hrtf_engine_->current_elevation()  : 0.f;
}
float HolographicBinauralNode::current_distance()  const noexcept {
    return raytracer_   ? raytracer_->distance()              : 1.f;
}
float HolographicBinauralNode::effective_sample_rate() const noexcept {
    return hrtf_engine_ ? hrtf_engine_->effective_sample_rate() : 48000.f;
}
bool HolographicBinauralNode::hrtf_updated_last_block() const noexcept {
    return hrtf_engine_ ? hrtf_engine_->hrtf_updated_last_block() : false;
}

}  // namespace claudio::spatial
