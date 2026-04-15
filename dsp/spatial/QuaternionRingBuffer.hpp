#pragma once
/**
 * QuaternionRingBuffer.hpp — Claudio Spatial Audio Engine
 *
 * Lock-free single-producer/single-consumer ring buffer for 6DoF head tracking.
 *
 * ╔══════════════════════════════════════════════════════════════════════╗
 * ║  COMPILER LAW — CompilerDrone Enforcement                           ║
 * ║  This file MUST NOT contain blocking locks (mutexes), condition        ║
 * ║  variables, or any other blocking synchronisation.                  ║
 * ║  Only std::atomic operations are permitted.                         ║
 * ║  Violations will fail the SpatialLatencyGate adversary validator.   ║
 * ╚══════════════════════════════════════════════════════════════════════╝
 */
#include <array>
#include <atomic>
#include <cstdint>
#include <cmath>

namespace claudio::spatial {

/// Six-Degrees-of-Freedom head orientation sample.
struct Quaternion6DoF {
    float    w{1.f}, x{0.f}, y{0.f}, z{0.f};  ///< Unit quaternion (rotation)
    float    tx{0.f}, ty{0.f}, tz{0.f};        ///< Translation in metres (source pos)
    uint64_t timestamp_us{0};                   ///< Timestamp (POSIX microseconds)
};

/**
 * Lock-free SPSC (Single-Producer / Single-Consumer) ring buffer.
 *
 * Thread-safety:
 *   - Exactly one thread may call push() concurrently with one thread
 *     calling pop() or pop_latest().  No other concurrent access is safe.
 *   - Uses acquire/release memory ordering to ensure buffer data visibility
 *     without locks, fences, or any blocking primitives beyond the atomics.
 *
 * Capacity must be a power of 2 to enable bitmask wrap-around (eliminates
 * modulo division from the hot path).
 */
template <std::size_t Capacity = 128>
class QuaternionRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "QuaternionRingBuffer: Capacity must be a power of 2");
    static constexpr std::size_t kMask = Capacity - 1u;

    std::array<Quaternion6DoF, Capacity> storage_{};

    // Cache-line-aligned counters to prevent false sharing between threads.
    alignas(64) std::atomic<uint32_t> write_pos_{0};
    alignas(64) std::atomic<uint32_t> read_pos_{0};

  public:
    QuaternionRingBuffer() = default;

    // Non-copyable / non-movable (atomics are neither).
    QuaternionRingBuffer(const QuaternionRingBuffer&)            = delete;
    QuaternionRingBuffer& operator=(const QuaternionRingBuffer&) = delete;

    /**
     * Push a 6DoF sample (tracking / UI thread).
     * @returns false if buffer is full; caller must decide to drop or overwrite.
     */
    bool push(const Quaternion6DoF& sample) noexcept {
        const uint32_t wp = write_pos_.load(std::memory_order_relaxed);
        const uint32_t rp = read_pos_.load(std::memory_order_acquire);
        if ((wp - rp) >= static_cast<uint32_t>(Capacity))
            return false;  // Full — drop this sample
        storage_[wp & kMask] = sample;
        write_pos_.store(wp + 1u, std::memory_order_release);
        return true;
    }

    /**
     * Pop the oldest pending sample (audio thread).
     * @returns false if buffer is empty (audio thread keeps last orientation).
     */
    bool pop(Quaternion6DoF& out) noexcept {
        const uint32_t rp = read_pos_.load(std::memory_order_relaxed);
        const uint32_t wp = write_pos_.load(std::memory_order_acquire);
        if (rp == wp) return false;
        out = storage_[rp & kMask];
        read_pos_.store(rp + 1u, std::memory_order_release);
        return true;
    }

    /**
     * Drain to the most recent sample only (audio thread best-effort).
     *
     * Discards all intermediate stale orientation frames, returning only the
     * newest pending entry.  This keeps the audio thread current without
     * processing a backlog when the tracking rate briefly exceeds the block rate.
     *
     * @returns false if no new data is available.
     */
    bool pop_latest(Quaternion6DoF& out) noexcept {
        const uint32_t rp = read_pos_.load(std::memory_order_relaxed);
        const uint32_t wp = write_pos_.load(std::memory_order_acquire);
        if (rp == wp) return false;
        // Jump directly to the newest entry (write_pos - 1).
        out = storage_[(wp - 1u) & kMask];
        read_pos_.store(wp, std::memory_order_release);
        return true;
    }

    bool empty() const noexcept {
        return read_pos_.load(std::memory_order_acquire) ==
               write_pos_.load(std::memory_order_acquire);
    }

    std::size_t size() const noexcept {
        const uint32_t wp = write_pos_.load(std::memory_order_acquire);
        const uint32_t rp = read_pos_.load(std::memory_order_acquire);
        return static_cast<std::size_t>(wp - rp);
    }
};

}  // namespace claudio::spatial
