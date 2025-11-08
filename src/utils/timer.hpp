#pragma once

/**
 * Compatibility header for timer
 * The full C++23 module is available in timer.cppm
 */

#include <chrono>
#include <string>

// Minimal compatibility
namespace bigbrother::utils {
    class Timer {
    public:
        using Clock = std::chrono::high_resolution_clock;
        using TimePoint = Clock::time_point;

        Timer() : start_{Clock::now()} {}

        auto elapsed() const -> double {
            auto end = Clock::now();
            return std::chrono::duration<double, std::micro>(end - start_).count();
        }

        auto elapsedSeconds() const -> double {
            return elapsed() / 1'000'000.0;
        }

        static auto now() -> int64_t {
            auto n = Clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                n.time_since_epoch()).count();
        }

        static auto timepoint() -> TimePoint {
            return Clock::now();
        }

    private:
        Clock::time_point start_;
    };

    // Minimal Profiler for compatibility
    class Profiler {
    public:
        static auto getInstance() -> Profiler& {
            static Profiler instance;
            return instance;
        }
        auto printStats() const -> void {}
        auto record(std::string const&, double) -> void {}
    };
}

// Compatibility macro
#define PROFILE_SCOPE(name) do {} while(0)
