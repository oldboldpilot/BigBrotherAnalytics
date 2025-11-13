/**
 * Quantization Benchmark - Runtime vs Pre-Quantized
 *
 * Compares runtime quantization vs pre-quantized weights for INT8/INT16
 *
 * Build:
 *   SKIP_CLANG_TIDY=1 cmake -G Ninja -B build && ninja -C build benchmark_int8_quantization
 *
 * Run:
 *   ./build/bin/benchmark_int8_quantization
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <array>
#include <vector>
#include <string>
#include <filesystem>

import bigbrother.ml.weight_loader;
import bigbrother.ml.neural_net_int8;
import bigbrother.ml.neural_net_int16;
import bigbrother.ml.neural_net_int8_prequant;
import bigbrother.ml.neural_net_int16_prequant;

using namespace bigbrother::ml;

// Test input (60 features, normalized)
constexpr std::array<float, 60> TEST_INPUT = {
    0.523f, 0.612f, 0.489f, 0.701f, 0.334f, 0.890f, 0.123f, 0.567f, 0.789f, 0.234f,
    0.456f, 0.678f, 0.901f, 0.345f, 0.678f, 0.912f, 0.234f, 0.567f, 0.890f, 0.123f,
    0.612f, 0.423f, 0.734f, 0.512f, 0.289f, 0.901f, 0.456f, 0.723f, 0.891f, 0.234f,
    0.534f, 0.712f, 0.389f, 0.601f, 0.234f, 0.790f, 0.023f, 0.467f, 0.689f, 0.134f,
    0.423f, 0.612f, 0.289f, 0.701f, 0.434f, 0.690f, 0.323f, 0.867f, 0.589f, 0.334f,
    0.556f, 0.678f, 0.901f, 0.245f, 0.578f, 0.812f, 0.134f, 0.467f, 0.790f, 0.023f
};

struct BenchResult {
    std::string name;
    double latency_us;
    double throughput;
    std::array<float, 3> predictions;
    size_t memory_kb;
};

template<typename Engine>
auto benchmark(const Engine& engine, const char* name, int iterations = 10000) -> BenchResult {
    // Warmup
    for (int i = 0; i < 100; ++i) {
        auto result = engine.predict(TEST_INPUT);
        (void)result;
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    std::array<float, 3> final_result{};

    for (int i = 0; i < iterations; ++i) {
        final_result = engine.predict(TEST_INPUT);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double avg_latency_us = static_cast<double>(duration_us) / iterations;
    double throughput = (iterations * 1e6) / duration_us;

    return BenchResult{
        .name = name,
        .latency_us = avg_latency_us,
        .throughput = throughput,
        .predictions = final_result,
        .memory_kb = 0
    };
}

auto main() -> int {
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Quantization Benchmark: Runtime vs Pre-Quantized        ║\n";
    std::cout << "║  BigBrotherAnalytics Neural Network Performance          ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";

    // Check if pre-quantized weight files exist
    const std::string int8_file = "models/weights/price_predictor_int8.bin";
    const std::string int16_file = "models/weights/price_predictor_int16.bin";

    bool prequant_available = std::filesystem::exists(int8_file) &&
                              std::filesystem::exists(int16_file);

    if (!prequant_available) {
        std::cout << "⚠ Pre-quantized weight files not found.\n";
        std::cout << "  Run: python scripts/ml/quantize_weights_offline.py\n\n";
    }

    // Load weights
    std::cout << "Loading FP32 weights for runtime quantization...\n";
    auto weights = PricePredictorConfig::createLoader().load();
    std::cout << "✓ Weights loaded (" << weights.total_params << " parameters)\n\n";

    // Initialize runtime quantization engines
    std::cout << "Initializing runtime quantization engines...\n";
    NeuralNetINT8 engine_int8(weights);
    std::cout << "✓ INT8 runtime engine ready\n";

    NeuralNetINT16 engine_int16(weights);
    std::cout << "✓ INT16 runtime engine ready\n\n";

    // Run runtime quantization benchmarks
    constexpr int ITERATIONS = 10000;
    std::cout << "Running runtime quantization benchmarks (" << ITERATIONS << " iterations)...\n";

    auto result_int8_runtime = benchmark(engine_int8, "INT8 Runtime", ITERATIONS);
    std::cout << "✓ INT8 runtime benchmark complete\n";

    auto result_int16_runtime = benchmark(engine_int16, "INT16 Runtime", ITERATIONS);
    std::cout << "✓ INT16 runtime benchmark complete\n\n";

    // Run pre-quantized benchmarks if available
    BenchResult result_int8_prequant{};
    BenchResult result_int16_prequant{};

    if (prequant_available) {
        std::cout << "Loading pre-quantized weight files...\n";
        NeuralNetINT8PreQuant engine_int8_pq(int8_file);
        std::cout << "✓ INT8 pre-quantized engine ready\n";

        NeuralNetINT16PreQuant engine_int16_pq(int16_file);
        std::cout << "✓ INT16 pre-quantized engine ready\n\n";

        std::cout << "Running pre-quantized benchmarks (" << ITERATIONS << " iterations)...\n";

        result_int8_prequant = benchmark(engine_int8_pq, "INT8 PreQuant", ITERATIONS);
        std::cout << "✓ INT8 pre-quantized benchmark complete\n";

        result_int16_prequant = benchmark(engine_int16_pq, "INT16 PreQuant", ITERATIONS);
        std::cout << "✓ INT16 pre-quantized benchmark complete\n\n";
    }

    // FP32 baseline (estimated from previous benchmarks)
    const double fp32_throughput = 357e6;  // 357M predictions/sec
    const double fp32_latency = 1e6 / fp32_throughput;  // μs

    // Display results
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║  BENCHMARK RESULTS                                        ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";

    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(20) << "Engine"
              << std::setw(15) << "Latency (μs)"
              << std::setw(15) << "Throughput"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "Memory\n";
    std::cout << std::string(77, '-') << "\n";

    // FP32 (baseline)
    std::cout << std::setw(20) << "FP32 (MKL)"
              << std::setw(15) << std::setprecision(4) << fp32_latency
              << std::setw(15) << (std::to_string(static_cast<int>(fp32_throughput / 1e6)) + " M/s")
              << std::setw(12) << "1.00×"
              << std::setw(15) << "228 KB\n";

    // Runtime quantization
    std::cout << std::setprecision(2);
    std::cout << std::setw(20) << "INT16 Runtime"
              << std::setw(15) << result_int16_runtime.latency_us
              << std::setw(15) << (std::to_string(static_cast<int>(result_int16_runtime.throughput / 1e3)) + " K/s")
              << std::setw(12) << (std::to_string(result_int16_runtime.throughput / fp32_throughput).substr(0, 6) + "×")
              << std::setw(15) << "114 KB\n";

    std::cout << std::setw(20) << "INT8 Runtime"
              << std::setw(15) << result_int8_runtime.latency_us
              << std::setw(15) << (std::to_string(static_cast<int>(result_int8_runtime.throughput / 1e3)) + " K/s")
              << std::setw(12) << (std::to_string(result_int8_runtime.throughput / fp32_throughput).substr(0, 6) + "×")
              << std::setw(15) << "57 KB\n";

    // Pre-quantized (if available)
    if (prequant_available) {
        std::cout << std::string(77, '-') << "\n";
        std::cout << std::setw(20) << "INT16 PreQuant"
                  << std::setw(15) << result_int16_prequant.latency_us
                  << std::setw(15) << (std::to_string(static_cast<int>(result_int16_prequant.throughput / 1e3)) + " K/s")
                  << std::setw(12) << (std::to_string(result_int16_prequant.throughput / fp32_throughput).substr(0, 6) + "×")
                  << std::setw(15) << "114 KB\n";

        std::cout << std::setw(20) << "INT8 PreQuant"
                  << std::setw(15) << result_int8_prequant.latency_us
                  << std::setw(15) << (std::to_string(static_cast<int>(result_int8_prequant.throughput / 1e3)) + " K/s")
                  << std::setw(12) << (std::to_string(result_int8_prequant.throughput / fp32_throughput).substr(0, 6) + "×")
                  << std::setw(15) << "57 KB\n";
    }

    std::cout << "\n";

    // Predictions comparison
    std::cout << "Predictions for test input:\n";
    std::cout << std::setprecision(6);
    std::cout << "  INT16 Runtime:  [" << result_int16_runtime.predictions[0] << ", "
              << result_int16_runtime.predictions[1] << ", "
              << result_int16_runtime.predictions[2] << "]\n";
    std::cout << "  INT8 Runtime:   [" << result_int8_runtime.predictions[0] << ", "
              << result_int8_runtime.predictions[1] << ", "
              << result_int8_runtime.predictions[2] << "]\n";

    if (prequant_available) {
        std::cout << "  INT16 PreQuant: [" << result_int16_prequant.predictions[0] << ", "
                  << result_int16_prequant.predictions[1] << ", "
                  << result_int16_prequant.predictions[2] << "]\n";
        std::cout << "  INT8 PreQuant:  [" << result_int8_prequant.predictions[0] << ", "
                  << result_int8_prequant.predictions[1] << ", "
                  << result_int8_prequant.predictions[2] << "]\n";
    }
    std::cout << "\n";

    // Analysis
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║  ANALYSIS                                                 ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";

    std::cout << std::setprecision(2);

    std::cout << "Memory Savings:\n";
    std::cout << "  - INT16: 50% smaller (114 KB vs 228 KB FP32)\n";
    std::cout << "  - INT8:  75% smaller (57 KB vs 228 KB FP32)\n\n";

    if (prequant_available) {
        double int16_improvement = result_int16_prequant.throughput / result_int16_runtime.throughput;
        double int8_improvement = result_int8_prequant.throughput / result_int8_runtime.throughput;

        std::cout << "Pre-Quantization Performance Improvement:\n";
        std::cout << "  - INT16 PreQuant vs Runtime: " << int16_improvement << "× faster\n";
        std::cout << "  - INT8 PreQuant vs Runtime: " << int8_improvement << "× faster\n\n";

        if (int16_improvement > 1.1 || int8_improvement > 1.1) {
            std::cout << "✓ Pre-quantization eliminates runtime quantization overhead!\n\n";
        }
    } else {
        std::cout << "To test pre-quantized performance:\n";
        std::cout << "  1. Run: python scripts/ml/quantize_weights_offline.py\n";
        std::cout << "  2. Re-run this benchmark\n\n";
    }

    std::cout << "Current Limitations:\n";
    std::cout << "  - Dynamic allocations in inference hot path\n";
    std::cout << "  - Per-layer activation quantization overhead\n";
    std::cout << "  - No AVX-512 VNNI support on this CPU\n\n";

    std::cout << "Future Optimizations:\n";
    std::cout << "  1. Fixed-size buffers (eliminate std::vector allocations)\n";
    std::cout << "  2. Fused operations (matmul + dequantize + ReLU)\n";
    std::cout << "  3. CPU with AVX-512 VNNI (INT8 VPDPBUSD instruction)\n";

    std::cout << "\n✅ Benchmark completed successfully!\n";

    return 0;
}
