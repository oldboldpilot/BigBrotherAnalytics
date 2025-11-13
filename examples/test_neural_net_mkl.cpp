/**
 * Test Neural Network MKL Implementation
 *
 * Demonstrates pure C++ neural network inference using Intel MKL BLAS.
 * Loads PyTorch-exported weights and performs price prediction.
 *
 * Usage:
 *   ./build/bin/test_neural_net_mkl [weights_dir]
 *
 * Example:
 *   ./build/bin/test_neural_net_mkl models/weights
 */

#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

import bigbrother.ml.neural_net_mkl;
import bigbrother.utils.logger;

using namespace bigbrother::ml;
using namespace bigbrother::utils;

// Generate synthetic test data (80 features)
auto generateTestInput() -> std::vector<float> {
    std::vector<float> input(80);

    // Simulate realistic market features
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);

    for (auto& val : input) {
        val = dist(gen);
    }

    // Add some structure to make it more realistic
    input[0] = 0.05f;  // Recent price movement
    input[1] = 0.02f;  // Volatility
    input[2] = -0.01f; // Delta
    input[3] = 0.5f;   // Gamma

    return input;
}

auto main(int argc, char* argv[]) -> int {
    auto& logger = Logger::getInstance();
    logger.initialize("logs/neural_net_test.log", LogLevel::INFO, true);

    logger.info("Testing Neural Network with Intel MKL BLAS");
    logger.info("═══════════════════════════════════════════");

    // Parse command line arguments
    std::filesystem::path weights_dir = "models/weights";
    if (argc > 1) {
        weights_dir = argv[1];
    }

    logger.info("Loading weights from: {}", weights_dir.string());

    // Create and load neural network
    auto net = NeuralNet::create();
    net.loadWeights(weights_dir);

    if (!net.isReady()) {
        logger.error("Failed to load neural network weights!");
        return 1;
    }

    logger.info("Neural network loaded successfully!");
    logger.info("");
    logger.info(net.getInfo());
    logger.info("");
    logger.info(net.getPerformanceEstimate());
    logger.info("");

    // Test 1: Single prediction
    logger.info("Test 1: Single Prediction");
    logger.info("─────────────────────────");

    auto test_input = generateTestInput();
    auto start = std::chrono::high_resolution_clock::now();
    auto result = net.predict(test_input);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    logger.info(formatPrediction(result));
    logger.info("Inference time: {} μs", duration_us);
    logger.info("");

    // Test 2: Batch prediction (100 samples)
    logger.info("Test 2: Batch Prediction (100 samples)");
    logger.info("───────────────────────────────────────");

    std::vector<float> batch_input(80 * 100);
    for (int i = 0; i < 100; ++i) {
        auto sample = generateTestInput();
        std::copy(sample.begin(), sample.end(), batch_input.begin() + i * 80);
    }

    start = std::chrono::high_resolution_clock::now();
    auto batch_results = net.predictBatch(batch_input, 100);
    end = std::chrono::high_resolution_clock::now();

    auto batch_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    logger.info("Processed {} predictions", batch_results.size());
    logger.info("Total time: {} ms", batch_duration_ms);
    logger.info("Average time per prediction: {:.2f} μs",
                batch_duration_ms * 1000.0 / batch_results.size());

    // Show first 5 predictions
    logger.info("");
    logger.info("First 5 predictions:");
    for (int i = 0; i < 5 && i < static_cast<int>(batch_results.size()); ++i) {
        auto const& r = batch_results[i];
        logger.info("  Sample {}: 1d={:.2f}%, 5d={:.2f}%, 20d={:.2f}% ({})",
                   i + 1, r.oneDay() * 100.0f, r.fiveDay() * 100.0f,
                   r.twentyDay() * 100.0f, r.getDirection());
    }

    logger.info("");

    // Test 3: Throughput test (1000 predictions)
    logger.info("Test 3: Throughput Test (1000 predictions)");
    logger.info("──────────────────────────────────────────");

    int const num_iterations = 1000;
    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        auto input = generateTestInput();
        auto pred = net.predict(input);
    }

    end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    double throughput = static_cast<double>(num_iterations) / (total_ms / 1000.0);

    logger.info("Total time: {} ms", total_ms);
    logger.info("Throughput: {} predictions/sec", static_cast<int>(throughput));
    logger.info("Average latency: {:.2f} μs", total_ms * 1000.0 / num_iterations);

    logger.info("");
    logger.info("═══════════════════════════════════════════");
    logger.info("All tests completed successfully!");

    return 0;
}
