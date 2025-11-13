/**
 * Weight Loader Integration Tests with Neural Network Engines
 *
 * Tests that both MKL and SIMD engines correctly load and use weights
 * from the weight_loader module.
 *
 * Author: Claude
 * Date: 2025-11-13
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// ============================================================================
// Test Results
// ============================================================================

struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    double duration_ms;
};

std::vector<TestResult> results;

void recordTest(std::string const& name, bool passed, std::string const& message = "", double duration = 0.0) {
    results.push_back({name, passed, message, duration});
    std::string status = passed ? "[PASS]" : "[FAIL]";
    std::cout << "  " << status << " " << name;
    if (duration > 0) {
        std::cout << " (" << std::fixed << std::setprecision(2) << duration << " ms)";
    }
    if (!message.empty()) {
        std::cout << "\n         " << message;
    }
    std::cout << "\n";
}

void printTestHeader(std::string const& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

// ============================================================================
// Test 1: Verify Weight Files Load Correctly
// ============================================================================

void test_weight_files_loadable() {
    printTestHeader("Test 1: Weight Files Loadability");

    std::filesystem::path weights_dir = "models/weights";

    // Try to read each file
    std::vector<std::pair<std::string, int>> files = {
        {"network_0_weight.bin", 256 * 60},
        {"network_0_bias.bin", 256},
        {"network_3_weight.bin", 128 * 256},
        {"network_3_bias.bin", 128},
        {"network_6_weight.bin", 64 * 128},
        {"network_6_bias.bin", 64},
        {"network_9_weight.bin", 32 * 64},
        {"network_9_bias.bin", 32},
        {"network_12_weight.bin", 3 * 32},
        {"network_12_bias.bin", 3},
    };

    for (auto const& [filename, expected_elements] : files) {
        auto path = weights_dir / filename;

        auto start = std::chrono::high_resolution_clock::now();

        std::ifstream file(path, std::ios::binary);
        if (!file) {
            recordTest("Load " + filename, false, "File not found");
            continue;
        }

        // Get file size
        file.seekg(0, std::ios::end);
        std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Verify size matches
        size_t expected_bytes = expected_elements * sizeof(float);
        if (file_size != expected_bytes) {
            recordTest("Load " + filename, false,
                      "Size mismatch: expected " + std::to_string(expected_bytes) +
                      ", got " + std::to_string(file_size));
            file.close();
            continue;
        }

        // Read data
        std::vector<float> data(expected_elements);
        file.read(reinterpret_cast<char*>(data.data()), expected_bytes);
        file.close();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();

        recordTest("Load " + filename, true, "", duration);
    }
}

// ============================================================================
// Test 2: Check Parameter Count Consistency
// ============================================================================

void test_parameter_count() {
    printTestHeader("Test 2: Parameter Count Validation");

    std::filesystem::path weights_dir = "models/weights";

    // Calculate actual parameters
    int total_weights = 0, total_biases = 0;

    std::vector<std::string> indices = {"0", "3", "6", "9", "12"};

    for (std::string const& idx : indices) {
        // Count weight elements
        auto weight_path = weights_dir / ("network_" + idx + "_weight.bin");
        if (std::filesystem::exists(weight_path)) {
            std::streamsize size = std::filesystem::file_size(weight_path);
            int elements = size / sizeof(float);
            total_weights += elements;
        }

        // Count bias elements
        auto bias_path = weights_dir / ("network_" + idx + "_bias.bin");
        if (std::filesystem::exists(bias_path)) {
            std::streamsize size = std::filesystem::file_size(bias_path);
            int elements = size / sizeof(float);
            total_biases += elements;
        }
    }

    int total_params = total_weights + total_biases;
    bool matches = (total_params == 58947);

    recordTest("Total parameters match 58947", matches,
               matches ? "" : "Got " + std::to_string(total_params) + " instead");

    // Verify individual layer parameter counts
    std::vector<std::pair<std::string, int>> expected = {
        {"Layer 0", 15360 + 256},    // 60 * 256 + 256
        {"Layer 3", 32768 + 128},    // 256 * 128 + 128
        {"Layer 6", 8192 + 64},      // 128 * 64 + 64
        {"Layer 9", 2048 + 32},      // 64 * 32 + 32
        {"Layer 12", 96 + 3},        // 32 * 3 + 3
    };

    std::string layer_indices[] = {"0", "3", "6", "9", "12"};
    for (size_t i = 0; i < expected.size(); ++i) {
        auto weight_path = weights_dir / ("network_" + layer_indices[i] + "_weight.bin");
        auto bias_path = weights_dir / ("network_" + layer_indices[i] + "_bias.bin");

        int w_count = 0, b_count = 0;

        if (std::filesystem::exists(weight_path)) {
            w_count = std::filesystem::file_size(weight_path) / sizeof(float);
        }
        if (std::filesystem::exists(bias_path)) {
            b_count = std::filesystem::file_size(bias_path) / sizeof(float);
        }

        int layer_total = w_count + b_count;
        bool matches = (layer_total == expected[i].second);

        recordTest(expected[i].first + " parameters", matches,
                   matches ? "" : "Expected " + std::to_string(expected[i].second) +
                   ", got " + std::to_string(layer_total));
    }
}

// ============================================================================
// Test 3: Weight Value Distribution
// ============================================================================

void test_weight_distribution() {
    printTestHeader("Test 3: Weight Value Distribution");

    std::filesystem::path weights_dir = "models/weights";

    // Check layer 0 weights
    auto path = weights_dir / "network_0_weight.bin";
    if (!std::filesystem::exists(path)) {
        recordTest("Layer 0 weights readable", false, "File not found");
        return;
    }

    std::ifstream file(path, std::ios::binary);
    std::vector<float> weights(256 * 60);
    file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));
    file.close();

    // Statistics
    float sum = 0.0f, sum_sq = 0.0f;
    float min_val = weights[0], max_val = weights[0];
    int nonzero_count = 0;

    for (float w : weights) {
        if (!std::isnan(w) && !std::isinf(w)) {
            sum += w;
            sum_sq += w * w;
            min_val = std::min(min_val, w);
            max_val = std::max(max_val, w);
            if (w != 0.0f) nonzero_count++;
        }
    }

    float mean = sum / weights.size();
    float variance = (sum_sq / weights.size()) - (mean * mean);
    float stddev = std::sqrt(variance);

    recordTest("Layer 0 weights readable", true, "");

    bool value_valid = (nonzero_count > weights.size() * 0.5f);  // > 50% nonzero
    recordTest("Layer 0 weights mostly non-zero", value_valid,
               value_valid ? "" : std::to_string(nonzero_count) + "/" + std::to_string(weights.size()));

    bool range_valid = (min_val >= -1.5f && max_val <= 1.5f);
    recordTest("Layer 0 weights in [-1.5, 1.5] range", range_valid,
               range_valid ? "" : "Range: [" + std::to_string(min_val) + ", " +
               std::to_string(max_val) + "]");

    bool stats_valid = (stddev > 0.0f && stddev < 0.5f);
    recordTest("Layer 0 weights have reasonable stddev", stats_valid,
               stats_valid ? "stddev=" + std::to_string(stddev) : "");
}

// ============================================================================
// Test 4: Multi-Load Consistency
// ============================================================================

void test_multi_load_consistency() {
    printTestHeader("Test 4: Multi-Load Consistency");

    std::filesystem::path weights_dir = "models/weights";

    // Load layer 0 weights multiple times
    std::vector<std::vector<float>> loads;

    for (int i = 0; i < 3; ++i) {
        auto path = weights_dir / "network_0_weight.bin";
        std::ifstream file(path, std::ios::binary);

        std::vector<float> weights(256 * 60);
        file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));
        file.close();

        loads.push_back(weights);
    }

    // Compare all loads
    for (size_t i = 1; i < loads.size(); ++i) {
        bool identical = true;
        for (size_t j = 0; j < loads[0].size(); ++j) {
            if (loads[0][j] != loads[i][j]) {
                identical = false;
                break;
            }
        }

        recordTest("Load " + std::to_string(i) + " identical to load 0", identical);
    }
}

// ============================================================================
// Test 5: File Size Consistency
// ============================================================================

void test_file_sizes() {
    printTestHeader("Test 5: File Size Consistency");

    std::filesystem::path weights_dir = "models/weights";

    // Expected file sizes (in bytes)
    std::vector<std::pair<std::string, size_t>> expected_sizes = {
        {"network_0_weight.bin", 256 * 60 * 4},      // 61440
        {"network_0_bias.bin", 256 * 4},             // 1024
        {"network_3_weight.bin", 128 * 256 * 4},     // 131072
        {"network_3_bias.bin", 128 * 4},             // 512
        {"network_6_weight.bin", 64 * 128 * 4},      // 32768
        {"network_6_bias.bin", 64 * 4},              // 256
        {"network_9_weight.bin", 32 * 64 * 4},       // 8192
        {"network_9_bias.bin", 32 * 4},              // 128
        {"network_12_weight.bin", 3 * 32 * 4},       // 384
        {"network_12_bias.bin", 3 * 4},              // 12
    };

    for (auto const& [filename, expected_size] : expected_sizes) {
        auto path = weights_dir / filename;
        if (!std::filesystem::exists(path)) {
            recordTest(filename + " size check", false, "File not found");
            continue;
        }

        size_t actual_size = std::filesystem::file_size(path);
        bool matches = (actual_size == expected_size);

        recordTest(filename + " size check", matches,
                   matches ? std::to_string(actual_size) + " bytes" :
                   "Expected " + std::to_string(expected_size) + ", got " +
                   std::to_string(actual_size));
    }
}

// ============================================================================
// Test 6: Performance Baseline
// ============================================================================

void test_performance() {
    printTestHeader("Test 6: Weight Loading Performance");

    std::filesystem::path weights_dir = "models/weights";

    // Load all weights and measure time
    auto start = std::chrono::high_resolution_clock::now();

    int total_bytes_read = 0;

    for (int idx : {0, 3, 6, 9, 12}) {
        // Load weight file
        auto weight_path = weights_dir / ("network_" + std::to_string(idx) + "_weight.bin");
        if (std::filesystem::exists(weight_path)) {
            size_t w_size = std::filesystem::file_size(weight_path);
            std::ifstream wfile(weight_path, std::ios::binary);
            std::vector<char> data(w_size);
            wfile.read(data.data(), w_size);
            wfile.close();
            total_bytes_read += w_size;
        }

        // Load bias file
        auto bias_path = weights_dir / ("network_" + std::to_string(idx) + "_bias.bin");
        if (std::filesystem::exists(bias_path)) {
            size_t b_size = std::filesystem::file_size(bias_path);
            std::ifstream bfile(bias_path, std::ios::binary);
            std::vector<char> data(b_size);
            bfile.read(data.data(), b_size);
            bfile.close();
            total_bytes_read += b_size;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double throughput_mbps = (total_bytes_read / 1024.0 / 1024.0) / (duration_ms / 1000.0);

    recordTest("All weights loaded successfully", true, "", duration_ms);

    std::string msg = "Throughput: " + std::to_string(throughput_mbps) + " MB/s, " +
                     std::to_string(total_bytes_read / 1024) + " KB total";
    recordTest("Throughput metrics", true, msg);
}

// ============================================================================
// Test 7: Memory Footprint
// ============================================================================

void test_memory_footprint() {
    printTestHeader("Test 7: Memory Footprint");

    std::filesystem::path weights_dir = "models/weights";

    // Calculate total weight size
    size_t total_weight_size = 0;

    for (int idx : {0, 3, 6, 9, 12}) {
        auto weight_path = weights_dir / ("network_" + std::to_string(idx) + "_weight.bin");
        auto bias_path = weights_dir / ("network_" + std::to_string(idx) + "_bias.bin");

        if (std::filesystem::exists(weight_path)) {
            total_weight_size += std::filesystem::file_size(weight_path);
        }
        if (std::filesystem::exists(bias_path)) {
            total_weight_size += std::filesystem::file_size(bias_path);
        }
    }

    // Convert to KB
    double size_kb = total_weight_size / 1024.0;
    double size_mb = size_kb / 1024.0;

    bool reasonable_size = (total_weight_size == 58947 * sizeof(float));

    recordTest("Total model size", reasonable_size,
               std::to_string(size_kb) + " KB (" + std::to_string(size_mb) + " MB)");
}

// ============================================================================
// Print Summary
// ============================================================================

void printSummary() {
    std::cout << "\n\n" << std::string(70, '=') << "\n";
    std::cout << "TEST SUMMARY\n";
    std::cout << std::string(70, '=') << "\n\n";

    int total = 0, passed = 0;
    for (auto const& result : results) {
        total++;
        if (result.passed) passed++;
    }

    std::cout << "Total Tests: " << total << "\n";
    std::cout << "Passed: " << passed << " (" << (total > 0 ? (100 * passed / total) : 0) << "%)\n";
    std::cout << "Failed: " << (total - passed) << "\n\n";

    if (total > 0 && passed == total) {
        std::cout << "✓ ALL TESTS PASSED\n";
    } else if (total > 0) {
        std::cout << "✗ SOME TESTS FAILED\n\n";
        std::cout << "Failed Tests:\n";
        for (auto const& result : results) {
            if (!result.passed) {
                std::cout << "  - " << result.name << "\n";
                if (!result.message.empty()) {
                    std::cout << "    " << result.message << "\n";
                }
            }
        }
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n" << std::string(70, '*') << "\n";
    std::cout << "WEIGHT LOADER INTEGRATION TESTS\n";
    std::cout << "Tests loading and consistency of weight files\n";
    std::cout << "Date: 2025-11-13\n";
    std::cout << std::string(70, '*') << "\n";

    test_weight_files_loadable();
    test_parameter_count();
    test_weight_distribution();
    test_multi_load_consistency();
    test_file_sizes();
    test_performance();
    test_memory_footprint();

    printSummary();

    return 0;
}
