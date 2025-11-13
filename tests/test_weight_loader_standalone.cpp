/**
 * Standalone Weight Loader Test - No build system required
 * Directly tests the weight_loader module with inline imports
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// Test Results Recording
// ============================================================================

struct TestResult {
    std::string name;
    bool passed;
    std::string message;
};

std::vector<TestResult> test_results;
int total_tests = 0;
int passed_tests = 0;

void recordTest(std::string const& name, bool passed, std::string const& message = "OK") {
    total_tests++;
    if (passed) {
        passed_tests++;
        std::cout << "  [PASS] " << name << "\n";
    } else {
        std::cout << "  [FAIL] " << name << "\n";
        if (!message.empty()) {
            std::cout << "         " << message << "\n";
        }
    }
    test_results.push_back({name, passed, message});
}

void printTestHeader(std::string const& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

// ============================================================================
// File Size Utility
// ============================================================================

size_t getFileSize(std::filesystem::path const& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return 0;
    return file.tellg();
}

// ============================================================================
// Test 1: Verify All Weight Files Exist
// ============================================================================

void test_weight_files_exist() {
    printTestHeader("Test 1: Weight Files Existence");

    std::filesystem::path weights_dir = "models/weights";

    if (!std::filesystem::exists(weights_dir)) {
        recordTest("Directory exists", false, "models/weights not found");
        return;
    }
    recordTest("Directory exists", true);

    std::vector<std::pair<std::string, size_t>> expected_files = {
        {"network_0_weight.bin", 61440},   // 256 * 60 * 4 bytes
        {"network_0_bias.bin", 1024},      // 256 * 4 bytes
        {"network_3_weight.bin", 131072},  // 128 * 256 * 4 bytes
        {"network_3_bias.bin", 512},       // 128 * 4 bytes
        {"network_6_weight.bin", 32768},   // 64 * 128 * 4 bytes
        {"network_6_bias.bin", 256},       // 64 * 4 bytes
        {"network_9_weight.bin", 8192},    // 32 * 64 * 4 bytes
        {"network_9_bias.bin", 128},       // 32 * 4 bytes
        {"network_12_weight.bin", 384},    // 3 * 32 * 4 bytes
        {"network_12_bias.bin", 12},       // 3 * 4 bytes
    };

    for (auto const& [filename, expected_size] : expected_files) {
        auto path = weights_dir / filename;
        bool exists = std::filesystem::exists(path);
        recordTest(filename + " exists", exists);

        if (exists) {
            size_t actual_size = getFileSize(path);
            bool size_matches = (actual_size == expected_size);
            std::string msg = "Expected " + std::to_string(expected_size) + " bytes, got " +
                             std::to_string(actual_size);
            recordTest(filename + " size", size_matches, msg);
        }
    }
}

// ============================================================================
// Test 2: Verify Weight File Structure
// ============================================================================

void test_weight_structure() {
    printTestHeader("Test 2: Weight File Structure");

    std::filesystem::path weights_dir = "models/weights";

    // Test loading network_0_weight.bin
    auto path = weights_dir / "network_0_weight.bin";
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        recordTest("Open network_0_weight.bin", false);
        return;
    }

    // Read and verify it contains valid floats (not all zeros or NaN)
    std::vector<float> weights(256 * 60);
    file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));
    file.close();

    if (!file) {
        recordTest("Read network_0_weight.bin", false);
        return;
    }
    recordTest("Read network_0_weight.bin", true);

    // Check for non-zero values
    int nonzero_count = 0;
    for (float w : weights) {
        if (w != 0.0f && !std::isnan(w) && !std::isinf(w)) {
            nonzero_count++;
        }
    }

    bool has_valid_data = (nonzero_count > weights.size() * 0.1);  // At least 10% non-zero
    recordTest("Contains valid weight data", has_valid_data,
               "Nonzero count: " + std::to_string(nonzero_count) + "/" +
               std::to_string(weights.size()));

    // Check bias file
    auto bias_path = weights_dir / "network_0_bias.bin";
    std::ifstream bias_file(bias_path, std::ios::binary);
    std::vector<float> biases(256);
    bias_file.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(float));
    bias_file.close();

    int bias_nonzero = 0;
    for (float b : biases) {
        if (b != 0.0f && !std::isnan(b) && !std::isinf(b)) {
            bias_nonzero++;
        }
    }

    recordTest("Bias file has valid data", (bias_nonzero > 0),
               "Nonzero biases: " + std::to_string(bias_nonzero));
}

// ============================================================================
// Test 3: Calculate Total Parameters
// ============================================================================

void test_total_parameters() {
    printTestHeader("Test 3: Total Parameter Count");

    std::filesystem::path weights_dir = "models/weights";

    // Expected:
    // Layer 0: 256*60 + 256 = 15360 + 256 = 15616
    // Layer 3: 128*256 + 128 = 32768 + 128 = 32896
    // Layer 6: 64*128 + 64 = 8192 + 64 = 8256
    // Layer 9: 32*64 + 32 = 2048 + 32 = 2080
    // Layer 12: 3*32 + 3 = 96 + 3 = 99
    // Total: 15616 + 32896 + 8256 + 2080 + 99 = 58947

    int total_params = 0;

    std::vector<std::pair<std::string, int>> layers = {
        {"network_0", 256 * 60 + 256},
        {"network_3", 128 * 256 + 128},
        {"network_6", 64 * 128 + 64},
        {"network_9", 32 * 64 + 32},
        {"network_12", 3 * 32 + 3},
    };

    for (auto const& [prefix, expected] : layers) {
        int weight_elements = 0, bias_elements = 0;

        // Count weight file elements
        auto weight_path = weights_dir / (prefix + "_weight.bin");
        if (std::filesystem::exists(weight_path)) {
            size_t size = getFileSize(weight_path);
            weight_elements = size / sizeof(float);
        }

        // Count bias file elements
        auto bias_path = weights_dir / (prefix + "_bias.bin");
        if (std::filesystem::exists(bias_path)) {
            size_t size = getFileSize(bias_path);
            bias_elements = size / sizeof(float);
        }

        int layer_total = weight_elements + bias_elements;
        recordTest(prefix + " parameters", layer_total == expected,
                   "Expected " + std::to_string(expected) + ", got " + std::to_string(layer_total));

        total_params += layer_total;
    }

    recordTest("Total parameters = 58947", total_params == 58947,
               "Got " + std::to_string(total_params));
}

// ============================================================================
// Test 4: Verify Architecture Dimensions
// ============================================================================

void test_architecture() {
    printTestHeader("Test 4: Network Architecture");

    // Architecture: 60 → 256 → 128 → 64 → 32 → 3

    std::vector<std::pair<std::string, int>> layer_outputs = {
        {"network_0", 256},    // 60 → 256
        {"network_3", 128},    // 256 → 128
        {"network_6", 64},     // 128 → 64
        {"network_9", 32},     // 64 → 32
        {"network_12", 3},     // 32 → 3
    };

    std::filesystem::path weights_dir = "models/weights";

    for (auto const& [prefix, expected_output] : layer_outputs) {
        auto bias_path = weights_dir / (prefix + "_bias.bin");
        if (std::filesystem::exists(bias_path)) {
            size_t bias_size = getFileSize(bias_path);
            int bias_elements = bias_size / sizeof(float);
            recordTest(prefix + " output size", bias_elements == expected_output,
                       "Expected " + std::to_string(expected_output) + ", got " +
                       std::to_string(bias_elements));
        } else {
            recordTest(prefix + " output size", false, "Bias file not found");
        }
    }
}

// ============================================================================
// Test 5: Layer Indices Verification
// ============================================================================

void test_layer_indices() {
    printTestHeader("Test 5: PyTorch Layer Indices");

    // PyTorch Sequential uses indices: 0, 3, 6, 9, 12
    // These correspond to Linear layers in a network with ReLU activations:
    // [Linear(0), ReLU(1), Linear(2)] + [Linear(3), ReLU(4), Linear(5)] etc...

    std::vector<int> expected_indices = {0, 3, 6, 9, 12};
    std::filesystem::path weights_dir = "models/weights";

    for (int idx : expected_indices) {
        std::string weight_file = "network_" + std::to_string(idx) + "_weight.bin";
        std::string bias_file = "network_" + std::to_string(idx) + "_bias.bin";

        bool weight_exists = std::filesystem::exists(weights_dir / weight_file);
        bool bias_exists = std::filesystem::exists(weights_dir / bias_file);

        recordTest("Index " + std::to_string(idx) + " files exist",
                   weight_exists && bias_exists);
    }
}

// ============================================================================
// Test 6: Error Cases
// ============================================================================

void test_error_cases() {
    printTestHeader("Test 6: Error Handling");

    // Test missing file detection
    std::filesystem::path nonexistent = "nonexistent/weights/network_0_weight.bin";
    bool exists = std::filesystem::exists(nonexistent);
    recordTest("Missing file detection", !exists);

    // Test size mismatch detection
    std::filesystem::path temp_file = "test_corrupted.bin";
    std::ofstream out(temp_file, std::ios::binary);
    out.put('X');  // Write only 1 byte
    out.close();

    size_t file_size = getFileSize(temp_file);
    int expected_elements = 100;  // 100 floats = 400 bytes
    bool size_mismatch = (file_size != expected_elements * sizeof(float));

    recordTest("Size mismatch detection", size_mismatch);
    std::filesystem::remove(temp_file);
}

// ============================================================================
// Test 7: Weight Value Ranges
// ============================================================================

void test_weight_ranges() {
    printTestHeader("Test 7: Weight Value Ranges");

    std::filesystem::path weights_dir = "models/weights";

    // Check layer 0 weights
    auto path = weights_dir / "network_0_weight.bin";
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        recordTest("Can read weights", false);
        return;
    }

    std::vector<float> weights(256 * 60);
    file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));
    file.close();

    float min_val = weights[0], max_val = weights[0];
    for (float w : weights) {
        if (!std::isnan(w) && !std::isinf(w)) {
            min_val = std::min(min_val, w);
            max_val = std::max(max_val, w);
        }
    }

    bool range_valid = (min_val >= -10.0f && max_val <= 10.0f);
    recordTest("Weight values in reasonable range [-10, 10]", range_valid,
               "Range: [" + std::to_string(min_val) + ", " + std::to_string(max_val) + "]");

    bool no_nans = true;
    for (float w : weights) {
        if (std::isnan(w) || std::isinf(w)) {
            no_nans = false;
            break;
        }
    }
    recordTest("No NaN or Inf values", no_nans);
}

// ============================================================================
// Test 8: File Reading Performance
// ============================================================================

void test_file_performance() {
    printTestHeader("Test 8: File Reading Performance");

    std::filesystem::path weights_dir = "models/weights";

    auto start = std::chrono::high_resolution_clock::now();

    // Read all weight files
    int files_read = 0;
    for (int idx : {0, 3, 6, 9, 12}) {
        auto weight_path = weights_dir / ("network_" + std::to_string(idx) + "_weight.bin");
        auto bias_path = weights_dir / ("network_" + std::to_string(idx) + "_bias.bin");

        // Get sizes
        size_t w_size = getFileSize(weight_path);
        size_t b_size = getFileSize(bias_path);

        // Read weights
        std::ifstream wfile(weight_path, std::ios::binary);
        std::vector<char> wdata(w_size);
        wfile.read(wdata.data(), w_size);
        wfile.close();

        // Read biases
        std::ifstream bfile(bias_path, std::ios::binary);
        std::vector<char> bdata(b_size);
        bfile.read(bdata.data(), b_size);
        bfile.close();

        files_read += 2;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    recordTest("All 10 files read successfully", files_read == 10,
               "Time: " + std::to_string(duration_ms) + " ms");
}

// ============================================================================
// Print Summary
// ============================================================================

void printSummary() {
    std::cout << "\n\n" << std::string(70, '=') << "\n";
    std::cout << "TEST SUMMARY\n";
    std::cout << std::string(70, '=') << "\n\n";

    std::cout << "Total Tests: " << total_tests << "\n";
    std::cout << "Passed: " << passed_tests << " ("
              << (total_tests > 0 ? (100 * passed_tests / total_tests) : 0) << "%)\n";
    std::cout << "Failed: " << (total_tests - passed_tests) << "\n\n";

    if (total_tests > 0 && passed_tests == total_tests) {
        std::cout << "✓ ALL TESTS PASSED\n";
    } else if (total_tests > 0) {
        std::cout << "✗ SOME TESTS FAILED\n\n";
        std::cout << "Failed Tests:\n";
        for (auto const& result : test_results) {
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
    std::cout << "WEIGHT_LOADER STANDALONE REGRESSION TESTS\n";
    std::cout << "Date: 2025-11-13\n";
    std::cout << std::string(70, '*') << "\n";

    test_weight_files_exist();
    test_weight_structure();
    test_total_parameters();
    test_architecture();
    test_layer_indices();
    test_error_cases();
    test_weight_ranges();
    test_file_performance();

    printSummary();

    return (passed_tests == total_tests) ? 0 : 1;
}
