/**
 * Comprehensive Regression Tests for weight_loader C++23 Module
 *
 * Tests:
 * 1. Fluent API with various configurations
 * 2. Weight file loading and validation
 * 3. Error handling (missing files, corrupted data)
 * 4. Parameter count validation
 * 5. Multiple usage patterns
 *
 * Author: Claude
 * Date: 2025-11-13
 */

#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

import bigbrother.ml.weight_loader;

using namespace bigbrother::ml;

// ============================================================================
// Test Framework
// ============================================================================

struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    double duration_ms;
};

std::vector<TestResult> test_results;
int total_tests = 0;
int passed_tests = 0;

void printTestHeader(std::string const& name) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "TEST: " << name << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void recordTest(std::string const& name, bool passed, std::string const& message, double duration_ms = 0.0) {
    total_tests++;
    if (passed) {
        passed_tests++;
        std::cout << "  [PASS] " << name << " (" << std::fixed << std::setprecision(2) << duration_ms << " ms)\n";
    } else {
        std::cout << "  [FAIL] " << name << "\n";
        std::cout << "         " << message << "\n";
    }
    test_results.push_back({name, passed, message, duration_ms});
}

// ============================================================================
// Test 1: Fluent API with Default Configuration
// ============================================================================

void test_fluent_api_default() {
    printTestHeader("Fluent API with Default PricePredictorConfig");

    try {
        auto weights = PricePredictorConfig::createLoader("models/weights")
            .verbose(true)
            .load();

        bool test_passed = (weights.input_size == 60 &&
                           weights.output_size == 3 &&
                           weights.num_layers == 5 &&
                           weights.total_params == 58947);

        recordTest(
            "Default config loads 60→256→128→64→32→3 network",
            test_passed,
            test_passed ? "Success" : "Parameter mismatch",
            0.0
        );

    } catch (std::exception const& e) {
        recordTest(
            "Default config loads 60→256→128→64→32→3 network",
            false,
            std::string("Exception: ") + e.what(),
            0.0
        );
    }
}

// ============================================================================
// Test 2: Custom Architecture (60 → 128 → 64 → 3)
// ============================================================================

void test_custom_architecture() {
    printTestHeader("Custom Architecture Configuration");

    try {
        // Test custom architecture with different layer sizes
        auto weights = WeightLoader::fromDirectory("models/weights")
            .withArchitecture(60, {128, 64}, 3)
            .verbose(true)
            .load();

        bool test_passed = (weights.input_size == 60 &&
                           weights.output_size == 3 &&
                           weights.num_layers == 3);

        recordTest(
            "Custom 60→128→64→3 architecture",
            test_passed,
            test_passed ? "Success" : "Architecture mismatch",
            0.0
        );

    } catch (std::exception const& e) {
        recordTest(
            "Custom 60→128→64→3 architecture",
            false,
            std::string("Exception: ") + e.what(),
            0.0
        );
    }
}

// ============================================================================
// Test 3: Custom Naming Schemes
// ============================================================================

void test_custom_naming_scheme() {
    printTestHeader("Custom Naming Schemes");

    try {
        // Test with the actual file naming convention
        auto weights = WeightLoader::fromDirectory("models/weights")
            .withArchitecture(60, {256, 128, 64, 32}, 3)
            .withNamingScheme("network_{}_weight.bin", "network_{}_bias.bin")
            .verbose(true)
            .load();

        bool test_passed = (weights.total_params == 58947);

        recordTest(
            "Custom naming scheme (network_{}.bin)",
            test_passed,
            test_passed ? "Success" : "Naming scheme didn't resolve files",
            0.0
        );

    } catch (std::exception const& e) {
        recordTest(
            "Custom naming scheme (network_{}.bin)",
            false,
            std::string("Exception: ") + e.what(),
            0.0
        );
    }
}

// ============================================================================
// Test 4: Verbose Mode On/Off
// ============================================================================

void test_verbose_mode() {
    printTestHeader("Verbose Mode Toggle");

    try {
        // Test with verbose OFF
        auto weights_quiet = WeightLoader::fromDirectory("models/weights")
            .withArchitecture(60, {256, 128, 64, 32}, 3)
            .verbose(false)
            .load();

        // Test with verbose ON
        std::cout << "\n  [With verbose=true]:\n";
        auto weights_verbose = WeightLoader::fromDirectory("models/weights")
            .withArchitecture(60, {256, 128, 64, 32}, 3)
            .verbose(true)
            .load();

        recordTest("Verbose mode toggle", true, "Success", 0.0);

    } catch (std::exception const& e) {
        recordTest("Verbose mode toggle", false, std::string("Exception: ") + e.what(), 0.0);
    }
}

// ============================================================================
// Test 5: Verify All Weight Files Exist and Load
// ============================================================================

void test_weight_file_loading() {
    printTestHeader("Weight File Loading Verification");

    try {
        auto weights = WeightLoader::fromDirectory("models/weights")
            .withArchitecture(60, {256, 128, 64, 32}, 3)
            .verbose(true)
            .load();

        // Verify all layers loaded
        bool test_passed = (weights.layer_weights.size() == 5 &&
                           weights.layer_biases.size() == 5);

        recordTest("All 5 layers loaded with weights and biases", test_passed,
                   test_passed ? "Success" : "Layer count mismatch", 0.0);

        // Verify each layer sizes
        std::vector<std::pair<int, int>> expected_sizes = {
            {15360, 256},  // Layer 0: 60 × 256 weights, 256 biases
            {32768, 128},  // Layer 1: 256 × 128 weights, 128 biases
            {8192, 64},    // Layer 2: 128 × 64 weights, 64 biases
            {2048, 32},    // Layer 3: 64 × 32 weights, 32 biases
            {96, 3}        // Layer 4: 32 × 3 weights, 3 biases
        };

        for (size_t i = 0; i < weights.layer_weights.size(); ++i) {
            std::string layer_name = "Layer " + std::to_string(i + 1);
            bool size_correct = (weights.layer_weights[i].size() == expected_sizes[i].first &&
                                weights.layer_biases[i].size() == expected_sizes[i].second);
            recordTest(layer_name + " sizes match expected", size_correct,
                      size_correct ? "✓" : "Size mismatch", 0.0);
        }

    } catch (std::exception const& e) {
        recordTest("Weight file loading", false, std::string("Exception: ") + e.what(), 0.0);
    }
}

// ============================================================================
// Test 6: Parameter Count Validation
// ============================================================================

void test_parameter_count() {
    printTestHeader("Total Parameter Count Validation");

    try {
        auto weights = WeightLoader::fromDirectory("models/weights")
            .withArchitecture(60, {256, 128, 64, 32}, 3)
            .load();

        // Expected: 15360 + 256 + 32768 + 128 + 8192 + 64 + 2048 + 32 + 96 + 3 = 58947
        bool test_passed = (weights.total_params == 58947);

        recordTest("Total parameters = 58947", test_passed,
                  test_passed ? "✓" : "Got " + std::to_string(weights.total_params), 0.0);

    } catch (std::exception const& e) {
        recordTest("Total parameters validation", false, std::string("Exception: ") + e.what(), 0.0);
    }
}

// ============================================================================
// Test 7: Error Handling - Missing Files
// ============================================================================

void test_missing_file_error() {
    printTestHeader("Error Handling - Missing Weight Files");

    try {
        // Try to load from non-existent directory
        auto weights = WeightLoader::fromDirectory("nonexistent/weights")
            .withArchitecture(60, {256, 128, 64, 32}, 3)
            .load();

        recordTest("Missing file detection", false, "Should have thrown exception", 0.0);

    } catch (std::runtime_error const& e) {
        bool correct_error = (std::string(e.what()).find("not found") != std::string::npos ||
                            std::string(e.what()).find("not exist") != std::string::npos);
        recordTest("Missing file detection", correct_error,
                  correct_error ? "✓ Proper error message" : "Wrong error message", 0.0);

    } catch (std::exception const& e) {
        recordTest("Missing file detection", false, "Wrong exception type", 0.0);
    }
}

// ============================================================================
// Test 8: Error Handling - Size Mismatch (Corrupted Files)
// ============================================================================

void test_corrupted_file_error() {
    printTestHeader("Error Handling - Corrupted Weight Files");

    try {
        // Create a temporary corrupted file
        std::filesystem::path temp_dir = "tests/temp_weights";
        std::filesystem::create_directories(temp_dir);

        // Create a file with wrong size (1 byte instead of 15360*4 = 61440 bytes)
        std::ofstream corrupted(temp_dir / "network_0_weight.bin", std::ios::binary);
        corrupted.put('X');
        corrupted.close();

        // Create valid bias file
        std::ofstream bias_file(temp_dir / "network_0_bias.bin", std::ios::binary);
        std::vector<float> dummy_bias(256, 0.0f);
        bias_file.write(reinterpret_cast<char*>(dummy_bias.data()), 256 * sizeof(float));
        bias_file.close();

        // Try to load
        auto weights = WeightLoader::fromDirectory(temp_dir)
            .withArchitecture(60, {256}, 3)
            .load();

        recordTest("Size mismatch detection", false, "Should have thrown exception", 0.0);

        // Cleanup
        std::filesystem::remove_all(temp_dir);

    } catch (std::runtime_error const& e) {
        bool correct_error = (std::string(e.what()).find("Size mismatch") != std::string::npos ||
                            std::string(e.what()).find("size") != std::string::npos);
        recordTest("Size mismatch detection", correct_error,
                  correct_error ? "✓ Detected file corruption" : "Wrong error", 0.0);
        std::filesystem::remove_all("tests/temp_weights");

    } catch (std::exception const& e) {
        recordTest("Size mismatch detection", false, "Wrong exception type", 0.0);
        std::filesystem::remove_all("tests/temp_weights");
    }
}

// ============================================================================
// Test 9: Verify() Method
// ============================================================================

void test_verify_method() {
    printTestHeader("Verify Method");

    try {
        auto loader = WeightLoader::fromDirectory("models/weights")
            .withArchitecture(60, {256, 128, 64, 32}, 3);

        bool verified = loader.verify();

        recordTest("verify() method returns true for valid weights", verified,
                  verified ? "✓" : "Verification failed", 0.0);

    } catch (std::exception const& e) {
        recordTest("verify() method", false, std::string("Exception: ") + e.what(), 0.0);
    }
}

// ============================================================================
// Test 10: WeightLoader::fromDirectory() Pattern
// ============================================================================

void test_fromdirectory_pattern() {
    printTestHeader("WeightLoader::fromDirectory() Pattern");

    try {
        auto weights = WeightLoader::fromDirectory("models/weights")
            .withArchitecture(60, {256, 128, 64, 32}, 3)
            .withLayerIndices({0, 3, 6, 9, 12})
            .verbose(false)
            .load();

        bool test_passed = (weights.total_params == 58947);

        recordTest("fromDirectory() with custom layer indices", test_passed,
                  test_passed ? "✓" : "Failed to load", 0.0);

    } catch (std::exception const& e) {
        recordTest("fromDirectory() pattern", false, std::string("Exception: ") + e.what(), 0.0);
    }
}

// ============================================================================
// Test 11: PricePredictorConfig Pattern
// ============================================================================

void test_pricepredictor_pattern() {
    printTestHeader("PricePredictorConfig::createLoader() Pattern");

    try {
        auto weights = PricePredictorConfig::createLoader("models/weights").load();

        bool test_passed = (weights.input_size == 60 &&
                           weights.output_size == 3 &&
                           weights.total_params == 58947);

        recordTest("PricePredictorConfig pattern", test_passed,
                  test_passed ? "✓" : "Failed to load", 0.0);

    } catch (std::exception const& e) {
        recordTest("PricePredictorConfig pattern", false, std::string("Exception: ") + e.what(), 0.0);
    }
}

// ============================================================================
// Test 12: Multiple Loads in Sequence
// ============================================================================

void test_sequential_loads() {
    printTestHeader("Sequential Multiple Loads");

    try {
        // Load multiple times to check for state issues
        for (int i = 0; i < 3; ++i) {
            auto weights = PricePredictorConfig::createLoader("models/weights").load();
            if (weights.total_params != 58947) {
                recordTest("Sequential load " + std::to_string(i + 1), false, "Parameter mismatch", 0.0);
                return;
            }
        }

        recordTest("3 sequential loads succeed", true, "✓", 0.0);

    } catch (std::exception const& e) {
        recordTest("Sequential loads", false, std::string("Exception: ") + e.what(), 0.0);
    }
}

// ============================================================================
// Test 13: Weight Values Sanity Check
// ============================================================================

void test_weight_values_sanity() {
    printTestHeader("Weight Values Sanity Check");

    try {
        auto weights = PricePredictorConfig::createLoader("models/weights").load();

        // Check that weights are reasonable (not all zeros, not NaN/Inf)
        for (size_t layer = 0; layer < weights.layer_weights.size(); ++layer) {
            auto const& w = weights.layer_weights[layer];
            auto const& b = weights.layer_biases[layer];

            bool has_nonzero = false;
            bool all_valid = true;

            for (float val : w) {
                if (val != 0.0f) has_nonzero = true;
                if (std::isnan(val) || std::isinf(val)) all_valid = false;
            }

            for (float val : b) {
                if (val != 0.0f) has_nonzero = true;
                if (std::isnan(val) || std::isinf(val)) all_valid = false;
            }

            std::string test_name = "Layer " + std::to_string(layer) + " - valid values";
            recordTest(test_name, all_valid && has_nonzero,
                      (all_valid && has_nonzero) ? "✓" : "Invalid values detected", 0.0);
        }

    } catch (std::exception const& e) {
        recordTest("Weight sanity check", false, std::string("Exception: ") + e.what(), 0.0);
    }
}

// ============================================================================
// Test 14: No Architecture Configured Error
// ============================================================================

void test_no_architecture_error() {
    printTestHeader("Error Handling - No Architecture Configured");

    try {
        auto weights = WeightLoader::fromDirectory("models/weights").load();

        recordTest("Architecture required check", false, "Should have thrown exception", 0.0);

    } catch (std::runtime_error const& e) {
        bool correct_error = (std::string(e.what()).find("Architecture") != std::string::npos ||
                            std::string(e.what()).find("configured") != std::string::npos);
        recordTest("Architecture required check", correct_error,
                  correct_error ? "✓ Proper error message" : "Wrong error", 0.0);

    } catch (std::exception const& e) {
        recordTest("Architecture required check", false, "Wrong exception type", 0.0);
    }
}

// ============================================================================
// Test Summary and Report
// ============================================================================

void printTestSummary() {
    std::cout << "\n\n" << std::string(70, '=') << "\n";
    std::cout << "TEST SUMMARY\n";
    std::cout << std::string(70, '=') << "\n\n";

    std::cout << "Total Tests: " << total_tests << "\n";
    std::cout << "Passed: " << passed_tests << " (" << (total_tests > 0 ? (100 * passed_tests / total_tests) : 0) << "%)\n";
    std::cout << "Failed: " << (total_tests - passed_tests) << "\n\n";

    if (total_tests > 0 && passed_tests == total_tests) {
        std::cout << "✓ ALL TESTS PASSED\n";
    } else {
        std::cout << "✗ SOME TESTS FAILED\n";
        std::cout << "\nFailed Tests:\n";
        for (auto const& result : test_results) {
            if (!result.passed) {
                std::cout << "  - " << result.name << "\n";
                std::cout << "    " << result.message << "\n";
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
    std::cout << "WEIGHT_LOADER C++23 MODULE - COMPREHENSIVE REGRESSION TESTS\n";
    std::cout << "Author: Claude\n";
    std::cout << "Date: 2025-11-13\n";
    std::cout << std::string(70, '*') << "\n";

    // Run all tests
    test_fluent_api_default();
    test_custom_architecture();
    test_custom_naming_scheme();
    test_verbose_mode();
    test_weight_file_loading();
    test_parameter_count();
    test_missing_file_error();
    test_corrupted_file_error();
    test_verify_method();
    test_fromdirectory_pattern();
    test_pricepredictor_pattern();
    test_sequential_loads();
    test_weight_values_sanity();
    test_no_architecture_error();

    // Print summary
    printTestSummary();

    return (passed_tests == total_tests) ? 0 : 1;
}
