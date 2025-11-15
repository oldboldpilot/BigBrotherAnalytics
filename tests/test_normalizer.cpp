/**
 * Unit Test: Composable Normalizer with Dataset-Level Min/Max
 *
 * Tests:
 * 1. Dataset-level min/max learning with fit()
 * 2. Forward transformation (raw → [0,1])
 * 3. Inverse transformation ([0,1] → raw)
 * 4. Batch operations
 * 5. Edge cases (constant features, empty data)
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-14
 */

#include <iostream>
#include <array>
#include <vector>
#include <cmath>
#include <iomanip>

import bigbrother.ml.normalizer;

using namespace bigbrother::ml;

// Helper: Check if two floats are approximately equal
auto approx_equal(float a, float b, float epsilon = 1e-5f) -> bool {
    return std::abs(a - b) < epsilon;
}

// Test 1: Basic fit() and transform()
auto test_basic_fit_transform() -> bool {
    std::cout << "\n[Test 1] Basic fit() and transform()\n";
    std::cout << "=========================================\n";

    // Training dataset: 3 samples of 5 features
    std::array<float, 5> sample1 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    std::array<float, 5> sample2 = {15.0f, 25.0f, 35.0f, 45.0f, 55.0f};
    std::array<float, 5> sample3 = {20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

    std::vector<std::array<float, 5>> training_data = {sample1, sample2, sample3};

    // Fit normalizer
    auto normalizer = Normalizer<5>::fit(training_data);

    std::cout << "Learned parameters:\n";
    auto min = normalizer.get_min();
    auto max = normalizer.get_max();
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "  Feature " << i << ": min=" << min[i] << ", max=" << max[i] << "\n";
    }

    // Transform sample1 (should be all zeros since it's the minimum)
    auto normalized1 = normalizer.transform(sample1);
    std::cout << "\nNormalized sample1 (should be ~0):\n  ";
    for (auto val : normalized1) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    // Transform sample3 (should be all ones since it's the maximum)
    auto normalized3 = normalizer.transform(sample3);
    std::cout << "Normalized sample3 (should be ~1):\n  ";
    for (auto val : normalized3) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    // Verify
    bool pass = true;
    for (size_t i = 0; i < 5; ++i) {
        if (!approx_equal(normalized1[i], 0.0f)) {
            std::cout << "❌ Feature " << i << " of sample1 not ~0\n";
            pass = false;
        }
        if (!approx_equal(normalized3[i], 1.0f)) {
            std::cout << "❌ Feature " << i << " of sample3 not ~1\n";
            pass = false;
        }
    }

    if (pass) {
        std::cout << "✅ Test 1 PASSED\n";
    }
    return pass;
}

// Test 2: Inverse transformation
auto test_inverse() -> bool {
    std::cout << "\n[Test 2] Inverse transformation\n";
    std::cout << "=========================================\n";

    // Training dataset
    std::array<float, 3> sample1 = {100.0f, 200.0f, 300.0f};
    std::array<float, 3> sample2 = {150.0f, 250.0f, 350.0f};
    std::array<float, 3> sample3 = {200.0f, 300.0f, 400.0f};

    std::vector<std::array<float, 3>> training_data = {sample1, sample2, sample3};

    // Fit normalizer
    auto normalizer = Normalizer<3>::fit(training_data);

    // Transform then inverse
    auto normalized = normalizer.transform(sample2);
    auto recovered = normalizer.inverse(normalized);

    std::cout << "Original:   ";
    for (auto val : sample2) std::cout << val << " ";
    std::cout << "\nNormalized: ";
    for (auto val : normalized) std::cout << val << " ";
    std::cout << "\nRecovered:  ";
    for (auto val : recovered) std::cout << val << " ";
    std::cout << "\n";

    // Verify round-trip
    bool pass = true;
    for (size_t i = 0; i < 3; ++i) {
        if (!approx_equal(sample2[i], recovered[i], 1e-3f)) {
            std::cout << "❌ Feature " << i << " round-trip failed\n";
            pass = false;
        }
    }

    if (pass) {
        std::cout << "✅ Test 2 PASSED (round-trip accuracy)\n";
    }
    return pass;
}

// Test 3: Batch operations
auto test_batch() -> bool {
    std::cout << "\n[Test 3] Batch operations\n";
    std::cout << "=========================================\n";

    // Training dataset
    std::array<float, 2> sample1 = {0.0f, 0.0f};
    std::array<float, 2> sample2 = {50.0f, 100.0f};
    std::array<float, 2> sample3 = {100.0f, 200.0f};

    std::vector<std::array<float, 2>> training_data = {sample1, sample2, sample3};

    auto normalizer = Normalizer<2>::fit(training_data);

    // Batch transform
    std::vector<std::array<float, 2>> batch = {sample1, sample2, sample3};
    auto normalized_batch = normalizer.transform_batch(batch);

    std::cout << "Batch transformation:\n";
    for (size_t i = 0; i < normalized_batch.size(); ++i) {
        std::cout << "  Sample " << i << ": ";
        for (auto val : normalized_batch[i]) {
            std::cout << std::fixed << std::setprecision(2) << val << " ";
        }
        std::cout << "\n";
    }

    // Batch inverse
    auto recovered_batch = normalizer.inverse_batch(normalized_batch);

    std::cout << "\nRecovered batch:\n";
    for (size_t i = 0; i < recovered_batch.size(); ++i) {
        std::cout << "  Sample " << i << ": ";
        for (auto val : recovered_batch[i]) {
            std::cout << std::fixed << std::setprecision(2) << val << " ";
        }
        std::cout << "\n";
    }

    // Verify
    bool pass = true;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            if (!approx_equal(batch[i][j], recovered_batch[i][j], 1e-3f)) {
                std::cout << "❌ Batch round-trip failed at sample " << i << ", feature " << j << "\n";
                pass = false;
            }
        }
    }

    if (pass) {
        std::cout << "✅ Test 3 PASSED (batch operations)\n";
    }
    return pass;
}

// Test 4: Edge case - constant feature
auto test_constant_feature() -> bool {
    std::cout << "\n[Test 4] Edge case: constant feature\n";
    std::cout << "=========================================\n";

    // All samples have same value for feature 1
    std::array<float, 3> sample1 = {10.0f, 42.0f, 30.0f};
    std::array<float, 3> sample2 = {20.0f, 42.0f, 40.0f};
    std::array<float, 3> sample3 = {30.0f, 42.0f, 50.0f};

    std::vector<std::array<float, 3>> training_data = {sample1, sample2, sample3};

    auto normalizer = Normalizer<3>::fit(training_data);

    // Transform (constant feature should not cause division by zero)
    auto normalized = normalizer.transform(sample2);

    std::cout << "Normalized sample (feature 1 is constant):\n  ";
    for (auto val : normalized) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    // Check that no NaN or Inf values exist
    bool pass = true;
    for (size_t i = 0; i < 3; ++i) {
        if (std::isnan(normalized[i]) || std::isinf(normalized[i])) {
            std::cout << "❌ Feature " << i << " is NaN or Inf\n";
            pass = false;
        }
    }

    if (pass) {
        std::cout << "✅ Test 4 PASSED (handles constant features)\n";
    }
    return pass;
}

// Test 5: Empty dataset edge case
auto test_empty_dataset() -> bool {
    std::cout << "\n[Test 5] Edge case: empty dataset\n";
    std::cout << "=========================================\n";

    std::vector<std::array<float, 3>> empty_data;

    auto normalizer = Normalizer<3>::fit(empty_data);

    // Should create identity normalizer (min=0, max=1, range=1)
    std::array<float, 3> test_input = {0.5f, 0.7f, 0.9f};
    auto normalized = normalizer.transform(test_input);

    std::cout << "Input:      ";
    for (auto val : test_input) std::cout << val << " ";
    std::cout << "\nNormalized: ";
    for (auto val : normalized) std::cout << val << " ";
    std::cout << "\n";

    // With identity, output should equal input
    bool pass = true;
    for (size_t i = 0; i < 3; ++i) {
        if (!approx_equal(test_input[i], normalized[i])) {
            std::cout << "❌ Identity transform failed for feature " << i << "\n";
            pass = false;
        }
    }

    if (pass) {
        std::cout << "✅ Test 5 PASSED (empty dataset → identity)\n";
    }
    return pass;
}

auto main() -> int {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║    Composable Normalizer Test Suite                        ║\n";
    std::cout << "║    Dataset-Level Min/Max with Automatic Inverse            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";

    int passed = 0;
    int total = 5;

    if (test_basic_fit_transform()) ++passed;
    if (test_inverse()) ++passed;
    if (test_batch()) ++passed;
    if (test_constant_feature()) ++passed;
    if (test_empty_dataset()) ++passed;

    std::cout << "\n";
    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << "Results: " << passed << "/" << total << " tests passed\n";
    std::cout << "════════════════════════════════════════════════════════════\n";

    return (passed == total) ? 0 : 1;
}
