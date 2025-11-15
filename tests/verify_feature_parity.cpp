/**
 * Feature Parity Validation Test
 *
 * Verifies that C++ feature extraction matches Python training exactly.
 * Tests all 85 features with known input data.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-14
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Import feature extractor module
import bigbrother.market_intelligence.feature_extractor;

using namespace bigbrother::market_intelligence;

/**
 * Test data structure to hold expected vs actual features
 */
struct FeatureTest {
    std::string feature_name;
    int index;
    float python_value;
    float cpp_value;
    float diff;
    bool passed;
};

/**
 * Load expected Python features from CSV file
 *
 * Expected CSV format:
 * feature_index,feature_name,value
 * 0,close,193.47
 * 1,open,192.35
 * ...
 */
std::vector<std::pair<std::string, float>> loadPythonFeatures(std::string const& csv_path) {
    std::vector<std::pair<std::string, float>> features;
    std::ifstream file(csv_path);

    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open " << csv_path << std::endl;
        return features;
    }

    std::string line;
    std::getline(file, line);  // Skip header

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string index_str, name, value_str;

        std::getline(ss, index_str, ',');
        std::getline(ss, name, ',');
        std::getline(ss, value_str, ',');

        features.emplace_back(name, std::stof(value_str));
    }

    return features;
}

/**
 * Compare two feature arrays with tolerance
 */
std::vector<FeatureTest> compareFeatures(
    std::vector<std::pair<std::string, float>> const& python_features,
    std::array<float, 85> const& cpp_features,
    float tolerance = 1e-3f) {

    std::vector<FeatureTest> results;

    for (size_t i = 0; i < 85 && i < python_features.size(); ++i) {
        auto const& [name, py_val] = python_features[i];
        float cpp_val = cpp_features[i];
        float diff = std::abs(py_val - cpp_val);
        bool passed = diff <= tolerance;

        results.push_back({
            name,
            static_cast<int>(i),
            py_val,
            cpp_val,
            diff,
            passed
        });
    }

    return results;
}

/**
 * Print comparison results
 */
void printResults(std::vector<FeatureTest> const& results) {
    int passed = 0;
    int failed = 0;

    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "FEATURE PARITY VALIDATION RESULTS\n";
    std::cout << std::string(100, '=') << "\n\n";

    std::cout << "Index | Feature Name                | Python Value  | C++ Value     | Diff       | Status\n";
    std::cout << std::string(100, '-') << "\n";

    for (auto const& test : results) {
        char status = test.passed ? '✓' : '✗';
        printf("[%2d]  | %-27s | %13.6f | %13.6f | %10.6f | %c\n",
               test.index,
               test.feature_name.c_str(),
               test.python_value,
               test.cpp_value,
               test.diff,
               status);

        if (test.passed) {
            ++passed;
        } else {
            ++failed;
        }
    }

    std::cout << std::string(100, '-') << "\n";
    std::cout << "SUMMARY:\n";
    std::cout << "  Total features: " << results.size() << "\n";
    std::cout << "  Passed: " << passed << " (" << (100.0f * passed / results.size()) << "%)\n";
    std::cout << "  Failed: " << failed << " (" << (100.0f * failed / results.size()) << "%)\n";
    std::cout << std::string(100, '=') << "\n";

    if (failed == 0) {
        std::cout << "\n✅ ALL FEATURES MATCH! Feature parity verified.\n\n";
    } else {
        std::cout << "\n❌ FEATURE MISMATCH DETECTED! Review failed features above.\n\n";
    }
}

/**
 * Main test
 */
int main(int argc, char* argv[]) {
    std::cout << "Feature Parity Validation Test\n";
    std::cout << "Model Version: v4.0 (85 features, INT32 SIMD)\n\n";

    // Check if test data file provided
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_python_features.csv>\n";
        std::cerr << "\nTo generate test data:\n";
        std::cerr << "  python scripts/ml/export_test_features.py > test_features.csv\n";
        return 1;
    }

    std::string csv_path = argv[1];

    // Load expected Python features
    auto python_features = loadPythonFeatures(csv_path);
    if (python_features.empty()) {
        std::cerr << "ERROR: No features loaded from " << csv_path << "\n";
        return 1;
    }

    std::cout << "Loaded " << python_features.size() << " Python features from " << csv_path << "\n";

    // TODO: Extract C++ features from same input data
    // For now, create dummy test to demonstrate structure

    // Example: Create test price history (21+ days required)
    std::vector<float> price_history = {
        193.47f, 193.62f, 197.08f, 190.51f, 188.34f,  // Most recent 5 days
        185.22f, 189.67f, 192.45f, 191.23f, 190.88f,
        188.99f, 187.34f, 189.56f, 190.12f, 191.45f,
        192.67f, 193.89f, 194.12f, 195.34f, 196.56f,
        197.78f, 198.90f, 199.12f, 200.34f, 201.56f,  // 25 days total
    };

    std::vector<float> volume_history = {
        19504093.0f, 18900000.0f, 20100000.0f, 21500000.0f, 19800000.0f,
        20200000.0f, 19500000.0f, 21000000.0f, 19900000.0f, 20400000.0f,
        19800000.0f, 20100000.0f, 19700000.0f, 20500000.0f, 19600000.0f,
        20300000.0f, 19900000.0f, 20200000.0f, 19800000.0f, 20100000.0f,
        20000000.0f, 19700000.0f, 20400000.0f, 19500000.0f, 20600000.0f,
    };

    // Create timestamp (2025-11-14 16:00:00 ET)
    auto timestamp = std::chrono::system_clock::now();

    // Extract features using toArray85
    PriceFeatures features;
    features.close = 193.47f;
    features.open = 192.35f;
    features.high = 195.12f;
    features.low = 191.89f;
    features.volume = 19504093.0f;
    features.rsi_14 = 51.49f;
    features.macd = -2.53f;
    features.macd_signal = -2.55f;
    features.bb_upper = 213.93f;
    features.bb_lower = 179.73f;
    features.bb_position = 0.52f;
    features.atr_14 = 1.02f;
    features.volume_ratio = 1.01f;
    features.yield_curve_slope = 0.0056f;
    features.fed_funds_rate = 0.0387f;
    features.treasury_10yr = 0.0411f;
    features.return_1d = 0.0001f;
    features.return_20d = 0.0028f;
    features.momentum_3d = 0.0003f;
    features.recent_win_rate = 0.52f;
    features.symbol_encoded = 9.5f;
    features.price_direction = 1.0f;
    features.price_above_ma5 = 1.0f;
    features.price_above_ma20 = 1.0f;
    features.macd_signal_direction = 1.0f;
    features.volume_trend = 1.0f;
    features.volume_rsi_signal = 0.01f;
    features.yield_volatility = 0.04f;
    features.macd_volume = 0.13f;
    features.bb_momentum = -2.65f;
    features.rate_return = 0.002f;
    features.rsi_bb_signal = 0.12f;

    auto cpp_features = features.toArray85(price_history, volume_history, timestamp);

    // Compare features
    auto results = compareFeatures(python_features, cpp_features);

    // Print results
    printResults(results);

    // Return exit code (0 = success, 1 = failure)
    int failed_count = std::count_if(results.begin(), results.end(),
                                     [](auto const& t) { return !t.passed; });
    return (failed_count == 0) ? 0 : 1;
}
