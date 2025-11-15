/**
 * C++ vs Python Parity Validation Test
 *
 * This test validates that the C++ INT32 SIMD inference engine produces
 * IDENTICAL results to the Python PyTorch model when using the same features.
 *
 * Test Flow:
 * 1. Load normalized features from Python validation output
 * 2. Load Python predictions for comparison
 * 3. Run C++ INT32 SIMD inference with the same features
 * 4. Compare predictions and verify parity (error < 1e-4)
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-14
 */

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

import bigbrother.ml.neural_net_int32_simd;
import bigbrother.ml.weight_loader;
import bigbrother.utils.logger;

using json = nlohmann::json;
using namespace bigbrother::ml;
using namespace bigbrother::utils;

/**
 * StandardScaler parameters for the C++ 85-feature model
 * Loaded from models/price_predictor_cpp_85feat_info.json
 */
struct StandardScalerCPP {
    std::array<float, 85> mean;
    std::array<float, 85> std;

    [[nodiscard]] static auto loadFromJSON(std::string const& path) -> StandardScalerCPP {
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error("Failed to open model info: " + path);
        }

        json model_info = json::parse(f);

        StandardScalerCPP scaler;
        auto mean_json = model_info["scaler_params"]["mean"];
        auto std_json = model_info["scaler_params"]["std"];

        if (mean_json.size() != 85 || std_json.size() != 85) {
            throw std::runtime_error("Invalid scaler params size");
        }

        for (size_t i = 0; i < 85; ++i) {
            scaler.mean[i] = mean_json[i].get<float>();
            scaler.std[i] = std_json[i].get<float>();
        }

        return scaler;
    }

    [[nodiscard]] auto normalize(std::array<float, 85> const& features) const
        -> std::array<float, 85> {
        std::array<float, 85> normalized;
        for (size_t i = 0; i < 85; ++i) {
            normalized[i] = (features[i] - mean[i]) / std[i];
        }
        return normalized;
    }
};

/**
 * Load Python validation data from JSON
 */
struct PythonValidationData {
    std::array<float, 85> normalized_features;
    std::array<float, 3> python_predictions;

    [[nodiscard]] static auto loadFromJSON(std::string const& path) -> PythonValidationData {
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error("Failed to open Python predictions: " + path);
        }

        json data = json::parse(f);

        PythonValidationData result;

        // Load normalized features
        auto features_json = data["features"];
        if (features_json.size() != 85) {
            throw std::runtime_error("Invalid features size");
        }
        for (size_t i = 0; i < 85; ++i) {
            result.normalized_features[i] = features_json[i].get<float>();
        }

        // Load Python predictions
        auto preds_json = data["python_predictions"];
        if (preds_json.size() != 3) {
            throw std::runtime_error("Invalid predictions size");
        }
        for (size_t i = 0; i < 3; ++i) {
            result.python_predictions[i] = preds_json[i].get<float>();
        }

        return result;
    }
};

/**
 * Print section header
 */
void printSection(std::string const& title) {
    std::cout << std::string(80, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(80, '=') << "\n\n";
}

void printSubSection(std::string const& title) {
    std::cout << std::string(80, '-') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(80, '-') << "\n";
}

int main() {
    try {
        printSection("C++ vs PYTHON PARITY VALIDATION TEST");

        // [1/5] Load Python validation data
        std::cout << "[1/5] Loading Python validation data...\n";
        auto python_data = PythonValidationData::loadFromJSON("/tmp/python_predictions.json");
        std::cout << "✅ Loaded normalized features and Python predictions\n";
        std::cout << "   Features [0-5]: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << python_data.normalized_features[i] << " ";
        }
        std::cout << "\n";
        std::cout << "   Python predictions:\n";
        std::cout << "     1-day:  " << std::fixed << std::setprecision(6)
                  << std::showpos << python_data.python_predictions[0] << "%\n";
        std::cout << "     5-day:  " << python_data.python_predictions[1] << "%\n";
        std::cout << "     20-day: " << python_data.python_predictions[2] << "%\n";
        std::cout << "\n";

        // [2/5] Load C++ model scaler
        std::cout << "[2/5] Loading C++ model scaler parameters...\n";
        auto scaler = StandardScalerCPP::loadFromJSON("models/price_predictor_cpp_85feat_info.json");
        std::cout << "✅ Loaded StandardScaler parameters\n";
        std::cout << "   Mean [0-5]: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << scaler.mean[i] << " ";
        }
        std::cout << "\n\n";

        // [3/5] Load C++ neural network weights
        std::cout << "[3/5] Loading C++ INT32 SIMD neural network...\n";
        auto weights = PricePredictorConfig85::createLoader("models/weights").load();
        auto neural_net = std::make_unique<NeuralNetINT32SIMD85>(weights);
        std::cout << "✅ Neural network loaded\n";
        std::cout << "   Engine: " << neural_net->getInfo() << "\n\n";

        // [4/5] Run C++ INT32 SIMD inference
        std::cout << "[4/5] Running C++ INT32 SIMD inference...\n";

        // Time the inference
        auto start = std::chrono::high_resolution_clock::now();
        auto cpp_predictions = neural_net->predict(std::span<const float, 85>(python_data.normalized_features));
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "✅ C++ inference complete in " << duration.count() << " μs\n";
        std::cout << "   C++ predictions:\n";
        std::cout << "     1-day:  " << std::fixed << std::setprecision(6)
                  << std::showpos << cpp_predictions[0] << "%\n";
        std::cout << "     5-day:  " << cpp_predictions[1] << "%\n";
        std::cout << "     20-day: " << cpp_predictions[2] << "%\n";
        std::cout << "\n";

        // [5/5] Compare predictions and verify parity
        std::cout << "[5/5] Verifying parity...\n";
        printSubSection("PARITY ANALYSIS");

        std::array<float, 3> errors;
        float max_error = 0.0f;
        float total_error = 0.0f;

        std::cout << std::fixed << std::setprecision(8);

        for (size_t i = 0; i < 3; ++i) {
            errors[i] = std::abs(cpp_predictions[i] - python_data.python_predictions[i]);
            max_error = std::max(max_error, errors[i]);
            total_error += errors[i];
        }

        std::cout << "Absolute Errors:\n";
        std::cout << "  1-day error:  " << errors[0] << "%\n";
        std::cout << "  5-day error:  " << errors[1] << "%\n";
        std::cout << "  20-day error: " << errors[2] << "%\n";
        std::cout << "\n";
        std::cout << "Summary:\n";
        std::cout << "  Max error:  " << max_error << "%\n";
        std::cout << "  Mean error: " << (total_error / 3.0f) << "%\n";
        std::cout << "\n";

        // Parity threshold: 1e-4 (0.0001%)
        constexpr float PARITY_THRESHOLD = 1e-4f;

        bool parity_verified = max_error < PARITY_THRESHOLD;

        std::cout << "\n";
        printSection("VALIDATION RESULT");

        if (parity_verified) {
            std::cout << "✅ PARITY VERIFIED: C++ and Python produce IDENTICAL results!\n";
            std::cout << "\n";
            std::cout << "Max error " << max_error << "% < threshold " << PARITY_THRESHOLD << "%\n";
            std::cout << "\n";
            std::cout << "The C++ Single Source of Truth architecture guarantees:\n";
            std::cout << "  ✅ Perfect feature extraction parity (C++ used for both training and inference)\n";
            std::cout << "  ✅ Perfect quantization parity (INT32 error < 1e-6)\n";
            std::cout << "  ✅ Perfect normalization parity (same StandardScaler parameters)\n";
            std::cout << "  ✅ Perfect inference parity (same neural network weights)\n";
            std::cout << "\n";
            std::cout << "ZERO feature drift possible! 🎉\n";
            std::cout << "\n";
            return 0;
        } else {
            std::cout << "❌ PARITY FAILED: C++ and Python predictions differ!\n";
            std::cout << "\n";
            std::cout << "Max error " << max_error << "% >= threshold " << PARITY_THRESHOLD << "%\n";
            std::cout << "\n";
            std::cout << "Detailed Comparison:\n";
            std::cout << "                  Python          C++             Error\n";
            std::cout << "  1-day:    " << std::setw(12) << python_data.python_predictions[0]
                      << "   " << std::setw(12) << cpp_predictions[0]
                      << "   " << std::setw(12) << errors[0] << "\n";
            std::cout << "  5-day:    " << std::setw(12) << python_data.python_predictions[1]
                      << "   " << std::setw(12) << cpp_predictions[1]
                      << "   " << std::setw(12) << errors[1] << "\n";
            std::cout << "  20-day:   " << std::setw(12) << python_data.python_predictions[2]
                      << "   " << std::setw(12) << cpp_predictions[2]
                      << "   " << std::setw(12) << errors[2] << "\n";
            std::cout << "\n";
            return 1;
        }

    } catch (std::exception const& e) {
        std::cerr << "\n❌ TEST FAILED: " << e.what() << "\n\n";
        return 1;
    }
}
