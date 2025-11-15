/**
 * Test to isolate INT32 SIMD constant output bug
 * Compile: clang++ -std=c++23 -mavx2 -o test_int32 test_int32_simd_bug.cpp -I../src -fprebuilt-module-path=../build/CMakeFiles/ml.dir
 */

#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include <cstdint>
#include <cmath>

// Test quantization function (copied from neural_net_int32_simd.cppm)
auto quantizeToInt32(std::vector<float> const& weights) -> std::pair<std::vector<int32_t>, float> {
    // Find maximum absolute value
    float max_abs = 0.0f;
    for (float w : weights) {
        max_abs = std::max(max_abs, std::abs(w));
    }

    // Compute scale: map to [-2^30, +2^30]
    constexpr int32_t MAX_INT32_QUANT = (1 << 30) - 1;  // 2^30 - 1
    float scale = max_abs / static_cast<float>(MAX_INT32_QUANT);

    if (scale == 0.0f) {
        scale = 1.0f;  // Avoid division by zero
    }

    float inv_scale = 1.0f / scale;

    // Quantize weights
    std::vector<int32_t> quantized(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        quantized[i] = static_cast<int32_t>(std::round(weights[i] * inv_scale));
    }

    return {quantized, scale};
}

// Simple scalar matmul for testing
void matmul_scalar(
    const int32_t* weights,
    const int32_t* input,
    const float* bias,
    float* output,
    int rows,
    int cols,
    float weight_scale,
    float input_scale
) {
    float combined_scale = weight_scale * input_scale;

    for (int i = 0; i < rows; ++i) {
        int64_t sum = 0;
        for (int j = 0; j < cols; ++j) {
            sum += static_cast<int64_t>(weights[i * cols + j]) * static_cast<int64_t>(input[j]);
        }
        output[i] = static_cast<float>(sum) * combined_scale + bias[i];
    }
}

int main() {
    std::cout << "=" << std::string(79, '=') << std::endl;
    std::cout << "INT32 SIMD BUG ISOLATION TEST" << std::endl;
    std::cout << "=" << std::string(79, '=') << std::endl << std::endl;

    // Test 1: Load first layer weights
    std::cout << "Test 1: Loading layer1_weight.bin..." << std::endl;
    std::ifstream weight_file("models/weights/layer1_weight.bin", std::ios::binary);
    if (!weight_file) {
        std::cerr << "ERROR: Cannot open models/weights/layer1_weight.bin" << std::endl;
        return 1;
    }

    // Layer 1: 85 inputs → 256 outputs = 21,760 weights
    constexpr int LAYER1_ROWS = 256;
    constexpr int LAYER1_COLS = 85;
    constexpr int LAYER1_SIZE = LAYER1_ROWS * LAYER1_COLS;

    std::vector<float> layer1_weights_fp32(LAYER1_SIZE);
    weight_file.read(reinterpret_cast<char*>(layer1_weights_fp32.data()), LAYER1_SIZE * sizeof(float));
    weight_file.close();

    std::cout << "  Loaded " << LAYER1_SIZE << " weights" << std::endl;
    std::cout << "  Sample weights[0:5]: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << layer1_weights_fp32[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "  Sample weights[47*256:47*256+5] (symbol_enc column): ";
    for (int i = 0; i < 5; ++i) {
        std::cout << layer1_weights_fp32[47 * LAYER1_ROWS + i] << " ";
    }
    std::cout << std::endl << std::endl;

    // Test 2: Quantize weights
    std::cout << "Test 2: Quantizing weights to INT32..." << std::endl;
    auto [weights_int32, weight_scale] = quantizeToInt32(layer1_weights_fp32);
    std::cout << "  Weight scale: " << weight_scale << std::endl;
    std::cout << "  Quantized weights[0:5]: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << weights_int32[i] << " ";
    }
    std::cout << std::endl << std::endl;

    // Test 3: Create test inputs for SPY, QQQ, IWM
    std::cout << "Test 3: Creating test inputs (SPY, QQQ, IWM)..." << std::endl;

    struct TestCase {
        std::string symbol;
        float symbol_enc;  // Feature at index 47
    };

    std::vector<TestCase> test_cases = {
        {"SPY", 0.0f},
        {"QQQ", 1.0f},
        {"IWM", 2.0f}
    };

    // Load biases
    std::ifstream bias_file("models/weights/layer1_bias.bin", std::ios::binary);
    std::vector<float> layer1_bias(LAYER1_ROWS);
    bias_file.read(reinterpret_cast<char*>(layer1_bias.data()), LAYER1_ROWS * sizeof(float));
    bias_file.close();

    for (auto const& test : test_cases) {
        std::cout << std::endl << "Testing " << test.symbol << " (symbol_enc=" << test.symbol_enc << "):" << std::endl;

        // Create input (zeros except symbol_enc at position 47)
        std::vector<float> input_fp32(LAYER1_COLS, 0.0f);
        input_fp32[47] = test.symbol_enc;

        std::cout << "  Input[47] (symbol_enc): " << input_fp32[47] << std::endl;

        // Quantize input
        auto [input_int32, input_scale] = quantizeToInt32(input_fp32);
        std::cout << "  Input scale: " << input_scale << std::endl;
        std::cout << "  Quantized input[47]: " << input_int32[47] << std::endl;

        // Run matrix multiplication
        std::vector<float> output(LAYER1_ROWS);
        matmul_scalar(
            weights_int32.data(),
            input_int32.data(),
            layer1_bias.data(),
            output.data(),
            LAYER1_ROWS,
            LAYER1_COLS,
            weight_scale,
            input_scale
        );

        // Show first 5 outputs
        std::cout << "  Output[0:5]: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;

        // Apply ReLU
        for (float& val : output) {
            val = std::max(0.0f, val);
        }

        std::cout << "  After ReLU[0:5]: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "=" << std::string(79, '=') << std::endl;
    std::cout << "EXPECTED RESULT:" << std::endl;
    std::cout << "=" << std::string(79, '=') << std::endl;
    std::cout << "SPY, QQQ, IWM should produce DIFFERENT outputs because symbol_enc differs." << std::endl;
    std::cout << "If outputs are the SAME, the bug is in how we handle the input." << std::endl;

    return 0;
}
