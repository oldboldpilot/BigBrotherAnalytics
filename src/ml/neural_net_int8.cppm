/**
 * INT8 Quantized Neural Network Engine
 *
 * High-performance neural network inference using INT8 quantization.
 * Expected performance: 2-3× faster than FP32 (without VNNI)
 *
 * Architecture: 60 → 256 → 128 → 64 → 32 → 3
 * Activations: ReLU (applied in FP32 after dequantization)
 *
 * @module bigbrother.ml.neural_net_int8
 */

module;

#include <array>
#include <vector>
#include <span>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <string>

export module bigbrother.ml.neural_net_int8;

import bigbrother.ml.weight_loader;
import bigbrother.ml.quantization;
import bigbrother.ml.activations;

export namespace bigbrother::ml {

/**
 * INT8 Quantized Neural Network for Price Prediction
 *
 * Uses symmetric INT8 quantization for weights and activations.
 * Matrix operations use AVX-512 integer SIMD for 2-3× speedup.
 *
 * Flexible architecture - adapts to loaded weights
 */
class NeuralNetINT8 {
public:
    static constexpr int INPUT_SIZE = 60;
    static constexpr int OUTPUT_SIZE = 3;

    /**
     * Constructor: Takes FP32 weights and quantizes them to INT8
     */
    explicit NeuralNetINT8(const NetworkWeights& fp32_weights)
        : fp32_weights_(fp32_weights)
    {
        quantizeWeights();
    }

    /**
     * Predict using INT8 quantized inference
     * Dynamically handles any number of layers
     *
     * @param input Feature vector (60 features, FP32)
     * @return Predictions (3 values: 1d, 5d, 20d, FP32)
     */
    auto predict(std::span<const float, INPUT_SIZE> input) const
        -> std::array<float, OUTPUT_SIZE>
    {
        const int num_layers = fp32_weights_.num_layers;

        // Quantize input
        quantization::QuantizationParams input_params =
            quantization::computeQuantizationParams(input);

        std::vector<int8_t> current_int8(INPUT_SIZE);
        quantization::quantize(input, std::span(current_int8), input_params);

        float current_scale = input_params.scale;

        // Process each layer
        for (int layer = 0; layer < num_layers; ++layer) {
            const int input_size = fp32_weights_.layer_weights[layer].size() /
                                   fp32_weights_.layer_biases[layer].size();
            const int output_size = fp32_weights_.layer_biases[layer].size();

            // INT8 matmul + dequantize
            std::vector<float> output_fp32(output_size);
            quantization::matmul_int8_dequantize(
                weights_int8_[layer].data(),
                current_int8.data(),
                fp32_weights_.layer_biases[layer].data(),
                output_fp32.data(),
                output_size,
                input_size,
                weight_scales_[layer],
                current_scale
            );

            // Apply ReLU (except on last layer)
            if (layer < num_layers - 1) {
                activations::relu(std::span(output_fp32));
            }

            // Quantize for next layer (except if this is the last layer)
            if (layer < num_layers - 1) {
                quantization::QuantizationParams params =
                    quantization::computeQuantizationParams(std::span(output_fp32));

                current_int8.resize(output_size);
                quantization::quantize(std::span(output_fp32),
                                      std::span(current_int8),
                                      params);
                current_scale = params.scale;
            } else {
                // Last layer - return results
                std::array<float, OUTPUT_SIZE> result;
                for (int i = 0; i < OUTPUT_SIZE; ++i) {
                    result[i] = output_fp32[i];
                }
                return result;
            }
        }

        // Should never reach here
        return std::array<float, OUTPUT_SIZE>{};
    }

    /**
     * Get quantization statistics for debugging
     */
    auto getQuantizationStats() const -> std::string {
        std::string stats;
        stats += "INT8 Quantized Neural Network\n";
        stats += "=============================\n";
        stats += "Architecture: ";

        stats += std::to_string(INPUT_SIZE);
        for (const auto& bias : fp32_weights_.layer_biases) {
            stats += " → " + std::to_string(bias.size());
        }
        stats += "\n\n";

        stats += "Weight Scales:\n";
        for (size_t i = 0; i < weight_scales_.size(); ++i) {
            stats += "  Layer " + std::to_string(i + 1) + ": " +
                     std::to_string(weight_scales_[i]) + "\n";
        }

        stats += "\nMemory Usage:\n";
        size_t fp32_bytes = 0;
        for (const auto& w : fp32_weights_.layer_weights) {
            fp32_bytes += w.size() * sizeof(float);
        }
        size_t int8_bytes = fp32_bytes / 4;

        stats += "  FP32 weights: " + std::to_string(fp32_bytes / 1024) + " KB\n";
        stats += "  INT8 weights: " + std::to_string(int8_bytes / 1024) + " KB\n";
        stats += "  Savings: " + std::to_string(100 - (int8_bytes * 100 / fp32_bytes)) + "%\n";

        return stats;
    }

private:
    /**
     * Quantize FP32 weights to INT8
     */
    void quantizeWeights() {
        const auto& weights = fp32_weights_.layer_weights;

        weights_int8_.resize(weights.size());
        weight_scales_.resize(weights.size());

        for (size_t i = 0; i < weights.size(); ++i) {
            std::span<const float> w = weights[i];
            quantization::QuantizationParams params =
                quantization::computeQuantizationParams(w);

            weights_int8_[i].resize(w.size());
            quantization::quantize(w, std::span(weights_int8_[i]), params);
            weight_scales_[i] = params.scale;
        }
    }

    // FP32 weights (keep for bias and reference)
    NetworkWeights fp32_weights_;

    // INT8 quantized weights
    std::vector<std::vector<int8_t>> weights_int8_;

    // Quantization scales for each layer
    std::vector<float> weight_scales_;
};

} // namespace bigbrother::ml
