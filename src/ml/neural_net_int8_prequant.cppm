/**
 * INT8 Pre-Quantized Neural Network Engine
 *
 * Loads pre-quantized INT8 weights from binary files - NO runtime quantization!
 * This eliminates the quantization overhead for maximum performance.
 *
 * Architecture: 60 → 256 → 128 → 64 → 32 → 3
 * Activations: ReLU (applied in FP32 after dequantization)
 *
 * @module bigbrother.ml.neural_net_int8_prequant
 */

module;

#include <array>
#include <vector>
#include <span>
#include <cstdint>
#include <string>

export module bigbrother.ml.neural_net_int8_prequant;

import bigbrother.ml.quantized_weight_loader;
import bigbrother.ml.quantization;
import bigbrother.ml.activations;

export namespace bigbrother::ml {

/**
 * INT8 Pre-Quantized Neural Network for Price Prediction
 *
 * Uses pre-quantized INT8 weights loaded from binary files.
 * No runtime quantization overhead - weights are already in INT8 format.
 *
 * Matrix operations use AVX-512 integer SIMD for high performance.
 */
class NeuralNetINT8PreQuant {
public:
    static constexpr int INPUT_SIZE = 60;
    static constexpr int OUTPUT_SIZE = 3;

    /**
     * Constructor: Load pre-quantized INT8 weights from binary file
     */
    explicit NeuralNetINT8PreQuant(const std::string& weight_file)
        : weights_(loadQuantizedWeightsINT8(weight_file))
    {
    }

    /**
     * Predict using pre-quantized INT8 inference
     *
     * @param input Feature vector (60 features, FP32)
     * @return Predictions (3 values: 1d, 5d, 20d, FP32)
     */
    auto predict(std::span<const float, INPUT_SIZE> input) const
        -> std::array<float, OUTPUT_SIZE>
    {
        const int num_layers = weights_.num_layers;

        // Quantize input only once
        quantization::QuantizationParams input_params =
            quantization::computeQuantizationParams(input);

        std::vector<int8_t> current_int8(INPUT_SIZE);
        quantization::quantize(input, std::span(current_int8), input_params);

        float current_scale = input_params.scale;

        // Process each layer
        for (int layer = 0; layer < num_layers; ++layer) {
            const int input_size = weights_.layer_input_sizes[layer];
            const int output_size = weights_.layer_output_sizes[layer];

            // INT8 matmul + dequantize (using pre-quantized weights)
            std::vector<float> output_fp32(output_size);
            quantization::matmul_int8_dequantize(
                weights_.layer_weights[layer].data(),
                current_int8.data(),
                weights_.layer_biases[layer].data(),
                output_fp32.data(),
                output_size,
                input_size,
                weights_.weight_scales[layer],  // Pre-computed scale
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
     * Get quantization statistics
     */
    auto getQuantizationStats() const -> std::string {
        std::string stats;
        stats += "INT8 Pre-Quantized Neural Network\n";
        stats += "==================================\n";
        stats += "Architecture: " + std::to_string(INPUT_SIZE);

        for (int i = 0; i < weights_.num_layers; ++i) {
            stats += " → " + std::to_string(weights_.layer_output_sizes[i]);
        }
        stats += "\n\n";

        stats += "Pre-Quantized Weight Scales:\n";
        for (int i = 0; i < weights_.num_layers; ++i) {
            stats += "  Layer " + std::to_string(i + 1) + ": " +
                     std::to_string(weights_.weight_scales[i]) + "\n";
        }

        stats += "\nMemory Usage:\n";
        size_t int8_bytes = 0;
        for (const auto& w : weights_.layer_weights) {
            int8_bytes += w.size() * sizeof(int8_t);
        }
        for (const auto& b : weights_.layer_biases) {
            int8_bytes += b.size() * sizeof(float);
        }

        size_t fp32_bytes = int8_bytes * 4;  // Estimate FP32 size

        stats += "  FP32 equivalent: " + std::to_string(fp32_bytes / 1024) + " KB\n";
        stats += "  INT8 pre-quantized: " + std::to_string(int8_bytes / 1024) + " KB\n";
        stats += "  Savings: 75%\n";
        stats += "\nAdvantages:\n";
        stats += "  - No runtime weight quantization overhead\n";
        stats += "  - Faster model initialization\n";
        stats += "  - Optimized for inference performance\n";

        return stats;
    }

private:
    QuantizedWeightsINT8 weights_;
};

} // namespace bigbrother::ml
