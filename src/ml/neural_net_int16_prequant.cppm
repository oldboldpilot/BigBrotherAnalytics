/**
 * INT16 Pre-Quantized Neural Network Engine
 *
 * Loads pre-quantized INT16 weights from binary files - NO runtime quantization!
 * Higher precision than INT8 with faster initialization.
 *
 * Architecture: 60 → 256 → 128 → 64 → 32 → 3
 *
 * @module bigbrother.ml.neural_net_int16_prequant
 */

module;

#include <array>
#include <vector>
#include <span>
#include <cstdint>
#include <string>

export module bigbrother.ml.neural_net_int16_prequant;

import bigbrother.ml.quantized_weight_loader;
import bigbrother.ml.quantization;
import bigbrother.ml.activations;

export namespace bigbrother::ml {

/**
 * INT16 Pre-Quantized Neural Network for Price Prediction
 *
 * Uses pre-quantized INT16 weights loaded from binary files.
 * No runtime quantization overhead - weights are already in INT16 format.
 */
class NeuralNetINT16PreQuant {
public:
    static constexpr int INPUT_SIZE = 60;
    static constexpr int OUTPUT_SIZE = 3;

    /**
     * Constructor: Load pre-quantized INT16 weights from binary file
     */
    explicit NeuralNetINT16PreQuant(const std::string& weight_file)
        : weights_(loadQuantizedWeightsINT16(weight_file))
    {
    }

    /**
     * Predict using pre-quantized INT16 inference
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
            quantization::computeQuantizationParams16(input);

        std::vector<int16_t> current_int16(INPUT_SIZE);
        quantization::quantize16(input, std::span(current_int16), input_params);

        float current_scale = input_params.scale;

        // Process each layer
        for (int layer = 0; layer < num_layers; ++layer) {
            const int input_size = weights_.layer_input_sizes[layer];
            const int output_size = weights_.layer_output_sizes[layer];

            // INT16 matmul + dequantize (using pre-quantized weights)
            std::vector<float> output_fp32(output_size);
            quantization::matmul_int16_dequantize(
                weights_.layer_weights[layer].data(),
                current_int16.data(),
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
                    quantization::computeQuantizationParams16(std::span(output_fp32));

                current_int16.resize(output_size);
                quantization::quantize16(std::span(output_fp32),
                                        std::span(current_int16),
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
        stats += "INT16 Pre-Quantized Neural Network\n";
        stats += "===================================\n";
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
        size_t int16_bytes = 0;
        for (const auto& w : weights_.layer_weights) {
            int16_bytes += w.size() * sizeof(int16_t);
        }
        for (const auto& b : weights_.layer_biases) {
            int16_bytes += b.size() * sizeof(float);
        }

        size_t fp32_bytes = int16_bytes * 2;  // Estimate FP32 size

        stats += "  FP32 equivalent: " + std::to_string(fp32_bytes / 1024) + " KB\n";
        stats += "  INT16 pre-quantized: " + std::to_string(int16_bytes / 1024) + " KB\n";
        stats += "  Savings: 50%\n";
        stats += "\nAdvantages:\n";
        stats += "  - No runtime weight quantization overhead\n";
        stats += "  - Faster model initialization\n";
        stats += "  - Higher precision than INT8\n";
        stats += "  - Optimized for inference performance\n";

        return stats;
    }

private:
    QuantizedWeightsINT16 weights_;
};

} // namespace bigbrother::ml
