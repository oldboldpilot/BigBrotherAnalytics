/**
 * INT16 Quantized Neural Network Engine
 *
 * Higher precision quantization than INT8
 * Architecture: 60 → 256 → 128 → 64 → 32 → 3
 *
 * @module bigbrother.ml.neural_net_int16
 */

module;

#include <array>
#include <vector>
#include <span>
#include <cstdint>
#include <string>

export module bigbrother.ml.neural_net_int16;

import bigbrother.ml.weight_loader;
import bigbrother.ml.quantization;
import bigbrother.ml.activations;

export namespace bigbrother::ml {

class NeuralNetINT16 {
public:
    static constexpr int INPUT_SIZE = 60;
    static constexpr int OUTPUT_SIZE = 3;

    explicit NeuralNetINT16(const NetworkWeights& fp32_weights)
        : fp32_weights_(fp32_weights)
    {
        quantizeWeights();
    }

    auto predict(std::span<const float, INPUT_SIZE> input) const
        -> std::array<float, OUTPUT_SIZE>
    {
        const int num_layers = fp32_weights_.num_layers;

        // Quantize input
        quantization::QuantizationParams input_params =
            quantization::computeQuantizationParams16(input);

        std::vector<int16_t> current_int16(INPUT_SIZE);
        quantization::quantize16(input, std::span(current_int16), input_params);

        float current_scale = input_params.scale;

        // Process each layer
        for (int layer = 0; layer < num_layers; ++layer) {
            const int input_size = fp32_weights_.layer_weights[layer].size() /
                                   fp32_weights_.layer_biases[layer].size();
            const int output_size = fp32_weights_.layer_biases[layer].size();

            // INT16 matmul + dequantize
            std::vector<float> output_fp32(output_size);
            quantization::matmul_int16_dequantize(
                weights_int16_[layer].data(),
                current_int16.data(),
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

            // Quantize for next layer
            if (layer < num_layers - 1) {
                quantization::QuantizationParams params =
                    quantization::computeQuantizationParams16(std::span(output_fp32));

                current_int16.resize(output_size);
                quantization::quantize16(std::span(output_fp32),
                                        std::span(current_int16),
                                        params);
                current_scale = params.scale;
            } else {
                std::array<float, OUTPUT_SIZE> result;
                for (int i = 0; i < OUTPUT_SIZE; ++i) {
                    result[i] = output_fp32[i];
                }
                return result;
            }
        }

        return std::array<float, OUTPUT_SIZE>{};
    }

    auto getQuantizationStats() const -> std::string {
        std::string stats;
        stats += "INT16 Quantized Neural Network\n";
        stats += "==============================\n";
        stats += "Architecture: " + std::to_string(INPUT_SIZE);
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
        size_t int16_bytes = fp32_bytes / 2;

        stats += "  FP32 weights: " + std::to_string(fp32_bytes / 1024) + " KB\n";
        stats += "  INT16 weights: " + std::to_string(int16_bytes / 1024) + " KB\n";
        stats += "  Savings: 50%\n";

        return stats;
    }

private:
    void quantizeWeights() {
        const auto& weights = fp32_weights_.layer_weights;

        weights_int16_.resize(weights.size());
        weight_scales_.resize(weights.size());

        for (size_t i = 0; i < weights.size(); ++i) {
            std::span<const float> w = weights[i];
            quantization::QuantizationParams params =
                quantization::computeQuantizationParams16(w);

            weights_int16_[i].resize(w.size());
            quantization::quantize16(w, std::span(weights_int16_[i]), params);
            weight_scales_[i] = params.scale;
        }
    }

    NetworkWeights fp32_weights_;
    std::vector<std::vector<int16_t>> weights_int16_;
    std::vector<float> weight_scales_;
};

} // namespace bigbrother::ml
