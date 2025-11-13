/**
 * Pre-Quantized Weight Loader
 *
 * Loads INT8/INT16 quantized weights from binary files created by
 * scripts/ml/quantize_weights_offline.py
 *
 * This eliminates runtime quantization overhead for faster inference.
 *
 * File format:
 *   [uint32] magic number (0x51494E54 = 'QINT')
 *   [uint32] version (1)
 *   [uint32] precision (8 or 16)
 *   [uint32] num_layers
 *
 *   For each layer:
 *     [uint32] weight_rows
 *     [uint32] weight_cols
 *     [float32] weight_scale
 *     [int8/int16 × rows × cols] quantized weights
 *     [uint32] bias_size
 *     [float32 × bias_size] biases (kept as FP32)
 *
 * @module bigbrother.ml.quantized_weight_loader
 */

module;

#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <bit>

export module bigbrother.ml.quantized_weight_loader;

export namespace bigbrother::ml {

/**
 * Pre-quantized INT8 weights for a neural network
 */
struct QuantizedWeightsINT8 {
    int num_layers{0};
    std::vector<std::vector<int8_t>> layer_weights;
    std::vector<std::vector<float>> layer_biases;
    std::vector<float> weight_scales;

    // Layer dimensions
    std::vector<int> layer_input_sizes;
    std::vector<int> layer_output_sizes;
};

/**
 * Pre-quantized INT16 weights for a neural network
 */
struct QuantizedWeightsINT16 {
    int num_layers{0};
    std::vector<std::vector<int16_t>> layer_weights;
    std::vector<std::vector<float>> layer_biases;
    std::vector<float> weight_scales;

    // Layer dimensions
    std::vector<int> layer_input_sizes;
    std::vector<int> layer_output_sizes;
};

/**
 * Load pre-quantized INT8 weights from binary file
 */
inline auto loadQuantizedWeightsINT8(const std::string& filepath)
    -> QuantizedWeightsINT8
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open quantized weight file: " + filepath);
    }

    QuantizedWeightsINT8 weights;

    // Read header
    uint32_t magic, version, precision, num_layers;
    file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&precision), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(uint32_t));

    if (magic != 0x51494E54) {
        throw std::runtime_error("Invalid magic number in quantized weight file");
    }
    if (version != 1) {
        throw std::runtime_error("Unsupported quantized weight file version");
    }
    if (precision != 8) {
        throw std::runtime_error("Expected INT8 precision, got " + std::to_string(precision));
    }

    weights.num_layers = static_cast<int>(num_layers);
    weights.layer_weights.resize(num_layers);
    weights.layer_biases.resize(num_layers);
    weights.weight_scales.resize(num_layers);
    weights.layer_input_sizes.resize(num_layers);
    weights.layer_output_sizes.resize(num_layers);

    // Read each layer
    for (uint32_t i = 0; i < num_layers; ++i) {
        uint32_t rows, cols;
        float scale;

        // Read layer metadata
        file.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&scale), sizeof(float));

        weights.layer_output_sizes[i] = static_cast<int>(rows);
        weights.layer_input_sizes[i] = static_cast<int>(cols);
        weights.weight_scales[i] = scale;

        // Read quantized weights
        size_t weight_count = rows * cols;
        weights.layer_weights[i].resize(weight_count);
        file.read(reinterpret_cast<char*>(weights.layer_weights[i].data()),
                  weight_count * sizeof(int8_t));

        // Read biases
        uint32_t bias_size;
        file.read(reinterpret_cast<char*>(&bias_size), sizeof(uint32_t));
        weights.layer_biases[i].resize(bias_size);
        file.read(reinterpret_cast<char*>(weights.layer_biases[i].data()),
                  bias_size * sizeof(float));
    }

    return weights;
}

/**
 * Load pre-quantized INT16 weights from binary file
 */
inline auto loadQuantizedWeightsINT16(const std::string& filepath)
    -> QuantizedWeightsINT16
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open quantized weight file: " + filepath);
    }

    QuantizedWeightsINT16 weights;

    // Read header
    uint32_t magic, version, precision, num_layers;
    file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&precision), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(uint32_t));

    if (magic != 0x51494E54) {
        throw std::runtime_error("Invalid magic number in quantized weight file");
    }
    if (version != 1) {
        throw std::runtime_error("Unsupported quantized weight file version");
    }
    if (precision != 16) {
        throw std::runtime_error("Expected INT16 precision, got " + std::to_string(precision));
    }

    weights.num_layers = static_cast<int>(num_layers);
    weights.layer_weights.resize(num_layers);
    weights.layer_biases.resize(num_layers);
    weights.weight_scales.resize(num_layers);
    weights.layer_input_sizes.resize(num_layers);
    weights.layer_output_sizes.resize(num_layers);

    // Read each layer
    for (uint32_t i = 0; i < num_layers; ++i) {
        uint32_t rows, cols;
        float scale;

        // Read layer metadata
        file.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&scale), sizeof(float));

        weights.layer_output_sizes[i] = static_cast<int>(rows);
        weights.layer_input_sizes[i] = static_cast<int>(cols);
        weights.weight_scales[i] = scale;

        // Read quantized weights
        size_t weight_count = rows * cols;
        weights.layer_weights[i].resize(weight_count);
        file.read(reinterpret_cast<char*>(weights.layer_weights[i].data()),
                  weight_count * sizeof(int16_t));

        // Read biases
        uint32_t bias_size;
        file.read(reinterpret_cast<char*>(&bias_size), sizeof(uint32_t));
        weights.layer_biases[i].resize(bias_size);
        file.read(reinterpret_cast<char*>(weights.layer_biases[i].data()),
                  bias_size * sizeof(float));
    }

    return weights;
}

} // namespace bigbrother::ml
