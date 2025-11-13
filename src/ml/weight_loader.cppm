/**
 * BigBrotherAnalytics - Unified ML Weight Loader Module
 *
 * Fluent API for loading PyTorch-exported weights across all inference engines.
 * Supports flexible architectures and multiple model configurations.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-13
 */

module;

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

export module bigbrother.ml.weight_loader;

export namespace bigbrother::ml {

/**
 * Layer configuration for neural network
 */
struct LayerConfig {
    int input_size;
    int output_size;
    std::filesystem::path weight_file;
    std::filesystem::path bias_file;
};

/**
 * Complete neural network weights
 */
struct NetworkWeights {
    std::vector<std::vector<float>> layer_weights;  // One per layer
    std::vector<std::vector<float>> layer_biases;   // One per layer

    int input_size;
    int output_size;
    int num_layers;
    int total_params;
};

/**
 * Weight loader with fluent API for flexible neural network configurations
 *
 * Example usage:
 *   auto weights = WeightLoader::fromDirectory("models/weights")
 *       .withArchitecture(60, {256, 128, 64, 32}, 3)
 *       .withNamingScheme("network_{}_weight.bin", "network_{}_bias.bin")
 *       .load();
 */
class WeightLoader {
public:
    /**
     * Start building a weight loader from a base directory
     */
    [[nodiscard]] static auto fromDirectory(std::filesystem::path base_dir) -> WeightLoader {
        WeightLoader loader;
        loader.base_dir_ = std::move(base_dir);
        return loader;
    }

    /**
     * Define the neural network architecture
     *
     * @param input_size Number of input features
     * @param hidden_layers Vector of hidden layer sizes
     * @param output_size Number of output neurons
     * @return *this for fluent chaining
     */
    auto withArchitecture(int input_size, std::vector<int> hidden_layers, int output_size) -> WeightLoader& {
        input_size_ = input_size;
        output_size_ = output_size;
        hidden_layers_ = std::move(hidden_layers);
        return *this;
    }

    /**
     * Set custom naming scheme for weight files
     * Use {} as placeholder for layer index
     *
     * @param weight_pattern Pattern for weight files (e.g., "layer_{}_weight.bin")
     * @param bias_pattern Pattern for bias files (e.g., "layer_{}_bias.bin")
     * @return *this for fluent chaining
     */
    auto withNamingScheme(std::string weight_pattern, std::string bias_pattern) -> WeightLoader& {
        weight_pattern_ = std::move(weight_pattern);
        bias_pattern_ = std::move(bias_pattern);
        return *this;
    }

    /**
     * Set custom layer indices (for PyTorch Sequential models)
     * Default: {0, 3, 6, 9, 12} for standard 5-layer network
     *
     * @param indices Vector of layer indices in file names
     * @return *this for fluent chaining
     */
    auto withLayerIndices(std::vector<int> indices) -> WeightLoader& {
        layer_indices_ = std::move(indices);
        return *this;
    }

    /**
     * Enable verbose logging
     */
    auto verbose(bool enable = true) -> WeightLoader& {
        verbose_ = enable;
        return *this;
    }

    /**
     * Load all weights based on configuration
     *
     * @return NetworkWeights containing all loaded weights and biases
     * @throws std::runtime_error if files don't exist or sizes mismatch
     */
    [[nodiscard]] auto load() const -> NetworkWeights {
        if (input_size_ == 0 || output_size_ == 0 || hidden_layers_.empty()) {
            throw std::runtime_error("Architecture not configured. Call withArchitecture() first.");
        }

        NetworkWeights weights;
        weights.input_size = input_size_;
        weights.output_size = output_size_;
        weights.num_layers = static_cast<int>(hidden_layers_.size()) + 1;  // +1 for output layer

        // Build layer configurations
        std::vector<LayerConfig> configs = buildLayerConfigs();

        // Load each layer
        for (size_t i = 0; i < configs.size(); ++i) {
            auto const& config = configs[i];

            if (verbose_) {
                std::cout << "[WeightLoader] Loading layer " << (i + 1) << "/" << configs.size()
                          << " (" << config.input_size << " → " << config.output_size << ")\n";
            }

            // Load weights and biases
            int weight_size = config.output_size * config.input_size;
            auto layer_weights = loadBinary(config.weight_file, weight_size);
            auto layer_biases = loadBinary(config.bias_file, config.output_size);

            weights.layer_weights.push_back(std::move(layer_weights));
            weights.layer_biases.push_back(std::move(layer_biases));
        }

        // Calculate total parameters
        weights.total_params = 0;
        for (size_t i = 0; i < weights.layer_weights.size(); ++i) {
            weights.total_params += static_cast<int>(weights.layer_weights[i].size());
            weights.total_params += static_cast<int>(weights.layer_biases[i].size());
        }

        if (verbose_) {
            std::cout << "[WeightLoader] ✓ Loaded " << weights.num_layers
                      << " layers with " << weights.total_params << " total parameters\n";
        }

        return weights;
    }

    /**
     * Verify that all weight files exist and have correct sizes
     *
     * @return true if all files are valid, false otherwise
     */
    [[nodiscard]] auto verify() const -> bool {
        try {
            load();
            return true;
        } catch (...) {
            return false;
        }
    }

private:
    WeightLoader() = default;

    /**
     * Build layer configurations based on architecture
     */
    [[nodiscard]] auto buildLayerConfigs() const -> std::vector<LayerConfig> {
        std::vector<LayerConfig> configs;

        // Determine layer indices
        std::vector<int> indices = layer_indices_;
        if (indices.empty()) {
            // Default PyTorch Sequential indices: 0, 3, 6, 9, 12...
            // (assuming ReLU between layers)
            for (size_t i = 0; i <= hidden_layers_.size(); ++i) {
                indices.push_back(static_cast<int>(i * 3));
            }
        }

        // Build all layers
        std::vector<int> all_sizes = {input_size_};
        all_sizes.insert(all_sizes.end(), hidden_layers_.begin(), hidden_layers_.end());
        all_sizes.push_back(output_size_);

        for (size_t i = 0; i < all_sizes.size() - 1; ++i) {
            LayerConfig config;
            config.input_size = all_sizes[i];
            config.output_size = all_sizes[i + 1];

            // Format file names
            int layer_idx = indices[i];
            config.weight_file = base_dir_ / formatString(weight_pattern_, layer_idx);
            config.bias_file = base_dir_ / formatString(bias_pattern_, layer_idx);

            configs.push_back(config);
        }

        return configs;
    }

    /**
     * Format string by replacing {} with index
     */
    [[nodiscard]] static auto formatString(std::string const& pattern, int index) -> std::string {
        std::string result = pattern;
        size_t pos = result.find("{}");
        if (pos != std::string::npos) {
            result.replace(pos, 2, std::to_string(index));
        }
        return result;
    }

    /**
     * Load binary weights from file
     */
    [[nodiscard]] static auto loadBinary(
        std::filesystem::path const& path,
        size_t expected_size
    ) -> std::vector<float> {
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("Weight file not found: " + path.string());
        }

        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("Failed to open: " + path.string());
        }

        auto file_size = file.tellg();
        auto expected_bytes = expected_size * sizeof(float);

        if (static_cast<size_t>(file_size) != expected_bytes) {
            throw std::runtime_error(
                "Size mismatch in " + path.string() +
                ": expected " + std::to_string(expected_bytes) + " bytes, " +
                "got " + std::to_string(file_size) + " bytes"
            );
        }

        file.seekg(0, std::ios::beg);
        std::vector<float> weights(expected_size);
        file.read(reinterpret_cast<char*>(weights.data()), expected_bytes);

        if (!file) {
            throw std::runtime_error("Failed to read: " + path.string());
        }

        return weights;
    }

    // Configuration
    std::filesystem::path base_dir_;
    int input_size_ = 0;
    int output_size_ = 0;
    std::vector<int> hidden_layers_;
    std::string weight_pattern_ = "network_{}_weight.bin";
    std::string bias_pattern_ = "network_{}_bias.bin";
    std::vector<int> layer_indices_;
    bool verbose_ = false;
};

/**
 * Predefined configuration for standard price predictor model (60 features)
 */
struct PricePredictorConfig {
    static constexpr int INPUT_SIZE = 60;
    static constexpr int OUTPUT_SIZE = 3;
    static constexpr std::array HIDDEN_LAYERS = {256, 128, 64, 32};

    [[nodiscard]] static auto createLoader(std::filesystem::path const& base_dir = "models/weights") -> WeightLoader {
        return WeightLoader::fromDirectory(base_dir)
            .withArchitecture(INPUT_SIZE, std::vector<int>{HIDDEN_LAYERS.begin(), HIDDEN_LAYERS.end()}, OUTPUT_SIZE)
            .withNamingScheme("network_{}_weight.bin", "network_{}_bias.bin");
    }
};

/**
 * Predefined configuration for clean 85-feature price predictor model (98% accuracy)
 *
 * Features:
 * - 58 base features (clean, no constant values)
 * - 3 temporal features (year, month, day)
 * - 20 first-order differences (price_diff_1d through 20d)
 * - 4 autocorrelation features (lags 1, 5, 10, 20)
 */
struct PricePredictorConfig85 {
    static constexpr int INPUT_SIZE = 85;
    static constexpr int OUTPUT_SIZE = 3;
    static constexpr std::array HIDDEN_LAYERS = {256, 128, 64, 32};

    [[nodiscard]] static auto createLoader(std::filesystem::path const& base_dir = "models/weights") -> WeightLoader {
        return WeightLoader::fromDirectory(base_dir)
            .withArchitecture(INPUT_SIZE, std::vector<int>{HIDDEN_LAYERS.begin(), HIDDEN_LAYERS.end()}, OUTPUT_SIZE)
            .withNamingScheme("layer{}_weight.bin", "layer{}_bias.bin")
            .withLayerIndices({1, 2, 3, 4, 5});  // Simple 1-indexed naming
    }
};

}  // namespace bigbrother::ml
