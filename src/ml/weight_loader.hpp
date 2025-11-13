/**
 * BigBrotherAnalytics - Unified ML Weight Loader
 *
 * Common infrastructure for loading PyTorch-exported weights across all inference engines.
 * Supports MKL, SIMD (AVX-512/AVX-2/SSE), and ONNX implementations.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-13
 */

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace bigbrother::ml {

/**
 * Neural network architecture constants
 * Model: price_predictor_best.pth (60 features)
 */
struct NetworkArchitecture {
    static constexpr int INPUT_SIZE = 60;
    static constexpr int LAYER1_SIZE = 256;
    static constexpr int LAYER2_SIZE = 128;
    static constexpr int LAYER3_SIZE = 64;
    static constexpr int LAYER4_SIZE = 32;
    static constexpr int OUTPUT_SIZE = 3;

    // Weight sizes
    static constexpr int LAYER1_WEIGHTS = LAYER1_SIZE * INPUT_SIZE;   // 256 × 60 = 15360
    static constexpr int LAYER2_WEIGHTS = LAYER2_SIZE * LAYER1_SIZE;  // 128 × 256 = 32768
    static constexpr int LAYER3_WEIGHTS = LAYER3_SIZE * LAYER2_SIZE;  // 64 × 128 = 8192
    static constexpr int LAYER4_WEIGHTS = LAYER4_SIZE * LAYER3_SIZE;  // 32 × 64 = 2048
    static constexpr int LAYER5_WEIGHTS = OUTPUT_SIZE * LAYER4_SIZE;  // 3 × 32 = 96

    static constexpr int TOTAL_PARAMS = 58947;
};

/**
 * Weight file paths (relative to models/ directory)
 */
struct WeightPaths {
    std::filesystem::path base_dir;

    explicit WeightPaths(std::filesystem::path const& base = "models/weights")
        : base_dir(base) {}

    [[nodiscard]] auto layer1_weight() const -> std::filesystem::path {
        return base_dir / "network_0_weight.bin";
    }
    [[nodiscard]] auto layer1_bias() const -> std::filesystem::path {
        return base_dir / "network_0_bias.bin";
    }
    [[nodiscard]] auto layer2_weight() const -> std::filesystem::path {
        return base_dir / "network_3_weight.bin";
    }
    [[nodiscard]] auto layer2_bias() const -> std::filesystem::path {
        return base_dir / "network_3_bias.bin";
    }
    [[nodiscard]] auto layer3_weight() const -> std::filesystem::path {
        return base_dir / "network_6_weight.bin";
    }
    [[nodiscard]] auto layer3_bias() const -> std::filesystem::path {
        return base_dir / "network_6_bias.bin";
    }
    [[nodiscard]] auto layer4_weight() const -> std::filesystem::path {
        return base_dir / "network_9_weight.bin";
    }
    [[nodiscard]] auto layer4_bias() const -> std::filesystem::path {
        return base_dir / "network_9_bias.bin";
    }
    [[nodiscard]] auto layer5_weight() const -> std::filesystem::path {
        return base_dir / "network_12_weight.bin";
    }
    [[nodiscard]] auto layer5_bias() const -> std::filesystem::path {
        return base_dir / "network_12_bias.bin";
    }
};

/**
 * Unified weight loader for all inference engines
 */
class WeightLoader {
public:
    /**
     * Load binary weights from file
     *
     * @param path Path to .bin file
     * @param expected_size Expected number of float32 values
     * @return Vector of weights
     * @throws std::runtime_error if file doesn't exist or size mismatch
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

    /**
     * Load all network weights from directory
     *
     * @param base_dir Base directory containing weight files
     * @return Structure containing all weights and biases
     */
    struct NetworkWeights {
        std::vector<float> layer1_weight;
        std::vector<float> layer1_bias;
        std::vector<float> layer2_weight;
        std::vector<float> layer2_bias;
        std::vector<float> layer3_weight;
        std::vector<float> layer3_bias;
        std::vector<float> layer4_weight;
        std::vector<float> layer4_bias;
        std::vector<float> layer5_weight;
        std::vector<float> layer5_bias;
    };

    [[nodiscard]] static auto loadAllWeights(
        std::filesystem::path const& base_dir = "models/weights"
    ) -> NetworkWeights {
        WeightPaths paths(base_dir);

        NetworkWeights weights;
        weights.layer1_weight = loadBinary(paths.layer1_weight(), NetworkArchitecture::LAYER1_WEIGHTS);
        weights.layer1_bias = loadBinary(paths.layer1_bias(), NetworkArchitecture::LAYER1_SIZE);
        weights.layer2_weight = loadBinary(paths.layer2_weight(), NetworkArchitecture::LAYER2_WEIGHTS);
        weights.layer2_bias = loadBinary(paths.layer2_bias(), NetworkArchitecture::LAYER2_SIZE);
        weights.layer3_weight = loadBinary(paths.layer3_weight(), NetworkArchitecture::LAYER3_WEIGHTS);
        weights.layer3_bias = loadBinary(paths.layer3_bias(), NetworkArchitecture::LAYER3_SIZE);
        weights.layer4_weight = loadBinary(paths.layer4_weight(), NetworkArchitecture::LAYER4_WEIGHTS);
        weights.layer4_bias = loadBinary(paths.layer4_bias(), NetworkArchitecture::LAYER4_SIZE);
        weights.layer5_weight = loadBinary(paths.layer5_weight(), NetworkArchitecture::LAYER5_WEIGHTS);
        weights.layer5_bias = loadBinary(paths.layer5_bias(), NetworkArchitecture::OUTPUT_SIZE);

        return weights;
    }

    /**
     * Verify weight file integrity
     *
     * @param base_dir Base directory containing weight files
     * @return true if all files exist and have correct sizes
     */
    [[nodiscard]] static auto verifyWeights(
        std::filesystem::path const& base_dir = "models/weights"
    ) -> bool {
        try {
            WeightPaths paths(base_dir);

            // Check all files exist and have correct sizes
            loadBinary(paths.layer1_weight(), NetworkArchitecture::LAYER1_WEIGHTS);
            loadBinary(paths.layer1_bias(), NetworkArchitecture::LAYER1_SIZE);
            loadBinary(paths.layer2_weight(), NetworkArchitecture::LAYER2_WEIGHTS);
            loadBinary(paths.layer2_bias(), NetworkArchitecture::LAYER2_SIZE);
            loadBinary(paths.layer3_weight(), NetworkArchitecture::LAYER3_WEIGHTS);
            loadBinary(paths.layer3_bias(), NetworkArchitecture::LAYER3_SIZE);
            loadBinary(paths.layer4_weight(), NetworkArchitecture::LAYER4_WEIGHTS);
            loadBinary(paths.layer4_bias(), NetworkArchitecture::LAYER4_SIZE);
            loadBinary(paths.layer5_weight(), NetworkArchitecture::LAYER5_WEIGHTS);
            loadBinary(paths.layer5_bias(), NetworkArchitecture::OUTPUT_SIZE);

            return true;
        } catch (...) {
            return false;
        }
    }
};

}  // namespace bigbrother::ml
