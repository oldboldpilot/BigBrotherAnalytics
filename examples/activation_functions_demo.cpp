/**
 * Activation Functions Library Demo
 *
 * Demonstrates usage of the C++23 activation functions module
 * with automatic SIMD optimization.
 *
 * Build:
 *   SKIP_CLANG_TIDY=1 cmake -G Ninja -B build && ninja -C build activation_functions_demo
 *
 * Run:
 *   ./build/bin/activation_functions_demo
 */

#include <array>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <span>

import bigbrother.ml.activations;

using namespace bigbrother::ml::activations;

void printArray(const char* label, std::span<const float> data, int max_print = 10) {
    std::cout << label << ": [";
    int n = std::min(max_print, static_cast<int>(data.size()));
    for (int i = 0; i < n; ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i];
        if (i < n - 1) std::cout << ", ";
    }
    if (data.size() > max_print) {
        std::cout << ", ... (" << data.size() << " total)";
    }
    std::cout << "]\n";
}

void demoBasicActivations() {
    std::cout << "\n=== Basic Activation Functions Demo ===\n\n";

    // Test input: [-2, -1, 0, 1, 2]
    std::array<float, 5> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};

    // ReLU
    {
        auto data = input;
        relu(std::span(data));
        std::cout << "ReLU:\n";
        std::cout << "  Input:  [-2.0, -1.0, 0.0, 1.0, 2.0]\n";
        printArray("  Output", std::span(data));
        std::cout << "  Expected: [0.0, 0.0, 0.0, 1.0, 2.0]\n";
    }

    // Leaky ReLU
    {
        auto data = input;
        leaky_relu(std::span(data), 0.1f);
        std::cout << "\nLeaky ReLU (alpha=0.1):\n";
        std::cout << "  Input:  [-2.0, -1.0, 0.0, 1.0, 2.0]\n";
        printArray("  Output", std::span(data));
        std::cout << "  Expected: [-0.2, -0.1, 0.0, 1.0, 2.0]\n";
    }

    // Sigmoid
    {
        auto data = input;
        sigmoid(std::span(data));
        std::cout << "\nSigmoid:\n";
        std::cout << "  Input:  [-2.0, -1.0, 0.0, 1.0, 2.0]\n";
        printArray("  Output", std::span(data));
        std::cout << "  Expected: [0.1192, 0.2689, 0.5000, 0.7311, 0.8808]\n";
    }

    // Tanh
    {
        auto data = input;
        tanh_activation(std::span(data));
        std::cout << "\nTanh:\n";
        std::cout << "  Input:  [-2.0, -1.0, 0.0, 1.0, 2.0]\n";
        printArray("  Output", std::span(data));
        std::cout << "  Expected: [-0.9640, -0.7616, 0.0000, 0.7616, 0.9640]\n";
    }

    // GELU
    {
        auto data = input;
        gelu(std::span(data));
        std::cout << "\nGELU (Gaussian Error Linear Unit):\n";
        std::cout << "  Input:  [-2.0, -1.0, 0.0, 1.0, 2.0]\n";
        printArray("  Output", std::span(data));
        std::cout << "  Expected: [-0.0454, -0.1588, 0.0000, 0.8412, 1.9545]\n";
    }

    // Swish
    {
        auto data = input;
        swish(std::span(data));
        std::cout << "\nSwish/SiLU:\n";
        std::cout << "  Input:  [-2.0, -1.0, 0.0, 1.0, 2.0]\n";
        printArray("  Output", std::span(data));
        std::cout << "  Expected: [-0.2384, -0.2689, 0.0000, 0.7311, 1.7616]\n";
    }

    // ELU
    {
        auto data = input;
        elu(std::span(data), 1.0f);
        std::cout << "\nELU (Exponential Linear Unit, alpha=1.0):\n";
        std::cout << "  Input:  [-2.0, -1.0, 0.0, 1.0, 2.0]\n";
        printArray("  Output", std::span(data));
        std::cout << "  Expected: [-0.8647, -0.6321, 0.0000, 1.0000, 2.0000]\n";
    }

    // Softmax
    {
        auto data = input;
        softmax(std::span(data));
        std::cout << "\nSoftmax (output sums to 1.0):\n";
        std::cout << "  Input:  [-2.0, -1.0, 0.0, 1.0, 2.0]\n";
        printArray("  Output", std::span(data));
        float sum = 0.0f;
        for (auto v : data) sum += v;
        std::cout << "  Sum: " << sum << " (should be 1.0)\n";
        std::cout << "  Expected: [0.0116, 0.0316, 0.0861, 0.2341, 0.6364]\n";
    }
}

void demoObjectOrientedAPI() {
    std::cout << "\n=== Object-Oriented API Demo ===\n\n";

    std::array<float, 8> data = {-3.0f, -2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};

    std::cout << "Input: ";
    printArray("", std::span(data));

    // Create activation function objects
    ActivationFunction relu_fn(ActivationType::ReLU);
    ActivationFunction leaky_fn(ActivationType::LeakyReLU, 0.01f);
    ActivationFunction gelu_fn(ActivationType::GELU);

    std::cout << "\n1. " << relu_fn.name() << ":\n";
    auto relu_data = data;
    relu_fn.apply(std::span(relu_data));
    printArray("   ", std::span(relu_data));

    std::cout << "\n2. " << leaky_fn.name() << ":\n";
    auto leaky_data = data;
    leaky_fn.apply(std::span(leaky_data));
    printArray("   ", std::span(leaky_data));

    std::cout << "\n3. " << gelu_fn.name() << ":\n";
    auto gelu_data = data;
    gelu_fn.apply(std::span(gelu_data));
    printArray("   ", std::span(gelu_data));
}

void demoPerformance() {
    std::cout << "\n=== Performance Benchmark ===\n\n";

    constexpr int SIZE = 10000;
    constexpr int ITERATIONS = 1000;

    std::vector<float> data(SIZE);
    for (int i = 0; i < SIZE; ++i) {
        data[i] = static_cast<float>(i) / SIZE * 4.0f - 2.0f; // Range [-2, 2]
    }

    // Detect instruction set
    auto isa = detectInstructionSet();
    std::cout << "Detected instruction set: ";
    switch (isa) {
        case InstructionSet::AVX512: std::cout << "AVX-512\n"; break;
        case InstructionSet::AVX2: std::cout << "AVX-2\n"; break;
        case InstructionSet::SSE: std::cout << "SSE\n"; break;
        case InstructionSet::Scalar: std::cout << "Scalar (no SIMD)\n"; break;
    }

    std::cout << "\nBenchmarking activation functions (" << SIZE << " elements, "
              << ITERATIONS << " iterations):\n\n";

    // Benchmark ReLU
    {
        auto test_data = data;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
            relu(std::span(test_data));
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double avg_us = static_cast<double>(duration) / ITERATIONS;
        double throughput = (SIZE * ITERATIONS) / (duration / 1e6); // elements/sec

        std::cout << "ReLU:\n";
        std::cout << "  Average time: " << std::fixed << std::setprecision(3)
                  << avg_us << " μs\n";
        std::cout << "  Throughput: " << std::setprecision(2)
                  << (throughput / 1e9) << " Gelements/sec\n\n";
    }

    // Benchmark GELU
    {
        auto test_data = data;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
            gelu(std::span(test_data));
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double avg_us = static_cast<double>(duration) / ITERATIONS;
        double throughput = (SIZE * ITERATIONS) / (duration / 1e6);

        std::cout << "GELU:\n";
        std::cout << "  Average time: " << std::fixed << std::setprecision(3)
                  << avg_us << " μs\n";
        std::cout << "  Throughput: " << std::setprecision(2)
                  << (throughput / 1e9) << " Gelements/sec\n\n";
    }

    // Benchmark Sigmoid
    {
        auto test_data = data;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
            sigmoid(std::span(test_data));
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double avg_us = static_cast<double>(duration) / ITERATIONS;
        double throughput = (SIZE * ITERATIONS) / (duration / 1e6);

        std::cout << "Sigmoid:\n";
        std::cout << "  Average time: " << std::fixed << std::setprecision(3)
                  << avg_us << " μs\n";
        std::cout << "  Throughput: " << std::setprecision(2)
                  << (throughput / 1e9) << " Gelements/sec\n";
    }
}

auto main() -> int {
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║    BigBrotherAnalytics Activation Functions Library       ║\n";
    std::cout << "║    C++23 Module with Auto SIMD Optimization                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";

    demoBasicActivations();
    demoObjectOrientedAPI();
    demoPerformance();

    std::cout << "\n✅ Demo completed successfully!\n";

    return 0;
}
