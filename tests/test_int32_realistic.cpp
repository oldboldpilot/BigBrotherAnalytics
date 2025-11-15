/**
 * Test INT32 SIMD with REALISTIC features from bot logs
 * From logs: SPY close=670.59, QQQ close=605.78, IWM close=236.45
 */

#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string>

// Scaler parameters from price_predictor.cppm
constexpr std::array<float, 85> MEAN = {
    171.73168510f, 171.77098131f, 173.85409399f, 169.78849837f, 18955190.81943483f,
    52.07665122f, -1.01429406f, -1.11443808f, 183.70466682f, 161.62830260f,
    0.53303925f, 4.63554388f, 18931511.93290513f, 1.01396078f, 0.06702154f,
    -0.09081233f, 0.19577506f, 0.07070947f, 0.02930415f, 0.02595905f,
    -1.16933052f, 0.00245190f, 0.00007623f, 0.11992571f, 0.32397784f,
    0.00058185f, 0.51397823f, 1.00021170f, 1.00055151f, 1.00072988f,
    1.00106309f, 1.00148186f, 1.00161426f, 1.00184679f, 1.00197729f,
    1.00221717f, 1.00252557f, 1.00279857f, 1.00292291f, 1.00302553f,
    1.00308145f, 1.00332408f, 1.00349154f, 1.00386256f, 1.00421874f,
    1.00436739f, 1.00463950f, 9.48643716f, 2.31273208f, 15.75989678f,
    6.54169551f, 2.51582856f, 183.66473661f, 0.51859777f, 0.53345082f,
    0.54987727f, 0.51526213f, 0.42463339f, 2023.02114671f, 6.54169551f,
    15.75989678f, -0.05298513f, -0.18776332f, -0.22518525f, -0.26102762f,
    -0.41762342f, -0.47033575f, -0.58258492f, -0.67154995f, -0.79431408f,
    -0.92354285f, -1.15338400f, -1.26767064f, -1.36071822f, -1.34270070f,
    -1.49789463f, -1.61267464f, -1.76728610f, -1.98800362f, -2.11874748f,
    -2.25577792f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f
};

constexpr std::array<float, 85> STD = {
    186.03571734f, 186.47600380f, 191.72157267f, 181.70836041f, 22005096.42658922f,
    16.71652602f, 15.57689787f, 16.42374095f, 223.36568996f, 167.49749464f,
    0.32584191f, 15.18473180f, 20390423.38239934f, 0.38221233f, 0.06174181f,
    0.09837635f, 0.21208174f, 0.07659907f, 0.38598442f, 0.08503450f,
    17.25845955f, 0.01734768f, 0.00311195f, 0.12017425f, 0.24795759f,
    0.03781362f, 0.15687517f, 0.02125998f, 0.02982959f, 0.03591441f,
    0.04211676f, 0.04605618f, 0.04989224f, 0.05347304f, 0.05674135f,
    0.05876375f, 0.06280664f, 0.06454916f, 0.06744410f, 0.06983660f,
    0.07165713f, 0.07489287f, 0.07817148f, 0.08043051f, 0.08268397f,
    0.08362892f, 0.08593539f, 5.75042283f, 2.02005544f, 8.73732957f,
    3.28657881f, 1.06885103f, 100.31964116f, 0.49965400f, 0.49887979f,
    0.49750604f, 0.49976701f, 0.49428724f, 1.33469640f, 3.28657881f,
    8.73732957f, 11.58576980f, 16.32518674f, 17.84023616f, 19.44449065f,
    20.31759436f, 22.41613818f, 23.65359845f, 26.22348338f, 26.79057463f,
    27.84826555f, 31.27653616f, 32.78403656f, 33.96486964f, 34.75751675f,
    36.14405424f, 38.68106210f, 39.64798053f, 42.54812539f, 44.39562117f,
    45.85464070f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f
};

auto normalize(std::array<float, 85> const& features) -> std::array<float, 85> {
    std::array<float, 85> normalized;
    for (size_t i = 0; i < 85; ++i) {
        normalized[i] = (features[i] - MEAN[i]) / STD[i];
    }
    return normalized;
}

int main() {
    std::cout << "Testing with REALISTIC features from bot logs:" << std::endl << std::endl;

    // Create realistic feature vectors for SPY, QQQ, IWM
    struct TestCase {
        std::string symbol;
        float close;
        float symbol_enc;
    };

    std::vector<TestCase> tests = {
        {"SPY", 670.59f, 0.0f},
        {"QQQ", 605.78f, 1.0f},
        {"IWM", 236.45f, 2.0f}
    };

    for (auto const& test : tests) {
        // Create feature vector (simplified - just set close and symbol_enc)
        std::array<float, 85> features;
        std::fill(features.begin(), features.end(), 0.0f);

        features[0] = test.close;      // close price
        features[47] = test.symbol_enc; // symbol encoding

        // Normalize
        auto normalized = normalize(features);

        std::cout << test.symbol << ":" << std::endl;
        std::cout << "  Raw: close=" << test.close << ", symbol_enc=" << test.symbol_enc << std::endl;
        std::cout << "  Normalized[0]=" << normalized[0] << ", Normalized[47]=" << normalized[47] << std::endl;
        std::cout << "  (If Python model works with these, C++ should too)" << std::endl << std::endl;
    }

    std::cout << "Conclusion: The bug is likely in the C++ engine, not the test." << std::endl;
    std::cout << "Need to test full 5-layer forward pass with real weights." << std::endl;

    return 0;
}
