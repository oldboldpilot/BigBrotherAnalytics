/**
 * Unit Tests for Correlation Engine
 *
 * Tests all correlation calculations and validates against known values.
 */

#include "../../src/correlation_engine/correlation.hpp"
#include "../../src/correlation_engine/correlation_fluent_api.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

using namespace bigbrother::correlation;
using namespace bigbrother::types;

constexpr double CORRELATION_TOLERANCE = 0.001;

/**
 * Test Pearson Correlation
 */
TEST(PearsonCorrelationTest, PerfectPositiveCorrelation) {
    // Perfectly correlated data
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0};  // y = 2*x

    auto result = CorrelationCalculator::pearson(x, y);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 1.0, CORRELATION_TOLERANCE)
        << "Perfect positive correlation should be 1.0";
}

TEST(PearsonCorrelationTest, PerfectNegativeCorrelation) {
    // Perfectly negatively correlated data
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {10.0, 8.0, 6.0, 4.0, 2.0};  // y = 12 - 2*x

    auto result = CorrelationCalculator::pearson(x, y);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, -1.0, CORRELATION_TOLERANCE)
        << "Perfect negative correlation should be -1.0";
}

TEST(PearsonCorrelationTest, NoCorrelation) {
    // Uncorrelated data
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {3.0, 1.0, 4.0, 2.0, 5.0};  // Random permutation

    auto result = CorrelationCalculator::pearson(x, y);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 0.0, 0.5)  // Very weak correlation
        << "Should have low correlation";
}

TEST(PearsonCorrelationTest, KnownValue) {
    // Known test case from statistics textbook
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> y = {2.1, 4.3, 6.2, 8.1, 9.8, 12.3, 14.1, 16.2, 18.0, 20.1};

    auto result = CorrelationCalculator::pearson(x, y);

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(*result, 0.99)  // Very strong positive correlation
        << "Should have strong positive correlation";
}

TEST(PearsonCorrelationTest, ZeroVariance) {
    // Constant series (zero variance)
    std::vector<double> x = {5.0, 5.0, 5.0, 5.0, 5.0};
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};

    auto result = CorrelationCalculator::pearson(x, y);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0.0)
        << "Correlation with constant series should be 0";
}

/**
 * Test Time-Lagged Correlation
 */
TEST(TimeLaggedTest, NoLag) {
    // Test with zero lag (should match regular correlation)
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0};

    auto cross_corr_result = CorrelationCalculator::crossCorrelation(x, y, 0);
    auto simple_corr_result = CorrelationCalculator::pearson(x, y);

    ASSERT_TRUE(cross_corr_result.has_value());
    ASSERT_TRUE(simple_corr_result.has_value());

    EXPECT_EQ(cross_corr_result->size(), 1);
    EXPECT_NEAR((*cross_corr_result)[0], *simple_corr_result, CORRELATION_TOLERANCE)
        << "Zero lag should match simple correlation";
}

TEST(TimeLaggedTest, OptimalLagDetection) {
    // Create data where Y lags X by 3 periods
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> y = {0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    // Y is X shifted by 3 periods

    auto result = CorrelationCalculator::findOptimalLag(x, y, 10);

    ASSERT_TRUE(result.has_value());

    auto const [optimal_lag, correlation] = *result;

    EXPECT_EQ(optimal_lag, 3)
        << "Optimal lag should be 3 periods";

    EXPECT_GT(correlation, 0.95)
        << "Correlation at optimal lag should be very high";
}

TEST(TimeLaggedTest, MultipleTimeFrames) {
    // Test correlations at different lags
    std::vector<double> x(100);
    std::vector<double> y(100);

    // Generate sine wave data
    for (size_t i = 0; i < 100; ++i) {
        x[i] = std::sin(static_cast<double>(i) * 0.1);
        y[i] = std::sin(static_cast<double>(i) * 0.1 + 0.5);  // Phase shifted
    }

    auto cross_corr = CorrelationCalculator::crossCorrelation(x, y, 20);

    ASSERT_TRUE(cross_corr.has_value());
    EXPECT_EQ(cross_corr->size(), 21);  // 0 to 20 inclusive

    // Correlations should vary with lag
    bool has_variation = false;
    for (size_t i = 1; i < cross_corr->size(); ++i) {
        if (std::abs((*cross_corr)[i] - (*cross_corr)[0]) > 0.1) {
            has_variation = true;
            break;
        }
    }

    EXPECT_TRUE(has_variation)
        << "Cross-correlation should vary with lag";
}

/**
 * Test Spearman Correlation
 */
TEST(SpearmanCorrelationTest, MonotonicRelationship) {
    // Non-linear but monotonic relationship
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {1.0, 4.0, 9.0, 16.0, 25.0};  // y = x²

    auto pearson_result = CorrelationCalculator::pearson(x, y);
    auto spearman_result = CorrelationCalculator::spearman(x, y);

    ASSERT_TRUE(pearson_result.has_value());
    ASSERT_TRUE(spearman_result.has_value());

    // Spearman should be 1.0 (perfect monotonic relationship)
    EXPECT_NEAR(*spearman_result, 1.0, CORRELATION_TOLERANCE);

    // Pearson should be less than 1.0 (non-linear)
    EXPECT_LT(*pearson_result, 0.99);
}

TEST(SpearmanCorrelationTest, HandlesOutliers) {
    // Data with outlier
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 100.0};  // Outlier: 100
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0};

    auto pearson_result = CorrelationCalculator::pearson(x, y);
    auto spearman_result = CorrelationCalculator::spearman(x, y);

    ASSERT_TRUE(pearson_result.has_value());
    ASSERT_TRUE(spearman_result.has_value());

    // Spearman should be more robust to outlier
    EXPECT_GT(*spearman_result, *pearson_result)
        << "Spearman should be more robust to outliers";
}

/**
 * Test Rolling Correlation
 */
TEST(RollingCorrelationTest, DetectRegimeChange) {
    // Create data with changing correlation
    std::vector<double> x, y;

    // First 50 points: positive correlation
    for (int i = 0; i < 50; ++i) {
        x.push_back(static_cast<double>(i));
        y.push_back(static_cast<double>(i) + (rand() % 10 - 5) * 0.1);
    }

    // Next 50 points: negative correlation
    for (int i = 50; i < 100; ++i) {
        x.push_back(static_cast<double>(i));
        y.push_back(100.0 - static_cast<double>(i) + (rand() % 10 - 5) * 0.1);
    }

    auto rolling_corr = CorrelationCalculator::rollingCorrelation(x, y, 20);

    ASSERT_TRUE(rolling_corr.has_value());

    // Should detect change from positive to negative correlation
    EXPECT_GT((*rolling_corr)[0], 0.0)
        << "Initial rolling correlation should be positive";

    EXPECT_LT((*rolling_corr)[rolling_corr->size() - 1], 0.0)
        << "Final rolling correlation should be negative";
}

/**
 * Test Correlation Matrix
 */
TEST(CorrelationMatrixTest, SymmetricMatrix) {
    // Create test time series
    std::vector<TimeSeries> series;

    for (int i = 0; i < 5; ++i) {
        TimeSeries ts;
        ts.symbol = "SYM" + std::to_string(i);

        for (int j = 0; j < 100; ++j) {
            ts.values.push_back(static_cast<double>(j) + i * 10.0 + (rand() % 10) * 0.1);
            ts.timestamps.push_back(j * 1'000'000);
        }

        series.push_back(ts);
    }

    auto matrix_result = CorrelationCalculator::correlationMatrix(series);

    ASSERT_TRUE(matrix_result.has_value());

    auto const& matrix = *matrix_result;

    // Check symmetry
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix.size(); ++j) {
            auto const& sym_i = matrix.getSymbols()[i];
            auto const& sym_j = matrix.getSymbols()[j];

            double const corr_ij = matrix.get(sym_i, sym_j);
            double const corr_ji = matrix.get(sym_j, sym_i);

            EXPECT_NEAR(corr_ij, corr_ji, CORRELATION_TOLERANCE)
                << "Matrix should be symmetric";
        }
    }

    // Check diagonal (self-correlation = 1.0)
    for (size_t i = 0; i < matrix.size(); ++i) {
        auto const& symbol = matrix.getSymbols()[i];
        double const self_corr = matrix.get(symbol, symbol);

        EXPECT_NEAR(self_corr, 1.0, CORRELATION_TOLERANCE)
            << "Self-correlation should be 1.0";
    }
}

/**
 * Test Fluent API
 */
TEST(FluentAPITest, SimpleCorrelation) {
    TimeSeries ts1;
    ts1.symbol = "AAPL";
    ts1.values = {100.0, 102.0, 104.0, 103.0, 105.0};
    ts1.timestamps = {0, 1, 2, 3, 4};

    TimeSeries ts2;
    ts2.symbol = "MSFT";
    ts2.values = {200.0, 204.0, 208.0, 206.0, 210.0};
    ts2.timestamps = {0, 1, 2, 3, 4};

    auto result = CorrelationAnalyzer()
        .withTimeSeries(ts1, ts2)
        .pearson()
        .calculate();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(*result, 0.9)  // Should be highly correlated
        << "Similar patterns should be highly correlated";
}

TEST(FluentAPITest, OptimalLagFinding) {
    // Create lagged data
    std::vector<double> x_vals, y_vals;
    std::vector<Timestamp> timestamps;

    for (int i = 0; i < 100; ++i) {
        x_vals.push_back(static_cast<double>(i));
        timestamps.push_back(i * 1'000'000);
    }

    // Y lags X by 5 periods
    for (int i = 0; i < 5; ++i) {
        y_vals.push_back(0.0);
    }
    for (int i = 0; i < 95; ++i) {
        y_vals.push_back(x_vals[i]);
    }

    TimeSeries ts1{"LEAD", x_vals, timestamps};
    TimeSeries ts2{"LAG", y_vals, timestamps};

    auto result = CorrelationAnalyzer()
        .withTimeSeries(ts1, ts2)
        .laggedUpTo(10)
        .findOptimalLag();

    ASSERT_TRUE(result.has_value());

    auto const [lag, corr] = *result;

    EXPECT_EQ(lag, 5)
        << "Should detect 5-period lag";

    EXPECT_GT(corr, 0.95)
        << "Correlation at optimal lag should be very high";
}

/**
 * Test Signal Generation
 */
TEST(SignalGenerationTest, GeneratesTradingSignals) {
    std::vector<CorrelationResult> correlations = {
        {
            .symbol1 = "NVDA",
            .symbol2 = "AMD",
            .correlation = 0.85,
            .p_value = 0.001,
            .sample_size = 252,
            .lag = 15,  // AMD lags NVDA by 15 minutes
            .type = CorrelationType::Pearson
        },
        {
            .symbol1 = "SPY",
            .symbol2 = "QQQ",
            .correlation = 0.92,
            .p_value = 0.0001,
            .sample_size = 252,
            .lag = 0,  // Contemporaneous
            .type = CorrelationType::Pearson
        }
    };

    auto signals = CorrelationSignalGenerator::generateSignals(
        correlations,
        0.6,  // min correlation
        0.7   // min confidence
    );

    EXPECT_GE(signals.size(), 1)
        << "Should generate at least one signal";

    // Check that lagged relationship is identified
    auto nvda_amd_signal = std::ranges::find_if(signals,
        [](auto const& s) {
            return s.leading_symbol == "NVDA" && s.lagging_symbol == "AMD";
        }
    );

    EXPECT_NE(nvda_amd_signal, signals.end())
        << "Should identify NVDA → AMD relationship";

    if (nvda_amd_signal != signals.end()) {
        EXPECT_EQ(nvda_amd_signal->optimal_lag, 15)
            << "Should preserve 15-minute lag";

        EXPECT_TRUE(nvda_amd_signal->isActionable())
            << "Strong correlation should be actionable";
    }
}

/**
 * Test Performance
 */
TEST(PerformanceTest, CorrelationSpeed) {
    // Test with realistic data size (252 trading days)
    std::vector<double> x(252), y(252);

    for (size_t i = 0; i < 252; ++i) {
        x[i] = 100.0 + static_cast<double>(i) * 0.5;
        y[i] = 200.0 + static_cast<double>(i) * 0.3;
    }

    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERATIONS = 10'000;
    for (int i = 0; i < ITERATIONS; ++i) {
        auto result = CorrelationCalculator::pearson(x, y);
        ASSERT_TRUE(result.has_value());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_time_us = static_cast<double>(duration.count()) / ITERATIONS;

    std::cout << "Pearson correlation (252 points) average time: "
              << avg_time_us << " μs" << std::endl;

    // Should be very fast (< 10 microseconds)
    EXPECT_LT(avg_time_us, 10.0)
        << "Correlation calculation too slow";
}

TEST(PerformanceTest, CorrelationMatrixSpeed) {
    // Test 100x100 correlation matrix
    std::vector<TimeSeries> series;

    for (int i = 0; i < 100; ++i) {
        TimeSeries ts;
        ts.symbol = "SYM" + std::to_string(i);

        for (int j = 0; j < 252; ++j) {
            ts.values.push_back(100.0 + j + i * 10.0 + (rand() % 10) * 0.1);
            ts.timestamps.push_back(j * 1'000'000);
        }

        series.push_back(ts);
    }

    auto start = std::chrono::high_resolution_clock::now();

    auto result = CorrelationCalculator::correlationMatrix(series);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "100x100 correlation matrix time: "
              << duration.count() << " ms" << std::endl;

    ASSERT_TRUE(result.has_value());

    // With OpenMP, should be < 1 second for 100x100
    EXPECT_LT(duration.count(), 1000)
        << "100x100 matrix calculation too slow";

    // Verify matrix is complete
    EXPECT_EQ(result->size(), 100);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
