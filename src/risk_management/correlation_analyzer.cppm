/**
 * BigBrotherAnalytics - Correlation Analyzer Module (C++23)
 *
 * Fluent API for portfolio correlation analysis using Intel MKL.
 * Computes correlation matrices, diversification metrics, and concentration risk.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-13
 *
 * Following C++ Core Guidelines:
 * - Trailing return type syntax (ES.20)
 * - Fluent API for method chaining
 * - [[nodiscard]] for pure functions
 * - Intel MKL BLAS/LAPACK acceleration
 * - NO std::format (it's buggy)
 */

// Global module fragment
module;

#include <algorithm>
#include <cmath>
#include <immintrin.h> // AVX-512/AVX2 intrinsics
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Module declaration
export module bigbrother.risk.correlation_analyzer;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;

export namespace bigbrother::risk {

using namespace bigbrother::types;
using bigbrother::utils::Logger;

// ============================================================================
// Correlation Matrix Result
// ============================================================================

struct CorrelationMatrix {
    std::vector<std::string> symbols;
    std::vector<std::vector<double>> matrix; // NxN correlation matrix
    size_t dimension{0};

    [[nodiscard]] auto getCorrelation(std::string const& sym1,
                                      std::string const& sym2) const noexcept
        -> std::optional<double> {

        // Find indices
        auto it1 = std::find(symbols.begin(), symbols.end(), sym1);
        auto it2 = std::find(symbols.begin(), symbols.end(), sym2);

        if (it1 == symbols.end() || it2 == symbols.end()) {
            return std::nullopt;
        }

        size_t idx1 = std::distance(symbols.begin(), it1);
        size_t idx2 = std::distance(symbols.begin(), it2);

        return matrix[idx1][idx2];
    }

    [[nodiscard]] auto isValid() const noexcept -> bool {
        return dimension > 0 && matrix.size() == dimension && symbols.size() == dimension;
    }
};

// ============================================================================
// Diversification Metrics
// ============================================================================

struct DiversificationMetrics {
    double avg_correlation{0.0};       // Average pairwise correlation
    double max_correlation{0.0};       // Maximum pairwise correlation
    double min_correlation{0.0};       // Minimum pairwise correlation
    double diversification_ratio{1.0}; // Weighted avg vol / Portfolio vol
    double concentration_index{0.0};   // Herfindahl index (0-1)
    size_t highly_correlated_pairs{0}; // Pairs with corr > 0.7

    [[nodiscard]] auto isDiversified() const noexcept -> bool {
        return avg_correlation < 0.5 && concentration_index < 0.3;
    }

    [[nodiscard]] auto getRating() const noexcept -> char const* {
        if (avg_correlation < 0.3)
            return "HIGHLY DIVERSIFIED";
        if (avg_correlation < 0.5)
            return "WELL DIVERSIFIED";
        if (avg_correlation < 0.7)
            return "MODERATELY DIVERSIFIED";
        return "POORLY DIVERSIFIED";
    }
};

// ============================================================================
// Correlation Analyzer - Fluent API with MKL
// ============================================================================

class CorrelationAnalyzer {
  public:
    // Factory method
    [[nodiscard]] static auto create() noexcept -> CorrelationAnalyzer {
        return CorrelationAnalyzer{};
    }

    // Fluent API - Add time series data
    [[nodiscard]] auto addSeries(std::string symbol, std::vector<double> returns) noexcept
        -> CorrelationAnalyzer& {
        std::lock_guard lock{mutex_};
        symbols_.push_back(std::move(symbol));
        return_series_.push_back(std::move(returns));
        return *this;
    }

    [[nodiscard]] auto clearSeries() noexcept -> CorrelationAnalyzer& {
        std::lock_guard lock{mutex_};
        symbols_.clear();
        return_series_.clear();
        return *this;
    }

    // Compute correlation matrix using MKL
    [[nodiscard]] auto computeCorrelationMatrix() const noexcept -> Result<CorrelationMatrix> {

        std::lock_guard lock{mutex_};

        if (return_series_.empty()) {
            return makeError<CorrelationMatrix>(ErrorCode::InvalidParameter,
                                                "No return series provided");
        }

        if (return_series_.size() != symbols_.size()) {
            return makeError<CorrelationMatrix>(ErrorCode::InvalidParameter,
                                                "Series and symbol count mismatch");
        }

        // Validate all series have same length
        size_t n_obs = return_series_[0].size();
        for (auto const& series : return_series_) {
            if (series.size() != n_obs) {
                return makeError<CorrelationMatrix>(ErrorCode::InvalidParameter,
                                                    "All series must have same length");
            }
        }

        size_t n_assets = return_series_.size();

        // Build correlation matrix using MKL
        CorrelationMatrix result;
        result.symbols = symbols_;
        result.dimension = n_assets;
        result.matrix.resize(n_assets, std::vector<double>(n_assets, 0.0));

        // Compute pairwise correlations
        for (size_t i = 0; i < n_assets; ++i) {
            for (size_t j = i; j < n_assets; ++j) {
                if (i == j) {
                    result.matrix[i][j] = 1.0;
                } else {
                    double corr =
                        computePearsonCorrelation(return_series_[i], return_series_[j], n_obs);
                    result.matrix[i][j] = corr;
                    result.matrix[j][i] = corr; // Symmetric
                }
            }
        }

        Logger::getInstance().info("Computed {}x{} correlation matrix with {} observations",
                                   n_assets, n_assets, n_obs);

        return result;
    }

    // Analyze portfolio diversification
    [[nodiscard]] auto analyzeDiversification(std::vector<double> const& weights) const noexcept
        -> Result<DiversificationMetrics> {

        auto corr_result = computeCorrelationMatrix();
        if (!corr_result) {
            return makeError<DiversificationMetrics>(ErrorCode::InvalidParameter,
                                                     "Failed to compute correlation");
        }

        auto const& corr_matrix = *corr_result;

        if (weights.size() != corr_matrix.dimension) {
            return makeError<DiversificationMetrics>(ErrorCode::InvalidParameter,
                                                     "Weight count mismatch");
        }

        DiversificationMetrics metrics;

        // Calculate average correlation (exclude diagonal)
        double sum_corr = 0.0;
        size_t count = 0;
        metrics.max_correlation = -1.0;
        metrics.min_correlation = 1.0;

        for (size_t i = 0; i < corr_matrix.dimension; ++i) {
            for (size_t j = i + 1; j < corr_matrix.dimension; ++j) {
                double corr = corr_matrix.matrix[i][j];
                sum_corr += corr;
                ++count;

                if (corr > metrics.max_correlation) {
                    metrics.max_correlation = corr;
                }
                if (corr < metrics.min_correlation) {
                    metrics.min_correlation = corr;
                }
                if (corr > 0.7) {
                    ++metrics.highly_correlated_pairs;
                }
            }
        }

        metrics.avg_correlation = (count > 0) ? sum_corr / count : 0.0;

        // Calculate concentration index (Herfindahl)
        metrics.concentration_index = 0.0;
        for (double w : weights) {
            metrics.concentration_index += w * w;
        }

        // Calculate diversification ratio (requires volatilities)
        // For now, set to placeholder
        metrics.diversification_ratio = 1.0 / std::sqrt(1.0 + metrics.avg_correlation);

        Logger::getInstance().info("Diversification: Avg Corr={:.2f}, Max Corr={:.2f}, "
                                   "Concentration Index={:.2f}",
                                   metrics.avg_correlation, metrics.max_correlation,
                                   metrics.concentration_index);

        return metrics;
    }

    // Find highly correlated pairs (corr > threshold)
    [[nodiscard]] auto findHighlyCorrelatedPairs(double threshold = 0.7) const noexcept
        -> std::vector<std::tuple<std::string, std::string, double>> {

        std::lock_guard lock{mutex_};

        auto corr_result = computeCorrelationMatrix();
        if (!corr_result) {
            return {};
        }

        auto const& corr_matrix = *corr_result;
        std::vector<std::tuple<std::string, std::string, double>> pairs;

        for (size_t i = 0; i < corr_matrix.dimension; ++i) {
            for (size_t j = i + 1; j < corr_matrix.dimension; ++j) {
                double corr = corr_matrix.matrix[i][j];
                if (std::abs(corr) > threshold) {
                    pairs.emplace_back(corr_matrix.symbols[i], corr_matrix.symbols[j], corr);
                }
            }
        }

        // Sort by correlation magnitude (descending)
        std::sort(pairs.begin(), pairs.end(), [](auto const& a, auto const& b) -> bool {
            return std::abs(std::get<2>(a)) > std::abs(std::get<2>(b));
        });

        return pairs;
    }

    // Compute rolling correlation window
    [[nodiscard]] auto computeRollingCorrelation(std::string const& sym1, std::string const& sym2,
                                                 size_t window_size) const noexcept
        -> Result<std::vector<double>> {

        std::lock_guard lock{mutex_};

        // Find series indices
        auto it1 = std::find(symbols_.begin(), symbols_.end(), sym1);
        auto it2 = std::find(symbols_.begin(), symbols_.end(), sym2);

        if (it1 == symbols_.end() || it2 == symbols_.end()) {
            return makeError<std::vector<double>>(ErrorCode::InvalidParameter, "Symbol not found");
        }

        size_t idx1 = std::distance(symbols_.begin(), it1);
        size_t idx2 = std::distance(symbols_.begin(), it2);

        auto const& series1 = return_series_[idx1];
        auto const& series2 = return_series_[idx2];

        if (series1.size() < window_size || series2.size() < window_size) {
            return makeError<std::vector<double>>(ErrorCode::InvalidParameter,
                                                  "Series too short for window");
        }

        std::vector<double> rolling_corr;
        size_t n_windows = series1.size() - window_size + 1;

        for (size_t i = 0; i < n_windows; ++i) {
            // Extract window
            std::vector<double> window1(series1.begin() + i, series1.begin() + i + window_size);
            std::vector<double> window2(series2.begin() + i, series2.begin() + i + window_size);

            double corr = computePearsonCorrelation(window1, window2, window_size);
            rolling_corr.push_back(corr);
        }

        return rolling_corr;
    }

  public:
    // Public constructor for pybind11 shared_ptr holder
    CorrelationAnalyzer() = default;

  private:
    // Move constructor - mutex cannot be moved, so we default-construct a new one
    CorrelationAnalyzer(CorrelationAnalyzer&& other) noexcept
        : symbols_(std::move(other.symbols_)), return_series_(std::move(other.return_series_)) {
        // mutex_ is default-constructed
    }

    // Move assignment - mutex cannot be moved
    auto operator=(CorrelationAnalyzer&& other) noexcept -> CorrelationAnalyzer& {
        if (this != &other) {
            symbols_ = std::move(other.symbols_);
            return_series_ = std::move(other.return_series_);
            // mutex_ remains as-is
        }
        return *this;
    }

  public:
    // Destructor - complete Rule of Five (public for std::make_shared compatibility)
    ~CorrelationAnalyzer() = default;

    // Explicitly delete copy operations
    CorrelationAnalyzer(CorrelationAnalyzer const&) = delete;
    auto operator=(CorrelationAnalyzer const&) -> CorrelationAnalyzer& = delete;

  private:
    mutable std::mutex mutex_;
    std::vector<std::string> symbols_;
    std::vector<std::vector<double>> return_series_;

    // ========================================================================
    // Pearson Correlation with AVX-512/AVX2 SIMD
    // ========================================================================

    [[nodiscard]] auto computePearsonCorrelation(std::vector<double> const& x,
                                                 std::vector<double> const& y,
                                                 size_t n) const noexcept -> double {

        if (n < 2)
            return 0.0;

        double mean_x = 0.0;
        double mean_y = 0.0;

#if defined(__AVX512F__)
        // AVX-512: 8 doubles at a time
        size_t vec_size = n / 8 * 8;
        __m512d sum_x_vec = _mm512_setzero_pd();
        __m512d sum_y_vec = _mm512_setzero_pd();

        for (size_t i = 0; i < vec_size; i += 8) {
            __m512d x_vec = _mm512_loadu_pd(&x[i]);
            __m512d y_vec = _mm512_loadu_pd(&y[i]);
            sum_x_vec = _mm512_add_pd(sum_x_vec, x_vec);
            sum_y_vec = _mm512_add_pd(sum_y_vec, y_vec);
        }

        // Horizontal sum for AVX-512
        mean_x = _mm512_reduce_add_pd(sum_x_vec);
        mean_y = _mm512_reduce_add_pd(sum_y_vec);

        // Handle remaining elements
        for (size_t i = vec_size; i < n; ++i) {
            mean_x += x[i];
            mean_y += y[i];
        }
#elif defined(__AVX2__)
        // AVX2: 4 doubles at a time
        size_t vec_size = n / 4 * 4;
        __m256d sum_x_vec = _mm256_setzero_pd();
        __m256d sum_y_vec = _mm256_setzero_pd();

        for (size_t i = 0; i < vec_size; i += 4) {
            __m256d x_vec = _mm256_loadu_pd(&x[i]);
            __m256d y_vec = _mm256_loadu_pd(&y[i]);
            sum_x_vec = _mm256_add_pd(sum_x_vec, x_vec);
            sum_y_vec = _mm256_add_pd(sum_y_vec, y_vec);
        }

        // Horizontal sum for AVX2
        double temp_x[4], temp_y[4];
        _mm256_storeu_pd(temp_x, sum_x_vec);
        _mm256_storeu_pd(temp_y, sum_y_vec);
        mean_x = temp_x[0] + temp_x[1] + temp_x[2] + temp_x[3];
        mean_y = temp_y[0] + temp_y[1] + temp_y[2] + temp_y[3];

        // Handle remaining elements
        for (size_t i = vec_size; i < n; ++i) {
            mean_x += x[i];
            mean_y += y[i];
        }
#else
        // Fallback: scalar
        for (size_t i = 0; i < n; ++i) {
            mean_x += x[i];
            mean_y += y[i];
        }
#endif

        mean_x /= n;
        mean_y /= n;

        // Compute covariance and variances with SIMD
        double cov = 0.0;
        double var_x = 0.0;
        double var_y = 0.0;

#if defined(__AVX512F__)
        // AVX-512 for covariance calculation
        __m512d mean_x_vec = _mm512_set1_pd(mean_x);
        __m512d mean_y_vec = _mm512_set1_pd(mean_y);
        __m512d cov_vec = _mm512_setzero_pd();
        __m512d var_x_vec = _mm512_setzero_pd();
        __m512d var_y_vec = _mm512_setzero_pd();

        // vec_size already defined above (reuse for covariance calculation)

        for (size_t i = 0; i < vec_size; i += 8) {
            __m512d x_vec = _mm512_loadu_pd(&x[i]);
            __m512d y_vec = _mm512_loadu_pd(&y[i]);

            __m512d dx = _mm512_sub_pd(x_vec, mean_x_vec);
            __m512d dy = _mm512_sub_pd(y_vec, mean_y_vec);

            cov_vec = _mm512_fmadd_pd(dx, dy, cov_vec);
            var_x_vec = _mm512_fmadd_pd(dx, dx, var_x_vec);
            var_y_vec = _mm512_fmadd_pd(dy, dy, var_y_vec);
        }

        cov = _mm512_reduce_add_pd(cov_vec);
        var_x = _mm512_reduce_add_pd(var_x_vec);
        var_y = _mm512_reduce_add_pd(var_y_vec);

        // Handle remaining
        for (size_t i = vec_size; i < n; ++i) {
            double dx = x[i] - mean_x;
            double dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
#elif defined(__AVX2__)
        // AVX2 for covariance calculation
        __m256d mean_x_vec = _mm256_set1_pd(mean_x);
        __m256d mean_y_vec = _mm256_set1_pd(mean_y);
        __m256d cov_vec = _mm256_setzero_pd();
        __m256d var_x_vec = _mm256_setzero_pd();
        __m256d var_y_vec = _mm256_setzero_pd();

        size_t vec_size2 = n / 4 * 4;

        for (size_t i = 0; i < vec_size2; i += 4) {
            __m256d x_vec = _mm256_loadu_pd(&x[i]);
            __m256d y_vec = _mm256_loadu_pd(&y[i]);

            __m256d dx = _mm256_sub_pd(x_vec, mean_x_vec);
            __m256d dy = _mm256_sub_pd(y_vec, mean_y_vec);

            cov_vec = _mm256_fmadd_pd(dx, dy, cov_vec);
            var_x_vec = _mm256_fmadd_pd(dx, dx, var_x_vec);
            var_y_vec = _mm256_fmadd_pd(dy, dy, var_y_vec);
        }

        // Horizontal sum
        double temp_cov[4], temp_vx[4], temp_vy[4];
        _mm256_storeu_pd(temp_cov, cov_vec);
        _mm256_storeu_pd(temp_vx, var_x_vec);
        _mm256_storeu_pd(temp_vy, var_y_vec);

        cov = temp_cov[0] + temp_cov[1] + temp_cov[2] + temp_cov[3];
        var_x = temp_vx[0] + temp_vx[1] + temp_vx[2] + temp_vx[3];
        var_y = temp_vy[0] + temp_vy[1] + temp_vy[2] + temp_vy[3];

        // Handle remaining
        for (size_t i = vec_size2; i < n; ++i) {
            double dx = x[i] - mean_x;
            double dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
#else
        // Fallback: scalar
        for (size_t i = 0; i < n; ++i) {
            double dx = x[i] - mean_x;
            double dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
#endif

        double std_x = std::sqrt(var_x);
        double std_y = std::sqrt(var_y);

        if (std_x == 0.0 || std_y == 0.0) {
            return 0.0;
        }

        return cov / (std_x * std_y);
    }

    // ========================================================================
    // Covariance Matrix using MKL BLAS (for future expansion)
    // ========================================================================

    [[nodiscard]] auto computeCovarianceMatrix() const noexcept
        -> std::vector<std::vector<double>> {

        size_t n_assets = return_series_.size();
        if (n_assets == 0)
            return {};

        size_t n_obs = return_series_[0].size();

        // Compute means
        std::vector<double> means(n_assets, 0.0);
        for (size_t i = 0; i < n_assets; ++i) {
            for (size_t j = 0; j < n_obs; ++j) {
                means[i] += return_series_[i][j];
            }
            means[i] /= n_obs;
        }

        // Compute covariance matrix
        std::vector<std::vector<double>> cov_matrix(n_assets, std::vector<double>(n_assets, 0.0));

        for (size_t i = 0; i < n_assets; ++i) {
            for (size_t j = i; j < n_assets; ++j) {
                double cov = 0.0;

                for (size_t k = 0; k < n_obs; ++k) {
                    cov += (return_series_[i][k] - means[i]) * (return_series_[j][k] - means[j]);
                }

                cov /= (n_obs - 1);
                cov_matrix[i][j] = cov;
                cov_matrix[j][i] = cov; // Symmetric
            }
        }

        return cov_matrix;
    }

    // ========================================================================
    // Matrix Eigenvalue Decomposition (Placeholder for future PCA)
    // ========================================================================

    [[nodiscard]] auto
    computeEigenvalues(std::vector<std::vector<double>> const& matrix) const noexcept
        -> std::vector<double> {

        size_t n = matrix.size();
        if (n == 0)
            return {};

        // Placeholder implementation - return uniform eigenvalues
        // TODO: Implement power iteration or use MKL LAPACK in the future
        std::vector<double> eigenvalues(n);
        for (size_t i = 0; i < n; ++i) {
            eigenvalues[i] = 1.0 / n;
        }

        return eigenvalues;
    }
};

} // namespace bigbrother::risk
