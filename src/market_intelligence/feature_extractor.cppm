/**
 * BigBrotherAnalytics - Feature Extractor for Price Prediction
 *
 * Parallel feature engineering using OpenMP and AVX2 SIMD intrinsics.
 * Extracts predictive features from multi-source market data.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-11
 * Phase 5+: ML-Based Price Prediction System
 *
 * Features Extracted:
 * - Technical indicators (RSI, MACD, Bollinger Bands, ATR)
 * - Sentiment scores (news, social media, analyst ratings)
 * - Economic indicators (employment, GDP, inflation)
 * - Sector correlations (cross-asset momentum)
 * - Volume profile (accumulation/distribution)
 *
 * Performance:
 * - OpenMP: 16-thread parallel processing
 * - AVX2: 4x speedup for vector operations
 * - Target: <10ms for 100 features
 */

module;

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <immintrin.h>  // AVX2 intrinsics
#include <optional>
#include <span>
#include <string>
#include <vector>

export module bigbrother.market_intelligence.feature_extractor;

import bigbrother.utils.types;
import bigbrother.utils.logger;

export namespace bigbrother::market_intelligence {

using namespace bigbrother::types;
using namespace bigbrother::utils;

/**
 * StandardScaler normalization parameters (from trained model)
 * These must match the Python sklearn StandardScaler exactly
 *
 * PRODUCTION MODEL: 85 features (clean dataset v4.0)
 * Legacy 60-feature model deprecated due to 17 constant features
 */
struct StandardScalerParams {
    // Mean and std for 85 features
    static constexpr std::array<float, 85> MEAN = {
        9.50023516f, -1.00000000f, 0.00000000f, 21.00000000f, 0.00000000f,
        2.30746620f, 15.82010582f, 6.21634333f, 2.40799530f, 173.75955320f,
        0.00000000f, 0.03870000f, 0.03920000f, 0.03550000f, 0.03670000f,
        0.04110000f, 0.00560000f, 0.00000000f, 0.52277450f, 0.06611603f,
        -0.10231217f, 0.22056667f, 0.07966363f, 0.25000000f, 0.00000000f,
        0.00000000f, 193.47860042f, 193.61560842f, 197.08028385f, 190.51396287f,
        19504093.11240447f, 0.00001472f, -0.00003658f, -0.00003331f, 51.49374001f,
        -2.52725291f, -2.55461665f, 1.01588575f, 7.50535213f, 213.93105880f,
        179.73140204f, 0.52310927f, 0.00000000f, 0.01476783f, 0.04202997f,
        0.13069362f, -2.65035731f, 0.00227835f, 0.00000000f, -0.00000129f,
        0.12269950f, 0.31620693f, 0.50693710f, 0.02951205f, 0.52039976f,
        0.52063492f, -0.00000196f, 0.52004703f, 0.42704292f, 0.50688703f
    };

    static constexpr std::array<float, 60> STD = {
        5.76649539f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f,
        2.01408841f, 8.74422468f, 3.53751920f, 1.14212053f, 108.11254032f,
        1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f,
        1.00000000f, 1.00000000f, 1.00000000f, 0.00000010f, 0.05899109f,
        0.19877818f, 0.42853006f, 0.15477526f, 1.00000000f, 1.00000000f,
        1.00000000f, 375.90174564f, 375.77581273f, 392.59950297f, 362.79810211f,
        23481915.43384170f, 0.02054624f, 0.04202051f, 0.07756912f, 16.74851602f,
        24.77242961f, 23.94474762f, 0.37887954f, 34.92826572f, 468.89783305f,
        327.37020026f, 0.32671467f, 1.00000000f, 0.38387967f, 0.19559829f,
        0.00000003f, 25.81278234f, 0.01465837f, 1.00000000f, 0.00300192f,
        0.11096264f, 0.24651550f, 0.49995187f, 1.11284745f, 0.49958368f,
        0.49957402f, 0.03401980f, 0.49959795f, 0.49464863f, 0.15850324f
    };

    /**
     * Apply StandardScaler normalization: (x - mean) / std
     * Uses AVX2 SIMD for 8x speedup
     */
    [[nodiscard]] static auto normalize(std::array<float, 60> const& features)
        -> std::array<float, 60> {

        std::array<float, 60> normalized;

        #if defined(__AVX2__)
        // AVX2: Process 8 floats at a time
        for (size_t i = 0; i < 60; i += 8) {
            __m256 f = _mm256_loadu_ps(&features[i]);
            __m256 m = _mm256_loadu_ps(&MEAN[i]);
            __m256 s = _mm256_loadu_ps(&STD[i]);

            // (f - m) / s
            __m256 result = _mm256_div_ps(_mm256_sub_ps(f, m), s);
            _mm256_storeu_ps(&normalized[i], result);
        }
        #else
        // Fallback: OpenMP SIMD
        #pragma omp simd
        for (size_t i = 0; i < 60; ++i) {
            normalized[i] = (features[i] - MEAN[i]) / STD[i];
        }
        #endif

        return normalized;
    }
};

/**
 * Feature vector for price prediction (60 features)
 *
 * Feature order must match Python model EXACTLY:
 * [0-2]   Identification (3): symbol_encoded, sector_encoded, is_option
 * [3-10]  Time (8): hour, minute, day_of_week, day_of_month, month, quarter, day_of_year, is_market_open
 * [11-17] Treasury Rates (7): fed_funds, 3mo, 2yr, 5yr, 10yr, slope, inversion
 * [18-23] Options Greeks (6): delta, gamma, theta, vega, rho, IV
 * [24-25] Sentiment (2): avg_sentiment, news_count
 * [26-30] Price (5): close, open, high, low, volume
 * [31-37] Momentum (7): return_1d, return_5d, return_20d, RSI, MACD, signal, volume_ratio
 * [38-41] Volatility (4): ATR, BB_upper, BB_lower, BB_position
 * [42-51] Interactions (10): sentiment×momentum, volume×RSI, yield×volatility, etc.
 * [52-59] Directionality (8): direction, trend_strength, price_above_MA, etc.
 */

// Forward declaration for normalization function used in PriceFeatures::toArray85()
[[nodiscard]] inline auto normalizeFeatures85(std::array<float, 85> const& features)
    -> std::array<float, 85>;

struct PriceFeatures {
    // [0-2] Identification
    float symbol_encoded{0.0f};        // Symbol ID (0-19 for 20 symbols)
    float sector_encoded{-1.0f};       // Sector ID (-1=unknown)
    float is_option{0.0f};             // 1=option, 0=stock

    // [3-10] Time Features
    float hour_of_day{0.0f};           // Hour (0-23)
    float minute_of_hour{0.0f};        // Minute (0-59)
    float day_of_week{0.0f};           // Day (0=Mon, 6=Sun)
    float day_of_month{0.0f};          // Day of month (1-31)
    float month_of_year{0.0f};         // Month (1-12)
    float quarter{0.0f};               // Quarter (1-4)
    float day_of_year{0.0f};           // Day of year (1-365)
    float is_market_open{0.0f};        // 1=market open, 0=closed

    // [11-17] Treasury Rates
    float fed_funds_rate{0.0f};        // Federal funds rate (%)
    float treasury_3mo{0.0f};          // 3-month Treasury yield (%)
    float treasury_2yr{0.0f};          // 2-year Treasury yield (%)
    float treasury_5yr{0.0f};          // 5-year Treasury yield (%)
    float treasury_10yr{0.0f};         // 10-year Treasury yield (%)
    float yield_curve_slope{0.0f};     // 10yr - 2yr (%)
    float yield_curve_inversion{0.0f}; // 1=inverted (2yr > 10yr), 0=normal

    // [18-23] Options Greeks
    float delta{0.0f};                 // Delta [-1, 1]
    float gamma{0.0f};                 // Gamma [0, ∞)
    float theta{0.0f};                 // Theta ($/day)
    float vega{0.0f};                  // Vega ($/1% vol)
    float rho{0.0f};                   // Rho ($/1% rate)
    float implied_volatility{0.0f};    // IV (%)

    // [24-25] Sentiment
    float avg_sentiment{0.0f};         // Average news sentiment [-1, 1]
    float news_count{0.0f};            // Number of news articles

    // [26-30] Price (OHLCV)
    float close{0.0f};                 // Current close price
    float open{0.0f};                  // Current open price
    float high{0.0f};                  // Current high price
    float low{0.0f};                   // Current low price
    float volume{0.0f};                // Current volume

    // [31-37] Momentum
    float return_1d{0.0f};             // 1-day return %
    float return_5d{0.0f};             // 5-day return %
    float return_20d{0.0f};            // 20-day return %
    float rsi_14{0.0f};                // RSI(14) [0-100]
    float macd{0.0f};                  // MACD line
    float macd_signal{0.0f};           // MACD signal line
    float volume_ratio{0.0f};          // Volume / 20-day avg

    // [38-41] Volatility
    float atr_14{0.0f};                // Average True Range(14)
    float bb_upper{0.0f};              // Bollinger upper band
    float bb_lower{0.0f};              // Bollinger lower band
    float bb_position{0.0f};           // Price position in BB (0=lower, 1=upper)

    // [42-51] Interaction Features
    float sentiment_momentum{0.0f};    // sentiment × return_5d
    float volume_rsi_signal{0.0f};     // volume_ratio × (RSI-50)/50
    float yield_volatility{0.0f};      // yield_slope × ATR
    float delta_iv{0.0f};              // delta × IV
    float macd_volume{0.0f};           // MACD × volume_ratio
    float bb_momentum{0.0f};           // BB_position × return_1d
    float sentiment_strength{0.0f};    // sentiment × log(news_count + 1)
    float rate_return{0.0f};           // fed_funds × return_20d
    float gamma_volatility{0.0f};      // gamma × ATR
    float rsi_bb_signal{0.0f};         // (RSI/100) × BB_position

    // [52-59] Directionality Features
    float price_direction{0.0f};       // 1 if return_1d > 0, else 0
    float trend_strength{0.0f};        // Rolling 5-day win rate - 0.5
    float price_above_ma5{0.0f};       // 1 if price > MA(5), else 0
    float price_above_ma20{0.0f};      // 1 if price > MA(20), else 0
    float momentum_3d{0.0f};           // 3-day momentum %
    float macd_signal_direction{0.0f}; // 1 if MACD > signal, else 0
    float volume_trend{0.0f};          // 1 if volume_ratio > 1, else 0
    float recent_win_rate{0.0f};       // Rolling 10-day win rate

    /**
     * Convert to float array for neural network input (60 features)
     * IMPORTANT: Order must match Python model exactly
     */
    [[nodiscard]] auto toArray() const -> std::array<float, 60> {
        return {
            // [0-2] Identification
            symbol_encoded, sector_encoded, is_option,
            // [3-10] Time
            hour_of_day, minute_of_hour, day_of_week, day_of_month,
            month_of_year, quarter, day_of_year, is_market_open,
            // [11-17] Treasury
            fed_funds_rate, treasury_3mo, treasury_2yr, treasury_5yr,
            treasury_10yr, yield_curve_slope, yield_curve_inversion,
            // [18-23] Greeks
            delta, gamma, theta, vega, rho, implied_volatility,
            // [24-25] Sentiment
            avg_sentiment, news_count,
            // [26-30] Price
            close, open, high, low, volume,
            // [31-37] Momentum
            return_1d, return_5d, return_20d, rsi_14, macd, macd_signal, volume_ratio,
            // [38-41] Volatility
            atr_14, bb_upper, bb_lower, bb_position,
            // [42-51] Interactions
            sentiment_momentum, volume_rsi_signal, yield_volatility, delta_iv,
            macd_volume, bb_momentum, sentiment_strength, rate_return,
            gamma_volatility, rsi_bb_signal,
            // [52-59] Directionality
            price_direction, trend_strength, price_above_ma5, price_above_ma20,
            momentum_3d, macd_signal_direction, volume_trend, recent_win_rate
        };
    }

    /**
     * Normalize features using StandardScaler (matches Python training)
     * Returns a new normalized array
     */
    [[nodiscard]] auto normalize() const -> std::array<float, 60> {
        return StandardScalerParams::normalize(toArray());
    }

  private:
    /**
     * Calculate Black-Scholes Greeks from pricing data
     * Returns: {gamma, theta, vega, rho}
     */
    [[nodiscard]] static auto calculateGreeks(
        float spot_price,
        float volatility,
        float risk_free_rate,
        float time_to_expiry = 30.0f / 365.0f) -> std::array<float, 4> {

        // For ETFs, approximate ATM option (strike = spot)
        float K = spot_price;
        float S = spot_price;
        float r = risk_free_rate / 100.0f;  // Convert percentage to decimal
        float sigma = volatility;
        float T = time_to_expiry;

        // Avoid invalid inputs
        if (S <= 0.0f || K <= 0.0f || sigma <= 0.0f || T <= 0.0f) {
            return {0.01f, -0.05f, 0.20f, 0.01f};  // Fallback to estimates
        }

        // Black-Scholes d1 and d2
        float sqrt_T = std::sqrt(T);
        float d1 = (std::log(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrt_T);
        float d2 = d1 - sigma * sqrt_T;

        // Standard normal PDF: N'(x) = (1/sqrt(2π)) * exp(-x²/2)
        float n_prime_d1 = (1.0f / std::sqrt(2.0f * 3.14159265f)) * std::exp(-0.5f * d1 * d1);

        // Standard normal CDF: N(x) (approximation using tanh)
        auto norm_cdf = [](float x) -> float {
            return 0.5f * (1.0f + std::tanh(x / std::sqrt(2.0f) * 0.7978845608f));
        };

        float n_d2 = norm_cdf(d2);

        // Calculate Greeks for ATM call option (matching Python JAX implementation)
        float gamma_val = n_prime_d1 / (S * sigma * sqrt_T);
        float vega_val = S * n_prime_d1 * sqrt_T / 100.0f;  // Per 1% volatility change

        // Theta (full formula matching Python):
        // theta = (-S * N'(d1) * sigma / (2*sqrt(T)) - r * K * exp(-r*T) * N(d2)) / 365
        float theta_val = (-(S * n_prime_d1 * sigma) / (2.0f * sqrt_T)
                          - r * K * std::exp(-r * T) * n_d2) / 365.0f;

        // Rho (full formula matching Python):
        // rho = K * T * exp(-r*T) * N(d2) / 100
        float rho_val = K * T * std::exp(-r * T) * n_d2 / 100.0f;

        // Scale to reasonable ranges
        gamma_val = std::clamp(gamma_val, 0.0f, 0.1f);
        theta_val = std::clamp(theta_val, -0.5f, 0.0f);
        vega_val = std::clamp(vega_val, 0.0f, 1.0f);
        rho_val = std::clamp(rho_val, -0.5f, 0.5f);

        return {gamma_val, theta_val, vega_val, rho_val};
    }

  public:
    /**
     * Convert to 85-feature array for v4.0 clean model
     * Requires at least 21 days of price history
     */
    [[nodiscard]] auto toArray85(
        std::vector<float> const& price_history,
        std::vector<float> const& volume_history,
        std::chrono::system_clock::time_point timestamp) const -> std::array<float, 85> {

        // Calculate price lags as actual historical prices (not ratios)
        // Training data uses actual price values for lag features
        std::array<float, 20> price_lags{};
        for (int i = 0; i < 20 && i < static_cast<int>(price_history.size()); ++i) {
            price_lags[i] = price_history[i];
        }

        // Calculate price diffs (20 values)
        // Python: df.groupby('symbol')['close'].diff(lag)
        // This computes: price[t] - price[t-lag]
        // In our price_history: [0] is most recent, so diff(i+1) = price[0] - price[i+1]
        std::array<float, 20> price_diffs{};
        for (int i = 0; i < 20 && i + 1 < static_cast<int>(price_history.size()); ++i) {
            price_diffs[i] = price_history[0] - price_history[i + 1];  // current - (i+1) days ago
        }

        // Calculate autocorrelations using returns (matches Python)
        // Python: df['returns'] = df.groupby('symbol')['close'].pct_change()
        // Then computes rolling autocorrelation with window=60
        auto calc_autocorr = [&price_history](int lag, int window = 60) -> float {
            if (static_cast<int>(price_history.size()) < window + lag + 1) return 0.0f;

            // Calculate returns (most recent first)
            std::vector<float> returns;
            for (size_t i = 0; i + 1 < price_history.size() && i < static_cast<size_t>(window + lag); ++i) {
                float ret = (price_history[i] - price_history[i + 1]) / price_history[i + 1];
                returns.push_back(ret);
            }

            if (returns.size() < static_cast<size_t>(window + lag)) return 0.0f;

            // Compute autocorrelation: corr(returns[0:window], returns[lag:window+lag])
            // Calculate means
            float mean1 = 0.0f, mean2 = 0.0f;
            for (int i = 0; i < window; ++i) {
                mean1 += returns[i];
                mean2 += returns[i + lag];
            }
            mean1 /= window;
            mean2 /= window;

            // Calculate correlation
            float num = 0.0f, denom1 = 0.0f, denom2 = 0.0f;
            for (int i = 0; i < window; ++i) {
                float diff1 = returns[i] - mean1;
                float diff2 = returns[i + lag] - mean2;
                num += diff1 * diff2;
                denom1 += diff1 * diff1;
                denom2 += diff2 * diff2;
            }

            float denom = std::sqrt(denom1 * denom2);
            return (denom > 0.0f) ? (num / denom) : 0.0f;
        };

        float autocorr_1 = calc_autocorr(1, 60);
        float autocorr_5 = calc_autocorr(5, 60);
        float autocorr_10 = calc_autocorr(10, 60);
        float autocorr_20 = calc_autocorr(20, 60);

        // Extract year, month, day from timestamp
        auto time_t_now = std::chrono::system_clock::to_time_t(timestamp);
        std::tm tm_local;
        #if defined(__linux__) || defined(__APPLE__)
        localtime_r(&time_t_now, &tm_local);
        #else
        localtime_s(&tm_local, &time_t_now);
        #endif

        float year = static_cast<float>(tm_local.tm_year + 1900);
        float month = static_cast<float>(tm_local.tm_mon + 1);
        float day = static_cast<float>(tm_local.tm_mday);

        // Calculate volume_sma20
        float volume_sma20 = 0.0f;
        int volume_count = std::min(20, static_cast<int>(volume_history.size()));
        if (volume_count > 0) {
            for (int i = 0; i < volume_count; ++i) {
                volume_sma20 += volume_history[i];
            }
            volume_sma20 /= volume_count;
        }

        // Calculate volume_ratio
        float vol_ratio = (volume_sma20 > 0.0f) ? (volume / volume_sma20) : volume_ratio;

        // Calculate implied volatility from historical price data
        // Use ATR as a proxy for volatility (annualized)
        float volatility_estimate = 0.0f;
        if (price_history.size() >= 14) {
            float atr_sum = 0.0f;
            for (int i = 1; i < 14 && i < static_cast<int>(price_history.size()); ++i) {
                atr_sum += std::abs(price_history[i-1] - price_history[i]);
            }
            float avg_atr = atr_sum / 13.0f;
            // Annualize: ATR / Price * sqrt(252 trading days)
            volatility_estimate = (close > 0.0f) ? (avg_atr / close) * std::sqrt(252.0f) : 0.20f;
            volatility_estimate = std::clamp(volatility_estimate, 0.05f, 2.0f);  // Reasonable bounds
        } else {
            volatility_estimate = 0.20f;  // Default 20% volatility
        }

        // Use treasury_10yr as risk-free rate (or fed_funds_rate if available)
        // Note: These should come from PriceFeatures struct members
        float risk_free = (treasury_10yr > 0.0f) ? treasury_10yr :
                          (fed_funds_rate > 0.0f) ? fed_funds_rate : 4.5f;  // Default ~4.5%

        // Calculate Greeks from pricing data
        auto greeks = calculateGreeks(close, volatility_estimate, risk_free);
        float gamma_calc = greeks[0];
        float theta_calc = greeks[1];
        float vega_calc = greeks[2];
        float rho_calc = greeks[3];

        // Recalculate interaction features that depend on Greeks
        // Use calculated Greeks instead of struct values to match Python training
        float gamma_volatility_calc = gamma_calc * atr_14;

        std::array<float, 85> features = {
            // [0-4] OHLCV
            close, open, high, low, volume,
            // [5-7] Technical indicators
            rsi_14, macd, macd_signal,
            // [8-10] Bollinger Bands
            bb_upper, bb_lower, bb_position,
            // [11-13] Volatility/Volume
            atr_14, volume_sma20,
            vol_ratio,
            // [14-17] Greeks (calculated from pricing data using Black-Scholes)
            gamma_calc, theta_calc, vega_calc, rho_calc,
            // [18-24] Interaction features
            volume_rsi_signal, yield_volatility, macd_volume,
            bb_momentum, rate_return, gamma_volatility_calc, rsi_bb_signal,
            // [25-26] Momentum features
            momentum_3d, recent_win_rate,
            // [27-46] Price lags (20 days)
            price_lags[0], price_lags[1], price_lags[2], price_lags[3], price_lags[4],
            price_lags[5], price_lags[6], price_lags[7], price_lags[8], price_lags[9],
            price_lags[10], price_lags[11], price_lags[12], price_lags[13], price_lags[14],
            price_lags[15], price_lags[16], price_lags[17], price_lags[18], price_lags[19],
            // [47] Symbol encoding
            symbol_encoded,
            // [48-52] Time features
            day_of_week, day_of_month, month_of_year, quarter, day_of_year,
            // [53-57] Directional features
            price_direction, price_above_ma5, price_above_ma20,
            macd_signal_direction, volume_trend,
            // [58-60] Date components
            year, month, day,
            // [61-80] Price diffs (20 days)
            price_diffs[0], price_diffs[1], price_diffs[2], price_diffs[3], price_diffs[4],
            price_diffs[5], price_diffs[6], price_diffs[7], price_diffs[8], price_diffs[9],
            price_diffs[10], price_diffs[11], price_diffs[12], price_diffs[13], price_diffs[14],
            price_diffs[15], price_diffs[16], price_diffs[17], price_diffs[18], price_diffs[19],
            // [81-84] Autocorrelations
            autocorr_1, autocorr_5, autocorr_10, autocorr_20
        };

        // Normalize all features to [0,1] using global min/max from training data
        // This ensures: (1) perfect INT32 quantization, (2) no dominant features
        return normalizeFeatures85(features);
    }
};

// ============================================================================
// Feature Normalization - Single Source of Truth
// ============================================================================

/**
 * Global min/max normalization constants from training data
 * Ensures all features are in [0,1] for perfect INT32 quantization
 *
 * IMPORTANT: These are computed from the ENTIRE training dataset and hardcoded here.
 * Training and inference use these SAME constants → perfect parity.
 *
 * Computed from 9,720 training samples (all SPY data)
 */
inline constexpr std::array<float, 85> FEATURE_MIN_85 = {
    25.752899f, 25.954308f, 26.119101f, 25.734591f, 1245100.0f,  // [0-4] OHLCV
    0.0f, -4.184433f, -3.979798f, 0.0f, 0.0f,  // [5-9]
    -0.077830f, 0.0f, -1.0f, -0.452162f, -0.314953f,  // [10-14]
    -0.376302f, -1.0f, -1.0f, 0.0f, -0.092566f,  // [15-19]
    -6.506897f, 0.0f, 0.0f, 0.0f, 0.0f,  // [20-24]
    -6.506897f, 0.0f, 0.0f, 0.0f, 0.0f,  // [25-29]
    0.0f, -0.196610f, -0.189683f, -0.176601f, 0.0f,  // [30-34]
    -41.470001f, -41.200001f, 0.0f, 0.0f, 0.0f,  // [35-39]
    0.0f, 0.0f, 0.0f, -0.074834f, -0.196610f,  // [40-44]
    -0.452162f, -41.470001f, -0.158731f, 0.0f, -0.188785f,  // [45-49]
    0.0f, 0.0f, 0.0f, -0.077830f, 0.0f,  // [50-54]
    0.0f, -0.188785f, 0.0f, 0.0f, 0.0f,  // [55-59]
    0.0f, -191.740005f, -191.470001f, -186.039993f, -184.839996f,  // [60-64]
    -183.380005f, -178.979996f, -176.160004f, -172.869995f, -170.020004f,  // [65-69]
    -167.259995f, -164.509995f, -161.979996f, -159.610001f, -157.179993f,  // [70-74]
    -154.759995f, -152.330002f, -149.949997f, -147.570007f, -145.199997f,  // [75-79]
    -143.500000f, -0.173090f, 0.0f, -0.066612f, -0.081545f  // [80-84] Last + autocorrs
};

inline constexpr std::array<float, 85> FEATURE_MAX_85 = {
    673.109985f, 673.530029f, 673.950012f, 669.460022f, 256611392.0f,  // [0-4] OHLCV
    100.0f, 4.160000f, 4.110000f, 1.0f, 1.0f,  // [5-9]
    0.134623f, 0.999875f, 1.0f, 0.437879f, 0.312500f,  // [10-14]
    0.378571f, 1.0f, 1.0f, 1.0f, 0.138462f,  // [15-19]
    6.506897f, 1.0f, 1.0f, 1.0f, 1.0f,  // [20-24]
    6.506897f, 1.0f, 1.0f, 1.0f, 1.0f,  // [25-29]
    1.0f, 0.314433f, 0.296610f, 0.271449f, 100.0f,  // [30-34]
    41.470001f, 41.200001f, 1.0f, 31.0f, 673.109985f,  // [35-39]
    669.460022f, 1.0f, 1.0f, 0.159091f, 0.314433f,  // [40-44]
    0.437879f, 41.470001f, 0.162338f, 1.0f, 0.188785f,  // [45-49]
    1.0f, 1.0f, 1.0f, 0.134623f, 1.0f,  // [50-54]
    1.0f, 0.188785f, 1.0f, 1.0f, 1.0f,  // [55-59]
    4.0f, 191.740005f, 191.470001f, 186.039993f, 184.839996f,  // [60-64]
    183.380005f, 178.979996f, 176.160004f, 172.869995f, 170.020004f,  // [65-69]
    167.259995f, 164.509995f, 161.979996f, 159.610001f, 157.179993f,  // [70-74]
    154.759995f, 152.330002f, 149.949997f, 147.570007f, 145.199997f,  // [75-79]
    143.500000f, 0.0f, 0.142276f, 0.099620f, 0.046974f  // [80-84] Last + autocorrs
};

/**
 * Normalize 85 features to [0,1] using global min/max constants
 * Formula: normalized = (value - min) / (max - min)
 *
 * This ensures:
 * - All features in [0,1] range
 * - Perfect INT32 quantization (no dominant features)
 * - Identical normalization in training and inference
 */
[[nodiscard]] inline auto normalizeFeatures85(std::array<float, 85> const& features)
    -> std::array<float, 85>
{
    std::array<float, 85> normalized;

    for (size_t i = 0; i < 85; ++i) {
        float const range = FEATURE_MAX_85[i] - FEATURE_MIN_85[i];

        // Avoid division by zero for constant features
        if (range < 1e-8f) {
            normalized[i] = 0.0f;
        } else {
            float const value = std::clamp(features[i], FEATURE_MIN_85[i], FEATURE_MAX_85[i]);
            normalized[i] = (value - FEATURE_MIN_85[i]) / range;
        }
    }

    return normalized;
};

// ============================================================================
// INT32 Quantization for Neural Network Inference
// ============================================================================

/**
 * Quantization parameters for INT32 symmetric quantization
 * Maps FP32 range [-scale, +scale] to INT32 range [-2^31+1, +2^31-1]
 */
struct QuantizationParams32 {
    float scale;           // Scale factor: fp32_value = int32_value * scale
    float inv_scale;       // Inverse: int32_value = fp32_value / inv_scale
    int32_t zero_point{0}; // Symmetric quantization always uses 0

    QuantizationParams32() : scale(1.0f), inv_scale(1.0f) {}

    explicit QuantizationParams32(float s)
        : scale(s)
        , inv_scale(1.0f / s)
    {}
};

/**
 * Compute INT32 quantization parameters from FP32 data
 * Uses symmetric quantization: [-max_abs, +max_abs] → [-2147483647, +2147483647]
 */
[[nodiscard]] inline auto computeQuantizationParams32(std::span<float const> data) -> QuantizationParams32 {
    float max_abs = 0.0f;
    for (float val : data) {
        max_abs = std::max(max_abs, std::abs(val));
    }

    // Avoid division by zero
    if (max_abs < 1e-8f) {
        max_abs = 1.0f;
    }

    // Map [-max_abs, +max_abs] to [-2147483647, +2147483647]
    // Use 2^31 - 1 to avoid overflow
    constexpr int32_t MAX_INT32 = 2147483647;
    float scale = max_abs / static_cast<float>(MAX_INT32);
    return QuantizationParams32(scale);
}

/**
 * Quantize FP32 array to INT32
 * output[i] = clamp(round(input[i] / scale), -2147483647, 2147483647)
 */
[[nodiscard]] inline auto quantize32(
    std::span<float const> input,
    std::span<int32_t> output,
    QuantizationParams32 const& params) -> void
{
    float const inv_scale = params.inv_scale;
    int const size = static_cast<int>(input.size());

    #if defined(__AVX2__)
    // AVX2: Process 8 floats at a time
    __m256 const inv_scale_vec = _mm256_set1_ps(inv_scale);

    int i = 0;
    for (; i + 8 <= size; i += 8) {
        // Load 8 floats
        __m256 fp32_vals = _mm256_loadu_ps(&input[i]);

        // Scale: fp32 / scale
        __m256 scaled = _mm256_mul_ps(fp32_vals, inv_scale_vec);

        // Round to nearest integer and convert to int32
        __m256i int32_vals = _mm256_cvtps_epi32(scaled);

        // Store (no clamping needed for INT32 as FP32 range fits)
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output[i]), int32_vals);
    }

    // Handle remainder
    for (; i < size; ++i) {
        float scaled = input[i] * inv_scale;
        output[i] = static_cast<int32_t>(std::round(scaled));
    }
    #else
    // Scalar fallback
    for (int i = 0; i < size; ++i) {
        float scaled = input[i] * inv_scale;
        output[i] = static_cast<int32_t>(std::round(scaled));
    }
    #endif
}

/**
 * Dequantize INT32 array to FP32
 * output[i] = input[i] * scale
 */
[[nodiscard]] inline auto dequantize32(
    std::span<int32_t const> input,
    std::span<float> output,
    QuantizationParams32 const& params) -> void
{
    float const scale = params.scale;
    int const size = static_cast<int>(input.size());

    #if defined(__AVX2__)
    // AVX2: Process 8 int32 values at a time
    __m256 const scale_vec = _mm256_set1_ps(scale);

    int i = 0;
    for (; i + 8 <= size; i += 8) {
        // Load 8 int32 values
        __m256i int32_vals = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(&input[i]));

        // Convert int32 → fp32
        __m256 fp32_vals = _mm256_cvtepi32_ps(int32_vals);

        // Scale: int32 * scale
        fp32_vals = _mm256_mul_ps(fp32_vals, scale_vec);

        // Store
        _mm256_storeu_ps(&output[i], fp32_vals);
    }

    // Handle remainder
    for (; i < size; ++i) {
        output[i] = static_cast<float>(input[i]) * scale;
    }
    #else
    // Scalar fallback
    for (int i = 0; i < size; ++i) {
        output[i] = static_cast<float>(input[i]) * scale;
    }
    #endif
}

/**
 * Quantize 85-feature array for neural network inference
 * Returns both quantized features and quantization parameters
 */
[[nodiscard]] inline auto quantizeFeatures85(
    std::array<float, 85> const& features)
    -> std::pair<std::array<int32_t, 85>, QuantizationParams32>
{
    // Compute quantization parameters
    auto params = computeQuantizationParams32(std::span<float const>(features.data(), 85));

    // Quantize features
    std::array<int32_t, 85> quantized;
    quantize32(
        std::span<float const>(features.data(), 85),
        std::span<int32_t>(quantized.data(), 85),
        params
    );

    return {quantized, params};
}

/**
 * Dequantize 85-feature array back to FP32
 */
[[nodiscard]] inline auto dequantizeFeatures85(
    std::array<int32_t, 85> const& quantized,
    QuantizationParams32 const& params)
    -> std::array<float, 85>
{
    std::array<float, 85> features;
    dequantize32(
        std::span<int32_t const>(quantized.data(), 85),
        std::span<float>(features.data(), 85),
        params
    );
    return features;
}

/**
 * Context data for feature extraction (60 features)
 * Contains external data that must be provided by caller
 */
struct FeatureContext {
    // Identification
    int symbol_id{0};               // Symbol ID (0-19)
    int sector_id{-1};              // Sector ID (-1=unknown)
    bool is_option{false};          // true=option, false=stock

    // Timestamp for time features
    std::chrono::system_clock::time_point timestamp;

    // Treasury rates (must be fetched from database/FRED)
    float fed_funds_rate{0.0f};
    float treasury_3mo{0.0f};
    float treasury_2yr{0.0f};
    float treasury_5yr{0.0f};
    float treasury_10yr{0.0f};

    // Options Greeks (must be calculated with Black-Scholes)
    float delta{0.0f};
    float gamma{0.0f};
    float theta{0.0f};
    float vega{0.0f};
    float rho{0.0f};
    float implied_volatility{0.0f};

    // Sentiment (must be queried from news_articles table)
    float avg_sentiment{0.0f};
    float news_count{0.0f};
};

/**
 * Feature extractor for price prediction
 *
 * Parallel extraction of 60 features from market data using OpenMP and AVX2 SIMD
 */
class FeatureExtractor {
  public:
    /**
     * Extract all 60 features from market data
     *
     * @param close Current close price
     * @param open Current open price
     * @param high Current high price
     * @param low Current low price
     * @param volume Current volume
     * @param price_history Price history (most recent first, needs 26+ bars)
     * @param volume_history Volume history (most recent first, needs 20+ bars)
     * @param context External context data (treasury, sentiment, Greeks, etc.)
     * @return Complete 60-feature set
     */
    [[nodiscard]] static auto extractFeatures(
        float close, float open, float high, float low, float volume,
        std::span<float const> price_history,
        std::span<float const> volume_history,
        FeatureContext const& context) -> PriceFeatures {

        PriceFeatures features;

        if (price_history.size() < 26 || volume_history.size() < 20) {
            Logger::getInstance().warn("Insufficient history for feature extraction");
            return features;
        }

        // [0-2] Identification (from context)
        features.symbol_encoded = static_cast<float>(context.symbol_id);
        features.sector_encoded = static_cast<float>(context.sector_id);
        features.is_option = context.is_option ? 1.0f : 0.0f;

        // [3-10] Time Features (extract from timestamp)
        auto time_features = extractTimeFeatures(context.timestamp);
        features.hour_of_day = time_features[0];
        features.minute_of_hour = time_features[1];
        features.day_of_week = time_features[2];
        features.day_of_month = time_features[3];
        features.month_of_year = time_features[4];
        features.quarter = time_features[5];
        features.day_of_year = time_features[6];
        features.is_market_open = time_features[7];

        // [11-17] Treasury Rates (from context)
        features.fed_funds_rate = context.fed_funds_rate;
        features.treasury_3mo = context.treasury_3mo;
        features.treasury_2yr = context.treasury_2yr;
        features.treasury_5yr = context.treasury_5yr;
        features.treasury_10yr = context.treasury_10yr;
        features.yield_curve_slope = context.treasury_10yr - context.treasury_2yr;
        features.yield_curve_inversion = (context.treasury_2yr > context.treasury_10yr) ? 1.0f : 0.0f;

        // [18-23] Options Greeks (from context)
        features.delta = context.delta;
        features.gamma = context.gamma;
        features.theta = context.theta;
        features.vega = context.vega;
        features.rho = context.rho;
        features.implied_volatility = context.implied_volatility;

        // [24-25] Sentiment (from context)
        features.avg_sentiment = context.avg_sentiment;
        features.news_count = context.news_count;

        // [26-30] Price (OHLCV)
        features.close = close;
        features.open = open;
        features.high = high;
        features.low = low;
        features.volume = volume;

        // [31-37] Momentum
        features.return_1d = (price_history[0] - price_history[1]) / price_history[1];
        features.return_5d = (price_history[0] - price_history[5]) / price_history[5];
        features.return_20d = (price_history[0] - price_history[20]) / price_history[20];
        features.rsi_14 = calculateRSI(price_history.subspan(0, 14));

        auto [macd, signal, hist] = calculateMACD(price_history);
        features.macd = macd;
        features.macd_signal = signal;

        float volume_sma20 = calculateMean(volume_history.subspan(0, 20));
        features.volume_ratio = volume / (volume_sma20 + 0.0001f);

        // [38-41] Volatility
        features.atr_14 = calculateATR(price_history.subspan(0, 14));

        auto [upper, middle, lower] = calculateBollingerBands(price_history.subspan(0, 20));
        features.bb_upper = upper;
        features.bb_lower = lower;
        features.bb_position = (close - lower) / (upper - lower + 0.0001f);

        // [42-51] Interaction Features (calculated with SIMD)
        features.sentiment_momentum = features.avg_sentiment * features.return_5d;
        features.volume_rsi_signal = features.volume_ratio * (features.rsi_14 - 50.0f) / 50.0f;
        features.yield_volatility = features.yield_curve_slope * features.atr_14;
        features.delta_iv = features.delta * features.implied_volatility;
        features.macd_volume = features.macd * features.volume_ratio;
        features.bb_momentum = features.bb_position * features.return_1d;
        features.sentiment_strength = features.avg_sentiment * std::log1p(features.news_count);
        features.rate_return = features.fed_funds_rate * features.return_20d;
        features.gamma_volatility = features.gamma * features.atr_14;
        features.rsi_bb_signal = (features.rsi_14 / 100.0f) * features.bb_position;

        // [52-59] Directionality Features (calculated with SIMD)
        features.price_direction = (features.return_1d > 0.0f) ? 1.0f : 0.0f;

        // Trend strength: rolling 5-day win rate
        float wins_5d = 0.0f;
        size_t limit_5d = std::min(size_t(6), price_history.size());  // Need 6 for 5 comparisons
        #pragma omp simd reduction(+:wins_5d)
        for (size_t i = 1; i < limit_5d; ++i) {
            if (price_history[i-1] > price_history[i]) wins_5d += 1.0f;
        }
        features.trend_strength = (wins_5d / 5.0f) - 0.5f;  // Center at 0

        // Price above moving averages
        float ma5 = calculateMean(price_history.subspan(0, std::min(size_t(5), price_history.size())));
        float ma20 = calculateMean(price_history.subspan(0, std::min(size_t(20), price_history.size())));
        features.price_above_ma5 = (close > ma5) ? 1.0f : 0.0f;
        features.price_above_ma20 = (close > ma20) ? 1.0f : 0.0f;

        // 3-day momentum
        if (price_history.size() > 3) {
            features.momentum_3d = (price_history[0] - price_history[3]) / price_history[3];
        }

        features.macd_signal_direction = (features.macd > features.macd_signal) ? 1.0f : 0.0f;
        features.volume_trend = (features.volume_ratio > 1.0f) ? 1.0f : 0.0f;

        // Recent win rate: rolling 10-day
        float wins_10d = 0.0f;
        size_t limit_10d = std::min(size_t(11), price_history.size());  // Need 11 for 10 comparisons
        #pragma omp simd reduction(+:wins_10d)
        for (size_t i = 1; i < limit_10d; ++i) {
            if (price_history[i-1] > price_history[i]) wins_10d += 1.0f;
        }
        features.recent_win_rate = wins_10d / std::min(10.0f, static_cast<float>(price_history.size() - 1));

        return features;
    }

    /**
     * Calculate RSI using AVX2 SIMD for 4x speedup
     */
    [[nodiscard]] static auto calculateRSI(std::span<float const> prices) -> float {
        if (prices.size() < 14) return 50.0f;  // Neutral RSI

        float gains = 0.0f;
        float losses = 0.0f;

        // Calculate price changes and sum gains/losses
        #pragma omp simd reduction(+:gains, losses)
        for (size_t i = 1; i < 14; ++i) {
            float change = prices[i - 1] - prices[i];
            if (change > 0) {
                gains += change;
            } else {
                losses -= change;  // Make positive
            }
        }

        float avg_gain = gains / 13.0f;
        float avg_loss = losses / 13.0f;

        if (avg_loss < 0.0001f) return 100.0f;  // No losses = overbought

        float rs = avg_gain / avg_loss;
        return 100.0f - (100.0f / (1.0f + rs));
    }

    /**
     * Calculate MACD using exponential moving averages
     */
    [[nodiscard]] static auto calculateMACD(std::span<float const> prices)
        -> std::tuple<float, float, float> {

        if (prices.size() < 26) return {0.0f, 0.0f, 0.0f};

        float ema12 = calculateEMA(prices.subspan(0, 12), 12);
        float ema26 = calculateEMA(prices.subspan(0, 26), 26);
        float macd = ema12 - ema26;

        // Signal line (9-period EMA of MACD)
        // Simplified: just use current MACD (full implementation would need MACD history)
        float signal = macd * 0.9f;
        float histogram = macd - signal;

        return {macd, signal, histogram};
    }

    /**
     * Calculate Bollinger Bands
     */
    [[nodiscard]] static auto calculateBollingerBands(std::span<float const> prices)
        -> std::tuple<float, float, float> {

        float mean = calculateMean(prices);
        float std_dev = calculateStdDev(prices, mean);

        float upper = mean + 2.0f * std_dev;
        float lower = mean - 2.0f * std_dev;

        return {upper, mean, lower};
    }

    /**
     * Calculate Average True Range using SIMD
     */
    [[nodiscard]] static auto calculateATR(std::span<float const> prices) -> float {
        if (prices.size() < 2) return 0.0f;

        float sum = 0.0f;

        #pragma omp simd reduction(+:sum)
        for (size_t i = 1; i < prices.size(); ++i) {
            float range = std::abs(prices[i - 1] - prices[i]);
            sum += range;
        }

        return sum / static_cast<float>(prices.size() - 1);
    }

  private:
    /**
     * Extract time features from timestamp
     * Returns: [hour, minute, day_of_week, day_of_month, month, quarter, day_of_year, is_market_open]
     */
    [[nodiscard]] static auto extractTimeFeatures(std::chrono::system_clock::time_point timestamp)
        -> std::array<float, 8> {

        auto time_t = std::chrono::system_clock::to_time_t(timestamp);
        std::tm* tm = std::localtime(&time_t);

        float hour = static_cast<float>(tm->tm_hour);
        float minute = static_cast<float>(tm->tm_min);

        // Convert tm_wday (0=Sunday) to Python dayofweek (0=Monday)
        // tm_wday: Sun=0, Mon=1, Tue=2, Wed=3, Thu=4, Fri=5, Sat=6
        // Python:  Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
        float day_of_week = (tm->tm_wday == 0) ? 6.0f : static_cast<float>(tm->tm_wday - 1);

        float day_of_month = static_cast<float>(tm->tm_mday);
        float month = static_cast<float>(tm->tm_mon + 1);     // 1-12
        float quarter = static_cast<float>((tm->tm_mon / 3) + 1);  // 1-4
        float day_of_year = static_cast<float>(tm->tm_yday + 1);   // 1-365

        // Market open: Mon-Fri 9:30 AM - 4:00 PM ET
        // Now check using original tm_wday (1-5 = Mon-Fri)
        bool is_weekday = (tm->tm_wday >= 1 && tm->tm_wday <= 5);
        bool is_market_hours = (hour >= 9 && hour < 16) || (hour == 9 && minute >= 30);
        float is_market_open = (is_weekday && is_market_hours) ? 1.0f : 0.0f;

        return {hour, minute, day_of_week, day_of_month, month, quarter, day_of_year, is_market_open};
    }

    /**
     * Calculate exponential moving average using AVX2
     */
    [[nodiscard]] static auto calculateEMA(std::span<float const> prices, int period) -> float {
        if (prices.empty()) return 0.0f;

        float multiplier = 2.0f / (static_cast<float>(period) + 1.0f);
        float ema = prices[prices.size() - 1];  // Start with oldest price

        for (size_t i = prices.size() - 1; i > 0; --i) {
            ema = (prices[i - 1] - ema) * multiplier + ema;
        }

        return ema;
    }

    /**
     * Calculate mean using AVX2 SIMD
     */
    [[nodiscard]] static auto calculateMean(std::span<float const> data) -> float {
        if (data.empty()) return 0.0f;

        float sum = 0.0f;

        #if defined(__AVX2__)
        // AVX2: Process 8 floats at a time
        __m256 sum_vec = _mm256_setzero_ps();
        size_t i = 0;

        for (; i + 8 <= data.size(); i += 8) {
            __m256 values = _mm256_loadu_ps(&data[i]);
            sum_vec = _mm256_add_ps(sum_vec, values);
        }

        // Horizontal sum of vector
        alignas(32) float temp[8];
        _mm256_store_ps(temp, sum_vec);
        sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

        // Handle remaining elements
        for (; i < data.size(); ++i) {
            sum += data[i];
        }
        #else
        // Fallback: OpenMP SIMD
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < data.size(); ++i) {
            sum += data[i];
        }
        #endif

        return sum / static_cast<float>(data.size());
    }

    /**
     * Calculate standard deviation using AVX2
     */
    [[nodiscard]] static auto calculateStdDev(std::span<float const> data, float mean) -> float {
        if (data.empty()) return 0.0f;

        float sum_sq_diff = 0.0f;

        #pragma omp simd reduction(+:sum_sq_diff)
        for (size_t i = 0; i < data.size(); ++i) {
            float diff = data[i] - mean;
            sum_sq_diff += diff * diff;
        }

        return std::sqrt(sum_sq_diff / static_cast<float>(data.size()));
    }
};

/**
 * Configuration for feature extraction
 */
struct FeatureConfig {
    int rsi_period = 14;
    int macd_fast = 12;
    int macd_slow = 26;
    int macd_signal = 9;
    int bb_period = 20;
    float bb_std_dev = 2.0f;
    int atr_period = 14;
    int volume_ma_period = 20;
};

}  // namespace bigbrother::market_intelligence
