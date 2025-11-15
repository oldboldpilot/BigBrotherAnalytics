/**
 * Python bindings for C++ feature extraction
 * Allows testing C++ and Python implementations side-by-side
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <array>
#include <vector>
#include <chrono>
#include <cmath>
#include <ctime>

namespace py = pybind11;

/**
 * Standalone C++ feature extraction (copied from feature_extractor.cppm)
 * This is a minimal version for testing purposes only
 */
class FeatureExtractorCPP {
public:
    /**
     * Calculate Black-Scholes Greeks from pricing data
     */
    static std::array<float, 4> calculateGreeks(
        float spot_price,
        float volatility,
        float risk_free_rate,
        float time_to_expiry = 30.0f / 365.0f) {

        float K = spot_price;
        float S = spot_price;
        float r = risk_free_rate / 100.0f;
        float sigma = volatility;
        float T = time_to_expiry;

        if (S <= 0.0f || K <= 0.0f || sigma <= 0.0f || T <= 0.0f) {
            return {0.01f, -0.05f, 0.20f, 0.01f};
        }

        float sqrt_T = std::sqrt(T);
        float d1 = (std::log(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrt_T);
        float d2 = d1 - sigma * sqrt_T;

        float n_prime_d1 = (1.0f / std::sqrt(2.0f * 3.14159265f)) * std::exp(-0.5f * d1 * d1);

        auto norm_cdf = [](float x) -> float {
            return 0.5f * (1.0f + std::tanh(x / std::sqrt(2.0f) * 0.7978845608f));
        };

        float n_d2 = norm_cdf(d2);

        float gamma_val = n_prime_d1 / (S * sigma * sqrt_T);
        float vega_val = S * n_prime_d1 * sqrt_T / 100.0f;
        float theta_val = (-(S * n_prime_d1 * sigma) / (2.0f * sqrt_T)
                          - r * K * std::exp(-r * T) * n_d2) / 365.0f;
        float rho_val = K * T * std::exp(-r * T) * n_d2 / 100.0f;

        gamma_val = std::clamp(gamma_val, 0.0f, 0.1f);
        theta_val = std::clamp(theta_val, -0.5f, 0.0f);
        vega_val = std::clamp(vega_val, 0.0f, 1.0f);
        rho_val = std::clamp(rho_val, -0.5f, 0.5f);

        return {gamma_val, theta_val, vega_val, rho_val};
    }

    /**
     * Extract 85 features (matching toArray85 from feature_extractor.cppm)
     */
    static std::array<float, 85> extractFeatures85(
        float close, float open, float high, float low, float volume,
        float rsi_14, float macd, float macd_signal,
        float bb_upper, float bb_lower, float bb_position,
        float atr_14, float volume_ratio,
        float volume_rsi_signal, float yield_volatility, float macd_volume,
        float bb_momentum, float rate_return, float rsi_bb_signal,
        float momentum_3d, float recent_win_rate,
        float symbol_encoded,
        float day_of_week, float day_of_month, float month_of_year, float quarter, float day_of_year,
        float price_direction, float price_above_ma5, float price_above_ma20,
        float macd_signal_direction, float volume_trend,
        int year, int month, int day,
        std::vector<float> const& price_history,
        std::vector<float> const& volume_history,
        float fed_funds_rate, float treasury_10yr) {

        // Price lags (actual historical prices)
        std::array<float, 20> price_lags{};
        for (int i = 0; i < 20 && i < static_cast<int>(price_history.size()); ++i) {
            price_lags[i] = price_history[i];
        }

        // Price diffs
        std::array<float, 20> price_diffs{};
        for (int i = 0; i < 20 && i + 1 < static_cast<int>(price_history.size()); ++i) {
            price_diffs[i] = price_history[0] - price_history[i + 1];
        }

        // Autocorrelations
        auto calc_autocorr = [&price_history](int lag, int window = 60) -> float {
            if (static_cast<int>(price_history.size()) < window + lag + 1) return 0.0f;

            std::vector<float> returns;
            for (size_t i = 0; i + 1 < price_history.size() && i < static_cast<size_t>(window + lag); ++i) {
                float ret = (price_history[i] - price_history[i + 1]) / price_history[i + 1];
                returns.push_back(ret);
            }

            if (returns.size() < static_cast<size_t>(window + lag)) return 0.0f;

            float mean1 = 0.0f, mean2 = 0.0f;
            for (int i = 0; i < window; ++i) {
                mean1 += returns[i];
                mean2 += returns[i + lag];
            }
            mean1 /= window;
            mean2 /= window;

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

        // Volume calculations
        float volume_sma20 = 0.0f;
        int volume_count = std::min(20, static_cast<int>(volume_history.size()));
        if (volume_count > 0) {
            for (int i = 0; i < volume_count; ++i) {
                volume_sma20 += volume_history[i];
            }
            volume_sma20 /= volume_count;
        }

        float vol_ratio = (volume_sma20 > 0.0f) ? (volume / volume_sma20) : volume_ratio;

        // Greeks calculation
        float volatility_estimate = 0.0f;
        if (price_history.size() >= 14) {
            float atr_sum = 0.0f;
            for (int i = 1; i < 14 && i < static_cast<int>(price_history.size()); ++i) {
                atr_sum += std::abs(price_history[i-1] - price_history[i]);
            }
            float avg_atr = atr_sum / 13.0f;
            volatility_estimate = (close > 0.0f) ? (avg_atr / close) * std::sqrt(252.0f) : 0.20f;
            volatility_estimate = std::clamp(volatility_estimate, 0.05f, 2.0f);
        } else {
            volatility_estimate = 0.20f;
        }

        float risk_free = (treasury_10yr > 0.0f) ? treasury_10yr :
                          (fed_funds_rate > 0.0f) ? fed_funds_rate : 4.5f;

        auto greeks = calculateGreeks(close, volatility_estimate, risk_free);
        float gamma_calc = greeks[0];
        float theta_calc = greeks[1];
        float vega_calc = greeks[2];
        float rho_calc = greeks[3];

        float gamma_volatility_calc = gamma_calc * atr_14;

        return {
            // [0-4] OHLCV
            close, open, high, low, volume,
            // [5-7] Technical indicators
            rsi_14, macd, macd_signal,
            // [8-10] Bollinger Bands
            bb_upper, bb_lower, bb_position,
            // [11-13] Volatility/Volume
            atr_14, volume_sma20, vol_ratio,
            // [14-17] Greeks
            gamma_calc, theta_calc, vega_calc, rho_calc,
            // [18-24] Interaction features
            volume_rsi_signal, yield_volatility, macd_volume,
            bb_momentum, rate_return, gamma_volatility_calc, rsi_bb_signal,
            // [25-26] Momentum features
            momentum_3d, recent_win_rate,
            // [27-46] Price lags
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
            static_cast<float>(year), static_cast<float>(month), static_cast<float>(day),
            // [61-80] Price diffs
            price_diffs[0], price_diffs[1], price_diffs[2], price_diffs[3], price_diffs[4],
            price_diffs[5], price_diffs[6], price_diffs[7], price_diffs[8], price_diffs[9],
            price_diffs[10], price_diffs[11], price_diffs[12], price_diffs[13], price_diffs[14],
            price_diffs[15], price_diffs[16], price_diffs[17], price_diffs[18], price_diffs[19],
            // [81-84] Autocorrelations
            autocorr_1, autocorr_5, autocorr_10, autocorr_20
        };
    }

    /**
     * Python-friendly wrapper returning numpy array
     */
    static py::array_t<float> extractFeatures85_py(
        float close, float open, float high, float low, float volume,
        float rsi_14, float macd, float macd_signal,
        float bb_upper, float bb_lower, float bb_position,
        float atr_14, float volume_ratio,
        float volume_rsi_signal, float yield_volatility, float macd_volume,
        float bb_momentum, float rate_return, float rsi_bb_signal,
        float momentum_3d, float recent_win_rate,
        float symbol_encoded,
        float day_of_week, float day_of_month, float month_of_year, float quarter, float day_of_year,
        float price_direction, float price_above_ma5, float price_above_ma20,
        float macd_signal_direction, float volume_trend,
        int year, int month, int day,
        py::array_t<float> price_history,
        py::array_t<float> volume_history,
        float fed_funds_rate, float treasury_10yr) {

        // Convert numpy arrays to std::vector
        auto price_buf = price_history.request();
        auto volume_buf = volume_history.request();

        std::vector<float> price_vec(static_cast<float*>(price_buf.ptr),
                                     static_cast<float*>(price_buf.ptr) + price_buf.size);
        std::vector<float> volume_vec(static_cast<float*>(volume_buf.ptr),
                                      static_cast<float*>(volume_buf.ptr) + volume_buf.size);

        // Extract features
        auto features = extractFeatures85(
            close, open, high, low, volume,
            rsi_14, macd, macd_signal,
            bb_upper, bb_lower, bb_position,
            atr_14, volume_ratio,
            volume_rsi_signal, yield_volatility, macd_volume,
            bb_momentum, rate_return, rsi_bb_signal,
            momentum_3d, recent_win_rate,
            symbol_encoded,
            day_of_week, day_of_month, month_of_year, quarter, day_of_year,
            price_direction, price_above_ma5, price_above_ma20,
            macd_signal_direction, volume_trend,
            year, month, day,
            price_vec, volume_vec,
            fed_funds_rate, treasury_10yr
        );

        // Convert to numpy array
        auto result = py::array_t<float>(85);
        auto result_buf = result.request();
        float* ptr = static_cast<float*>(result_buf.ptr);
        std::copy(features.begin(), features.end(), ptr);

        return result;
    }

    /**
     * INT32 Quantization - Symmetric quantization for neural network inference
     */
    struct QuantizationParams32 {
        float scale;
        float inv_scale;

        QuantizationParams32() : scale(1.0f), inv_scale(1.0f) {}
        explicit QuantizationParams32(float s) : scale(s), inv_scale(1.0f / s) {}
    };

    /**
     * Compute INT32 quantization parameters from FP32 features
     */
    static QuantizationParams32 computeQuantizationParams32(py::array_t<float> features) {
        auto buf = features.request();
        float* ptr = static_cast<float*>(buf.ptr);
        size_t size = buf.size;

        float max_abs = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            max_abs = std::max(max_abs, std::abs(ptr[i]));
        }

        if (max_abs < 1e-8f) {
            max_abs = 1.0f;
        }

        constexpr int32_t MAX_INT32 = 2147483647;
        float scale = max_abs / static_cast<float>(MAX_INT32);
        return QuantizationParams32(scale);
    }

    /**
     * Quantize FP32 features to INT32
     * Returns tuple of (quantized_features, scale)
     */
    static py::tuple quantizeFeatures85_py(py::array_t<float> features) {
        auto buf = features.request();
        if (buf.size != 85) {
            throw std::runtime_error("Expected 85 features, got " + std::to_string(buf.size));
        }

        float* input = static_cast<float*>(buf.ptr);

        // Compute quantization parameters
        auto params = computeQuantizationParams32(features);

        // Quantize
        auto quantized = py::array_t<int32_t>(85);
        auto q_buf = quantized.request();
        int32_t* output = static_cast<int32_t*>(q_buf.ptr);

        for (size_t i = 0; i < 85; ++i) {
            float scaled = input[i] * params.inv_scale;
            output[i] = static_cast<int32_t>(std::round(scaled));
        }

        return py::make_tuple(quantized, params.scale);
    }

    /**
     * Dequantize INT32 features back to FP32
     */
    static py::array_t<float> dequantizeFeatures85_py(py::array_t<int32_t> quantized, float scale) {
        auto buf = quantized.request();
        if (buf.size != 85) {
            throw std::runtime_error("Expected 85 features, got " + std::to_string(buf.size));
        }

        int32_t* input = static_cast<int32_t*>(buf.ptr);

        // Dequantize
        auto features = py::array_t<float>(85);
        auto f_buf = features.request();
        float* output = static_cast<float*>(f_buf.ptr);

        for (size_t i = 0; i < 85; ++i) {
            output[i] = static_cast<float>(input[i]) * scale;
        }

        return features;
    }
};

PYBIND11_MODULE(feature_extractor_cpp, m) {
    m.doc() = "C++ feature extraction module with INT32 quantization support";

    m.def("extract_features_85", &FeatureExtractorCPP::extractFeatures85_py,
          "Extract 85 features using C++ implementation",
          py::arg("close"), py::arg("open"), py::arg("high"), py::arg("low"), py::arg("volume"),
          py::arg("rsi_14"), py::arg("macd"), py::arg("macd_signal"),
          py::arg("bb_upper"), py::arg("bb_lower"), py::arg("bb_position"),
          py::arg("atr_14"), py::arg("volume_ratio"),
          py::arg("volume_rsi_signal"), py::arg("yield_volatility"), py::arg("macd_volume"),
          py::arg("bb_momentum"), py::arg("rate_return"), py::arg("rsi_bb_signal"),
          py::arg("momentum_3d"), py::arg("recent_win_rate"),
          py::arg("symbol_encoded"),
          py::arg("day_of_week"), py::arg("day_of_month"), py::arg("month_of_year"),
          py::arg("quarter"), py::arg("day_of_year"),
          py::arg("price_direction"), py::arg("price_above_ma5"), py::arg("price_above_ma20"),
          py::arg("macd_signal_direction"), py::arg("volume_trend"),
          py::arg("year"), py::arg("month"), py::arg("day"),
          py::arg("price_history"), py::arg("volume_history"),
          py::arg("fed_funds_rate"), py::arg("treasury_10yr"));

    m.def("quantize_features_85", &FeatureExtractorCPP::quantizeFeatures85_py,
          "Quantize 85 FP32 features to INT32 (returns tuple of quantized_features, scale)",
          py::arg("features"));

    m.def("dequantize_features_85", &FeatureExtractorCPP::dequantizeFeatures85_py,
          "Dequantize 85 INT32 features back to FP32",
          py::arg("quantized"), py::arg("scale"));
}
