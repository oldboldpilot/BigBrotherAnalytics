/**
 * BigBrotherAnalytics - FRED Rates Python Bindings
 *
 * pybind11 bindings for FRED risk-free rates fetcher.
 * Exposes C++ FRED API client to Python for dashboard integration.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-11
 * Phase 5+: Risk-Free Rate Integration
 *
 * Usage (Python):
 *   from build import fred_rates_py
 *
 *   # Create fetcher with API key
 *   config = fred_rates_py.FREDConfig()
 *   config.api_key = "your_api_key_here"
 *
 *   fetcher = fred_rates_py.FREDRatesFetcher(config)
 *
 *   # Fetch specific rate
 *   rate_data = fetcher.fetch_latest_rate(fred_rates_py.RateSeries.ThreeMonthTreasury)
 *   print(f"3-Month Treasury: {rate_data.rate_value * 100:.3f}%")
 *
 *   # Fetch all rates
 *   all_rates = fetcher.fetch_all_rates()
 *   for series, data in all_rates.items():
 *       print(f"{data.series_name}: {data.rate_value * 100:.3f}%")
 *
 *   # Get risk-free rate for options pricing
 *   risk_free = fetcher.get_risk_free_rate()
 *   print(f"Risk-free rate: {risk_free * 100:.3f}%")
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <map>
#include <string>

import bigbrother.market_intelligence.fred_rates;
import bigbrother.utils.types;

namespace py = pybind11;
using namespace bigbrother::market_intelligence;
using namespace bigbrother::types;

/**
 * Helper function to convert Result<T> to Python
 * Raises exception if error, returns value if success
 */
template<typename T>
auto unwrapResult(Result<T>&& result) -> T {
    if (result) {
        return std::move(*result);
    } else {
        throw std::runtime_error(result.error().message);
    }
}

/**
 * FRED Rates Python Module
 */
PYBIND11_MODULE(fred_rates_py, m) {
    m.doc() = "FRED (Federal Reserve Economic Data) API client for risk-free rates";

    // ========================================================================
    // Enums
    // ========================================================================

    py::enum_<RateSeries>(m, "RateSeries",
                         "FRED rate series identifiers")
        .value("ThreeMonthTreasury", RateSeries::ThreeMonthTreasury,
               "3-Month Treasury Bill (DGS3MO)")
        .value("TwoYearTreasury", RateSeries::TwoYearTreasury,
               "2-Year Treasury Note (DGS2)")
        .value("FiveYearTreasury", RateSeries::FiveYearTreasury,
               "5-Year Treasury Note (DGS5)")
        .value("TenYearTreasury", RateSeries::TenYearTreasury,
               "10-Year Treasury Note (DGS10)")
        .value("ThirtyYearTreasury", RateSeries::ThirtyYearTreasury,
               "30-Year Treasury Bond (DGS30)")
        .value("FedFundsRate", RateSeries::FedFundsRate,
               "Effective Federal Funds Rate (DFF)")
        .export_values();

    // ========================================================================
    // Data Structures
    // ========================================================================

    py::class_<RateData>(m, "RateData",
                        "FRED rate data structure")
        .def(py::init<>())
        .def_readwrite("series", &RateData::series,
                      "Rate series enum")
        .def_readwrite("series_id", &RateData::series_id,
                      "FRED series ID (e.g., 'DGS3MO')")
        .def_readwrite("series_name", &RateData::series_name,
                      "Human-readable series name (e.g., '3-Month Treasury Bill')")
        .def_readwrite("rate_value", &RateData::rate_value,
                      "Rate as decimal (e.g., 0.05 for 5%)")
        .def_readwrite("last_updated", &RateData::last_updated,
                      "Timestamp when rate was fetched")
        .def_readwrite("observation_date", &RateData::observation_date,
                      "Date of the rate observation")
        .def("__repr__", [](RateData const& rd) {
            return std::format("<RateData series={} rate={:.3f}% updated={}>",
                             rd.series_id, rd.rate_value * 100.0, rd.last_updated);
        });

    py::class_<FREDConfig>(m, "FREDConfig",
                          "FRED API configuration")
        .def(py::init<>())
        .def_readwrite("api_key", &FREDConfig::api_key,
                      "FRED API key (get free at https://fred.stlouisfed.org)")
        .def_readwrite("base_url", &FREDConfig::base_url,
                      "FRED API base URL (default: https://api.stlouisfed.org/fred/series/observations)")
        .def_readwrite("timeout_seconds", &FREDConfig::timeout_seconds,
                      "HTTP request timeout in seconds (default: 10)")
        .def_readwrite("max_observations", &FREDConfig::max_observations,
                      "Maximum observations to fetch (default: 5)")
        .def("__repr__", [](FREDConfig const& c) {
            return std::format("<FREDConfig api_key={} timeout={}s>",
                             c.api_key.empty() ? "NOT_SET" : "***",
                             c.timeout_seconds);
        });

    // ========================================================================
    // FRED Rates Fetcher
    // ========================================================================

    py::class_<FREDRatesFetcher>(m, "FREDRatesFetcher",
                                "FRED API client for fetching risk-free rates")
        .def(py::init<FREDConfig>(),
             py::arg("config"),
             "Create FRED rates fetcher with configuration")

        .def("fetch_latest_rate",
             [](FREDRatesFetcher& fetcher, RateSeries series) -> RateData {
                 return unwrapResult(fetcher.fetchLatestRate(series));
             },
             py::arg("series"),
             "Fetch latest rate for a specific series\n\n"
             "Args:\n"
             "    series (RateSeries): Rate series to fetch\n\n"
             "Returns:\n"
             "    RateData: Rate data with value and metadata\n\n"
             "Raises:\n"
             "    RuntimeError: If API request fails\n\n"
             "Example:\n"
             "    rate_data = fetcher.fetch_latest_rate(RateSeries.ThreeMonthTreasury)\n"
             "    print(f'3M Treasury: {rate_data.rate_value * 100:.3f}%')")

        .def("fetch_all_rates",
             &FREDRatesFetcher::fetchAllRates,
             "Fetch all available rate series\n\n"
             "Returns:\n"
             "    dict[RateSeries, RateData]: Map of series to rate data\n\n"
             "Example:\n"
             "    rates = fetcher.fetch_all_rates()\n"
             "    for series, data in rates.items():\n"
             "        print(f'{data.series_name}: {data.rate_value * 100:.3f}%')")

        .def("get_risk_free_rate",
             &FREDRatesFetcher::getRiskFreeRate,
             py::arg("maturity") = RateSeries::ThreeMonthTreasury,
             "Get risk-free rate for options pricing\n\n"
             "Args:\n"
             "    maturity (RateSeries): Desired maturity (default: 3-month)\n\n"
             "Returns:\n"
             "    float: Risk-free rate as decimal (e.g., 0.05 for 5%)\n\n"
             "Note:\n"
             "    Returns 0.04 (4%) as default if rate unavailable\n\n"
             "Example:\n"
             "    rf_rate = fetcher.get_risk_free_rate()\n"
             "    option_price = black_scholes(S, K, T, rf_rate, sigma)")

        .def("clear_cache",
             &FREDRatesFetcher::clearCache,
             "Clear internal rate cache\n\n"
             "Forces fresh fetch on next request (cache TTL: 1 hour)")

        .def("__repr__", [](FREDRatesFetcher const&) {
            return "<FREDRatesFetcher ready>";
        });

    // ========================================================================
    // Helper Functions
    // ========================================================================

    m.def("get_series_id",
          &FREDRatesFetcher::getSeriesId,
          py::arg("series"),
          "Get FRED series ID from enum\n\n"
          "Args:\n"
          "    series (RateSeries): Rate series\n\n"
          "Returns:\n"
          "    str: FRED series ID (e.g., 'DGS3MO')\n\n"
          "Example:\n"
          "    series_id = get_series_id(RateSeries.TenYearTreasury)\n"
          "    # Returns: 'DGS10'");

    m.def("get_series_name",
          &FREDRatesFetcher::getSeriesName,
          py::arg("series"),
          "Get human-readable series name\n\n"
          "Args:\n"
          "    series (RateSeries): Rate series\n\n"
          "Returns:\n"
          "    str: Human-readable name (e.g., '10-Year Treasury Note')\n\n"
          "Example:\n"
          "    name = get_series_name(RateSeries.TenYearTreasury)\n"
          "    # Returns: '10-Year Treasury Note'");

    // ========================================================================
    // Module-Level Convenience Functions
    // ========================================================================

    m.def("fetch_rate_simple",
          [](std::string const& api_key, RateSeries series) -> RateData {
              FREDConfig config;
              config.api_key = api_key;
              FREDRatesFetcher fetcher(config);
              return unwrapResult(fetcher.fetchLatestRate(series));
          },
          py::arg("api_key"),
          py::arg("series") = RateSeries::ThreeMonthTreasury,
          "Convenience function to fetch a rate with minimal setup\n\n"
          "Args:\n"
          "    api_key (str): FRED API key\n"
          "    series (RateSeries): Rate series to fetch (default: 3-month Treasury)\n\n"
          "Returns:\n"
          "    RateData: Rate data\n\n"
          "Example:\n"
          "    rate = fetch_rate_simple('your_api_key', RateSeries.ThreeMonthTreasury)\n"
          "    print(f'Rate: {rate.rate_value * 100:.3f}%')");

    m.def("get_treasury_curve",
          [](std::string const& api_key) -> std::map<RateSeries, RateData> {
              FREDConfig config;
              config.api_key = api_key;
              FREDRatesFetcher fetcher(config);
              return fetcher.fetchAllRates();
          },
          py::arg("api_key"),
          "Fetch entire Treasury yield curve\n\n"
          "Args:\n"
          "    api_key (str): FRED API key\n\n"
          "Returns:\n"
          "    dict[RateSeries, RateData]: Complete yield curve\n\n"
          "Example:\n"
          "    curve = get_treasury_curve('your_api_key')\n"
          "    for series, data in curve.items():\n"
          "        print(f'{data.series_name}: {data.rate_value * 100:.3f}%')");

    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "Olumuyiwa Oluwasanmi";
}
