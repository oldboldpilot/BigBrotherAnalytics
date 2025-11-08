// Auto-generated: 2025-11-07 17:32:56.528381
// Risk-free rate from FRED as of 2025-11-06
// Data source: https://fred.stlouisfed.org/series/DGS10

#pragma once

namespace bigbrother::constants {

// 10-Year Treasury Rate (updated 2025-11-06)
constexpr double RISK_FREE_RATE = 0.041100;  // 4.110%

// For different option expirations, use appropriate tenor:
constexpr double RISK_FREE_RATE_3M = 0.041100;   // Update with DTB3
constexpr double RISK_FREE_RATE_1Y = 0.041100;   // Update with DGS1
constexpr double RISK_FREE_RATE_10Y = 0.041100;  // DGS10

// Last updated
constexpr const char* RATE_DATE = "2025-11-06";

} // namespace bigbrother::constants
