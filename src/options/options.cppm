/**
 * @file options.cppm
 * @brief Options Trading Module - Main Export Interface
 *
 * Comprehensive options trading system with:
 * - Trinomial tree pricing (American & European)
 * - Greeks calculation (delta, gamma, theta, vega, rho)
 * - Professional trading strategies
 * - Real-time portfolio monitoring
 * - ML-driven option selection
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-14
 */

export module bigbrother.options;

// Re-export all options components
export import options:trinomial_pricer;
export import options:strategies;
export import options:greeks_calculator;

// Main options namespace
export namespace bigbrother::options {
    // All types and classes are exported through the partition modules
}
