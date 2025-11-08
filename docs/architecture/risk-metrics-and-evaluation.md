# Risk Metrics and Evaluation Framework

## Overview

This document details the comprehensive risk evaluation framework implemented in BigBrotherAnalytics for options and stock trading.

---

## Table of Contents

1. [Position-Level Risk Metrics](#position-level-risk-metrics)
2. [Portfolio-Level Risk Metrics](#portfolio-level-risk-metrics)
3. [Value at Risk (VaR) Implementation](#value-at-risk-implementation)
4. [Stress Testing](#stress-testing)
5. [Correlation Analysis](#correlation-analysis)
6. [Scenario Analysis](#scenario-analysis)
7. [Real-Time Risk Monitoring](#real-time-risk-monitoring)

---

## Position-Level Risk Metrics

### 1. Maximum Theoretical Risk (MTR)

**Definition:** Maximum possible loss for a position

**Calculations by Strategy:**

```cpp
// Vertical Spreads
double calculateMaxLoss_VerticalSpread(
    double spread_width,
    double net_premium,
    bool is_debit
) {
    if (is_debit) {
        return net_premium * 100;  // Debit paid
    } else {
        return (spread_width - net_premium) * 100;  // Credit spread
    }
}

// Iron Condor
double calculateMaxLoss_IronCondor(
    double put_spread_width,
    double call_spread_width,
    double net_credit
) {
    double max_spread = std::max(put_spread_width, call_spread_width);
    return (max_spread - net_credit) * 100;
}

// Naked Short Put
double calculateMaxLoss_NakedPut(double strike) {
    return strike * 100;  // Stock can go to $0
}

// Naked Short Call
double calculateMaxLoss_NakedCall() {
    return std::numeric_limits<double>::infinity();  // Unlimited
}
```

### 2. Probability of Profit (POP)

**Delta Approximation:**
```cpp
double calculatePOP_DeltaApprox(double net_delta) {
    // Delta represents ~probability of ITM
    // POP ≈ 1 - |Delta|
    return 1.0 - std::abs(net_delta);
}
```

**Monte Carlo Simulation:**
```cpp
double calculatePOP_MonteCarlo(
    const Position& position,
    int num_simulations = 10000
) {
    int profitable_outcomes = 0;

    for (int i = 0; i < num_simulations; ++i) {
        double final_price = simulatePricePath(position);
        double pnl = calculatePnL(position, final_price);

        if (pnl > 0) {
            profitable_outcomes++;
        }
    }

    return static_cast<double>(profitable_outcomes) / num_simulations;
}
```

**Normal Distribution (Log-Normal Model):**
```cpp
double calculatePOP_LogNormal(
    double current_price,
    double breakeven_price,
    double volatility,
    double time_to_expiration
) {
    double drift = -0.5 * volatility * volatility * time_to_expiration;
    double diffusion = volatility * std::sqrt(time_to_expiration);

    double d = (std::log(breakeven_price / current_price) - drift) / diffusion;

    return normalCDF(-d);  // Probability of being above breakeven
}
```

### 3. Risk/Reward Ratio

```cpp
struct RiskRewardMetrics {
    double max_profit;
    double max_loss;
    double risk_reward_ratio;
    double expected_value;
    double pop;
};

RiskRewardMetrics calculateRiskReward(const Position& pos) {
    RiskRewardMetrics metrics;

    metrics.max_profit = calculateMaxProfit(pos);
    metrics.max_loss = calculateMaxLoss(pos);
    metrics.risk_reward_ratio = metrics.max_profit / metrics.max_loss;
    metrics.pop = calculatePOP(pos);

    // Expected Value
    metrics.expected_value =
        metrics.pop * metrics.max_profit -
        (1 - metrics.pop) * metrics.max_loss;

    return metrics;
}
```

### 4. Time Decay Analysis

**Theta Decay Curve:**
```cpp
std::vector<double> projectThetaDecay(
    const Position& position,
    int days_forward
) {
    std::vector<double> values;
    values.reserve(days_forward);

    for (int day = 0; day <= days_forward; ++day) {
        double value_at_day = calculatePositionValue(
            position,
            position.current_price,  // Assume no price change
            position.days_to_expiration - day
        );
        values.push_back(value_at_day);
    }

    return values;
}
```

**Theta Efficiency:**
```cpp
double calculateThetaEfficiency(const Position& pos) {
    // Theta earned per dollar at risk
    return std::abs(pos.theta) / pos.buying_power_reduction;
}
```

---

## Portfolio-Level Risk Metrics

### 1. Portfolio Greeks

```cpp
struct PortfolioGreeks {
    double delta;
    double gamma;
    double theta;
    double vega;
    double rho;

    // Beta-weighted to SPX
    double beta_weighted_delta;

    // Per $1000 capital
    double delta_per_1k;
    double theta_per_day;
};

PortfolioGreeks calculatePortfolioGreeks(
    const std::vector<Position>& positions,
    double portfolio_value
) {
    PortfolioGreeks greeks{};

    for (const auto& pos : positions) {
        greeks.delta += pos.delta * pos.quantity * 100;
        greeks.gamma += pos.gamma * pos.quantity * 100;
        greeks.theta += pos.theta * pos.quantity * 100;
        greeks.vega  += pos.vega * pos.quantity * 100;
        greeks.rho   += pos.rho * pos.quantity * 100;

        // Beta-weighted delta
        greeks.beta_weighted_delta +=
            pos.delta * pos.beta * pos.quantity * 100;
    }

    greeks.delta_per_1k = (greeks.delta / portfolio_value) * 1000;
    greeks.theta_per_day = greeks.theta;

    return greeks;
}
```

### 2. Buying Power Usage (BPU)

```cpp
struct BuyingPowerMetrics {
    double total_bp_available;
    double bp_used;
    double bp_remaining;
    double bp_usage_percent;
    double reserved_for_adjustments;  // 20% buffer
};

BuyingPowerMetrics calculateBPU(
    double account_value,
    const std::vector<Position>& positions
) {
    BuyingPowerMetrics metrics;

    // Calculate total margin requirement
    metrics.bp_used = 0;
    for (const auto& pos : positions) {
        metrics.bp_used += pos.buying_power_reduction;
    }

    // Account type determines BP
    bool is_portfolio_margin = (account_value >= 125000);

    if (is_portfolio_margin) {
        metrics.total_bp_available = account_value * 6.0;  // 6:1 leverage
    } else {
        metrics.total_bp_available = account_value * 2.0;  // Reg T margin
    }

    metrics.bp_remaining = metrics.total_bp_available - metrics.bp_used;
    metrics.bp_usage_percent = (metrics.bp_used / metrics.total_bp_available) * 100;

    // Reserve 20% for adjustments/margin calls
    metrics.reserved_for_adjustments = metrics.total_bp_available * 0.20;

    return metrics;
}
```

### 3. Concentration Risk

```cpp
struct ConcentrationMetrics {
    std::map<std::string, double> exposure_by_symbol;
    std::map<std::string, double> exposure_by_sector;
    double max_single_position_pct;
    bool is_diversified;
};

ConcentrationMetrics analyzeConcentration(
    const std::vector<Position>& positions,
    double portfolio_value
) {
    ConcentrationMetrics metrics;

    // Calculate exposure by symbol
    for (const auto& pos : positions) {
        double notional = pos.quantity * pos.current_price * 100;
        metrics.exposure_by_symbol[pos.symbol] += notional;
    }

    // Find max concentration
    metrics.max_single_position_pct = 0;
    for (const auto& [symbol, exposure] : metrics.exposure_by_symbol) {
        double pct = (exposure / portfolio_value) * 100;
        metrics.max_single_position_pct = std::max(
            metrics.max_single_position_pct, pct
        );
    }

    // Diversification threshold: no single position > 10%
    metrics.is_diversified = (metrics.max_single_position_pct < 10.0);

    return metrics;
}
```

---

## Value at Risk (VaR) Implementation

### 1. Parametric VaR (Variance-Covariance Method)

```cpp
double calculateParametricVaR(
    const PortfolioGreeks& greeks,
    double stock_price,
    double volatility,
    double confidence_level,  // 0.95 or 0.99
    int time_horizon_days = 1
) {
    // Expected stock move at confidence level
    double z_score = inverseNormalCDF(confidence_level);

    double expected_move = stock_price * volatility *
                          std::sqrt(time_horizon_days / 365.0) * z_score;

    // Linear approximation using delta
    double portfolio_change = greeks.delta * expected_move;

    // Add gamma adjustment for large moves
    double gamma_adjustment = 0.5 * greeks.gamma *
                             expected_move * expected_move;

    double VaR = -(portfolio_change + gamma_adjustment);

    return std::max(VaR, 0.0);
}
```

### 2. Historical VaR

```cpp
double calculateHistoricalVaR(
    const std::vector<double>& historical_pnl,
    double confidence_level
) {
    // Sort P/L in ascending order
    std::vector<double> sorted_pnl = historical_pnl;
    std::sort(sorted_pnl.begin(), sorted_pnl.end());

    // Find percentile
    size_t index = static_cast<size_t>(
        (1.0 - confidence_level) * sorted_pnl.size()
    );

    return -sorted_pnl[index];
}
```

### 3. Monte Carlo VaR

```cpp
double calculateMonteCarloVaR(
    const Portfolio& portfolio,
    int num_simulations,
    double confidence_level,
    int time_horizon_days
) {
    std::vector<double> simulated_pnl;
    simulated_pnl.reserve(num_simulations);

    for (int i = 0; i < num_simulations; ++i) {
        // Simulate price paths for all underlyings
        auto simulated_prices = simulateCorrelatedPrices(
            portfolio.underlyings,
            time_horizon_days
        );

        // Calculate portfolio value at simulated prices
        double pnl = calculatePortfolioPnL(portfolio, simulated_prices);
        simulated_pnl.push_back(pnl);
    }

    return calculateHistoricalVaR(simulated_pnl, confidence_level);
}
```

### 4. Conditional VaR (Expected Shortfall)

```cpp
double calculateCVaR(
    const std::vector<double>& pnl_distribution,
    double confidence_level
) {
    std::vector<double> sorted_pnl = pnl_distribution;
    std::sort(sorted_pnl.begin(), sorted_pnl.end());

    size_t cutoff_index = static_cast<size_t>(
        (1.0 - confidence_level) * sorted_pnl.size()
    );

    // Average of worst losses (beyond VaR)
    double sum_tail_losses = 0;
    for (size_t i = 0; i <= cutoff_index; ++i) {
        sum_tail_losses += sorted_pnl[i];
    }

    return -sum_tail_losses / (cutoff_index + 1);
}
```

---

## Stress Testing

### 1. Price Stress Tests

```cpp
struct StressTestResult {
    double scenario_pnl;
    double scenario_delta;
    double scenario_gamma;
    std::string scenario_name;
};

std::vector<StressTestResult> runPriceStressTests(
    const Portfolio& portfolio
) {
    std::vector<StressTestResult> results;

    // Scenarios
    std::vector<double> price_moves = {
        -0.20, -0.15, -0.10, -0.05, -0.02,  // Down scenarios
        0.0,                                  // Current
        0.02, 0.05, 0.10, 0.15, 0.20         // Up scenarios
    };

    for (double move : price_moves) {
        double stressed_price = portfolio.current_price * (1 + move);

        StressTestResult result;
        result.scenario_name = std::to_string(move * 100) + "%";
        result.scenario_pnl = calculatePortfolioPnL(portfolio, stressed_price);
        result.scenario_delta = recalculateDelta(portfolio, stressed_price);
        result.scenario_gamma = recalculateGamma(portfolio, stressed_price);

        results.push_back(result);
    }

    return results;
}
```

### 2. Volatility Stress Tests

```cpp
std::vector<StressTestResult> runVolatilityStressTests(
    const Portfolio& portfolio
) {
    std::vector<double> iv_changes = {
        -0.50, -0.30, -0.20, -0.10,  // IV crush scenarios
        0.0,                          // Current
        0.10, 0.20, 0.30, 0.50       // IV expansion
    };

    std::vector<StressTestResult> results;

    for (double iv_change : iv_changes) {
        double stressed_iv = portfolio.current_iv * (1 + iv_change);

        StressTestResult result;
        result.scenario_name = "IV " + std::to_string(iv_change * 100) + "%";
        result.scenario_pnl = recalculatePortfolioValue(
            portfolio,
            portfolio.current_price,
            stressed_iv
        );

        results.push_back(result);
    }

    return results;
}
```

### 3. Time Stress Tests

```cpp
std::vector<double> projectPortfolioValue(
    const Portfolio& portfolio,
    int days_forward
) {
    std::vector<double> values;

    for (int day = 0; day <= days_forward; ++day) {
        double value = calculatePortfolioValue(
            portfolio,
            portfolio.current_price,  // No price change assumption
            portfolio.days_to_expiration - day,
            portfolio.current_iv
        );
        values.push_back(value);
    }

    return values;
}
```

### 4. Combined Stress Tests (Worst Case)

```cpp
struct WorstCaseScenario {
    double worst_pnl;
    std::string scenario_description;
    double price_at_worst;
    double iv_at_worst;
};

WorstCaseScenario findWorstCaseScenario(const Portfolio& portfolio) {
    WorstCaseScenario worst;
    worst.worst_pnl = 0;

    // Test combinations
    std::vector<double> price_moves = {-0.30, -0.20, -0.10, 0, 0.10, 0.20, 0.30};
    std::vector<double> iv_changes = {-0.50, -0.30, 0, 0.30, 0.50};

    for (double price_move : price_moves) {
        for (double iv_change : iv_changes) {
            double stressed_price = portfolio.current_price * (1 + price_move);
            double stressed_iv = portfolio.current_iv * (1 + iv_change);

            double pnl = recalculatePortfolioValue(
                portfolio, stressed_price, stressed_iv
            );

            if (pnl < worst.worst_pnl) {
                worst.worst_pnl = pnl;
                worst.price_at_worst = stressed_price;
                worst.iv_at_worst = stressed_iv;
                worst.scenario_description =
                    "Price: " + std::to_string(price_move * 100) + "%, " +
                    "IV: " + std::to_string(iv_change * 100) + "%";
            }
        }
    }

    return worst;
}
```

---

## Correlation Analysis

### 1. Position Correlation Matrix

```cpp
Eigen::MatrixXd calculateCorrelationMatrix(
    const std::vector<std::string>& symbols,
    const Eigen::MatrixXd& price_history  // rows: time, cols: symbols
) {
    int n = symbols.size();
    Eigen::MatrixXd returns = calculateReturns(price_history);

    // Calculate correlation matrix
    Eigen::MatrixXd corr_matrix(n, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                corr_matrix(i, j) = 1.0;
            } else {
                corr_matrix(i, j) = calculatePearsonCorrelation(
                    returns.col(i), returns.col(j)
                );
            }
        }
    }

    return corr_matrix;
}
```

### 2. Beta Calculation

```cpp
double calculateBeta(
    const Eigen::VectorXd& asset_returns,
    const Eigen::VectorXd& market_returns
) {
    double covariance = calculateCovariance(asset_returns, market_returns);
    double market_variance = calculateVariance(market_returns);

    return covariance / market_variance;
}
```

### 3. Portfolio Variance (with Correlations)

```cpp
double calculatePortfolioVariance(
    const std::vector<double>& weights,
    const std::vector<double>& volatilities,
    const Eigen::MatrixXd& correlation_matrix
) {
    int n = weights.size();
    double variance = 0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            variance += weights[i] * weights[j] *
                       volatilities[i] * volatilities[j] *
                       correlation_matrix(i, j);
        }
    }

    return variance;
}
```

---

## Scenario Analysis

### 1. Historical Scenarios

**2020 COVID Crash:**
```cpp
struct Scenario {
    std::string name;
    double stock_move_pct;
    double iv_change_pct;
    int days_elapsed;
};

Scenario covid_crash{
    .name = "COVID-19 Crash (Feb-Mar 2020)",
    .stock_move_pct = -0.35,  // SPY down 35%
    .iv_change_pct = 3.50,     // VIX from 15 to 85 (~467% increase)
    .days_elapsed = 20
};
```

**2008 Financial Crisis:**
```cpp
Scenario financial_crisis{
    .name = "2008 Financial Crisis",
    .stock_move_pct = -0.57,   // SPY down 57% peak-to-trough
    .iv_change_pct = 4.00,      // VIX peaked at 80+
    .days_elapsed = 365
};
```

**Flash Crash 2010:**
```cpp
Scenario flash_crash{
    .name = "Flash Crash (May 6, 2010)",
    .stock_move_pct = -0.09,   // Intraday drop
    .iv_change_pct = 0.50,      // Brief IV spike
    .days_elapsed = 1           // Recovered same day
};
```

### 2. Hypothetical Scenarios

```cpp
std::vector<Scenario> generateHypotheticalScenarios() {
    return {
        {"Black Swan Down", -0.40, 5.00, 10},
        {"Black Swan Up", 0.30, 2.00, 10},
        {"Gradual Bear Market", -0.25, 0.80, 180},
        {"Melt-Up Rally", 0.40, -0.30, 90},
        {"Volatility Collapse", 0.00, -0.60, 30},
        {"Volatility Spike", 0.00, 1.50, 5}
    };
}
```

### 3. Scenario Impact Calculator

```cpp
double calculateScenarioImpact(
    const Portfolio& portfolio,
    const Scenario& scenario
) {
    double current_value = portfolio.current_value;

    // Apply price move
    double new_price = portfolio.current_price *
                      (1 + scenario.stock_move_pct);

    // Apply IV change
    double new_iv = portfolio.current_iv *
                    (1 + scenario.iv_change_pct);

    // Apply time decay
    int new_dte = portfolio.days_to_expiration - scenario.days_elapsed;
    new_dte = std::max(new_dte, 0);

    // Recalculate portfolio value
    double stressed_value = recalculatePortfolioValue(
        portfolio, new_price, new_iv, new_dte
    );

    return stressed_value - current_value;
}
```

---

## Real-Time Risk Monitoring

### 1. Risk Limits

```cpp
struct RiskLimits {
    double max_loss_per_position;    // $500
    double max_loss_per_day;         // $2,000
    double max_loss_per_week;        // $5,000
    double max_portfolio_delta;      // ±500
    double max_portfolio_gamma;      // 100
    double max_portfolio_vega;       // 1,000
    double max_bp_usage_pct;         // 50%
    double max_position_concentration; // 10%
};

bool checkRiskLimits(
    const Portfolio& portfolio,
    const RiskLimits& limits
) {
    auto greeks = calculatePortfolioGreeks(portfolio);
    auto bp_metrics = calculateBPU(portfolio);

    bool within_limits = true;

    // Check delta limit
    if (std::abs(greeks.delta) > limits.max_portfolio_delta) {
        LOG_ERROR("Portfolio delta {} exceeds limit {}",
                  greeks.delta, limits.max_portfolio_delta);
        within_limits = false;
    }

    // Check gamma limit
    if (std::abs(greeks.gamma) > limits.max_portfolio_gamma) {
        LOG_ERROR("Portfolio gamma {} exceeds limit {}",
                  greeks.gamma, limits.max_portfolio_gamma);
        within_limits = false;
    }

    // Check BP usage
    if (bp_metrics.bp_usage_percent > limits.max_bp_usage_pct) {
        LOG_ERROR("BP usage {}% exceeds limit {}%",
                  bp_metrics.bp_usage_percent, limits.max_bp_usage_pct);
        within_limits = false;
    }

    return within_limits;
}
```

### 2. Real-Time Alerts

```cpp
struct RiskAlert {
    enum class Severity { INFO, WARNING, CRITICAL };

    Severity severity;
    std::string message;
    std::string metric_name;
    double current_value;
    double threshold;
    std::chrono::system_clock::time_point timestamp;
};

class RiskMonitor {
public:
    void checkAndAlert(const Portfolio& portfolio) {
        // Check Greeks
        if (std::abs(portfolio.delta) > delta_warning_threshold_) {
            emitAlert(RiskAlert{
                .severity = RiskAlert::Severity::WARNING,
                .message = "High delta exposure",
                .metric_name = "Portfolio Delta",
                .current_value = portfolio.delta,
                .threshold = delta_warning_threshold_
            });
        }

        // Check Theta
        double daily_theta = portfolio.theta;
        if (daily_theta < theta_minimum_threshold_) {
            emitAlert(RiskAlert{
                .severity = RiskAlert::Severity::INFO,
                .message = "Low theta generation",
                .metric_name = "Daily Theta",
                .current_value = daily_theta,
                .threshold = theta_minimum_threshold_
            });
        }

        // Check drawdown
        double current_drawdown = calculateCurrentDrawdown(portfolio);
        if (current_drawdown > max_drawdown_threshold_) {
            emitAlert(RiskAlert{
                .severity = RiskAlert::Severity::CRITICAL,
                .message = "Maximum drawdown exceeded",
                .metric_name = "Drawdown",
                .current_value = current_drawdown,
                .threshold = max_drawdown_threshold_
            });
        }
    }

private:
    double delta_warning_threshold_ = 500;
    double theta_minimum_threshold_ = 50;  // $50/day minimum
    double max_drawdown_threshold_ = 0.15;  // 15%
};
```

---

## Advanced Risk Metrics

### 1. Sortino Ratio

**Better than Sharpe for asymmetric returns:**

```cpp
double calculateSortinoRatio(
    const std::vector<double>& returns,
    double risk_free_rate = 0.04
) {
    double mean_return = calculateMean(returns);

    // Downside deviation (only negative returns)
    double sum_downside_squared = 0;
    int downside_count = 0;

    for (double ret : returns) {
        if (ret < 0) {
            sum_downside_squared += ret * ret;
            downside_count++;
        }
    }

    double downside_dev = std::sqrt(
        sum_downside_squared / downside_count
    );

    return (mean_return - risk_free_rate) / downside_dev;
}
```

### 2. Calmar Ratio

```cpp
double calculateCalmarRatio(
    double annualized_return,
    double max_drawdown
) {
    // Annualized return / maximum drawdown
    return annualized_return / std::abs(max_drawdown);
}
```

### 3. Omega Ratio

```cpp
double calculateOmegaRatio(
    const std::vector<double>& returns,
    double threshold = 0.0
) {
    double gains = 0;
    double losses = 0;

    for (double ret : returns) {
        if (ret > threshold) {
            gains += (ret - threshold);
        } else {
            losses += (threshold - ret);
        }
    }

    return gains / losses;
}
```

---

## Position Sizing Framework

### 1. Fixed Fractional Method

```cpp
double calculatePositionSize_FixedFractional(
    double account_value,
    double risk_per_trade_pct,  // e.g., 2%
    double max_loss_per_contract
) {
    double max_risk_dollars = account_value * risk_per_trade_pct;
    int contracts = static_cast<int>(max_risk_dollars / max_loss_per_contract);

    return std::max(contracts, 1);
}
```

### 2. Delta-Adjusted Sizing

```cpp
int calculatePositionSize_DeltaAdjusted(
    double target_delta,
    double delta_per_contract,
    int max_contracts
) {
    int contracts = static_cast<int>(
        std::abs(target_delta / delta_per_contract)
    );

    return std::min(contracts, max_contracts);
}
```

### 3. Volatility-Adjusted Sizing

```cpp
int calculatePositionSize_VolatilityAdjusted(
    double account_value,
    double base_position_size,
    double current_volatility,
    double average_volatility
) {
    // Reduce size when volatility is high
    double vol_ratio = average_volatility / current_volatility;
    double adjusted_size = base_position_size * vol_ratio;

    return static_cast<int>(adjusted_size);
}
```

---

## Risk-Adjusted Performance Metrics

### 1. Risk-Adjusted Return

```cpp
struct PerformanceMetrics {
    double total_return;
    double annualized_return;
    double volatility;
    double sharpe_ratio;
    double sortino_ratio;
    double max_drawdown;
    double calmar_ratio;
    double win_rate;
    double profit_factor;
    double expectancy;
};

PerformanceMetrics calculatePerformanceMetrics(
    const std::vector<Trade>& trades,
    double initial_capital
) {
    PerformanceMetrics metrics;

    // Calculate returns
    std::vector<double> returns;
    for (const auto& trade : trades) {
        double trade_return = trade.pnl / initial_capital;
        returns.push_back(trade_return);
    }

    // Total and annualized return
    double final_capital = calculateFinalCapital(trades, initial_capital);
    metrics.total_return = (final_capital / initial_capital) - 1.0;

    double years = calculateTradingYears(trades);
    metrics.annualized_return = std::pow(1 + metrics.total_return, 1.0/years) - 1;

    // Volatility
    metrics.volatility = calculateStandardDeviation(returns) *
                        std::sqrt(252);  // Annualized

    // Sharpe ratio
    metrics.sharpe_ratio = calculateSharpeRatio(returns);

    // Sortino ratio
    metrics.sortino_ratio = calculateSortinoRatio(returns);

    // Drawdown
    metrics.max_drawdown = calculateMaxDrawdown(trades, initial_capital);

    // Calmar ratio
    metrics.calmar_ratio = metrics.annualized_return /
                          std::abs(metrics.max_drawdown);

    // Win rate
    int wins = std::count_if(trades.begin(), trades.end(),
                            [](const Trade& t) { return t.pnl > 0; });
    metrics.win_rate = static_cast<double>(wins) / trades.size();

    // Profit factor
    double gross_profit = 0, gross_loss = 0;
    for (const auto& trade : trades) {
        if (trade.pnl > 0) gross_profit += trade.pnl;
        else gross_loss += std::abs(trade.pnl);
    }
    metrics.profit_factor = gross_profit / gross_loss;

    // Expectancy
    double avg_win = gross_profit / wins;
    double avg_loss = gross_loss / (trades.size() - wins);
    metrics.expectancy = (metrics.win_rate * avg_win) -
                        ((1 - metrics.win_rate) * avg_loss);

    return metrics;
}
```

---

## Margin and Leverage Risk

### 1. Margin Call Probability

```cpp
double estimateMarginCallProbability(
    double account_value,
    double bp_used,
    double maintenance_margin_pct = 0.25
) {
    double required_equity = bp_used * maintenance_margin_pct;
    double excess_equity = account_value - required_equity;

    if (excess_equity <= 0) {
        return 1.0;  // Already in margin call
    }

    // Calculate price move that would trigger margin call
    // (Simplified - actual calculation depends on position structure)
    double buffer_pct = excess_equity / account_value;

    // Use historical volatility to estimate probability
    // of buffer being depleted

    return calculateProbabilityOfMove(buffer_pct, portfolio_volatility);
}
```

### 2. Leverage Ratio

```cpp
double calculateLeverageRatio(const Portfolio& portfolio) {
    double total_notional = 0;

    for (const auto& position : portfolio.positions) {
        total_notional += std::abs(position.delta) *
                         position.underlying_price * 100;
    }

    return total_notional / portfolio.account_value;
}
```

---

## Implementation Files

```
src/risk_management/
├── position_sizer.cpp         # Position sizing algorithms
├── stop_loss.cpp              # Stop-loss logic
├── portfolio_constraints.cpp  # Risk limit enforcement
├── kelly_criterion.cpp        # Kelly formula
├── monte_carlo.cpp            # MC VaR, stress testing
├── risk_monitor.cpp           # Real-time risk tracking (to implement)
├── correlation_analyzer.cpp   # Correlation matrix (to implement)
└── var_calculator.cpp         # VaR calculations (to implement)
```

**Integration Points:**
- Market Intelligence Engine → Risk metrics for trade signals
- Trading Decision Engine → Risk limits for order sizing
- Portfolio Optimizer → Risk-adjusted position sizing
- Explainability Layer → Risk attribution for decisions

---

## Best Practices

### Risk Management Rules

1. **Never risk more than 2% per trade**
2. **Maintain 20-30% BP buffer for adjustments**
3. **Diversify across 5-10 uncorrelated underlyings**
4. **Monitor portfolio Greeks daily**
5. **Set hard stop-losses at 2-3x initial credit**
6. **Close positions at 50% profit (for credit strategies)**
7. **Avoid earnings in high IV rank environments**
8. **Size positions using Kelly / Fixed Fractional**

### Greeks Targets (Portfolio)

```
Delta: ±200 (relatively neutral)
Gamma: < 50 (avoid pin risk)
Theta: > $100/day (minimum income target)
Vega: < ±1000 (limit IV exposure)
```

### Backtesting Requirements

1. **Out-of-sample validation**
2. **Include commissions and slippage**
3. **Account for assignment risk**
4. **Model early exercise (American options)**
5. **Use bid/ask spreads, not mid-price**
6. **Simulate margin calls**
7. **Test across different market regimes**

---

## Performance Tracking

### Trade Journal Fields

```cpp
struct TradeRecord {
    std::string symbol;
    std::string strategy;
    std::chrono::system_clock::time_point entry_date;
    std::chrono::system_clock::time_point exit_date;

    double entry_price;
    double exit_price;
    int quantity;

    double pnl;
    double pnl_percent;
    double max_profit_reached;
    double max_loss_reached;

    double entry_iv;
    double exit_iv;
    double iv_rank_at_entry;

    PortfolioGreeks entry_greeks;
    PortfolioGreeks exit_greeks;

    std::string exit_reason;  // "Profit target", "Stop loss", "Expiration"

    // Post-analysis
    double expectancy_at_entry;
    double actual_outcome;
    bool met_criteria;
};
```

---

This framework provides comprehensive risk evaluation for the BigBrotherAnalytics platform, ensuring robust risk management across all trading strategies.
