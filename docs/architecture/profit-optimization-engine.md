# Profit Maximization & Risk Minimization Engine

## Executive Summary

The Profit Maximization & Risk Minimization Engine is a quantitative portfolio optimization system that dynamically allocates capital across stocks, options, and interest rate instruments to maximize risk-adjusted returns. The engine combines Modern Portfolio Theory, options Greeks optimization, and machine learning to construct optimal portfolios that maximize Sharpe ratio while respecting risk constraints.

**Key Features:**
- Multi-asset optimization (stocks, options, bonds, hybrids)
- Real-time portfolio rebalancing based on market conditions
- Greeks-balanced options portfolios (delta-neutral, theta-positive)
- Interest rate risk hedging with bonds/options
- Kelly criterion-based position sizing
- Monte Carlo simulation for robust optimization
- Machine learning for return prediction and covariance estimation

---

## Table of Contents

1. [Mathematical Framework](#mathematical-framework)
2. [Optimization Objectives](#optimization-objectives)
3. [Multi-Asset Optimization](#multi-asset-optimization)
4. [Options Portfolio Optimization](#options-portfolio-optimization)
5. [Interest Rate Integration](#interest-rate-integration)
6. [Hybrid Strategies](#hybrid-strategies)
7. [Tier 1 Implementation (POC)](#tier-1-implementation-poc)
8. [Advanced Features (Tier 2+)](#advanced-features-tier-2)

---

## Mathematical Framework

### 1. Mean-Variance Optimization (Markowitz)

**Objective:** Maximize Sharpe ratio

```
maximize: (μᵀw - r_f) / √(wᵀΣw)

subject to:
  Σw_i = 1           (fully invested)
  w_i ≥ 0            (long-only) or w_i ∈ ℝ (long-short)
  0 ≤ w_i ≤ w_max    (position limits)

where:
w = portfolio weights vector
μ = expected returns vector
Σ = covariance matrix
r_f = risk-free rate
```

**Implementation:**
```cpp
struct OptimizationResult {
    Eigen::VectorXd weights;
    double expected_return;
    double expected_volatility;
    double sharpe_ratio;
    double portfolio_var_95;
};

OptimizationResult optimizePortfolio(
    const Eigen::VectorXd& expected_returns,
    const Eigen::MatrixXd& covariance_matrix,
    double risk_free_rate,
    const PortfolioConstraints& constraints
) {
    // Use quadratic programming solver
    // Libraries: OSQP, qpOASES, or Intel MKL

    OptimizationResult result;

    // Objective: maximize Sharpe ratio
    // Equivalent to: minimize variance for target return

    // Convert to quadratic programming form:
    // minimize: (1/2) * wᵀΣw - λ*μᵀw
    // where λ is Lagrange multiplier for return target

    // Solve using Sequential Quadratic Programming (SQP)
    result.weights = solveQP(covariance_matrix, expected_returns, constraints);

    // Calculate performance metrics
    result.expected_return = expected_returns.dot(result.weights);
    result.expected_volatility = std::sqrt(
        result.weights.transpose() * covariance_matrix * result.weights
    );
    result.sharpe_ratio = (result.expected_return - risk_free_rate) /
                          result.expected_volatility;

    return result;
}
```

### 2. Black-Litterman Model

**Advantage:** Incorporate market equilibrium + investor views

```
Combined Return Estimate:
E[R] = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]

where:
Π = market equilibrium returns (from CAPM)
P = matrix linking views to assets
Q = investor views on returns
Ω = uncertainty in views
τ = scaling factor (typically 0.025)
```

**Use Case:**
- Start with market-implied returns
- Add views from ML models (e.g., "Tech will outperform by 2% next month")
- Combine to get robust return estimates

**Implementation:**
```cpp
Eigen::VectorXd blackLittermanReturns(
    const Eigen::VectorXd& market_cap_weights,
    const Eigen::MatrixXd& covariance,
    const Eigen::MatrixXd& view_matrix,  // P
    const Eigen::VectorXd& view_returns, // Q
    const Eigen::MatrixXd& view_uncertainty, // Ω
    double tau = 0.025
) {
    // Market equilibrium returns (reverse optimization)
    Eigen::VectorXd pi = calculateEquilibriumReturns(
        market_cap_weights, covariance, tau
    );

    // Combine equilibrium with views
    auto cov_scaled = tau * covariance;
    auto precision_prior = cov_scaled.inverse();
    auto precision_views = view_matrix.transpose() *
                          view_uncertainty.inverse() *
                          view_matrix;

    auto combined_precision = precision_prior + precision_views;
    auto combined_cov = combined_precision.inverse();

    auto mean_prior = precision_prior * pi;
    auto mean_views = view_matrix.transpose() *
                     view_uncertainty.inverse() *
                     view_returns;

    Eigen::VectorXd posterior_returns = combined_cov *
                                        (mean_prior + mean_views);

    return posterior_returns;
}
```

### 3. Risk Parity

**Concept:** Allocate risk equally across assets

```
Risk Contribution of asset i:
RC_i = w_i * (Σw)_i

Target: RC_i = RC_j for all i,j

Approximately:
w_i ∝ 1/σ_i
```

**Implementation:**
```cpp
Eigen::VectorXd calculateRiskParityWeights(
    const Eigen::MatrixXd& covariance
) {
    int n = covariance.rows();
    Eigen::VectorXd volatilities(n);

    // Extract volatilities
    for (int i = 0; i < n; ++i) {
        volatilities(i) = std::sqrt(covariance(i, i));
    }

    // Inverse volatility weighting
    Eigen::VectorXd weights = volatilities.array().inverse();

    // Normalize to sum to 1
    weights /= weights.sum();

    return weights;
}
```

---

## Optimization Objectives

### 1. Maximum Sharpe Ratio

**Goal:** Best risk-adjusted returns

```cpp
struct SharpeOptimization {
    static OptimizationResult optimize(
        const Eigen::VectorXd& returns,
        const Eigen::MatrixXd& cov,
        double rf_rate
    ) {
        // Analytical solution exists for unconstrained case
        Eigen::VectorXd excess_returns = returns.array() - rf_rate;
        Eigen::VectorXd weights = cov.inverse() * excess_returns;

        // Normalize
        weights /= weights.sum();

        return createResult(weights, returns, cov, rf_rate);
    }
};
```

### 2. Minimum Variance

**Goal:** Lowest volatility portfolio

```cpp
struct MinimumVarianceOptimization {
    static OptimizationResult optimize(
        const Eigen::MatrixXd& cov,
        const PortfolioConstraints& constraints
    ) {
        // minimize: wᵀΣw
        // subject to: Σw_i = 1

        Eigen::VectorXd ones = Eigen::VectorXd::Ones(cov.rows());
        Eigen::VectorXd weights = cov.inverse() * ones;
        weights /= weights.sum();

        return createResult(weights, cov);
    }
};
```

### 3. Maximum Return for Target Risk

**Goal:** Highest return at specified risk level

```cpp
struct TargetRiskOptimization {
    static OptimizationResult optimize(
        const Eigen::VectorXd& returns,
        const Eigen::MatrixXd& cov,
        double target_volatility
    ) {
        // maximize: μᵀw
        // subject to: √(wᵀΣw) = σ_target
        //             Σw_i = 1

        // Use Lagrange multipliers or SQP solver
        return solveWithConstraint(returns, cov, target_volatility);
    }
};
```

### 4. Conditional Value at Risk (CVaR) Minimization

**Goal:** Minimize tail risk

```cpp
struct CVaROptimization {
    static OptimizationResult optimize(
        const std::vector<Eigen::VectorXd>& scenario_returns,
        double confidence_level = 0.95
    ) {
        // minimize: CVaR_α(w)
        // subject to: Σw_i = 1

        // CVaR is convex → can use linear programming
        return solveCVaRLP(scenario_returns, confidence_level);
    }
};
```

---

## Multi-Asset Optimization

### 1. Stock + Options Portfolio

**Objective:** Combine directional bets (stocks) with income generation (options)

```cpp
struct MultiAssetAllocation {
    double stock_allocation;     // e.g., 60%
    double options_allocation;   // e.g., 30%
    double cash_allocation;      // e.g., 10%

    // Within options allocation
    double straddle_pct;         // Volatility plays
    double iron_condor_pct;      // Income generation
    double vertical_spread_pct;  // Directional with defined risk
};

MultiAssetAllocation optimizeMultiAsset(
    const MarketConditions& market,
    double account_value,
    const RiskProfile& risk_tolerance
) {
    MultiAssetAllocation allocation;

    // Factor-based allocation
    double volatility_regime = classifyVolatilityRegime(market);
    double trend_strength = calculateTrendStrength(market);

    if (volatility_regime == VolatilityRegime::HIGH) {
        // High IV → sell premium
        allocation.options_allocation = 0.40;  // Higher options
        allocation.stock_allocation = 0.50;
        allocation.cash_allocation = 0.10;

        // Within options: favor credit strategies
        allocation.iron_condor_pct = 0.60;
        allocation.vertical_spread_pct = 0.30;
        allocation.straddle_pct = 0.10;
    }
    else if (volatility_regime == VolatilityRegime::LOW) {
        // Low IV → buy options, more stocks
        allocation.stock_allocation = 0.70;
        allocation.options_allocation = 0.20;
        allocation.cash_allocation = 0.10;

        // Within options: buy vol
        allocation.straddle_pct = 0.50;
        allocation.vertical_spread_pct = 0.40;
        allocation.iron_condor_pct = 0.10;
    }
    else {
        // Normal regime → balanced
        allocation.stock_allocation = 0.60;
        allocation.options_allocation = 0.30;
        allocation.cash_allocation = 0.10;

        allocation.iron_condor_pct = 0.40;
        allocation.vertical_spread_pct = 0.40;
        allocation.straddle_pct = 0.20;
    }

    return allocation;
}
```

### 2. Expected Utility Maximization

**For risk-averse investors:**

```cpp
double calculateExpectedUtility(
    const Eigen::VectorXd& weights,
    const Eigen::VectorXd& returns,
    const Eigen::MatrixXd& cov,
    double risk_aversion  // λ, typically 2-10
) {
    double expected_return = returns.dot(weights);
    double variance = weights.transpose() * cov * weights;

    // CARA utility: U = E[R] - (λ/2)*Var[R]
    return expected_return - 0.5 * risk_aversion * variance;
}

Eigen::VectorXd maximizeUtility(
    const Eigen::VectorXd& returns,
    const Eigen::MatrixXd& cov,
    double risk_aversion
) {
    // Analytical solution
    Eigen::VectorXd weights = (1.0 / risk_aversion) * cov.inverse() * returns;

    // Normalize if needed
    if (std::abs(weights.sum() - 1.0) > 1e-6) {
        weights /= weights.sum();
    }

    return weights;
}
```

---

## Options Portfolio Optimization

### 1. Greeks-Balanced Portfolio

**Objective:** Maximize theta while maintaining delta-neutral, controlling gamma/vega

```cpp
struct GreeksConstraints {
    double target_delta;        // 0 for delta-neutral
    double delta_tolerance;     // ±50
    double max_gamma;           // 100
    double max_vega;            // 1000
    double min_theta;           // $100/day
};

struct OptionsPosition {
    std::string symbol;
    std::string strategy;  // "iron_condor", "strangle", etc.
    int quantity;
    double delta;
    double gamma;
    double theta;
    double vega;
    double premium;
    double max_loss;
};

class GreeksOptimizer {
public:
    std::vector<OptionsPosition> optimizeGreeks(
        const std::vector<OptionsPosition>& candidates,
        double capital_available,
        const GreeksConstraints& constraints
    ) {
        // Formulate as mixed-integer programming problem:
        //
        // maximize: Σ(θ_i * n_i)  (total theta)
        //
        // subject to:
        //   |Σ(δ_i * n_i)| ≤ delta_tolerance
        //   Σ(γ_i * n_i) ≤ max_gamma
        //   Σ(ν_i * n_i) ≤ max_vega
        //   Σ(BPR_i * n_i) ≤ capital_available
        //   n_i ∈ {0, 1, 2, ...}  (integer positions)

        return solveMIP(candidates, capital_available, constraints);
    }

private:
    std::vector<OptionsPosition> solveMIP(
        const std::vector<OptionsPosition>& candidates,
        double capital,
        const GreeksConstraints& constraints
    ) {
        // Use COIN-OR CBC, Gurobi, or CPLEX for MIP
        // For Tier 1: Use greedy heuristic

        return greedyGreeksOptimization(candidates, capital, constraints);
    }

    std::vector<OptionsPosition> greedyGreeksOptimization(
        const std::vector<OptionsPosition>& candidates,
        double capital,
        const GreeksConstraints& constraints
    ) {
        std::vector<OptionsPosition> selected;

        // Sort by theta efficiency (theta / BPR)
        auto sorted = candidates;
        std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) {
                return (a.theta / a.max_loss) > (b.theta / b.max_loss);
            });

        double current_delta = 0;
        double current_gamma = 0;
        double current_vega = 0;
        double current_theta = 0;
        double capital_used = 0;

        for (const auto& candidate : sorted) {
            // Check if adding this position violates constraints
            double new_delta = current_delta + candidate.delta;
            double new_gamma = current_gamma + candidate.gamma;
            double new_vega = current_vega + candidate.vega;
            double new_capital = capital_used + candidate.max_loss;

            if (std::abs(new_delta) <= constraints.delta_tolerance &&
                new_gamma <= constraints.max_gamma &&
                new_vega <= constraints.max_vega &&
                new_capital <= capital) {

                selected.push_back(candidate);
                current_delta = new_delta;
                current_gamma = new_gamma;
                current_vega = new_vega;
                current_theta += candidate.theta;
                capital_used = new_capital;
            }
        }

        return selected;
    }
};
```

### 2. Volatility Targeting

**Strategy:** Adjust position sizes based on realized vs implied volatility

```cpp
class VolatilityTargetingOptimizer {
public:
    struct VolatilityMetrics {
        double realized_vol_30d;
        double implied_vol;
        double iv_rank;
        double iv_percentile;
        double vol_risk_premium;  // IV - RV
    };

    std::vector<OptionsPosition> selectStrategies(
        const VolatilityMetrics& vol_metrics,
        double capital
    ) {
        std::vector<OptionsPosition> positions;

        // High IV environment (IV Rank > 50)
        if (vol_metrics.iv_rank > 50) {
            // Sell premium strategies
            positions.push_back(
                constructIronCondor(vol_metrics, capital * 0.40)
            );
            positions.push_back(
                constructShortStrangle(vol_metrics, capital * 0.30)
            );
            positions.push_back(
                constructCreditSpread(vol_metrics, capital * 0.30)
            );
        }
        // Low IV environment (IV Rank < 30)
        else if (vol_metrics.iv_rank < 30) {
            // Buy options strategies
            positions.push_back(
                constructLongStraddle(vol_metrics, capital * 0.30)
            );
            positions.push_back(
                constructLongStrangle(vol_metrics, capital * 0.40)
            );
            positions.push_back(
                constructDebitSpread(vol_metrics, capital * 0.30)
            );
        }
        // Neutral IV (30-50)
        else {
            // Mixed strategies
            positions.push_back(
                constructIronCondor(vol_metrics, capital * 0.50)
            );
            positions.push_back(
                constructCalendarSpread(vol_metrics, capital * 0.30)
            );
            positions.push_back(
                constructRatioSpread(vol_metrics, capital * 0.20)
            );
        }

        return positions;
    }

private:
    OptionsPosition constructIronCondor(
        const VolatilityMetrics& vol,
        double capital
    ) {
        // Strike selection based on expected move
        double expected_move = calculateExpectedMove(vol);

        OptionsPosition pos;
        pos.strategy = "iron_condor";

        // Place short strikes at 1 SD
        // Place long strikes at 1.5 SD (or width that fits capital)

        pos.delta = 0;  // Approximately delta-neutral
        pos.gamma = calculateGamma(pos);
        pos.theta = calculateTheta(pos);  // Positive
        pos.vega = calculateVega(pos);    // Negative (short vol)

        return pos;
    }
};
```

### 3. Theta-Maximization Portfolio

**Goal:** Maximum daily income from time decay

```cpp
struct ThetaOptimizationResult {
    std::vector<OptionsPosition> positions;
    double daily_theta;
    double portfolio_delta;
    double capital_efficiency;  // theta / capital_at_risk
};

ThetaOptimizationResult maximizeTheta(
    const std::vector<std::string>& underlyings,
    const MarketData& market_data,
    double capital,
    const GreeksConstraints& constraints
) {
    std::vector<OptionsPosition> candidates;

    // Generate candidate strategies for each underlying
    for (const auto& symbol : underlyings) {
        auto vol_metrics = getVolatilityMetrics(symbol, market_data);

        // Only consider selling premium in high IV
        if (vol_metrics.iv_rank > 40) {
            candidates.push_back(
                generateIronCondor(symbol, vol_metrics, 0.16)  // 16 delta
            );
            candidates.push_back(
                generateIronCondor(symbol, vol_metrics, 0.10)  // 10 delta
            );
            candidates.push_back(
                generateShortStrangle(symbol, vol_metrics, 0.16)
            );
            candidates.push_back(
                generateCreditSpread(symbol, vol_metrics, "put")
            );
            candidates.push_back(
                generateCreditSpread(symbol, vol_metrics, "call")
            );
        }
    }

    // Optimize for maximum theta
    auto selected = GreeksOptimizer().optimizeGreeks(
        candidates, capital, constraints
    );

    ThetaOptimizationResult result;
    result.positions = selected;

    // Calculate portfolio metrics
    for (const auto& pos : selected) {
        result.daily_theta += pos.theta;
        result.portfolio_delta += pos.delta;
    }

    result.capital_efficiency = result.daily_theta /
                               calculateTotalCapitalAtRisk(selected);

    return result;
}
```

---

## Interest Rate Integration

### 1. Bond-Equity Allocation

**Dynamic allocation based on yield curve:**

```cpp
struct BondEquityAllocation {
    double stock_weight;
    double bond_weight;
    double cash_weight;
};

BondEquityAllocation optimizeBondEquity(
    const YieldCurve& yields,
    const EquityMarketData& equity_data,
    double risk_aversion
) {
    BondEquityAllocation allocation;

    // Calculate equity risk premium
    double earnings_yield = equity_data.earnings / equity_data.price;
    double treasury_10y = yields.getYield(10.0);  // 10-year rate

    double equity_risk_premium = earnings_yield - treasury_10y;

    // Fed Model comparison
    if (equity_risk_premium > 0.02) {
        // Stocks attractive
        allocation.stock_weight = 0.70;
        allocation.bond_weight = 0.25;
        allocation.cash_weight = 0.05;
    }
    else if (equity_risk_premium < -0.01) {
        // Bonds attractive
        allocation.stock_weight = 0.40;
        allocation.bond_weight = 0.55;
        allocation.cash_weight = 0.05;
    }
    else {
        // Balanced
        allocation.stock_weight = 0.60;
        allocation.bond_weight = 0.35;
        allocation.cash_weight = 0.05;
    }

    // Adjust for risk aversion
    allocation = adjustForRiskAversion(allocation, risk_aversion);

    return allocation;
}
```

### 2. Duration Matching

**Immunize portfolio against interest rate changes:**

```cpp
struct DurationMetrics {
    double macaulay_duration;
    double modified_duration;
    double convexity;
    double dv01;  // Dollar value of 1 bp change
};

DurationMetrics calculatePortfolioDuration(
    const std::vector<Bond>& bonds,
    const std::vector<double>& weights
) {
    DurationMetrics metrics{};

    for (size_t i = 0; i < bonds.size(); ++i) {
        metrics.macaulay_duration += weights[i] * bonds[i].duration;
        metrics.convexity += weights[i] * bonds[i].convexity;
    }

    metrics.modified_duration = metrics.macaulay_duration /
                               (1 + bonds[0].yield);

    return metrics;
}

std::vector<double> matchTargetDuration(
    const std::vector<Bond>& bonds,
    double target_duration
) {
    // Solve for weights that achieve target duration
    // minimize: distance from target
    // subject to: Σw_i*D_i = D_target, Σw_i = 1

    return solveDurationMatching(bonds, target_duration);
}
```

### 3. Interest Rate Options (Swaptions, Caps, Floors)

**Hedge interest rate risk:**

```cpp
struct InterestRateHedge {
    enum class Type { CAP, FLOOR, SWAPTION };

    Type type;
    double notional;
    double strike_rate;
    double expiration_years;
    double premium;
};

InterestRateHedge selectRateHedge(
    const Portfolio& portfolio,
    const YieldCurve& current_yields
) {
    InterestRateHedge hedge;

    // Calculate portfolio duration
    double portfolio_duration = calculateDuration(portfolio);

    // If long duration → buy caps (hedge against rising rates)
    if (portfolio_duration > 5.0) {
        hedge.type = InterestRateHedge::Type::CAP;
        hedge.strike_rate = current_yields.getYield(10.0) + 0.01;  // 100 bp OTM
        hedge.notional = portfolio.bond_value;
    }
    // If short duration → buy floors (hedge against falling rates)
    else if (portfolio_duration < 2.0) {
        hedge.type = InterestRateHedge::Type::FLOOR;
        hedge.strike_rate = current_yields.getYield(10.0) - 0.01;
        hedge.notional = portfolio.bond_value;
    }

    return hedge;
}
```

---

## Hybrid Strategies

### 1. Covered Call Optimization

**Enhance stock returns with call premium:**

```cpp
struct CoveredCallStrategy {
    std::string stock_symbol;
    int shares_owned;
    double strike_selection;  // % OTM
    int days_to_expiration;

    double expected_return;
    double downside_protection;  // Premium as % of stock price
};

CoveredCallStrategy optimizeCoveredCall(
    const StockPosition& stock_position,
    const OptionsChain& chain
) {
    CoveredCallStrategy strategy;
    strategy.stock_symbol = stock_position.symbol;
    strategy.shares_owned = stock_position.shares;

    double best_sharpe = -std::numeric_limits<double>::infinity();

    // Evaluate different strikes and expirations
    for (double otm_pct : {0.02, 0.05, 0.10, 0.15}) {
        for (int dte : {7, 14, 21, 30, 45}) {
            auto call = findCall(chain, stock_position.price * (1 + otm_pct), dte);

            if (!call) continue;

            // Calculate metrics
            double max_return = (call->strike - stock_position.price) +
                               call->premium;
            double prob_assignment = 1.0 - call->delta;

            double expected_return = prob_assignment * max_return +
                                    (1 - prob_assignment) * call->premium;

            double volatility = stock_position.volatility;

            double sharpe = expected_return / (volatility * std::sqrt(dte / 365.0));

            if (sharpe > best_sharpe) {
                best_sharpe = sharpe;
                strategy.strike_selection = otm_pct;
                strategy.days_to_expiration = dte;
                strategy.expected_return = expected_return;
                strategy.downside_protection = call->premium / stock_position.price;
            }
        }
    }

    return strategy;
}
```

### 2. Protective Put Selection

**Optimal strike selection for portfolio insurance:**

```cpp
struct ProtectivePutStrategy {
    double strike_price;
    double premium_cost;
    double protection_level;  // % below current price
    int expiration_days;

    double annualized_cost;  // Premium / time
    double cost_efficiency;  // Protection / premium
};

ProtectivePutStrategy optimizeProtectivePut(
    const StockPosition& stock_position,
    double max_acceptable_loss_pct,  // e.g., 10%
    const OptionsChain& chain
) {
    ProtectivePutStrategy strategy;

    double target_strike = stock_position.price * (1 - max_acceptable_loss_pct);

    double best_efficiency = -std::numeric_limits<double>::infinity();

    // Evaluate different expirations
    for (int dte : {30, 60, 90, 180}) {
        auto put_option = findPut(chain, target_strike, dte);

        if (!put_option) continue;

        double protection_amt = stock_position.price - put_option->strike;
        double cost_efficiency = protection_amt / put_option->premium;
        double annualized_cost = (put_option->premium / stock_position.price) *
                                (365.0 / dte);

        // Optimize for cost efficiency
        if (cost_efficiency > best_efficiency &&
            annualized_cost < 0.15) {  // Max 15% annualized cost

            best_efficiency = cost_efficiency;
            strategy.strike_price = put_option->strike;
            strategy.premium_cost = put_option->premium;
            strategy.protection_level = max_acceptable_loss_pct;
            strategy.expiration_days = dte;
            strategy.annualized_cost = annualized_cost;
            strategy.cost_efficiency = cost_efficiency;
        }
    }

    return strategy;
}
```

### 3. Collar Strategy Optimization

**Cap upside, protect downside:**

```cpp
struct CollarStrategy {
    double protective_put_strike;
    double covered_call_strike;
    double net_cost;  // Can be zero or credit
    double protected_range;
};

CollarStrategy optimizeCollar(
    const StockPosition& stock_position,
    const OptionsChain& chain,
    bool target_zero_cost = true
) {
    CollarStrategy collar;

    double current_price = stock_position.price;

    if (target_zero_cost) {
        // Find put/call combination with net zero cost

        for (double put_otm : {0.05, 0.10, 0.15, 0.20}) {
            double put_strike = current_price * (1 - put_otm);
            auto put = findPut(chain, put_strike, 45);

            if (!put) continue;

            // Find call strike that offsets put premium
            double target_call_premium = put->premium;

            for (double call_otm : {0.05, 0.10, 0.15, 0.20, 0.25}) {
                double call_strike = current_price * (1 + call_otm);
                auto call = findCall(chain, call_strike, 45);

                if (!call) continue;

                double net_cost = put->premium - call->premium;

                // Check if close to zero cost
                if (std::abs(net_cost) < 0.50) {  // Within $0.50
                    collar.protective_put_strike = put_strike;
                    collar.covered_call_strike = call_strike;
                    collar.net_cost = net_cost;
                    collar.protected_range = (call_strike - put_strike) /
                                            current_price;

                    return collar;
                }
            }
        }
    }

    return collar;
}
```

### 4. Convertible Bond Arbitrage

**Exploit pricing inefficiencies:**

```cpp
struct ConvertibleBondPosition {
    std::string bond_id;
    double bond_price;
    double conversion_ratio;
    double stock_price;
    double conversion_value;  // conversion_ratio * stock_price
    double conversion_premium;  // (bond_price - conversion_value) / conversion_value

    // Hedge ratios
    double delta;
    double gamma;

    // Arbitrage opportunity
    double theoretical_value;
    double mispricing;
};

ConvertibleBondPosition analyzeConvertible(
    const ConvertibleBond& bond,
    const Stock& stock,
    const OptionsChain& chain
) {
    ConvertibleBondPosition pos;

    pos.bond_price = bond.price;
    pos.conversion_ratio = bond.conversion_ratio;
    pos.stock_price = stock.price;
    pos.conversion_value = pos.conversion_ratio * pos.stock_price;
    pos.conversion_premium = (pos.bond_price - pos.conversion_value) /
                             pos.conversion_value;

    // Calculate embedded option value
    // Convertible = Bond + Embedded Call Option

    double straight_bond_value = calculateBondValue(
        bond.coupon, bond.maturity, bond.credit_spread
    );

    double embedded_call_value = pos.bond_price - straight_bond_value;

    // Compare to market call options
    auto synthetic_call = findEquivalentCall(
        chain, bond.conversion_ratio, bond.maturity
    );

    if (synthetic_call) {
        pos.theoretical_value = straight_bond_value + synthetic_call->price;
        pos.mispricing = pos.bond_price - pos.theoretical_value;
    }

    // Calculate hedge ratio (delta)
    pos.delta = pos.conversion_ratio *
               calculateConvertibleDelta(bond, stock);

    return pos;
}
```

---

## Dynamic Rebalancing

### 1. Threshold Rebalancing

**Rebalance when weights drift beyond thresholds:**

```cpp
class ThresholdRebalancer {
public:
    struct RebalanceDecision {
        bool should_rebalance;
        std::vector<Trade> trades;
        double expected_cost;  // Commissions + slippage
        double expected_benefit;  // From returning to optimal
    };

    RebalanceDecision checkRebalance(
        const Portfolio& current_portfolio,
        const Eigen::VectorXd& target_weights,
        double threshold = 0.05  // 5% drift
    ) {
        RebalanceDecision decision;
        decision.should_rebalance = false;

        auto current_weights = calculateCurrentWeights(current_portfolio);

        double max_drift = 0;
        for (size_t i = 0; i < current_weights.size(); ++i) {
            double drift = std::abs(current_weights(i) - target_weights(i));
            max_drift = std::max(max_drift, drift);
        }

        if (max_drift > threshold) {
            decision.should_rebalance = true;
            decision.trades = generateRebalanceTrades(
                current_portfolio, target_weights
            );

            decision.expected_cost = estimateTransactionCosts(decision.trades);
            decision.expected_benefit = estimateRebalanceBenefit(
                current_portfolio, target_weights
            );
        }

        return decision;
    }
};
```

### 2. Time-Based Rebalancing

**Periodic rebalancing (monthly, quarterly):**

```cpp
class PeriodicRebalancer {
public:
    enum class Frequency { MONTHLY, QUARTERLY, ANNUALLY };

    void scheduleRebalance(
        const Portfolio& portfolio,
        Frequency freq
    ) {
        int days_between_rebalance;

        switch (freq) {
            case Frequency::MONTHLY:
                days_between_rebalance = 30;
                break;
            case Frequency::QUARTERLY:
                days_between_rebalance = 90;
                break;
            case Frequency::ANNUALLY:
                days_between_rebalance = 365;
                break;
        }

        // Schedule next rebalance
        auto last_rebalance = portfolio.last_rebalance_date;
        auto next_rebalance = last_rebalance +
                             std::chrono::days(days_between_rebalance);

        scheduleTask(next_rebalance, [this, portfolio]() {
            performRebalance(portfolio);
        });
    }
};
```

### 3. Volatility-Triggered Rebalancing

**Rebalance when volatility regime changes:**

```cpp
class VolatilityRebalancer {
public:
    bool checkVolatilityRegimeChange(
        const MarketData& current_data,
        const MarketData& baseline_data
    ) {
        double current_vix = current_data.vix;
        double baseline_vix = baseline_data.vix;

        // Regime change if VIX moves > 30%
        double vix_change = std::abs(current_vix - baseline_vix) / baseline_vix;

        if (vix_change > 0.30) {
            // Significant regime change → rebalance
            return true;
        }

        return false;
    }

    Portfolio rebalanceForVolRegime(
        const Portfolio& current_portfolio,
        VolatilityRegime new_regime
    ) {
        if (new_regime == VolatilityRegime::HIGH) {
            // Reduce equity exposure, increase options selling
            return constructHighVolPortfolio(current_portfolio);
        }
        else if (new_regime == VolatilityRegime::LOW) {
            // Increase equity, reduce options, or buy volatility
            return constructLowVolPortfolio(current_portfolio);
        }

        return current_portfolio;
    }
};
```

---

## Machine Learning Integration

### 1. Return Prediction

**Use ML models to forecast returns:**

```cpp
class MLReturnPredictor {
public:
    Eigen::VectorXd predictReturns(
        const std::vector<std::string>& symbols,
        const MarketFeatures& features
    ) {
        Eigen::VectorXd predictions(symbols.size());

        for (size_t i = 0; i < symbols.size(); ++i) {
            auto symbol_features = extractFeatures(symbols[i], features);

            // Use ensemble of models
            double xgboost_pred = xgboost_model_.predict(symbol_features);
            double lstm_pred = lstm_model_.predict(symbol_features);
            double transformer_pred = transformer_model_.predict(symbol_features);

            // Weighted ensemble
            predictions(i) = 0.40 * xgboost_pred +
                            0.30 * lstm_pred +
                            0.30 * transformer_pred;
        }

        return predictions;
    }

private:
    XGBoostModel xgboost_model_;
    LSTMModel lstm_model_;
    TransformerModel transformer_model_;
};
```

### 2. Covariance Matrix Estimation

**Shrinkage estimators for stable covariance:**

```cpp
Eigen::MatrixXd estimateCovariance(
    const Eigen::MatrixXd& returns_history,
    double shrinkage_intensity = 0.0  // 0 = sample, 1 = identity
) {
    // Ledoit-Wolf shrinkage
    Eigen::MatrixXd sample_cov = calculateSampleCovariance(returns_history);

    // Target: constant correlation model or identity
    Eigen::MatrixXd target = constructTarget(sample_cov);

    // Optimal shrinkage intensity (if not provided)
    if (shrinkage_intensity == 0.0) {
        shrinkage_intensity = calculateOptimalShrinkage(
            returns_history, sample_cov, target
        );
    }

    // Shrunk covariance
    Eigen::MatrixXd shrunk_cov = shrinkage_intensity * target +
                                 (1 - shrinkage_intensity) * sample_cov;

    return shrunk_cov;
}
```

---

## Tier 1 Implementation (POC)

### Phase 1: Basic Optimization (Months 1-2)

**Scope:**
- Mean-variance optimization for stock portfolio
- Simple options selection (iron condors only)
- Fixed allocation (60/30/10 stock/options/cash)
- Threshold rebalancing

**Implementation:**
```cpp
// File: src/portfolio_optimizer/basic_optimizer.cpp

class Tier1Optimizer {
public:
    Portfolio optimizePortfolio(
        const std::vector<std::string>& stock_universe,
        const MarketData& market_data,
        double capital
    ) {
        Portfolio portfolio;

        // 1. Stock allocation (60%)
        double stock_capital = capital * 0.60;
        auto stock_weights = optimizeStocks(stock_universe, market_data);
        portfolio.stock_positions = allocateStocks(
            stock_weights, stock_capital, market_data
        );

        // 2. Options allocation (30%)
        double options_capital = capital * 0.30;
        portfolio.options_positions = selectIronCondors(
            stock_universe, options_capital, market_data
        );

        // 3. Cash (10%)
        portfolio.cash = capital * 0.10;

        return portfolio;
    }

private:
    Eigen::VectorXd optimizeStocks(
        const std::vector<std::string>& universe,
        const MarketData& data
    ) {
        // Simple equal-risk contribution
        return calculateEqualRiskWeights(universe, data);
    }

    std::vector<OptionsPosition> selectIronCondors(
        const std::vector<std::string>& universe,
        double capital,
        const MarketData& data
    ) {
        std::vector<OptionsPosition> positions;

        for (const auto& symbol : universe) {
            auto vol_metrics = getVolatilityMetrics(symbol, data);

            // Only trade high IV rank
            if (vol_metrics.iv_rank > 50) {
                auto ic = constructIronCondor(symbol, vol_metrics, data);
                positions.push_back(ic);
            }
        }

        // Size positions equally
        int num_positions = positions.size();
        double capital_per_position = capital / num_positions;

        for (auto& pos : positions) {
            pos.quantity = calculateQuantity(pos, capital_per_position);
        }

        return positions;
    }
};
```

### Phase 2: Greeks Optimization (Months 3-4)

**Scope:**
- Delta-neutral portfolio construction
- Theta maximization with constraints
- Gamma/vega risk management
- Multiple options strategies (strangles, credit spreads)

**Implementation:**
```cpp
// File: src/portfolio_optimizer/greeks_optimizer.cpp

class Tier1GreeksOptimizer {
public:
    Portfolio optimizeWithGreeks(
        const std::vector<std::string>& universe,
        const MarketData& market_data,
        double capital,
        const GreeksConstraints& constraints
    ) {
        Portfolio portfolio;

        // Generate candidate strategies
        auto candidates = generateAllCandidates(universe, market_data);

        // Optimize for theta while respecting Greeks constraints
        auto selected = greedyGreeksOptimization(
            candidates, capital, constraints
        );

        portfolio.options_positions = selected;

        // Calculate portfolio Greeks
        portfolio.greeks = aggregateGreeks(selected);

        // Add delta hedge if needed
        if (std::abs(portfolio.greeks.delta) > constraints.delta_tolerance) {
            portfolio.stock_hedge = calculateDeltaHedge(portfolio);
        }

        return portfolio;
    }
};
```

---

## Advanced Features (Tier 2+)

### 1. Multi-Period Optimization

**Optimize over multiple time horizons:**

```cpp
class MultiPeriodOptimizer {
public:
    struct Horizon {
        std::string name;
        int days;
        double weight;  // Importance of this horizon
    };

    Portfolio optimizeMultiPeriod(
        const std::vector<Horizon>& horizons,
        const MarketData& data,
        double capital
    ) {
        // Objective: maximize weighted sum of Sharpe ratios
        // across horizons

        std::vector<Portfolio> horizon_portfolios;

        for (const auto& horizon : horizons) {
            auto returns = predictReturns(data, horizon.days);
            auto cov = estimateCovariance(data, horizon.days);

            auto portfolio = optimizePortfolio(returns, cov, capital);
            horizon_portfolios.push_back(portfolio);
        }

        // Combine portfolios weighted by importance
        return combinePortfolios(horizon_portfolios, horizons);
    }
};
```

### 2. Transaction Cost Optimization

**Account for bid-ask spread, commissions, market impact:**

```cpp
double calculateTransactionCost(
    const Trade& trade,
    const MarketData& data
) {
    // Commission
    double commission = 0.65 * trade.num_legs;  // $0.65 per leg

    // Bid-ask spread cost
    double spread_cost = data.bid_ask_spread * trade.quantity * 100 * 0.5;

    // Market impact (for large orders)
    double market_impact = 0;
    if (trade.quantity > 50) {
        double adv = data.average_daily_volume;  // Average daily volume
        double participation_rate = (trade.quantity * 100) / adv;

        // Permanent impact (Almgren-Chriss model)
        market_impact = 0.01 * data.volatility * std::sqrt(participation_rate);
    }

    return commission + spread_cost + market_impact;
}

Portfolio optimizeWithTransactionCosts(
    const Eigen::VectorXd& target_weights,
    const Portfolio& current_portfolio,
    const MarketData& data
) {
    // Trade-off: benefit of rebalancing vs cost

    auto trades = generateRebalanceTrades(current_portfolio, target_weights);

    double total_cost = 0;
    for (const auto& trade : trades) {
        total_cost += calculateTransactionCost(trade, data);
    }

    double expected_benefit = estimateRebalanceBenefit(
        current_portfolio, target_weights
    );

    // Only rebalance if benefit > cost
    if (expected_benefit > total_cost * 1.5) {  // 50% margin
        return executeRebalance(current_portfolio, trades);
    }

    return current_portfolio;  // No rebalance
}
```

### 3. Robust Optimization

**Optimization under uncertainty:**

```cpp
class RobustOptimizer {
public:
    // Worst-case optimization
    Portfolio optimizeWorstCase(
        const std::vector<Eigen::VectorXd>& return_scenarios,
        const std::vector<Eigen::MatrixXd>& cov_scenarios,
        double capital
    ) {
        // maximize: min_scenario (Sharpe_ratio_scenario(w))

        double best_worst_case = -std::numeric_limits<double>::infinity();
        Eigen::VectorXd best_weights;

        // Grid search or global optimization
        for (auto& candidate_weights : generateCandidates()) {
            double worst_sharpe = std::numeric_limits<double>::infinity();

            for (size_t i = 0; i < return_scenarios.size(); ++i) {
                double sharpe = calculateSharpe(
                    candidate_weights,
                    return_scenarios[i],
                    cov_scenarios[i]
                );

                worst_sharpe = std::min(worst_sharpe, sharpe);
            }

            if (worst_sharpe > best_worst_case) {
                best_worst_case = worst_sharpe;
                best_weights = candidate_weights;
            }
        }

        return constructPortfolio(best_weights, capital);
    }
};
```

---

## Real-Time Optimization Engine

### 1. Streaming Data Integration

```cpp
class RealtimeOptimizer {
public:
    void onMarketDataUpdate(const MarketDataUpdate& update) {
        // Update internal state
        updatePrices(update);
        updateVolatility(update);

        // Check if reoptimization needed
        if (shouldReoptimize(update)) {
            auto new_portfolio = optimize();
            generateRebalanceSignals(new_portfolio);
        }
    }

private:
    bool shouldReoptimize(const MarketDataUpdate& update) {
        // Triggers:
        // 1. Portfolio delta exceeds limits
        // 2. Volatility regime change
        // 3. Scheduled rebalance time
        // 4. Position P/L hits stop-loss or profit target

        auto greeks = calculateCurrentGreeks();

        if (std::abs(greeks.delta) > delta_limit_) {
            return true;
        }

        if (detectVolatilityRegimeChange(update)) {
            return true;
        }

        return false;
    }

    double delta_limit_ = 500;
};
```

### 2. Risk-Adjusted Position Scaling

**Scale positions based on realized risk:**

```cpp
class AdaptivePositionSizer {
public:
    int calculateOptimalSize(
        const OptionsStrategy& strategy,
        const Portfolio& current_portfolio,
        double capital_available
    ) {
        // Base size from Kelly criterion
        double kelly_fraction = calculateKellyFraction(strategy);

        // Adjust for current portfolio risk
        double current_vol = calculatePortfolioVolatility(current_portfolio);
        double target_vol = target_volatility_;

        double vol_adjustment = target_vol / current_vol;
        vol_adjustment = std::clamp(vol_adjustment, 0.5, 2.0);  // Limit range

        // Adjust for correlation with existing positions
        double correlation_adj = calculateCorrelationAdjustment(
            strategy, current_portfolio
        );

        double adjusted_fraction = kelly_fraction *
                                  vol_adjustment *
                                  correlation_adj;

        // Conservative Kelly (use fraction of Kelly)
        adjusted_fraction *= kelly_fraction_multiplier_;  // e.g., 0.25

        int contracts = static_cast<int>(
            (capital_available * adjusted_fraction) / strategy.max_loss
        );

        return std::max(contracts, 1);
    }

private:
    double target_volatility_ = 0.15;  // 15% annualized
    double kelly_fraction_multiplier_ = 0.25;  // Conservative
};
```

---

## Profit Maximization Strategies

### 1. Theta Harvesting

**Daily income from time decay:**

```cpp
class ThetaHarvestingStrategy {
public:
    struct DailyTarget {
        double min_theta_dollars;  // e.g., $100/day
        double max_capital_at_risk;  // e.g., $20,000
        double min_theta_efficiency;  // theta / capital ≥ 0.5%
    };

    std::vector<OptionsPosition> constructThetaPortfolio(
        const std::vector<std::string>& underlyings,
        const MarketData& data,
        const DailyTarget& targets
    ) {
        std::vector<OptionsPosition> portfolio;

        double total_theta = 0;
        double total_capital_at_risk = 0;

        for (const auto& symbol : underlyings) {
            auto vol_metrics = getVolatilityMetrics(symbol, data);

            // Only sell premium in high IV
            if (vol_metrics.iv_rank < 40) continue;

            // Construct iron condor
            auto ic = optimizeIronCondorForTheta(symbol, vol_metrics, data);

            // Check efficiency
            double efficiency = ic.theta / ic.max_loss;
            if (efficiency < targets.min_theta_efficiency) continue;

            // Add to portfolio
            portfolio.push_back(ic);
            total_theta += ic.theta;
            total_capital_at_risk += ic.max_loss;

            // Check if targets met
            if (total_theta >= targets.min_theta_dollars &&
                total_capital_at_risk <= targets.max_capital_at_risk) {
                break;
            }
        }

        return portfolio;
    }
};
```

### 2. Volatility Arbitrage

**Exploit differences between realized and implied volatility:**

```cpp
class VolatilityArbitrageStrategy {
public:
    struct VolArb Signal {
        std::string symbol;
        double implied_vol;
        double realized_vol_30d;
        double vol_spread;  // IV - RV
        double signal_strength;  // Normalized spread
    };

    std::vector<VolArbSignal> findOpportunities(
        const std::vector<std::string>& universe,
        const MarketData& data
    ) {
        std::vector<VolArbSignal> signals;

        for (const auto& symbol : universe) {
            VolArbSignal signal;
            signal.symbol = symbol;
            signal.implied_vol = data.getImpliedVol(symbol);
            signal.realized_vol_30d = calculateRealizedVol(symbol, 30, data);
            signal.vol_spread = signal.implied_vol - signal.realized_vol_30d;

            // Normalize by historical spread distribution
            signal.signal_strength = normalizeSpread(
                signal.vol_spread, symbol, data
            );

            // Strong signal if |z-score| > 2
            if (std::abs(signal.signal_strength) > 2.0) {
                signals.push_back(signal);
            }
        }

        return signals;
    }

    OptionsPosition constructVolArbTrade(const VolArbSignal& signal) {
        if (signal.vol_spread > 0) {
            // IV > RV → Sell volatility
            return constructShortStraddle(signal.symbol);
        } else {
            // RV > IV → Buy volatility
            return constructLongStraddle(signal.symbol);
        }
    }
};
```

### 3. Mean Reversion Strategies

**Profit from overbought/oversold conditions:**

```cpp
class MeanReversionOptimizer {
public:
    struct MeanReversionSignal {
        std::string symbol;
        double current_price;
        double mean_price;
        double std_dev;
        double z_score;
        double reversion_probability;
    };

    std::vector<MeanReversionSignal> findMeanReversionOpportunities(
        const std::vector<std::string>& universe,
        const MarketData& data,
        int lookback_days = 20
    ) {
        std::vector<MeanReversionSignal> signals;

        for (const auto& symbol : universe) {
            MeanReversionSignal signal;
            signal.symbol = symbol;
            signal.current_price = data.getPrice(symbol);

            auto price_history = data.getPriceHistory(symbol, lookback_days);
            signal.mean_price = calculateMean(price_history);
            signal.std_dev = calculateStdDev(price_history);

            signal.z_score = (signal.current_price - signal.mean_price) /
                            signal.std_dev;

            // Estimate reversion probability
            signal.reversion_probability = estimateReversionProb(
                signal.z_score, lookback_days
            );

            // Strong signal if |z-score| > 2.5
            if (std::abs(signal.z_score) > 2.5 &&
                signal.reversion_probability > 0.60) {
                signals.push_back(signal);
            }
        }

        return signals;
    }

    OptionsPosition constructReversionTrade(
        const MeanReversionSignal& signal
    ) {
        if (signal.z_score > 2.5) {
            // Overbought → bearish trade
            return constructBearCallSpread(signal.symbol);
        } else if (signal.z_score < -2.5) {
            // Oversold → bullish trade
            return constructBullPutSpread(signal.symbol);
        }

        return {};
    }
};
```

---

## Risk Minimization Techniques

### 1. Tail Risk Hedging

**Protect against black swan events:**

```cpp
class TailRiskHedge {
public:
    struct HedgeStrategy {
        std::vector<OptionsPosition> protective_puts;
        double hedge_cost_annual;
        double downside_protection_pct;
        double drag_on_returns;
    };

    HedgeStrategy constructTailHedge(
        const Portfolio& portfolio,
        double max_acceptable_drawdown = 0.20  // 20%
    ) {
        HedgeStrategy hedge;

        // Buy far OTM puts on SPY/SPX
        double portfolio_beta_weighted_value =
            calculateBetaWeightedValue(portfolio);

        // Target: protect 80% of beta-weighted value
        double notional_to_protect = portfolio_beta_weighted_value * 0.80;

        // Buy puts at 20% OTM, quarterly expiration
        auto spy_price = getPrice("SPY");
        double put_strike = spy_price * 0.80;  // 20% OTM

        auto put_options = findPuts("SPY", put_strike, 90);  // 90 DTE

        if (put_options) {
            int contracts_needed = static_cast<int>(
                notional_to_protect / (spy_price * 100)
            );

            OptionsPosition protective_put;
            protective_put.symbol = "SPY";
            protective_put.strategy = "protective_put";
            protective_put.quantity = contracts_needed;
            protective_put.premium = put_options->price;

            hedge.protective_puts.push_back(protective_put);

            // Calculate cost
            hedge.hedge_cost_annual = (protective_put.premium * contracts_needed * 100) *
                                     (365.0 / 90.0);  // Roll quarterly

            hedge.downside_protection_pct = 0.20;  // Protects below 20% drop
            hedge.drag_on_returns = hedge.hedge_cost_annual /
                                   portfolio.total_value;
        }

        return hedge;
    }
};
```

### 2. Diversification Optimization

**Maximize diversification benefit:**

```cpp
double calculateDiversificationRatio(
    const Eigen::VectorXd& weights,
    const std::vector<double>& individual_vols,
    double portfolio_vol
) {
    double weighted_vol_sum = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        weighted_vol_sum += weights(i) * individual_vols[i];
    }

    return weighted_vol_sum / portfolio_vol;
}

Eigen::VectorXd maximizeDiversification(
    const std::vector<double>& individual_vols,
    const Eigen::MatrixXd& correlation_matrix
) {
    // maximize: Diversification Ratio
    //         = (Σw_i*σ_i) / √(wᵀΣw)

    // This is equivalent to maximizing:
    // (Σw_i*σ_i)² - wᵀΣw

    // Use nonlinear optimization (SLSQP, COBYLA)
    return solveNonlinearProgram(individual_vols, correlation_matrix);
}
```

### 3. Drawdown Control

**Limit maximum drawdown:**

```cpp
class DrawdownController {
public:
    void checkDrawdown(Portfolio& portfolio) {
        double current_drawdown = calculateCurrentDrawdown(portfolio);

        if (current_drawdown > max_drawdown_threshold_) {
            // Reduce risk exposure
            reduceRisk(portfolio, current_drawdown);
        }
    }

private:
    void reduceRisk(Portfolio& portfolio, double current_dd) {
        double risk_reduction_factor = 1.0 - (current_dd / max_drawdown_threshold_);

        // Scale down risky positions
        for (auto& position : portfolio.stock_positions) {
            position.shares = static_cast<int>(
                position.shares * risk_reduction_factor
            );
        }

        for (auto& position : portfolio.options_positions) {
            // Close most aggressive positions first
            if (position.max_loss > avg_position_risk_) {
                closePosition(position);
            }
        }

        // Move to cash
        double freed_capital = calculateFreedCapital(portfolio);
        portfolio.cash += freed_capital;

        LOG_WARN("Drawdown control activated: {}% DD, reduced risk by {}%",
                 current_dd * 100, (1 - risk_reduction_factor) * 100);
    }

    double max_drawdown_threshold_ = 0.15;  // 15%
    double avg_position_risk_;
};
```

---

## Tier 1 Implementation Roadmap

### Month 1-2: Core Framework

**Deliverables:**
1. ✅ **Stock Portfolio Optimizer** (Equal-risk weighting)
   - File: `src/portfolio_optimizer/stock_optimizer.cpp`
   - Status: Stub implementation exists

2. **Basic Options Selection** (Iron condors only)
   - File: `src/portfolio_optimizer/options_selector.cpp`
   - Implementation: IV rank filtering + standard strikes

3. **Portfolio Metrics Calculator**
   - File: `src/portfolio_optimizer/metrics.cpp`
   - Calculate: Sharpe, Sortino, Greeks, VaR

4. **Threshold Rebalancer**
   - File: `src/portfolio_optimizer/rebalancer.cpp`
   - Trigger: 5% weight drift

**Testing:**
- Backtest on 2020-2024 data
- Verify theta generation > $50/day per $10k capital
- Sharpe ratio > 1.5

### Month 3-4: Greeks Optimization

**Deliverables:**
1. **Greeks-Balanced Optimizer**
   - Implement greedy Greeks optimization
   - Target: Delta-neutral, theta-positive

2. **Multiple Options Strategies**
   - Add: Strangles, credit spreads, calendars
   - Strategy selection based on IV rank

3. **Position Sizing (Kelly)**
   - Kelly criterion implementation
   - Conservative fractional Kelly (1/4 or 1/2)

4. **Real-Time Monitoring**
   - Greeks tracking dashboard
   - Alerts for limit violations

**Testing:**
- 45-day rolling backtest
- Compare against buy-and-hold SPY
- Target: 2x returns with < 1.5x volatility

---

## Integration with Existing Components

### 1. Market Intelligence Engine → Optimizer

```cpp
class IntegratedOptimizer {
public:
    Portfolio optimize(
        const MarketIntelligence& mi_signals,
        const CorrelationAnalysis& correlation_data,
        double capital
    ) {
        // Use MI signals to adjust expected returns
        auto adjusted_returns = incorporateMISignals(
            base_returns_, mi_signals
        );

        // Use correlation data to adjust covariance
        auto adjusted_cov = incorporateCorrelations(
            base_cov_, correlation_data
        );

        // Optimize
        return optimizePortfolio(adjusted_returns, adjusted_cov, capital);
    }

private:
    Eigen::VectorXd incorporateMISignals(
        const Eigen::VectorXd& base_returns,
        const MarketIntelligence& mi
    ) {
        Eigen::VectorXd adjusted = base_returns;

        // Adjust returns based on MI predictions
        for (const auto& signal : mi.signals) {
            int idx = getSymbolIndex(signal.symbol);
            if (idx >= 0) {
                // Blend base forecast with MI signal
                adjusted(idx) = 0.60 * base_returns(idx) +
                               0.40 * signal.predicted_return;
            }
        }

        return adjusted;
    }
};
```

### 2. Correlation Engine → Risk Model

```cpp
Eigen::MatrixXd enhanceCovarianceWithCorrelations(
    const Eigen::MatrixXd& sample_cov,
    const CorrelationEngine& correlation_engine
) {
    Eigen::MatrixXd enhanced_cov = sample_cov;

    // Incorporate leading indicators
    auto correlations = correlation_engine.getTimelaggedCorrelations();

    for (const auto& corr : correlations) {
        int i = getSymbolIndex(corr.symbol1);
        int j = getSymbolIndex(corr.symbol2);

        if (i >= 0 && j >= 0) {
            // Adjust correlation based on time-lagged relationship
            double adjusted_corr = blendCorrelations(
                sample_cov(i, j),
                corr.coefficient,
                corr.confidence
            );

            enhanced_cov(i, j) = adjusted_corr *
                                std::sqrt(sample_cov(i, i)) *
                                std::sqrt(sample_cov(j, j));
            enhanced_cov(j, i) = enhanced_cov(i, j);
        }
    }

    return enhanced_cov;
}
```

---

## Performance Metrics and Evaluation

### 1. Ex-Ante Metrics (Before Trade)

```cpp
struct ExAnteMetrics {
    double expected_return;
    double expected_sharpe;
    double expected_max_drawdown;
    double win_probability;
    double expected_value;
    double kelly_fraction;
};

ExAnteMetrics evaluateBeforeTrade(
    const Portfolio& proposed_portfolio,
    const std::vector<Scenario>& scenarios
) {
    ExAnteMetrics metrics;

    // Monte Carlo simulation
    std::vector<double> simulated_returns;

    for (int i = 0; i < 10000; ++i) {
        auto scenario = sampleScenario(scenarios);
        double portfolio_return = simulatePortfolioReturn(
            proposed_portfolio, scenario
        );
        simulated_returns.push_back(portfolio_return);
    }

    metrics.expected_return = calculateMean(simulated_returns);
    metrics.expected_sharpe = calculateSharpe(simulated_returns);
    metrics.expected_max_drawdown = calculateMaxDD(simulated_returns);
    metrics.win_probability = calculateWinProb(simulated_returns);

    return metrics;
}
```

### 2. Ex-Post Analysis (After Trade)

```cpp
struct ExPostAnalysis {
    double actual_return;
    double actual_sharpe;
    double actual_max_dd;
    double forecast_error;
    double attribution;  // What drove performance?
};

ExPostAnalysis analyzePerformance(
    const Portfolio& portfolio,
    const ExAnteMetrics& forecast,
    const std::vector<Trade>& executed_trades
) {
    ExPostAnalysis analysis;

    // Calculate actual performance
    analysis.actual_return = calculateRealizedReturn(executed_trades);
    analysis.actual_sharpe = calculateRealizedSharpe(executed_trades);
    analysis.actual_max_dd = calculateRealizedMaxDD(executed_trades);

    // Compare to forecast
    analysis.forecast_error = analysis.actual_return - forecast.expected_return;

    // Attribution analysis
    analysis.attribution = attributePerformance(
        portfolio, executed_trades
    );

    // Store for model improvement
    storeForLearning(forecast, analysis);

    return analysis;
}
```

---

## Implementation Architecture

### File Structure

```
src/portfolio_optimizer/
├── optimizer_base.hpp           # Base class for all optimizers
├── stock_optimizer.cpp          # Stock-only optimization
├── options_optimizer.cpp        # Options portfolio optimization
├── greeks_optimizer.cpp         # Greeks-balanced optimization
├── multi_asset_optimizer.cpp    # Combined stock+options+bonds
├── rebalancer.cpp              # Rebalancing logic
├── position_sizer.cpp          # Kelly, fixed fractional sizing
├── risk_controller.cpp         # Drawdown control, risk limits
├── tail_risk_hedging.cpp       # Black swan protection
├── volatility_arbitrage.cpp    # Vol arb strategies
├── mean_reversion.cpp          # Mean reversion strategies
├── theta_harvesting.cpp        # Theta optimization
└── performance_evaluator.cpp   # Ex-ante and ex-post analysis

Tier 1 Priority:
1. optimizer_base.hpp
2. stock_optimizer.cpp (equal-risk)
3. options_optimizer.cpp (iron condors only)
4. greeks_optimizer.cpp (greedy algorithm)
5. rebalancer.cpp (threshold-based)
6. position_sizer.cpp (Kelly criterion)
```

### API Design

```cpp
namespace bigbrother {
namespace optimizer {

class PortfolioOptimizer {
public:
    // Main optimization interface
    virtual Portfolio optimize(
        const std::vector<std::string>& universe,
        const MarketData& market_data,
        double capital_available,
        const OptimizationConstraints& constraints
    ) = 0;

    // Evaluate proposed portfolio
    virtual ExAnteMetrics evaluate(
        const Portfolio& proposed_portfolio,
        const MarketData& data
    ) = 0;

    // Generate rebalancing trades
    virtual std::vector<Trade> rebalance(
        const Portfolio& current_portfolio,
        const Portfolio& target_portfolio
    ) = 0;
};

// Tier 1 implementation
class BasicOptimizer : public PortfolioOptimizer {
public:
    Portfolio optimize(...) override {
        // Simple implementation for POC
        return optimizeBasic(...);
    }
};

// Tier 2+ implementation
class AdvancedOptimizer : public PortfolioOptimizer {
public:
    Portfolio optimize(...) override {
        // ML-enhanced, multi-period, robust optimization
        return optimizeAdvanced(...);
    }
};

}} // namespace bigbrother::optimizer
```

---

## Integration Examples

### Example 1: Daily Optimization Workflow

```cpp
void dailyOptimizationWorkflow() {
    // 1. Fetch latest market data
    auto market_data = marketDataClient.fetchLatest();

    // 2. Get ML predictions from Market Intelligence
    auto mi_signals = marketIntelligenceEngine.getPredictions();

    // 3. Get correlation insights
    auto correlations = correlationEngine.getLatestCorrelations();

    // 4. Optimize portfolio
    OptimizationConstraints constraints{
        .max_position_size = 0.10,  // 10% per position
        .delta_tolerance = 100,
        .max_theta = -1000,  // Limit short premium
        .min_sharpe = 1.5
    };

    auto optimized_portfolio = optimizer.optimize(
        stock_universe_,
        market_data,
        account_value_,
        constraints
    );

    // 5. Generate rebalance trades
    auto trades = optimizer.rebalance(current_portfolio_, optimized_portfolio);

    // 6. Risk check before execution
    if (riskController.approve(trades)) {
        executeTradesin(trades);
    }

    // 7. Log and monitor
    logOptimizationDecision(optimized_portfolio, trades);
}
```

### Example 2: Volatility Event Optimization

```cpp
void onVolatilitySpike(double new_vix) {
    if (new_vix > vix_threshold_) {
        // Volatility spike → reoptimize immediately

        // 1. Close short volatility positions
        closePositionsWithNegativeVega();

        // 2. Add protective puts
        auto tail_hedge = tailRiskHedger.constructTailHedge(portfolio_);
        executeTrades(tail_hedge.protective_puts);

        // 3. Reduce overall exposure
        double target_reduction = 0.30;  // Reduce 30%
        auto reduction_trades = generateExposureReduction(target_reduction);
        executeTrades(reduction_trades);

        // 4. Reoptimize remaining capital
        double remaining_capital = portfolio_.total_value -
                                  calculateCapitalAtRisk(portfolio_);

        auto new_portfolio = optimizer.optimize(
            stock_universe_,
            current_market_data_,
            remaining_capital,
            high_volatility_constraints_
        );

        // 5. Execute
        auto rebalance_trades = optimizer.rebalance(portfolio_, new_portfolio);
        executeTrades(rebalance_trades);
    }
}
```

---

## Performance Targets (Tier 1 POC)

### Expected Performance Metrics

```
Target Annual Return:     25-40%
Target Sharpe Ratio:      1.5-2.5
Target Max Drawdown:      < 15%
Target Win Rate:          65-75%
Target Profit Factor:     > 2.0

Daily Theta Generation:   $100-200 per $20k capital
Options ROC:              15-30% per trade
Stock ROC:                10-20% annualized
```

### Risk Limits

```cpp
struct Tier1RiskLimits {
    // Position limits
    double max_single_position = 0.10;  // 10% of portfolio
    double max_sector_exposure = 0.30;   // 30% in one sector

    // Greeks limits
    double max_portfolio_delta = 500;
    double max_portfolio_gamma = 100;
    double max_portfolio_vega = 1000;
    double min_daily_theta = -50;  // Can pay up to $50/day for long options

    // Capital limits
    double max_bp_usage = 0.50;  // Use max 50% of buying power
    double min_cash_reserve = 0.10;  // Keep 10% cash

    // Loss limits
    double max_loss_per_position = 0.02;  // 2% of portfolio
    double max_loss_per_day = 0.05;  // 5% of portfolio
    double max_monthly_dd = 0.10;  // 10% monthly drawdown
};
```

---

## Machine Learning Enhancements (Tier 2+)

### 1. Reinforcement Learning for Strategy Selection

```cpp
class RLStrategySelector {
public:
    OptionsPosition selectStrategy(
        const MarketState& state,
        const RLPolicy& policy
    ) {
        // State: IV rank, trend, volatility, Greeks, etc.
        auto state_vector = encodeState(state);

        // Policy: trained RL agent (PPO, SAC, etc.)
        int action = policy.selectAction(state_vector);

        // Action space: {iron_condor, strangle, spread, straddle, ...}
        return constructStrategyFromAction(action, state);
    }
};
```

### 2. Neural Network Return Forecasting

```cpp
class NeuralReturnForecaster {
public:
    Eigen::VectorXd forecastReturns(
        const std::vector<std::string>& symbols,
        const MarketFeatures& features,
        int horizon_days
    ) {
        Eigen::VectorXd forecasts(symbols.size());

        for (size_t i = 0; i < symbols.size(); ++i) {
            auto features_tensor = prepareFeatures(symbols[i], features);

            // LSTM or Transformer prediction
            double forecast = model_->predict(features_tensor);

            forecasts(i) = forecast;
        }

        return forecasts;
    }

private:
    std::unique_ptr<NeuralNetwork> model_;
};
```

---

## Human-in-the-Loop Decision Making

### Overview

When the optimization engine encounters high uncertainty or conflicting signals, it defers to human judgment. The system presents alternatives, calculates expected outcomes, and learns from human decisions to gradually improve autonomous decision-making.

### 1. Uncertainty Detection

```cpp
struct UncertaintyMetrics {
    double prediction_confidence;     // 0-1
    double signal_agreement;          // -1 to 1
    double strategy_diversity;        // How different are top strategies?
    double market_regime_clarity;     // Clear trend/range vs transitional
    bool requires_human_input;
};

class UncertaintyDetector {
public:
    UncertaintyMetrics assessUncertainty(
        const std::vector<Signal>& signals,
        const std::vector<Strategy>& candidate_strategies,
        const MarketData& market_data
    ) {
        UncertaintyMetrics metrics;

        // 1. Check prediction confidence
        metrics.prediction_confidence = calculateEnsembleAgreement(signals);

        // 2. Check signal agreement
        int bullish = 0, bearish = 0, neutral = 0;
        for (const auto& signal : signals) {
            if (signal.direction == Direction::BULLISH) bullish++;
            else if (signal.direction == Direction::BEARISH) bearish++;
            else neutral++;
        }

        int total = signals.size();
        double max_agreement = std::max({bullish, bearish, neutral}) /
                               static_cast<double>(total);

        metrics.signal_agreement = (max_agreement - 0.33) / 0.67;  // Normalize

        // 3. Strategy diversity
        metrics.strategy_diversity = calculateStrategyDiversity(
            candidate_strategies
        );

        // 4. Market regime clarity
        metrics.market_regime_clarity = assessRegimeClarity(market_data);

        // Decide if human input needed
        metrics.requires_human_input =
            (metrics.prediction_confidence < 0.60) ||      // Low confidence
            (metrics.signal_agreement < 0.40) ||           // Conflicting signals
            (metrics.strategy_diversity > 0.70) ||         // Very different strategies
            (metrics.market_regime_clarity < 0.50);        // Unclear regime

        return metrics;
    }

private:
    double calculateStrategyDiversity(
        const std::vector<Strategy>& strategies
    ) {
        if (strategies.size() < 2) return 0.0;

        // Compare expected outcomes
        std::vector<double> expected_returns;
        std::vector<double> max_losses;

        for (const auto& strat : strategies) {
            expected_returns.push_back(strat.expected_return);
            max_losses.push_back(strat.max_loss);
        }

        double return_std = calculateStdDev(expected_returns);
        double loss_std = calculateStdDev(max_losses);

        // Normalized diversity score
        return (return_std + loss_std) / 2.0;
    }
};
```

### 2. Alternative Presentation

```cpp
struct DecisionAlternative {
    std::string strategy_name;
    std::string description;

    // Expected outcomes
    double expected_profit;
    double expected_loss;
    double probability_of_profit;
    double sharpe_ratio;

    // Greeks profile
    PortfolioGreeks greeks;

    // Capital requirements
    double buying_power_required;
    double max_loss;

    // Confidence and risks
    double confidence_level;
    std::vector<std::string> key_risks;
    std::vector<std::string> key_opportunities;
};

class DecisionPresenter {
public:
    void presentAlternatives(
        const std::vector<DecisionAlternative>& alternatives,
        const UncertaintyMetrics& uncertainty,
        const MarketContext& context
    ) {
        LOG_INFO("═══════════════════════════════════════");
        LOG_INFO("HUMAN DECISION REQUIRED");
        LOG_INFO("Uncertainty Reason: {}", getUncertaintyReason(uncertainty));
        LOG_INFO("═══════════════════════════════════════");
        LOG_INFO("");

        LOG_INFO("Market Context:");
        LOG_INFO("  Current Price: ${:.2f}", context.current_price);
        LOG_INFO("  IV Rank: {:.0f}", context.iv_rank);
        LOG_INFO("  Trend: {}", context.trend_description);
        LOG_INFO("  Regime: {}", context.market_regime);
        LOG_INFO("");

        // Present each alternative
        for (size_t i = 0; i < alternatives.size(); ++i) {
            const auto& alt = alternatives[i];

            LOG_INFO("Option {}: {}", i + 1, alt.strategy_name);
            LOG_INFO("  Description: {}", alt.description);
            LOG_INFO("  Expected Profit: ${:.2f} (POP: {:.1f}%)",
                     alt.expected_profit, alt.probability_of_profit * 100);
            LOG_INFO("  Max Loss: ${:.2f}", alt.max_loss);
            LOG_INFO("  Risk/Reward: {:.2f}",
                     alt.expected_profit / alt.max_loss);
            LOG_INFO("  Sharpe Ratio: {:.2f}", alt.sharpe_ratio);
            LOG_INFO("  Confidence: {:.0f}%", alt.confidence_level * 100);
            LOG_INFO("  Greeks: Δ={:.0f} Γ={:.1f} Θ=${:.2f}/day ν={:.0f}",
                     alt.greeks.delta, alt.greeks.gamma,
                     alt.greeks.theta, alt.greeks.vega);
            LOG_INFO("  Capital Required: ${:.0f}", alt.buying_power_required);

            if (!alt.key_risks.empty()) {
                LOG_INFO("  Key Risks:");
                for (const auto& risk : alt.key_risks) {
                    LOG_INFO("    - {}", risk);
                }
            }

            if (!alt.key_opportunities.empty()) {
                LOG_INFO("  Key Opportunities:");
                for (const auto& opp : alt.key_opportunities) {
                    LOG_INFO("    + {}", opp);
                }
            }

            LOG_INFO("");
        }

        LOG_INFO("Please select:");
        for (size_t i = 0; i < alternatives.size(); ++i) {
            LOG_INFO("  [{}] {}", i + 1, alternatives[i].strategy_name);
        }
        LOG_INFO("  [0] Skip this trade (no action)");
        LOG_INFO("  [A] Request more analysis");
    }
};
```

### 3. Human Decision Capture

```cpp
struct HumanDecision {
    std::chrono::system_clock::time_point decision_time;
    std::string symbol;
    std::vector<DecisionAlternative> alternatives_presented;
    int selected_alternative;  // 0 = skip, 1-N = alternative index
    std::string rationale;  // Human explanation
    UncertaintyMetrics uncertainty_at_decision;

    // Follow-up tracking
    Trade executed_trade;
    double actual_pnl;
    bool was_successful;
};

class HumanDecisionLogger {
public:
    HumanDecision captureDecision(
        const std::vector<DecisionAlternative>& alternatives,
        const UncertaintyMetrics& uncertainty
    ) {
        HumanDecision decision;
        decision.decision_time = std::chrono::system_clock::now();
        decision.alternatives_presented = alternatives;
        decision.uncertainty_at_decision = uncertainty;

        // Present alternatives
        DecisionPresenter presenter;
        presenter.presentAlternatives(alternatives, uncertainty, market_context_);

        // Get human input
        decision.selected_alternative = promptForSelection();

        if (decision.selected_alternative == -1) {  // 'A' - request analysis
            // Provide detailed analysis
            performDeepAnalysis(alternatives);
            decision.selected_alternative = promptForSelection();
        }

        // Get rationale
        decision.rationale = promptForRationale();

        // Store decision
        storeDecision(decision);

        return decision;
    }

    void recordOutcome(
        const HumanDecision& decision,
        const Trade& executed_trade,
        double actual_pnl
    ) {
        decision.executed_trade = executed_trade;
        decision.actual_pnl = actual_pnl;

        // Determine if successful
        decision.was_successful = (actual_pnl > 0);

        // Update database
        updateDecisionOutcome(decision);

        // Trigger learning
        decisionLearner_.learn(decision);
    }

private:
    DecisionLearner decisionLearner_;
    MarketContext market_context_;
};
```

### 4. Learning from Human Decisions

```cpp
class DecisionLearner {
public:
    void learn(const HumanDecision& decision) {
        // Store decision for analysis
        decision_history_.push_back(decision);

        // Analyze patterns
        if (decision_history_.size() >= 20) {
            analyzeDecisionPatterns();
        }
    }

private:
    void analyzeDecisionPatterns() {
        // Find patterns in human decisions

        // 1. When does human agree with model?
        auto agreement_patterns = findAgreementPatterns();

        // 2. When does human override model?
        auto override_patterns = findOverridePatterns();

        // 3. What factors influence human decisions?
        auto factor_importance = calculateFactorImportance();

        // 4. Update confidence thresholds
        updateConfidenceThresholds(agreement_patterns, override_patterns);

        // 5. Train ML model to mimic human decisions
        if (decision_history_.size() >= 100) {
            trainDecisionMimicModel();
        }
    }

    void updateConfidenceThresholds(
        const PatternAnalysis& agreement,
        const PatternAnalysis& override
    ) {
        // Lower threshold if human consistently agrees
        if (agreement.rate > 0.80) {
            confidence_threshold_ *= 0.95;  // Become more autonomous
        }

        // Raise threshold if human frequently overrides
        if (override.rate > 0.50) {
            confidence_threshold_ *= 1.10;  // Ask more often
        }

        // Clamp thresholds
        confidence_threshold_ = std::clamp(
            confidence_threshold_, 0.40, 0.80
        );
    }

    std::vector<HumanDecision> decision_history_;
    double confidence_threshold_ = 0.60;  // Initial threshold
};
```

### 5. Gradual Automation

```cpp
class GradualAutomation {
public:
    bool shouldAskHuman(
        const UncertaintyMetrics& uncertainty,
        const MarketContext& context
    ) {
        // Check if we've seen similar situations before
        auto similar_past_decisions = findSimilarDecisions(context);

        if (similar_past_decisions.size() >= 5) {
            // Check if human was consistent in similar situations
            double consistency = calculateConsistency(similar_past_decisions);

            if (consistency > 0.85) {
                // Human was consistent → automate this case
                LOG_INFO("Similar situation seen {} times, human was consistent ({:.0f}%). Automating.",
                         similar_past_decisions.size(), consistency * 100);
                return false;
            }
        }

        // Check base uncertainty criteria
        return uncertainty.requires_human_input;
    }

    int predictHumanChoice(
        const std::vector<DecisionAlternative>& alternatives,
        const MarketContext& context
    ) {
        if (!trained_model_) {
            return -1;  // No prediction available
        }

        // Use ML model to predict human choice
        auto features = extractDecisionFeatures(alternatives, context);
        int predicted_choice = trained_model_->predict(features);

        return predicted_choice;
    }

private:
    std::unique_ptr<MLModel> trained_model_;
};
```

### 6. Integration with Trading Engine

```cpp
class HumanLoopTrading Engine {
public:
    void executeWithHumanLoop(
        const std::vector<Signal>& signals,
        const Portfolio& current_portfolio
    ) {
        // Generate candidate strategies
        auto candidates = generateCandidateStrategies(signals);

        // Optimize each candidate
        std::vector<DecisionAlternative> alternatives;
        for (const auto& candidate : candidates) {
            auto optimized = optimizeStrategy(candidate);
            alternatives.push_back(toAlternative(optimized));
        }

        // Rank alternatives
        std::sort(alternatives.begin(), alternatives.end(),
            [](const auto& a, const auto& b) {
                return a.sharpe_ratio > b.sharpe_ratio;
            });

        // Keep top 3-5 alternatives
        alternatives.resize(std::min(alternatives.size(), size_t(5)));

        // Check uncertainty
        auto uncertainty = uncertaintyDetector_.assessUncertainty(
            signals, candidates, market_data_
        );

        int selected_index;

        if (gradualAutomation_.shouldAskHuman(uncertainty, market_context_)) {
            // Ask human
            LOG_WARN("High uncertainty detected - requesting human decision");

            auto decision = humanDecisionLogger_.captureDecision(
                alternatives, uncertainty
            );

            selected_index = decision.selected_alternative;
        } else {
            // Automated decision
            selected_index = selectBestAlternative(alternatives);

            // Or use human-mimicking model
            if (use_human_mimic_model_) {
                selected_index = gradualAutomation_.predictHumanChoice(
                    alternatives, market_context_
                );
            }

            LOG_INFO("Automated decision: selected {} (confidence: {:.1f}%)",
                     alternatives[selected_index].strategy_name,
                     alternatives[selected_index].confidence_level * 100);
        }

        // Execute selected strategy (if not skipped)
        if (selected_index > 0) {
            executeStrategy(alternatives[selected_index - 1]);
        } else {
            LOG_INFO("Trade skipped (human decision or low confidence)");
        }
    }

private:
    UncertaintyDetector uncertaintyDetector_;
    HumanDecisionLogger humanDecisionLogger_;
    GradualAutomation gradualAutomation_;
    bool use_human_mimic_model_ = false;  // Enable after 100+ decisions
};
```

### 7. Decision Dashboard (Web UI)

**Streamlit Dashboard for Human Decisions:**

```python
# File: poc/human_decision_dashboard.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def human_decision_dashboard():
    st.title("🧠 Human Decision Interface")

    # Fetch pending decisions
    pending = get_pending_decisions()

    if not pending:
        st.success("No decisions pending - system operating autonomously")
        return

    decision = pending[0]

    st.header(f"Decision Required: {decision['symbol']}")

    # Show market context
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"${decision['price']:.2f}")
    col2.metric("IV Rank", f"{decision['iv_rank']:.0f}")
    col3.metric("Trend", decision['trend'])
    col4.metric("Regime", decision['regime'])

    # Show uncertainty metrics
    st.subheader("⚠️ Uncertainty Analysis")
    uncertainty = decision['uncertainty']

    st.write(f"**Prediction Confidence:** {uncertainty['confidence']:.1%}")
    st.write(f"**Signal Agreement:** {uncertainty['agreement']:.1%}")
    st.write(f"**Reason:** {uncertainty['reason']}")

    # Present alternatives
    st.subheader("📊 Strategy Alternatives")

    alternatives = decision['alternatives']

    for i, alt in enumerate(alternatives):
        with st.expander(f"Option {i+1}: {alt['name']} (Confidence: {alt['confidence']:.0%})"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Expected Outcomes:**")
                st.write(f"- Expected Profit: ${alt['expected_profit']:.2f}")
                st.write(f"- Max Loss: ${alt['max_loss']:.2f}")
                st.write(f"- Probability of Profit: {alt['pop']:.1%}")
                st.write(f"- Sharpe Ratio: {alt['sharpe']:.2f}")

            with col2:
                st.write("**Greeks:**")
                st.write(f"- Delta: {alt['delta']:.0f}")
                st.write(f"- Gamma: {alt['gamma']:.1f}")
                st.write(f"- Theta: ${alt['theta']:.2f}/day")
                st.write(f"- Vega: {alt['vega']:.0f}")

            st.write("**Capital Required:**", f"${alt['capital']:.0f}")

            if alt['risks']:
                st.write("⚠️ **Key Risks:**")
                for risk in alt['risks']:
                    st.write(f"  - {risk}")

            if alt['opportunities']:
                st.write("💡 **Key Opportunities:**")
                for opp in alt['opportunities']:
                    st.write(f"  - {opp}")

            # P/L chart
            fig = create_pnl_chart(alt)
            st.plotly_chart(fig)

    # Decision input
    st.subheader("Your Decision")

    selected = st.radio(
        "Select strategy:",
        options=[f"Option {i+1}: {alt['name']}" for i, alt in enumerate(alternatives)] +
                ["Skip this trade", "Request more analysis"],
        index=0
    )

    rationale = st.text_area(
        "Rationale (optional):",
        placeholder="Explain your reasoning..."
    )

    if st.button("Submit Decision", type="primary"):
        # Record decision
        selection_idx = parse_selection(selected)
        record_human_decision(decision['id'], selection_idx, rationale)

        st.success("✅ Decision recorded!")
        st.balloons()

        # Execute if not skipped
        if selection_idx > 0:
            execute_strategy(alternatives[selection_idx - 1])
            st.info(f"🚀 Executing: {alternatives[selection_idx - 1]['name']}")
```

### 8. Performance Tracking

```cpp
struct HumanLoopMetrics {
    int total_decisions_requested;
    int decisions_automated;
    int decisions_human;
    double human_decision_win_rate;
    double automated_decision_win_rate;
    double average_decision_time_seconds;
    int consecutive_correct_automations;
};

class HumanLoopAnalyzer {
public:
    HumanLoopMetrics analyzePerformance() {
        HumanLoopMetrics metrics{};

        auto all_decisions = loadAllDecisions();

        for (const auto& decision : all_decisions) {
            metrics.total_decisions_requested++;

            if (decision.was_automated) {
                metrics.decisions_automated++;
                if (decision.was_successful) {
                    metrics.automated_decision_win_rate++;
                }
            } else {
                metrics.decisions_human++;
                if (decision.was_successful) {
                    metrics.human_decision_win_rate++;
                }
            }
        }

        // Calculate win rates
        metrics.human_decision_win_rate /= metrics.decisions_human;
        metrics.automated_decision_win_rate /= metrics.decisions_automated;

        return metrics;
    }

    void generateReport() {
        auto metrics = analyzePerformance();

        LOG_INFO("═══════════════════════════════════════");
        LOG_INFO("Human-in-the-Loop Performance Report");
        LOG_INFO("═══════════════════════════════════════");
        LOG_INFO("Total Decisions: {}", metrics.total_decisions_requested);
        LOG_INFO("  Automated: {} ({:.1f}%)",
                 metrics.decisions_automated,
                 100.0 * metrics.decisions_automated / metrics.total_decisions_requested);
        LOG_INFO("  Human: {} ({:.1f}%)",
                 metrics.decisions_human,
                 100.0 * metrics.decisions_human / metrics.total_decisions_requested);
        LOG_INFO("");
        LOG_INFO("Win Rates:");
        LOG_INFO("  Human Decisions: {:.1f}%",
                 metrics.human_decision_win_rate * 100);
        LOG_INFO("  Automated Decisions: {:.1f}%",
                 metrics.automated_decision_win_rate * 100);
        LOG_INFO("");

        // Improvement over time
        if (metrics.automated_decision_win_rate >= metrics.human_decision_win_rate * 0.95) {
            LOG_INFO("✅ System learning successfully - automation quality high");
        } else {
            LOG_WARN("⚠️ Automated decisions underperforming - more human input needed");
        }
    }
};
```

### 9. Tier 1 Implementation Scope

**What to Implement in Tier 1:**
- ✅ Uncertainty detection (signal disagreement, low confidence)
- ✅ Alternative presentation (console logs)
- ✅ Human decision capture (CLI prompts)
- ✅ Decision recording to database
- ✅ Basic performance tracking

**Defer to Tier 2:**
- Web dashboard (Streamlit interface)
- ML model to mimic human decisions
- Sophisticated pattern analysis
- Automated threshold adjustment

**Files for Tier 1:**
```
src/trading_decision/
├── uncertainty_detector.cpp      # NEW - assess when human input needed
├── decision_presenter.cpp        # NEW - format alternatives for human
├── human_decision_logger.cpp     # NEW - record human choices
└── human_loop_engine.cpp         # NEW - orchestrate human-in-loop flow

Database schema:
CREATE TABLE human_decisions (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP,
    symbol VARCHAR(10),
    uncertainty_score DOUBLE,
    alternatives_json TEXT,  -- Store all alternatives
    selected_index INTEGER,
    rationale TEXT,
    executed_trade_id INTEGER,
    actual_pnl DOUBLE,
    was_successful BOOLEAN
);
```

---

## Summary: What Goes in Tier 1

### ✅ Tier 1 POC (Months 1-4)

**Stock Optimization:**
- Equal-risk contribution weighting
- Mean-variance optimization (basic)
- Rebalancing (threshold-based)

**Options Optimization:**
- Iron condor construction (IV rank > 50)
- Delta-neutral targeting (greedy algorithm)
- Theta maximization (simple scoring)

**Risk Management:**
- Position limits (10% max per position)
- Greeks constraints (delta, gamma, vega)
- Kelly criterion position sizing
- Drawdown control (15% threshold)

**Performance Tracking:**
- Sharpe ratio calculation
- Win rate / profit factor
- Greeks monitoring
- P/L attribution

**Integration:**
- Read MI signals (simple blend with base forecasts)
- Use correlation data (adjust covariance matrix)
- Risk approval before trade execution

**Human-in-the-Loop:**
- Uncertainty detection (low confidence, conflicting signals)
- Present top 3-5 alternatives with pros/cons
- Capture human decisions with rationale
- Record outcomes for learning
- Basic performance tracking (human vs automated win rates)

### 🔮 Tier 2+ (Month 5+, After Profitability Proven)

- Advanced QP/MIP solvers (Gurobi, CPLEX)
- Black-Litterman with ML views
- Multi-period optimization
- Transaction cost modeling
- Reinforcement learning strategy selection
- Neural network return forecasting
- Robust optimization under uncertainty
- Convertible bond arbitrage
- Interest rate derivatives
- Multi-asset correlation trading

---

## Dependencies

**Libraries Required:**
- **Eigen3** - Linear algebra (INSTALLED)
- **OSQP** or **qpOASES** - Quadratic programming (Tier 1)
- **COIN-OR CBC** - Mixed-integer programming (Tier 2)
- **Intel MKL** - BLAS/LAPACK acceleration (OPTIONAL)

**Integration Points:**
- Market Intelligence Engine (predictions)
- Correlation Engine (time-lagged correlations)
- Risk Management System (limits, VaR)
- Trading Execution (order placement)

---

This profit optimization engine provides the quantitative foundation for systematic, risk-adjusted trading across multiple asset classes.
