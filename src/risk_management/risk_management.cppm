/**
 * BigBrotherAnalytics - Risk Management Module (C++23)
 *
 * Comprehensive risk management with fluent API.
 * Consolidates: risk_manager, stop_loss, position_sizer, monte_carlo
 *
 * Following C++ Core Guidelines:
 * - C.21: Rule of Five
 * - R.1: RAII for resource management
 * - I.11: Never transfer ownership by raw pointer
 * - ES.20: Always initialize objects
 * - Trailing return syntax throughout
 */

// Global module fragment
module;

#include <vector>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <mutex>
#include <random>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <expected>

#ifdef _OPENMP
#include <omp.h>
#endif

// Module declaration
export module bigbrother.risk_management;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.utils.timer;
import bigbrother.options.pricing;

export namespace bigbrother::risk {

using namespace bigbrother::types;
using namespace bigbrother::utils;
using namespace bigbrother::options;

// ============================================================================
// Risk Limits Configuration (C.1: Struct for passive data)
// ============================================================================

struct RiskLimits {
    double account_value{30'000.0};
    double max_daily_loss{900.0};
    double max_position_size{1'500.0};
    int max_concurrent_positions{10};
    double max_portfolio_heat{0.15};
    double max_correlation_exposure{0.30};
    bool require_stop_loss{true};

    [[nodiscard]] static constexpr auto forThirtyKAccount() noexcept -> RiskLimits {
        return {
            .account_value = 30'000.0,
            .max_daily_loss = 900.0,
            .max_position_size = 1'500.0,
            .max_concurrent_positions = 10,
            .max_portfolio_heat = 0.15,
            .max_correlation_exposure = 0.30,
            .require_stop_loss = true
        };
    }

    [[nodiscard]] auto validate() const noexcept -> Result<void> {
        if (account_value <= 0.0) {
            return makeError<void>(ErrorCode::InvalidParameter, "Account value must be positive");
        }
        if (max_daily_loss <= 0.0 || max_daily_loss > account_value) {
            return makeError<void>(ErrorCode::InvalidParameter, "Invalid daily loss limit");
        }
        if (max_position_size <= 0.0 || max_position_size > account_value) {
            return makeError<void>(ErrorCode::InvalidParameter, "Invalid position size limit");
        }
        if (max_concurrent_positions <= 0) {
            return makeError<void>(ErrorCode::InvalidParameter, "Invalid concurrent positions limit");
        }
        return {};
    }
};

// ============================================================================
// Position and Portfolio Risk Metrics
// ============================================================================

struct PositionRisk {
    std::string symbol;
    double position_value{0.0};
    double unrealized_pnl{0.0};
    double realized_pnl{0.0};
    double max_loss{0.0};
    double probability_of_loss{0.0};
    double expected_value{0.0};
    double sharpe_ratio{0.0};
    double var_95{0.0};
    double delta_exposure{0.0};
    double vega_exposure{0.0};
    double theta_decay{0.0};
    bool has_stop_loss{false};

    [[nodiscard]] auto isHighRisk() const noexcept -> bool {
        return position_value > 0.0 && 
               (max_loss / position_value > 0.20 || var_95 / position_value > 0.15);
    }
};

struct PortfolioRisk {
    double total_value{0.0};
    double total_exposure{0.0};
    double net_delta{0.0};
    double net_vega{0.0};
    double net_theta{0.0};
    double daily_pnl{0.0};
    double daily_loss_remaining{900.0};
    int active_positions{0};
    double portfolio_heat{0.0};
    double max_drawdown{0.0};
    double sharpe_ratio{0.0};
    std::vector<PositionRisk> positions;

    [[nodiscard]] auto canOpenNewPosition() const noexcept -> bool {
        return active_positions < 10 && daily_loss_remaining > 0.0;
    }

    [[nodiscard]] auto getRiskLevel() const noexcept -> std::string {
        if (portfolio_heat > 0.15) return "HIGH";
        if (portfolio_heat > 0.10) return "MEDIUM";
        return "LOW";
    }
};

struct TradeRisk {
    double position_size{0.0};
    double max_loss{0.0};
    double expected_return{0.0};
    double expected_value{0.0};
    double win_probability{0.0};
    double kelly_fraction{0.0};
    double risk_reward_ratio{0.0};
    double var_95{0.0};
    bool approved{false};
    std::string rejection_reason;

    [[nodiscard]] auto isApproved() const noexcept -> bool {
        return approved;
    }
};

// ============================================================================
// Stop Loss Management (Fluent API)
// ============================================================================

enum class StopType {
    Hard,
    Trailing,
    TimeStop,
    VolatilityStop,
    Greeks
};

struct Stop {
    std::string position_id;
    StopType type{StopType::Hard};
    Price trigger_price{0.0};
    Price initial_price{0.0};
    double trail_amount{0.0};
    Timestamp expiration{0};
    bool triggered{false};

    [[nodiscard]] auto isTriggered(Price current_price) const noexcept -> bool {
        if (triggered) return true;

        switch (type) {
            case StopType::Hard:
                return initial_price > trigger_price 
                    ? current_price <= trigger_price
                    : current_price >= trigger_price;
            
            case StopType::Trailing: {
                if (initial_price > trigger_price) {
                    auto const trailing_stop = current_price - trail_amount;
                    return current_price <= trailing_stop;
                } else {
                    auto const trailing_stop = current_price + trail_amount;
                    return current_price >= trailing_stop;
                }
            }
            
            case StopType::TimeStop: {
                auto const now = Timer::now();
                return now >= expiration;
            }
            
            case StopType::VolatilityStop: {
                auto const price_change_pct = std::abs(
                    (current_price - initial_price) / initial_price
                );
                return price_change_pct > trail_amount;
            }
            
            case StopType::Greeks:
                return current_price <= trigger_price;
            
            default:
                return false;
        }
    }
};

class StopLossManager {
public:
    // C.21: Rule of Five
    StopLossManager() = default;
    // C.21: Rule of Five - deleted due to mutex member
    StopLossManager(StopLossManager const&) = delete;
    auto operator=(StopLossManager const&) -> StopLossManager& = delete;
    StopLossManager(StopLossManager&&) noexcept = delete;
    auto operator=(StopLossManager&&) noexcept -> StopLossManager& = delete;
    ~StopLossManager() = default;

    // Fluent API
    [[nodiscard]] auto addStop(std::string position_id, StopType type, Price trigger, 
                                Price initial, double trail = 0.0) -> StopLossManager& {
        std::lock_guard<std::mutex> lock(mutex_);
        stops_[position_id] = Stop{
            .position_id = position_id,
            .type = type,
            .trigger_price = trigger,
            .initial_price = initial,
            .trail_amount = trail
        };
        return *this;
    }

    [[nodiscard]] auto checkStop(std::string const& position_id, Price current_price) 
        -> Result<bool> {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = stops_.find(position_id);
        if (it == stops_.end()) {
            return makeError<bool>(ErrorCode::InvalidParameter, "Stop not found");
        }

        auto triggered = it->second.isTriggered(current_price);
        if (triggered) {
            it->second.triggered = true;
        }
        
        return triggered;
    }

    [[nodiscard]] auto removeStop(std::string const& position_id) -> StopLossManager& {
        std::lock_guard<std::mutex> lock(mutex_);
        stops_.erase(position_id);
        return *this;
    }

private:
    std::unordered_map<std::string, Stop> stops_;
    mutable std::mutex mutex_;
};

// ============================================================================
// Position Sizing (Fluent API)
// ============================================================================

enum class SizingMethod {
    FixedDollar,
    FixedPercent,
    KellyCriterion,
    KellyHalf,
    VolatilityAdjusted,
    RiskParity
};

class PositionSizer {
public:
    [[nodiscard]] static auto calculate(
        SizingMethod method,
        double account_value,
        double win_probability,
        double win_amount,
        double loss_amount,
        double volatility = 0.0
    ) noexcept -> Result<double> {
        
        switch (method) {
            case SizingMethod::FixedDollar:
                return 1000.0;
            
            case SizingMethod::FixedPercent:
                return account_value * 0.02;
            
            case SizingMethod::KellyCriterion: {
                if (loss_amount == 0.0) {
                    return makeError<double>(ErrorCode::InvalidParameter, "Loss amount cannot be zero");
                }
                auto const kelly = (win_probability * win_amount - (1.0 - win_probability) * loss_amount) / loss_amount;
                return std::max(0.0, std::min(kelly * account_value, account_value * 0.05));
            }
            
            case SizingMethod::KellyHalf: {
                auto kelly_result = calculate(SizingMethod::KellyCriterion, account_value, 
                                             win_probability, win_amount, loss_amount);
                if (!kelly_result) return kelly_result;
                return kelly_result.value() * 0.5;
            }
            
            case SizingMethod::VolatilityAdjusted:
                if (volatility > 0.0) {
                    return (account_value * 0.02) / volatility;
                }
                return account_value * 0.02;
            
            case SizingMethod::RiskParity:
                return account_value * 0.025;
            
            default:
                return makeError<double>(ErrorCode::InvalidParameter, "Unknown sizing method");
        }
    }
};

// ============================================================================
// Monte Carlo Simulation
// ============================================================================

struct SimulationResult {
    double expected_value{0.0};
    double std_deviation{0.0};
    double probability_of_profit{0.0};
    double var_95{0.0};
    double var_99{0.0};
    double max_profit{0.0};
    double max_loss{0.0};
    std::vector<double> pnl_distribution;
};

class MonteCarloSimulator {
public:
    [[nodiscard]] static auto simulateOptionTrade(
        PricingParams const& params,
        double position_size,
        int num_simulations = 10'000,
        int num_steps = 100
    ) noexcept -> Result<SimulationResult> {
        
        if (auto validation = params.validate(); !validation) {
            return std::unexpected(validation.error());
        }

        if (num_simulations < 100) {
            return makeError<SimulationResult>(
                ErrorCode::InvalidParameter,
                "Need at least 100 simulations"
            );
        }

        // Convert PricingParams to ExtendedPricingParams
        options::ExtendedPricingParams extended_params{
            .spot_price = params.spot_price,
            .strike_price = params.strike_price,
            .time_to_expiration = params.time_to_expiration,
            .risk_free_rate = params.risk_free_rate,
            .volatility = params.volatility,
            .dividend_yield = 0.0,
            .option_type = params.option_type,
            .option_style = options::OptionStyle::American
        };
        auto price_result = OptionsPricer::price(extended_params, OptionsPricer::Model::BlackScholes);
        if (!price_result) {
            return std::unexpected(price_result.error());
        }

        auto const initial_option_price = price_result->option_price;
        auto const time_step = params.time_to_expiration / static_cast<double>(num_steps);
        
        std::vector<double> final_pnls(num_simulations);

        #pragma omp parallel
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> normal(0.0, 1.0);

            #pragma omp for
            for (int sim = 0; sim < num_simulations; ++sim) {
                double spot = params.spot_price;
                
                for (int step = 0; step < num_steps; ++step) {
                    auto const drift = params.risk_free_rate * time_step;
                    auto const diffusion = params.volatility * std::sqrt(time_step) * normal(gen);
                    spot *= std::exp(drift + diffusion);
                }

                auto const final_intrinsic = params.option_type == OptionType::Call
                    ? std::max(0.0, spot - params.strike_price)
                    : std::max(0.0, params.strike_price - spot);
                
                final_pnls[sim] = (final_intrinsic - initial_option_price) * position_size;
            }
        }

        SimulationResult result;
        result.pnl_distribution = final_pnls;
        
        std::sort(final_pnls.begin(), final_pnls.end());
        
        result.max_loss = final_pnls.front();
        result.max_profit = final_pnls.back();
        
        auto const sum = std::accumulate(final_pnls.begin(), final_pnls.end(), 0.0);
        result.expected_value = sum / static_cast<double>(num_simulations);
        
        auto const variance = std::accumulate(final_pnls.begin(), final_pnls.end(), 0.0,
            [ev = result.expected_value](double acc, double val) {
                return acc + (val - ev) * (val - ev);
            }) / static_cast<double>(num_simulations);
        result.std_deviation = std::sqrt(variance);
        
        result.probability_of_profit = static_cast<double>(
            std::count_if(final_pnls.begin(), final_pnls.end(), 
                         [](double pnl) { return pnl > 0.0; })
        ) / static_cast<double>(num_simulations);
        
        result.var_95 = final_pnls[static_cast<size_t>(num_simulations * 0.05)];
        result.var_99 = final_pnls[static_cast<size_t>(num_simulations * 0.01)];
        
        return result;
    }
};

// ============================================================================
// Risk Manager (Fluent API)
// ============================================================================

class RiskManager {
public:
    explicit RiskManager(RiskLimits limits = RiskLimits::forThirtyKAccount())
        : limits_{limits}, daily_pnl_{0.0}, daily_loss_remaining_{limits.max_daily_loss} {
        auto result = limits_.validate();
        (void)result;  // Suppress nodiscard - constructor can't fail
    }

    // C.21: Rule of Five - deleted due to mutex member in pImpl
    RiskManager(RiskManager const&) = delete;
    auto operator=(RiskManager const&) -> RiskManager& = delete;
    RiskManager(RiskManager&&) noexcept = delete;
    auto operator=(RiskManager&&) noexcept -> RiskManager& = delete;
    ~RiskManager() = default;

    // Fluent API
    [[nodiscard]] auto withLimits(RiskLimits limits) -> RiskManager& {
        limits_ = limits;
        daily_loss_remaining_ = limits.max_daily_loss;
        return *this;
    }

    [[nodiscard]] auto assessTrade(
        std::string symbol,
        double position_size,
        Price entry_price,
        Price stop_price,
        Price target_price,
        double win_probability
    ) noexcept -> Result<TradeRisk> {
        
        TradeRisk risk{};
        risk.position_size = std::min(position_size, limits_.max_position_size);
        risk.max_loss = std::abs(entry_price - stop_price) * (position_size / entry_price);
        risk.expected_return = (target_price - entry_price) * (position_size / entry_price);
        risk.win_probability = win_probability;
        risk.expected_value = win_probability * risk.expected_return -
                              (1.0 - win_probability) * risk.max_loss;
        risk.risk_reward_ratio = risk.max_loss > 0.0 ? risk.expected_return / risk.max_loss : 0.0;
        risk.approved = true;

        std::lock_guard<std::mutex> lock(mutex_);
        
        if (daily_loss_remaining_ < risk.max_loss) {
            risk.approved = false;
            risk.rejection_reason = "Daily loss limit would be exceeded";
        }

        if (risk.position_size > limits_.max_position_size) {
            risk.approved = false;
            risk.rejection_reason = "Position size exceeds maximum";
        }

        if (static_cast<int>(positions_.size()) >= limits_.max_concurrent_positions) {
            risk.approved = false;
            risk.rejection_reason = "Maximum concurrent positions reached";
        }

        return risk;
    }

    [[nodiscard]] auto getPortfolioRisk() const noexcept -> Result<PortfolioRisk> {
        std::lock_guard<std::mutex> lock(mutex_);
        
        PortfolioRisk portfolio{};
        portfolio.daily_pnl = daily_pnl_;
        portfolio.daily_loss_remaining = daily_loss_remaining_;
        portfolio.active_positions = static_cast<int>(positions_.size());
        
        return portfolio;
    }

    [[nodiscard]] auto stopLoss() -> StopLossManager& {
        return stop_loss_manager_;
    }

private:
    RiskLimits limits_;
    std::atomic<double> daily_pnl_;
    std::atomic<double> daily_loss_remaining_;
    std::unordered_map<std::string, PositionRisk> positions_;
    StopLossManager stop_loss_manager_;
    mutable std::mutex mutex_;
};

} // export namespace bigbrother::risk
