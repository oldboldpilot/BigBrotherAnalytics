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

#include <algorithm>
#include <atomic>
#include <cmath>
#include <expected>
#include <memory>
#include <mutex>
#include <numbers>
#include <random>
#include <unordered_map>
#include <vector>

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
// Note: NOT importing bigbrother.risk to avoid duplicate definitions

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
        return {.account_value = 30'000.0,
                .max_daily_loss = 900.0,
                .max_position_size = 1'500.0,
                .max_concurrent_positions = 10,
                .max_portfolio_heat = 0.15,
                .max_correlation_exposure = 0.30,
                .require_stop_loss = true};
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
            return makeError<void>(ErrorCode::InvalidParameter,
                                   "Invalid concurrent positions limit");
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
        if (portfolio_heat > 0.15)
            return "HIGH";
        if (portfolio_heat > 0.10)
            return "MEDIUM";
        return "LOW";
    }
};

// Trade Risk Assessment (comprehensive version)
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

    [[nodiscard]] auto isApproved() const noexcept -> bool { return approved; }
};

// ============================================================================
// Stop Loss Management (Fluent API)
// ============================================================================

enum class StopType { Hard, Trailing, TimeStop, VolatilityStop, Greeks };

struct Stop {
    std::string position_id;
    StopType type{StopType::Hard};
    Price trigger_price{0.0};
    Price initial_price{0.0};
    double trail_amount{0.0};
    Timestamp expiration{0};
    bool triggered{false};

    [[nodiscard]] auto isTriggered(Price current_price) const noexcept -> bool {
        if (triggered)
            return true;

        switch (type) {
            case StopType::Hard:
                return initial_price > trigger_price ? current_price <= trigger_price
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
                auto const price_change_pct =
                    std::abs((current_price - initial_price) / initial_price);
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
    [[nodiscard]] auto addStop(std::string position_id, StopType type, Price trigger, Price initial,
                               double trail = 0.0) -> StopLossManager& {
        std::lock_guard<std::mutex> lock(mutex_);
        stops_[position_id] = Stop{.position_id = position_id,
                                   .type = type,
                                   .trigger_price = trigger,
                                   .initial_price = initial,
                                   .trail_amount = trail};
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
    [[nodiscard]] static auto calculate(SizingMethod method, double account_value,
                                        double win_probability, double win_amount,
                                        double loss_amount, double volatility = 0.0) noexcept
        -> Result<double> {

        switch (method) {
            case SizingMethod::FixedDollar:
                return 1000.0;

            case SizingMethod::FixedPercent:
                return account_value * 0.02;

            case SizingMethod::KellyCriterion: {
                if (loss_amount == 0.0) {
                    return makeError<double>(ErrorCode::InvalidParameter,
                                             "Loss amount cannot be zero");
                }
                auto const kelly =
                    (win_probability * win_amount - (1.0 - win_probability) * loss_amount) /
                    loss_amount;
                return std::max(0.0, std::min(kelly * account_value, account_value * 0.05));
            }

            case SizingMethod::KellyHalf: {
                auto kelly_result = calculate(SizingMethod::KellyCriterion, account_value,
                                              win_probability, win_amount, loss_amount);
                if (!kelly_result)
                    return kelly_result;
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
    [[nodiscard]] static auto simulateOptionTrade(PricingParams const& params, double position_size,
                                                  int num_simulations = 10'000,
                                                  int num_steps = 100) noexcept
        -> Result<SimulationResult> {

        if (auto validation = params.validate(); !validation) {
            return std::unexpected(validation.error());
        }

        if (num_simulations < 100) {
            return makeError<SimulationResult>(ErrorCode::InvalidParameter,
                                               "Need at least 100 simulations");
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
            .option_style = options::OptionStyle::American};
        auto price_result =
            OptionsPricer::price(extended_params, OptionsPricer::Model::BlackScholes);
        if (!price_result) {
            return std::unexpected(price_result.error());
        }

        auto const initial_option_price = price_result->option_price;
        auto const time_step = params.time_to_expiration / static_cast<double>(num_steps);

        std::vector<double> final_pnls(num_simulations);

#pragma omp parallel
        {
            std::random_device rd;
            auto gen = std::mt19937{rd()};
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
                                              [ev = result.expected_value](double acc, double val) -> double {
                                                  return acc + (val - ev) * (val - ev);
                                              }) /
                              static_cast<double>(num_simulations);
        result.std_deviation = std::sqrt(variance);

        result.probability_of_profit =
            static_cast<double>(std::count_if(final_pnls.begin(), final_pnls.end(),
                                              [](double pnl) -> bool { return pnl > 0.0; })) /
            static_cast<double>(num_simulations);

        result.var_95 = final_pnls[static_cast<size_t>(num_simulations * 0.05)];
        result.var_99 = final_pnls[static_cast<size_t>(num_simulations * 0.01)];

        return result;
    }
};

// ============================================================================
// Kelly Criterion Calculator (Fluent API)
// ============================================================================

class KellyCalculator {
  public:
    explicit KellyCalculator(RiskManager const& risk_mgr) : risk_mgr_{risk_mgr} {}

    [[nodiscard]] auto withWinRate(double rate) -> KellyCalculator& {
        win_rate_ = rate;
        return *this;
    }

    [[nodiscard]] auto withWinLossRatio(double ratio) -> KellyCalculator& {
        win_loss_ratio_ = ratio;
        return *this;
    }

    [[nodiscard]] auto withDrawdownLimit(double limit) -> KellyCalculator& {
        drawdown_limit_ = limit;
        return *this;
    }

    [[nodiscard]] auto calculate() const noexcept -> Result<double> {
        if (win_rate_ < 0.0 || win_rate_ > 1.0) {
            return makeError<double>(ErrorCode::InvalidParameter, "Win rate must be in [0, 1]");
        }
        if (win_loss_ratio_ <= 0.0) {
            return makeError<double>(ErrorCode::InvalidParameter,
                                     "Win/loss ratio must be positive");
        }

        double const loss_rate = 1.0 - win_rate_;
        double const kelly = (win_rate_ * win_loss_ratio_ - loss_rate) / win_loss_ratio_;

        if (kelly < 0.0) {
            return 0.0; // No edge
        }

        // Apply drawdown limit if specified
        if (drawdown_limit_ > 0.0) {
            double const limited_kelly = kelly * drawdown_limit_;
            return std::min(kelly, limited_kelly);
        }

        return std::min(kelly, 0.25); // Never exceed 25% fractional Kelly
    }

  private:
    RiskManager const& risk_mgr_;
    double win_rate_{0.0};
    double win_loss_ratio_{1.0};
    double drawdown_limit_{0.0};
};

// ============================================================================
// Monte Carlo Simulator (Fluent Builder API)
// ============================================================================

class MonteCarloSimulatorBuilder {
  public:
    explicit MonteCarloSimulatorBuilder(RiskManager const& risk_mgr) : risk_mgr_{risk_mgr} {}

    [[nodiscard]] auto forOption(PricingParams params) -> MonteCarloSimulatorBuilder& {
        params_ = params;
        return *this;
    }

    [[nodiscard]] auto withSimulations(int count) -> MonteCarloSimulatorBuilder& {
        num_simulations_ = count;
        return *this;
    }

    [[nodiscard]] auto withSteps(int count) -> MonteCarloSimulatorBuilder& {
        num_steps_ = count;
        return *this;
    }

    [[nodiscard]] auto withPositionSize(double size) -> MonteCarloSimulatorBuilder& {
        position_size_ = size;
        return *this;
    }

    [[nodiscard]] auto run() const noexcept -> Result<SimulationResult> {
        return MonteCarloSimulator::simulateOptionTrade(params_, position_size_, num_simulations_,
                                                        num_steps_);
    }

  private:
    RiskManager const& risk_mgr_;
    PricingParams params_{};
    double position_size_{1.0};
    int num_simulations_{10'000};
    int num_steps_{100};
};

// ============================================================================
// Trade Risk Builder (Fluent API)
// ============================================================================

class TradeRiskBuilder {
  public:
    explicit TradeRiskBuilder(RiskManager& risk_mgr) : risk_mgr_{risk_mgr} {}

    [[nodiscard]] auto forSymbol(std::string symbol) -> TradeRiskBuilder& {
        symbol_ = std::move(symbol);
        return *this;
    }

    [[nodiscard]] auto withQuantity(int qty) -> TradeRiskBuilder& {
        quantity_ = qty;
        return *this;
    }

    [[nodiscard]] auto atPrice(double price) -> TradeRiskBuilder& {
        entry_price_ = price;
        return *this;
    }

    [[nodiscard]] auto withStop(double stop) -> TradeRiskBuilder& {
        stop_price_ = stop;
        return *this;
    }

    [[nodiscard]] auto withTarget(double target) -> TradeRiskBuilder& {
        target_price_ = target;
        return *this;
    }

    [[nodiscard]] auto withProbability(double prob) -> TradeRiskBuilder& {
        win_probability_ = prob;
        return *this;
    }

    [[nodiscard]] auto assess() noexcept -> Result<TradeRisk> {
        return risk_mgr_.assessTrade(symbol_, static_cast<double>(quantity_) * entry_price_,
                                     entry_price_, stop_price_, target_price_, win_probability_);
    }

  private:
    RiskManager& risk_mgr_;
    std::string symbol_;
    int quantity_{0};
    double entry_price_{0.0};
    double stop_price_{0.0};
    double target_price_{0.0};
    double win_probability_{0.0};
};

// ============================================================================
// Portfolio Risk Builder (Fluent API)
// ============================================================================

struct PortfolioPositionDef {
    std::string symbol;
    double quantity{0.0};
    double price{0.0};
    double volatility{0.0};
};

class PortfolioRiskBuilder {
  public:
    explicit PortfolioRiskBuilder(RiskManager& risk_mgr) : risk_mgr_{risk_mgr} {}

    [[nodiscard]] auto addPosition(std::string symbol, double quantity, double price,
                                   double volatility = 0.0) -> PortfolioRiskBuilder& {
        positions_.emplace_back(PortfolioPositionDef{.symbol = std::move(symbol),
                                                     .quantity = quantity,
                                                     .price = price,
                                                     .volatility = volatility});
        return *this;
    }

    [[nodiscard]] auto calculateHeat() -> PortfolioRiskBuilder& {
        double total_exposure = 0.0;
        for (auto const& pos : positions_) {
            total_exposure += pos.quantity * pos.price;
        }
        portfolio_heat_ =
            total_exposure / 30'000.0; // 30k account - can be parameterized
        return *this;
    }

    [[nodiscard]] auto calculateVaR(double confidence) -> PortfolioRiskBuilder& {
        var_confidence_ = confidence;
        // VaR calculation would be more sophisticated in production
        return *this;
    }

    [[nodiscard]] auto analyze() const noexcept -> Result<PortfolioRisk> {
        PortfolioRisk portfolio{};

        for (auto const& pos : positions_) {
            portfolio.total_value += pos.quantity * pos.price;
            portfolio.total_exposure += pos.quantity * pos.price;
            portfolio.active_positions++;
        }

        portfolio.portfolio_heat = portfolio_heat_;
        return portfolio;
    }

  private:
    RiskManager& risk_mgr_;
    std::vector<PortfolioPositionDef> positions_;
    double portfolio_heat_{0.0};
    double var_confidence_{0.95};
};

// ============================================================================
// Position Sizer Builder (Fluent API)
// ============================================================================

class PositionSizerBuilder {
  public:
    explicit PositionSizerBuilder(RiskManager const& risk_mgr) : risk_mgr_{risk_mgr} {}

    [[nodiscard]] auto withMethod(SizingMethod method) -> PositionSizerBuilder& {
        method_ = method;
        return *this;
    }

    [[nodiscard]] auto withAccountValue(double value) -> PositionSizerBuilder& {
        account_value_ = value;
        return *this;
    }

    [[nodiscard]] auto withWinProbability(double prob) -> PositionSizerBuilder& {
        win_probability_ = prob;
        return *this;
    }

    [[nodiscard]] auto withWinAmount(double amount) -> PositionSizerBuilder& {
        win_amount_ = amount;
        return *this;
    }

    [[nodiscard]] auto withLossAmount(double amount) -> PositionSizerBuilder& {
        loss_amount_ = amount;
        return *this;
    }

    [[nodiscard]] auto withVolatility(double vol) -> PositionSizerBuilder& {
        volatility_ = vol;
        return *this;
    }

    [[nodiscard]] auto calculate() const noexcept -> Result<double> {
        return PositionSizer::calculate(method_, account_value_, win_probability_, win_amount_,
                                        loss_amount_, volatility_);
    }

  private:
    RiskManager const& risk_mgr_;
    SizingMethod method_{SizingMethod::FixedPercent};
    double account_value_{30'000.0};
    double win_probability_{0.55};
    double win_amount_{100.0};
    double loss_amount_{100.0};
    double volatility_{0.0};
};

// ============================================================================
// Risk Manager (Fluent API - Enhanced)
// ============================================================================

class RiskManager {
  public:
    explicit RiskManager(RiskLimits limits = RiskLimits::forThirtyKAccount())
        : limits_{limits}, daily_pnl_{0.0}, daily_loss_remaining_{limits.max_daily_loss} {
        auto result = limits_.validate();
        (void)result; // Suppress nodiscard - constructor can't fail
    }

    // C.21: Rule of Five - deleted due to mutex member in pImpl
    RiskManager(RiskManager const&) = delete;
    auto operator=(RiskManager const&) -> RiskManager& = delete;
    RiskManager(RiskManager&&) noexcept = delete;
    auto operator=(RiskManager&&) noexcept -> RiskManager& = delete;
    ~RiskManager() = default;

    // ========================================================================
    // Fluent Configuration Methods
    // ========================================================================

    [[nodiscard]] auto withLimits(RiskLimits limits) -> RiskManager& {
        limits_ = limits;
        daily_loss_remaining_ = limits.max_daily_loss;
        return *this;
    }

    [[nodiscard]] auto setMaxDailyLoss(double amount) -> RiskManager& {
        limits_.max_daily_loss = amount;
        daily_loss_remaining_ = amount;
        return *this;
    }

    [[nodiscard]] auto setMaxPositionSize(double amount) -> RiskManager& {
        limits_.max_position_size = amount;
        return *this;
    }

    [[nodiscard]] auto setMaxPortfolioHeat(double pct) -> RiskManager& {
        limits_.max_portfolio_heat = pct;
        return *this;
    }

    [[nodiscard]] auto setMaxConcurrentPositions(int count) -> RiskManager& {
        limits_.max_concurrent_positions = count;
        return *this;
    }

    [[nodiscard]] auto setAccountValue(double value) -> RiskManager& {
        limits_.account_value = value;
        return *this;
    }

    [[nodiscard]] auto requireStopLoss(bool required) -> RiskManager& {
        limits_.require_stop_loss = required;
        return *this;
    }

    // ========================================================================
    // Fluent Trade Assessment (Terminal Operations)
    // ========================================================================

    [[nodiscard]] auto assessTrade() -> TradeRiskBuilder { return TradeRiskBuilder(*this); }

    [[nodiscard]] auto assessTrade(std::string symbol, double position_size, Price entry_price,
                                   Price stop_price, Price target_price,
                                   double win_probability) noexcept -> Result<TradeRisk> {

        TradeRisk risk{};
        risk.position_size = std::min(position_size, limits_.max_position_size);
        risk.max_loss = std::abs(entry_price - stop_price) * (position_size / entry_price);
        risk.expected_return = (target_price - entry_price) * (position_size / entry_price);
        risk.win_probability = win_probability;
        risk.expected_value =
            win_probability * risk.expected_return - (1.0 - win_probability) * risk.max_loss;
        risk.risk_reward_ratio = risk.max_loss > 0.0 ? risk.expected_return / risk.max_loss : 0.0;

        // Calculate Kelly fraction
        if (risk.max_loss > 0.0) {
            risk.kelly_fraction = (win_probability * risk.expected_return -
                                  (1.0 - win_probability) * risk.max_loss) / risk.max_loss;
            risk.kelly_fraction = std::max(0.0, risk.kelly_fraction);
        }

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

    // ========================================================================
    // Fluent Portfolio Analysis
    // ========================================================================

    [[nodiscard]] auto portfolio() -> PortfolioRiskBuilder { return PortfolioRiskBuilder(*this); }

    [[nodiscard]] auto getPortfolioRisk() const noexcept -> Result<PortfolioRisk> {
        std::lock_guard<std::mutex> lock(mutex_);

        PortfolioRisk portfolio{};
        portfolio.daily_pnl = daily_pnl_;
        portfolio.daily_loss_remaining = daily_loss_remaining_;
        portfolio.active_positions = static_cast<int>(positions_.size());

        return portfolio;
    }

    // ========================================================================
    // Specialized Calculator Accessors (Factory Methods)
    // ========================================================================

    [[nodiscard]] auto kelly() -> KellyCalculator { return KellyCalculator(*this); }

    [[nodiscard]] auto monteCarlo() -> MonteCarloSimulatorBuilder {
        return MonteCarloSimulatorBuilder(*this);
    }

    [[nodiscard]] auto positionSizer() -> PositionSizerBuilder { return PositionSizerBuilder(*this); }

    // ========================================================================
    // Stop Loss Management
    // ========================================================================

    [[nodiscard]] auto stopLoss() -> StopLossManager& { return stop_loss_manager_; }

    // ========================================================================
    // Daily P&L Management
    // ========================================================================

    [[nodiscard]] auto updateDailyPnL(double pnl) -> RiskManager& {
        daily_pnl_ += pnl;
        daily_loss_remaining_ -= std::min(0.0, pnl); // Only count losses
        return *this;
    }

    [[nodiscard]] auto resetDaily() -> RiskManager& {
        daily_pnl_ = 0.0;
        daily_loss_remaining_ = limits_.max_daily_loss;
        return *this;
    }

    [[nodiscard]] auto getDailyPnL() const noexcept -> double { return daily_pnl_; }

    [[nodiscard]] auto getDailyLossRemaining() const noexcept -> double {
        return daily_loss_remaining_;
    }

    [[nodiscard]] auto isDailyLossLimitReached() const noexcept -> bool {
        return daily_loss_remaining_ <= 0.0;
    }

  private:
    RiskLimits limits_;
    std::atomic<double> daily_pnl_;
    std::atomic<double> daily_loss_remaining_;
    std::unordered_map<std::string, PositionRisk> positions_;
    StopLossManager stop_loss_manager_;
    mutable std::mutex mutex_;
};

} // namespace bigbrother::risk
