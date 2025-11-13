"""
BigBrotherAnalytics - Paper Trading Configuration

Centralized configuration for paper trading environment.
Manages position limits, risk parameters, and trading constraints.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-12
"""

from dataclasses import dataclass
from datetime import time
from typing import Dict, List, Optional


@dataclass
class PaperTradingConfig:
    """Paper trading configuration with strict risk limits"""

    # Account Configuration
    account_id: str
    initial_capital: float = 30_000.00  # Starting capital
    position_limit: float = 2_000.00  # Max per position ($2K limit)
    daily_loss_limit: float = 900.00  # Max daily loss (3% of capital)

    # Risk Management
    max_portfolio_heat: float = 0.15  # Max 15% portfolio heat
    max_sector_allocation: float = 0.25  # Max 25% per sector
    min_sector_allocation: float = 0.05  # Min 5% per sector
    max_single_position_pct: float = 0.067  # Max 6.7% per position ($2K / $30K)

    # Trading Hours (Market hours: 9:30 AM - 4:00 PM ET)
    trading_start_time: time = time(9, 30)  # 9:30 AM ET
    trading_end_time: time = time(16, 0)  # 4:00 PM ET
    close_positions_before_market_close: bool = True
    minutes_before_close_to_exit: int = 15  # Exit 15 min before close

    # ML Model Configuration
    ml_model_path: str = "models/price_predictor.onnx"
    ml_confidence_threshold: float = 0.55  # Min 55% confidence
    ml_min_win_rate: float = 0.55  # Target 55% win rate

    # Options Strategy Configuration
    max_strategies_active: int = 10  # Max concurrent strategies
    allowed_strategies: List[str] = None  # None = all 52 strategies allowed

    # Position Sizing
    kelly_criterion_enabled: bool = True
    kelly_fraction: float = 0.25  # Use 25% of Kelly (conservative)
    fixed_position_size: Optional[float] = None  # Override Kelly if set

    # Monitoring and Alerts
    enable_real_time_monitoring: bool = True
    alert_on_daily_loss_pct: float = 0.02  # Alert at 2% daily loss
    alert_on_position_loss_pct: float = 0.10  # Alert at 10% position loss

    # Data Collection
    log_all_trades: bool = True
    log_directory: str = "logs/paper_trading"
    metrics_update_interval_seconds: int = 60  # Update metrics every 60s

    # API Configuration
    schwab_api_rate_limit_per_minute: int = 120
    news_api_daily_limit: int = 96

    def __post_init__(self):
        """Initialize allowed strategies if not set"""
        if self.allowed_strategies is None:
            # All 52 strategies from options_strategies.cppm
            self.allowed_strategies = [
                # Tier 1: Basic 2-Leg Strategies (8 strategies)
                "LONG_CALL", "LONG_PUT", "SHORT_CALL", "SHORT_PUT",
                "BULL_CALL_SPREAD", "BEAR_PUT_SPREAD",
                "BULL_PUT_SPREAD", "BEAR_CALL_SPREAD",

                # Tier 1: Combos (4 strategies)
                "LONG_STRADDLE", "SHORT_STRADDLE",
                "LONG_STRANGLE", "SHORT_STRANGLE",

                # Tier 2: Volatility Strategies Part 1 (8 strategies)
                "CALENDAR_SPREAD_CALL", "CALENDAR_SPREAD_PUT",
                "DIAGONAL_SPREAD_CALL", "DIAGONAL_SPREAD_PUT",
                "RATIO_CALL_SPREAD", "RATIO_PUT_SPREAD",
                "CALL_BACKSPREAD", "PUT_BACKSPREAD",

                # Tier 3: Butterflies & Condors (12 strategies)
                "LONG_CALL_BUTTERFLY", "SHORT_CALL_BUTTERFLY",
                "LONG_PUT_BUTTERFLY", "SHORT_PUT_BUTTERFLY",
                "IRON_BUTTERFLY", "REVERSE_IRON_BUTTERFLY",
                "LONG_CALL_CONDOR", "SHORT_CALL_CONDOR",
                "LONG_PUT_CONDOR", "SHORT_PUT_CONDOR",
                "IRON_CONDOR", "REVERSE_IRON_CONDOR",

                # Tier 4: Advanced Multi-Leg (12 strategies)
                "CALL_RATIO_BACKSPREAD", "PUT_RATIO_BACKSPREAD",
                "LONG_CALL_LADDER", "LONG_PUT_LADDER",
                "SHORT_CALL_LADDER", "SHORT_PUT_LADDER",
                "CALL_BUTTERFLY_SPREAD", "PUT_BUTTERFLY_SPREAD",
                "CHRISTMAS_TREE_CALL", "CHRISTMAS_TREE_PUT",
                "JADE_LIZARD", "REVERSE_JADE_LIZARD",

                # Tier 5: Exotic & Income (8 strategies)
                "COLLAR", "PROTECTIVE_COLLAR",
                "COVERED_CALL", "CASH_SECURED_PUT",
                "SYNTHETIC_LONG", "SYNTHETIC_SHORT",
                "CONVERSION", "REVERSAL",
            ]

    def validate(self) -> tuple[bool, str]:
        """
        Validate configuration parameters

        Returns:
            (is_valid, error_message)
        """
        if self.position_limit > self.initial_capital:
            return False, "Position limit cannot exceed initial capital"

        if self.daily_loss_limit > self.initial_capital:
            return False, "Daily loss limit cannot exceed initial capital"

        if self.max_portfolio_heat > 1.0 or self.max_portfolio_heat < 0:
            return False, "Portfolio heat must be between 0 and 1"

        if self.kelly_fraction > 1.0 or self.kelly_fraction < 0:
            return False, "Kelly fraction must be between 0 and 1"

        if self.ml_confidence_threshold > 1.0 or self.ml_confidence_threshold < 0:
            return False, "ML confidence threshold must be between 0 and 1"

        if self.trading_start_time >= self.trading_end_time:
            return False, "Trading start time must be before end time"

        return True, "Configuration valid"

    def get_max_position_size(self, current_portfolio_value: float) -> float:
        """
        Calculate maximum position size based on current portfolio value

        Args:
            current_portfolio_value: Current total portfolio value

        Returns:
            Maximum position size in dollars
        """
        # Always enforce the absolute $2,000 limit
        position_pct_limit = current_portfolio_value * self.max_single_position_pct
        return min(self.position_limit, position_pct_limit)

    def is_trading_hours(self, current_time: time) -> bool:
        """
        Check if current time is within trading hours

        Args:
            current_time: Current time to check

        Returns:
            True if within trading hours
        """
        return self.trading_start_time <= current_time <= self.trading_end_time

    def should_close_positions(self, current_time: time) -> bool:
        """
        Check if we should close positions (near market close)

        Args:
            current_time: Current time to check

        Returns:
            True if positions should be closed
        """
        if not self.close_positions_before_market_close:
            return False

        # Calculate time delta in minutes
        close_hour = self.trading_end_time.hour
        close_minute = self.trading_end_time.minute
        current_hour = current_time.hour
        current_minute = current_time.minute

        minutes_to_close = (close_hour - current_hour) * 60 + (close_minute - current_minute)

        return minutes_to_close <= self.minutes_before_close_to_exit


# Default paper trading configuration
DEFAULT_PAPER_TRADING_CONFIG = PaperTradingConfig(
    account_id="PAPER_TRADING_ACCOUNT",
    initial_capital=30_000.00,
    position_limit=2_000.00,
    daily_loss_limit=900.00,
)


def load_config(config_path: Optional[str] = None) -> PaperTradingConfig:
    """
    Load paper trading configuration from file or use defaults

    Args:
        config_path: Optional path to config file (JSON format)

    Returns:
        PaperTradingConfig instance
    """
    if config_path is None:
        return DEFAULT_PAPER_TRADING_CONFIG

    import json
    from pathlib import Path

    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Config file not found: {config_path}, using defaults")
        return DEFAULT_PAPER_TRADING_CONFIG

    with open(config_file) as f:
        config_data = json.load(f)

    # Convert time strings to time objects if present
    if "trading_start_time" in config_data:
        hour, minute = map(int, config_data["trading_start_time"].split(":"))
        config_data["trading_start_time"] = time(hour, minute)

    if "trading_end_time" in config_data:
        hour, minute = map(int, config_data["trading_end_time"].split(":"))
        config_data["trading_end_time"] = time(hour, minute)

    config = PaperTradingConfig(**config_data)

    # Validate configuration
    is_valid, error_msg = config.validate()
    if not is_valid:
        raise ValueError(f"Invalid configuration: {error_msg}")

    return config


def save_config(config: PaperTradingConfig, config_path: str):
    """
    Save paper trading configuration to JSON file

    Args:
        config: PaperTradingConfig to save
        config_path: Path to save config file
    """
    import json
    from pathlib import Path

    config_dict = {
        "account_id": config.account_id,
        "initial_capital": config.initial_capital,
        "position_limit": config.position_limit,
        "daily_loss_limit": config.daily_loss_limit,
        "max_portfolio_heat": config.max_portfolio_heat,
        "max_sector_allocation": config.max_sector_allocation,
        "min_sector_allocation": config.min_sector_allocation,
        "max_single_position_pct": config.max_single_position_pct,
        "trading_start_time": config.trading_start_time.strftime("%H:%M"),
        "trading_end_time": config.trading_end_time.strftime("%H:%M"),
        "close_positions_before_market_close": config.close_positions_before_market_close,
        "minutes_before_close_to_exit": config.minutes_before_close_to_exit,
        "ml_model_path": config.ml_model_path,
        "ml_confidence_threshold": config.ml_confidence_threshold,
        "ml_min_win_rate": config.ml_min_win_rate,
        "max_strategies_active": config.max_strategies_active,
        "allowed_strategies": config.allowed_strategies,
        "kelly_criterion_enabled": config.kelly_criterion_enabled,
        "kelly_fraction": config.kelly_fraction,
        "fixed_position_size": config.fixed_position_size,
        "enable_real_time_monitoring": config.enable_real_time_monitoring,
        "alert_on_daily_loss_pct": config.alert_on_daily_loss_pct,
        "alert_on_position_loss_pct": config.alert_on_position_loss_pct,
        "log_all_trades": config.log_all_trades,
        "log_directory": config.log_directory,
        "metrics_update_interval_seconds": config.metrics_update_interval_seconds,
        "schwab_api_rate_limit_per_minute": config.schwab_api_rate_limit_per_minute,
        "news_api_daily_limit": config.news_api_daily_limit,
    }

    Path(config_path).parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"Configuration saved to: {config_path}")


if __name__ == "__main__":
    # Test configuration
    config = DEFAULT_PAPER_TRADING_CONFIG

    is_valid, msg = config.validate()
    print(f"Configuration valid: {is_valid}")
    if not is_valid:
        print(f"Error: {msg}")
    else:
        print("✅ All validation checks passed")
        print(f"\nConfiguration:")
        print(f"  Initial Capital: ${config.initial_capital:,.2f}")
        print(f"  Position Limit: ${config.position_limit:,.2f}")
        print(f"  Daily Loss Limit: ${config.daily_loss_limit:,.2f}")
        print(f"  Max Portfolio Heat: {config.max_portfolio_heat:.1%}")
        print(f"  ML Confidence Threshold: {config.ml_confidence_threshold:.1%}")
        print(f"  Trading Hours: {config.trading_start_time} - {config.trading_end_time}")
        print(f"  Allowed Strategies: {len(config.allowed_strategies)} strategies")

        # Test trading hours
        from datetime import datetime
        test_times = [
            time(9, 0),   # Before market open
            time(10, 0),  # During trading
            time(15, 50), # Near close (should trigger exit)
            time(16, 30), # After close
        ]

        print("\nTrading Hours Check:")
        for t in test_times:
            in_hours = config.is_trading_hours(t)
            should_close = config.should_close_positions(t)
            print(f"  {t}: Trading={'✅' if in_hours else '❌'}, Close={'⚠️' if should_close else '  '}")
