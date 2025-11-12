#!/usr/bin/env python3
"""
Test Price Predictor with Live FRED Rates

Demonstrates integration of:
- FRED risk-free rates
- Feature extraction (technical + sentiment + economic)
- Price prediction (1-day, 5-day, 20-day forecasts)
- Trading signals generation

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-11
"""

import sys
from pathlib import Path

# Add build directory for C++ bindings
sys.path.insert(0, str(Path(__file__).parent.parent / 'build'))

try:
    from fred_rates_py import FREDRatesFetcher, FREDConfig, RateSeries
    FRED_AVAILABLE = True
except ImportError:
    print("⚠️  FRED bindings not available")
    FRED_AVAILABLE = False

import yaml
import numpy as np
from datetime import datetime


def load_fred_api_key():
    """Load FRED API key from api_keys.yaml"""
    api_keys_path = Path(__file__).parent.parent / 'api_keys.yaml'

    if not api_keys_path.exists():
        return None

    with open(api_keys_path) as f:
        keys = yaml.safe_load(f)

    return keys.get('fred_api_key')


def fetch_current_rates():
    """Fetch current risk-free rates from FRED"""
    if not FRED_AVAILABLE:
        return None

    api_key = load_fred_api_key()
    if not api_key:
        return None

    try:
        config = FREDConfig()
        config.api_key = api_key

        fetcher = FREDRatesFetcher(config)
        rates = fetcher.fetch_all_rates()

        return {
            'treasury_3m': rates[RateSeries.ThreeMonthTreasury].rate_value if RateSeries.ThreeMonthTreasury in rates else 0.04,
            'treasury_2y': rates[RateSeries.TwoYearTreasury].rate_value if RateSeries.TwoYearTreasury in rates else 0.04,
            'treasury_10y': rates[RateSeries.TenYearTreasury].rate_value if RateSeries.TenYearTreasury in rates else 0.04,
            'fed_funds': rates[RateSeries.FedFundsRate].rate_value if RateSeries.FedFundsRate in rates else 0.04,
        }

    except Exception as e:
        print(f"Failed to fetch FRED rates: {e}")
        return None


def generate_sample_features(symbol: str, rates: dict = None):
    """
    Generate sample feature vector for testing

    In production, these would come from:
    - Real-time market data (prices, volumes)
    - Sentiment analysis (news, social media)
    - Economic data (employment, GDP, inflation)
    - Sector correlations
    """

    # Use real FRED rates if available, otherwise defaults
    if rates is None:
        rates = {
            'treasury_3m': 0.0392,
            'treasury_2y': 0.0355,
            'treasury_10y': 0.0411,
            'fed_funds': 0.0387,
        }

    # Sample feature vector (25 features)
    features = {
        # Technical indicators (10)
        'rsi_14': 55.3,                 # RSI slightly above neutral
        'macd': 0.42,                   # Positive momentum
        'macd_signal': 0.35,            # Below MACD (bullish)
        'macd_histogram': 0.07,         # Positive histogram
        'bb_upper': 152.50,             # Bollinger upper
        'bb_middle': 150.00,            # Bollinger middle
        'bb_lower': 147.50,             # Bollinger lower
        'atr_14': 2.15,                 # Average volatility
        'volume_ratio': 1.23,           # Above average volume
        'momentum_5d': 0.018,           # 1.8% gain over 5 days

        # Sentiment (5)
        'news_sentiment': 0.35,         # Positive news sentiment
        'social_sentiment': 0.15,       # Slightly positive social
        'analyst_rating': 3.8,          # Buy rating (1-5 scale)
        'put_call_ratio': 0.85,         # Slightly bullish
        'vix_level': 16.5,              # Low fear

        # Economic indicators (5)
        'employment_change': 185000,    # Strong job growth
        'gdp_growth': 0.025,            # 2.5% quarterly
        'inflation_rate': 0.031,        # 3.1% CPI
        'fed_rate': rates['fed_funds'], # Live from FRED
        'treasury_yield_10y': rates['treasury_10y'],  # Live from FRED

        # Sector correlation (5)
        'sector_momentum': 0.042,       # 4.2% sector gain
        'spy_correlation': 0.78,        # High correlation with market
        'sector_beta': 1.15,            # Slightly more volatile than market
        'peer_avg_return': 0.025,       # Peers up 2.5%
        'market_regime': 0.65,          # Bullish regime
    }

    return features


def predict_price_movements(symbol: str, features: dict):
    """
    Predict price movements using ML model (placeholder)

    In production, this would call the C++ CUDA-accelerated neural network
    """

    # Placeholder prediction logic
    # Real implementation would use the C++ PricePredictor module

    # Calculate feature-based score
    technical_score = (
        (features['rsi_14'] - 50) / 50 * 0.3 +
        features['macd_histogram'] * 10 * 0.3 +
        (features['volume_ratio'] - 1.0) * 0.2
    )

    sentiment_score = (
        features['news_sentiment'] * 0.4 +
        features['social_sentiment'] * 0.3 +
        (features['analyst_rating'] - 3.0) / 2.0 * 0.3
    )

    economic_score = (
        features['gdp_growth'] * 20 * 0.4 +
        (1.0 - features['inflation_rate']) * 0.3 +
        features['sector_momentum'] * 5 * 0.3
    )

    overall_score = (technical_score * 0.4 + sentiment_score * 0.3 + economic_score * 0.3)

    # Generate predictions
    predictions = {
        'symbol': symbol,
        'day_1_change': overall_score * 1.5,      # 1-day forecast
        'day_5_change': overall_score * 4.0,      # 5-day forecast
        'day_20_change': overall_score * 10.0,    # 20-day forecast
        'confidence_1d': min(0.95, 0.6 + abs(overall_score) * 0.2),
        'confidence_5d': min(0.90, 0.5 + abs(overall_score) * 0.2),
        'confidence_20d': min(0.85, 0.4 + abs(overall_score) * 0.2),
        'timestamp': datetime.now().isoformat(),
    }

    # Generate signals
    def get_signal(change: float) -> str:
        if change > 5.0: return "STRONG_BUY"
        if change > 2.0: return "BUY"
        if change < -5.0: return "STRONG_SELL"
        if change < -2.0: return "SELL"
        return "HOLD"

    predictions['signal_1d'] = get_signal(predictions['day_1_change'])
    predictions['signal_5d'] = get_signal(predictions['day_5_change'])
    predictions['signal_20d'] = get_signal(predictions['day_20_change'])

    return predictions


def print_prediction(pred: dict):
    """Print formatted prediction"""
    print(f"\n{'═' * 70}")
    print(f"Price Prediction for {pred['symbol']}")
    print(f"{'═' * 70}")
    print(f"Timestamp: {pred['timestamp']}")
    print()

    print("Price Change Forecasts:")
    print(f"  1-Day:   {pred['day_1_change']:+6.2f}%  (Confidence: {pred['confidence_1d']:.1%})  [{pred['signal_1d']}]")
    print(f"  5-Day:   {pred['day_5_change']:+6.2f}%  (Confidence: {pred['confidence_5d']:.1%})  [{pred['signal_5d']}]")
    print(f"  20-Day:  {pred['day_20_change']:+6.2f}%  (Confidence: {pred['confidence_20d']:.1%})  [{pred['signal_20d']}]")
    print()

    # Overall signal
    weighted_change = (
        pred['day_1_change'] * pred['confidence_1d'] * 0.5 +
        pred['day_5_change'] * pred['confidence_5d'] * 0.3 +
        pred['day_20_change'] * pred['confidence_20d'] * 0.2
    )

    if weighted_change > 5.0:
        overall = "STRONG_BUY"
        color = "🟢"
    elif weighted_change > 2.0:
        overall = "BUY"
        color = "🟢"
    elif weighted_change < -5.0:
        overall = "STRONG_SELL"
        color = "🔴"
    elif weighted_change < -2.0:
        overall = "SELL"
        color = "🔴"
    else:
        overall = "HOLD"
        color = "🟡"

    print(f"Overall Signal: {color} {overall} (Weighted Change: {weighted_change:+.2f}%)")
    print(f"{'═' * 70}\n")


def main():
    print(f"{'═' * 70}")
    print("BigBrotherAnalytics - Price Predictor Test")
    print(f"{'═' * 70}\n")

    # Fetch live FRED rates
    print("📊 Fetching live risk-free rates from FRED...")
    rates = fetch_current_rates()

    if rates:
        print("✅ Live FRED Rates:")
        print(f"   3-Month Treasury: {rates['treasury_3m'] * 100:.3f}%")
        print(f"   2-Year Treasury:  {rates['treasury_2y'] * 100:.3f}%")
        print(f"   10-Year Treasury: {rates['treasury_10y'] * 100:.3f}%")
        print(f"   Federal Funds:    {rates['fed_funds'] * 100:.3f}%")
    else:
        print("⚠️  Using default rates (FRED API unavailable)")
        rates = None

    # Test symbols
    test_symbols = ['AAPL', 'NVDA', 'TSLA']

    for symbol in test_symbols:
        # Generate features
        features = generate_sample_features(symbol, rates)

        # Predict
        prediction = predict_price_movements(symbol, features)

        # Display
        print_prediction(prediction)

    print("\n💡 Note: This is using placeholder prediction logic.")
    print("   For production, this would call the C++ CUDA-accelerated neural network.")
    print("   Install CUDA toolkit to enable GPU acceleration (2-10x speedup).\n")


if __name__ == "__main__":
    main()
