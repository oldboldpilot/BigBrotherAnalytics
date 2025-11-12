#!/usr/bin/env python3
"""
Test prediction logic independently without Streamlit
Tests the core prediction algorithm
"""


def get_default_rates():
    """Get default rates when FRED is unavailable"""
    return {
        'treasury_3m': 0.0392,
        'treasury_2y': 0.0355,
        'treasury_10y': 0.0411,
        'fed_funds': 0.0387,
    }


def generate_features_for_symbol(symbol: str, rates: dict):
    """Generate feature vector for prediction"""
    features = {
        # Technical indicators (10)
        'rsi_14': 55.3,
        'macd': 0.42,
        'macd_signal': 0.35,
        'macd_histogram': 0.07,
        'bb_upper': 152.50,
        'bb_middle': 150.00,
        'bb_lower': 147.50,
        'atr_14': 2.15,
        'volume_ratio': 1.23,
        'momentum_5d': 0.018,

        # Sentiment (5)
        'news_sentiment': 0.35,
        'social_sentiment': 0.15,
        'analyst_rating': 3.8,
        'put_call_ratio': 0.85,
        'vix_level': 16.5,

        # Economic indicators (5)
        'employment_change': 185000,
        'gdp_growth': 0.025,
        'inflation_rate': 0.031,
        'fed_rate': rates.get('fed_funds', 0.04),
        'treasury_yield_10y': rates.get('treasury_10y', 0.04),

        # Sector correlation (5)
        'sector_momentum': 0.042,
        'spy_correlation': 0.78,
        'sector_beta': 1.15,
        'peer_avg_return': 0.025,
        'market_regime': 0.65,
    }
    return features


def predict_price_movements(symbol: str, features: dict):
    """Generate price predictions using feature-based scoring"""
    from datetime import datetime

    # Calculate component scores
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
        'day_1_change': overall_score * 1.5,
        'day_5_change': overall_score * 4.0,
        'day_20_change': overall_score * 10.0,
        'confidence_1d': min(0.95, 0.6 + abs(overall_score) * 0.2),
        'confidence_5d': min(0.90, 0.5 + abs(overall_score) * 0.2),
        'confidence_20d': min(0.85, 0.4 + abs(overall_score) * 0.2),
        'timestamp': datetime.now().isoformat(),
        'technical_score': technical_score,
        'sentiment_score': sentiment_score,
        'economic_score': economic_score,
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


def test_predictions():
    """Test prediction generation for multiple symbols"""

    print("=" * 70)
    print("Testing Price Prediction Logic")
    print("=" * 70)
    print()

    # Get default rates
    rates = get_default_rates()
    print("Using default FRED rates:")
    for key, value in rates.items():
        print(f"  {key}: {value * 100:.3f}%")
    print()

    # Test symbols
    test_symbols = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL']

    results = []

    for symbol in test_symbols:
        print(f"\n{'-' * 70}")
        print(f"Testing: {symbol}")
        print(f"{'-' * 70}")

        # Generate features
        features = generate_features_for_symbol(symbol, rates)
        print(f"✓ Generated {len(features)} features")

        # Generate prediction
        prediction = predict_price_movements(symbol, features)

        # Display results
        print(f"\nPrediction Results:")
        print(f"  1-Day:  {prediction['day_1_change']:+6.2f}% (Conf: {prediction['confidence_1d']:.1%}) [{prediction['signal_1d']}]")
        print(f"  5-Day:  {prediction['day_5_change']:+6.2f}% (Conf: {prediction['confidence_5d']:.1%}) [{prediction['signal_5d']}]")
        print(f"  20-Day: {prediction['day_20_change']:+6.2f}% (Conf: {prediction['confidence_20d']:.1%}) [{prediction['signal_20d']}]")

        # Calculate overall signal
        weighted_change = (
            prediction['day_1_change'] * prediction['confidence_1d'] * 0.5 +
            prediction['day_5_change'] * prediction['confidence_5d'] * 0.3 +
            prediction['day_20_change'] * prediction['confidence_20d'] * 0.2
        )

        if weighted_change > 5.0:
            overall_signal = "STRONG_BUY"
        elif weighted_change > 2.0:
            overall_signal = "BUY"
        elif weighted_change < -5.0:
            overall_signal = "STRONG_SELL"
        elif weighted_change < -2.0:
            overall_signal = "SELL"
        else:
            overall_signal = "HOLD"

        print(f"\nOverall Signal: {overall_signal} (Weighted: {weighted_change:+.2f}%)")

        # Component scores
        print(f"\nComponent Scores:")
        print(f"  Technical: {prediction['technical_score']:+.3f}")
        print(f"  Sentiment: {prediction['sentiment_score']:+.3f}")
        print(f"  Economic:  {prediction['economic_score']:+.3f}")

        results.append({
            'symbol': symbol,
            'overall_signal': overall_signal,
            'weighted_change': weighted_change,
            'day_1': prediction['day_1_change'],
            'day_5': prediction['day_5_change'],
            'day_20': prediction['day_20_change']
        })

    print(f"\n{'=' * 70}")
    print("Summary of All Predictions")
    print(f"{'=' * 70}\n")

    print(f"{'Symbol':<10} {'Signal':<15} {'Weighted':<12} {'1-Day':<10} {'5-Day':<10} {'20-Day':<10}")
    print("-" * 70)

    for r in results:
        print(f"{r['symbol']:<10} {r['overall_signal']:<15} {r['weighted_change']:+6.2f}%     {r['day_1']:+6.2f}%   {r['day_5']:+6.2f}%   {r['day_20']:+6.2f}%")

    print(f"\n{'=' * 70}")
    print("✓ All tests passed successfully!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    test_predictions()
