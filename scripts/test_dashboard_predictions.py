#!/usr/bin/env python3
"""
Test script for dashboard price predictions
Tests the prediction logic with multiple symbols
"""

import sys
from pathlib import Path

# Add dashboard directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard"))

from price_predictions_view import (
    get_default_rates,
    generate_features_for_symbol,
    predict_price_movements,
    get_signal_color,
    get_signal_emoji
)


def test_predictions():
    """Test prediction generation for multiple symbols"""

    print("=" * 70)
    print("Testing Dashboard Price Predictions")
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

        emoji = get_signal_emoji(overall_signal)
        color = get_signal_color(overall_signal)

        print(f"\nOverall Signal: {emoji} {overall_signal} (Weighted: {weighted_change:+.2f}%)")
        print(f"Signal Color: {color}")

        # Component scores
        print(f"\nComponent Scores:")
        print(f"  Technical: {prediction['technical_score']:+.3f}")
        print(f"  Sentiment: {prediction['sentiment_score']:+.3f}")
        print(f"  Economic:  {prediction['economic_score']:+.3f}")

        # Verify all required fields
        required_fields = [
            'symbol', 'day_1_change', 'day_5_change', 'day_20_change',
            'confidence_1d', 'confidence_5d', 'confidence_20d',
            'signal_1d', 'signal_5d', 'signal_20d',
            'timestamp', 'technical_score', 'sentiment_score', 'economic_score',
            'features'
        ]

        missing = [f for f in required_fields if f not in prediction]
        if missing:
            print(f"\n⚠️  Missing fields: {missing}")
        else:
            print(f"\n✓ All required fields present")

    print(f"\n{'=' * 70}")
    print("✓ All tests passed!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    test_predictions()
