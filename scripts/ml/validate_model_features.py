#!/usr/bin/env python3
"""
Validate Custom Model Features

This script shows exactly what features the C++ engine must provide
for inference with the custom 42-feature model.
"""

import torch
import json
import numpy as np
from pathlib import Path

print("="*80)
print("CUSTOM MODEL FEATURE VALIDATION")
print("="*80)

# Load model checkpoint
checkpoint_path = 'models/price_predictor_best.pth'
print(f"\nLoading model from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Get feature list
feature_cols = checkpoint['feature_cols']
print(f"\nTotal Features Required: {len(feature_cols)}")

# Load feature metadata
with open('models/custom_features_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"\n{'='*80}")
print("FEATURE SPECIFICATION FOR C++ ENGINE")
print(f"{'='*80}\n")

# Categorize features
categories = {
    'Identification': ['symbol_encoded', 'sector_encoded', 'is_option'],
    'Time': ['hour_of_day', 'minute_of_hour', 'day_of_week', 'day_of_month',
             'month_of_year', 'quarter', 'day_of_year', 'is_market_open'],
    'Treasury Rates': ['fed_funds_rate', 'treasury_3mo', 'treasury_2yr',
                       'treasury_5yr', 'treasury_10yr', 'yield_curve_slope',
                       'yield_curve_inversion'],
    'Options Greeks': ['delta', 'gamma', 'theta', 'vega', 'rho', 'implied_volatility'],
    'Sentiment': ['avg_sentiment', 'news_count'],
    'Price': ['close', 'open', 'high', 'low', 'volume'],
    'Momentum': ['return_1d', 'return_5d', 'return_20d', 'rsi_14', 'macd',
                 'macd_signal', 'volume_ratio'],
    'Volatility': ['atr_14', 'bb_upper', 'bb_lower', 'bb_position']
}

# Print each category
for i, (category, features) in enumerate(categories.items(), 1):
    matching = [f for f in feature_cols if f in features]
    print(f"{i}. {category} ({len(matching)} features)")
    for j, feat in enumerate(matching, 1):
        idx = feature_cols.index(feat)
        print(f"   [{idx:2d}] {feat}")
    print()

# Generate C++ header snippet
print("="*80)
print("C++ FEATURE EXTRACTION EXAMPLE")
print("="*80)
print("""
// Feature vector must be exactly 42 elements in this order:
struct ModelFeatures {
    // Identification (3)
    int symbol_encoded;          // 0-19 for 20 symbols
    int sector_encoded;          // -1 to 55 for sectors
    int is_option;               // 0=stock, 1=option

    // Time (8)
    int hour_of_day;             // 0-23
    int minute_of_hour;          // 0-59
    int day_of_week;             // 0=Mon, 6=Sun
    int day_of_month;            // 1-31
    int month_of_year;           // 1-12
    int quarter;                 // 1-4
    int day_of_year;             // 1-365
    int is_market_open;          // 0/1

    // Treasury Rates (7)
    double fed_funds_rate;       // e.g., 0.0387
    double treasury_3mo;         // e.g., 0.0392
    double treasury_2yr;         // e.g., 0.0355
    double treasury_5yr;         // e.g., 0.0367
    double treasury_10yr;        // e.g., 0.0411
    double yield_curve_slope;    // 10yr - 2yr
    int yield_curve_inversion;   // 1 if 2yr > 10yr

    // Options Greeks (6)
    double delta;                // Call delta (0-1)
    double gamma;                // Rate of delta change
    double theta;                // Time decay per day
    double vega;                 // IV sensitivity
    double rho;                  // Rate sensitivity
    double implied_volatility;   // IV percentage

    // Sentiment (2)
    double avg_sentiment;        // -1 to +1
    int news_count;              // Number of articles

    // Price (5)
    double close;                // Close price
    double open;                 // Open price
    double high;                 // High price
    double low;                  // Low price
    long volume;                 // Volume

    // Momentum (7)
    double return_1d;            // 1-day return
    double return_5d;            // 5-day return
    double return_20d;           // 20-day return
    double rsi_14;               // RSI (0-100)
    double macd;                 // MACD line
    double macd_signal;          // MACD signal
    double volume_ratio;         // Volume / 20-day avg

    // Volatility (4)
    double atr_14;               // ATR
    double bb_upper;             // Bollinger upper
    double bb_lower;             // Bollinger lower
    double bb_position;          // Position in bands (0-1)
};

// Convert to float array for ONNX
std::vector<float> toOnnxInput() const {
    return {
        static_cast<float>(symbol_encoded),
        static_cast<float>(sector_encoded),
        static_cast<float>(is_option),
        static_cast<float>(hour_of_day),
        static_cast<float>(minute_of_hour),
        static_cast<float>(day_of_week),
        static_cast<float>(day_of_month),
        static_cast<float>(month_of_year),
        static_cast<float>(quarter),
        static_cast<float>(day_of_year),
        static_cast<float>(is_market_open),
        static_cast<float>(fed_funds_rate),
        static_cast<float>(treasury_3mo),
        static_cast<float>(treasury_2yr),
        static_cast<float>(treasury_5yr),
        static_cast<float>(treasury_10yr),
        static_cast<float>(yield_curve_slope),
        static_cast<float>(yield_curve_inversion),
        static_cast<float>(delta),
        static_cast<float>(gamma),
        static_cast<float>(theta),
        static_cast<float>(vega),
        static_cast<float>(rho),
        static_cast<float>(implied_volatility),
        static_cast<float>(avg_sentiment),
        static_cast<float>(news_count),
        static_cast<float>(close),
        static_cast<float>(open),
        static_cast<float>(high),
        static_cast<float>(low),
        static_cast<float>(volume),
        static_cast<float>(return_1d),
        static_cast<float>(return_5d),
        static_cast<float>(return_20d),
        static_cast<float>(rsi_14),
        static_cast<float>(macd),
        static_cast<float>(macd_signal),
        static_cast<float>(volume_ratio),
        static_cast<float>(atr_14),
        static_cast<float>(bb_upper),
        static_cast<float>(bb_lower),
        static_cast<float>(bb_position)
    };
}
""")

# Create example feature vector
print("\n" + "="*80)
print("EXAMPLE FEATURE VECTOR (SPY)")
print("="*80)

example_features = {
    'symbol_encoded': 18,  # SPY
    'sector_encoded': -1,  # ETF (no sector)
    'is_option': 0,
    'hour_of_day': 16,  # 4 PM close
    'minute_of_hour': 0,
    'day_of_week': 1,  # Tuesday
    'day_of_month': 12,
    'month_of_year': 11,  # November
    'quarter': 4,
    'day_of_year': 317,
    'is_market_open': 1,
    'fed_funds_rate': 0.0387,
    'treasury_3mo': 0.0392,
    'treasury_2yr': 0.0355,
    'treasury_5yr': 0.0367,
    'treasury_10yr': 0.0411,
    'yield_curve_slope': 0.0056,
    'yield_curve_inversion': 0,
    'delta': 0.523,
    'gamma': 0.012,
    'theta': -0.05,
    'vega': 0.25,
    'rho': 0.08,
    'implied_volatility': 0.18,
    'avg_sentiment': 0.15,
    'news_count': 5,
    'close': 585.50,
    'open': 583.20,
    'high': 586.80,
    'low': 582.90,
    'volume': 45000000,
    'return_1d': 0.0042,
    'return_5d': 0.0123,
    'return_20d': 0.0456,
    'rsi_14': 58.3,
    'macd': 2.1,
    'macd_signal': 1.8,
    'volume_ratio': 1.05,
    'atr_14': 8.5,
    'bb_upper': 595.0,
    'bb_lower': 575.0,
    'bb_position': 0.525
}

print("\nFeature values:")
for i, feat in enumerate(feature_cols):
    value = example_features.get(feat, 0.0)
    print(f"[{i:2d}] {feat:25s} = {value}")

# Test with actual model
print("\n" + "="*80)
print("MODEL INFERENCE TEST")
print("="*80)

# Create feature vector
feature_vector = np.array([example_features.get(f, 0.0) for f in feature_cols])

# Normalize using scaler from checkpoint
scaler = checkpoint['scaler']
feature_vector_normalized = scaler.transform(feature_vector.reshape(1, -1))

# Load model
from scripts.ml.convert_to_onnx import CustomPricePredictor
model = CustomPricePredictor(input_size=len(feature_cols))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference
with torch.no_grad():
    input_tensor = torch.FloatTensor(feature_vector_normalized)
    output = model(input_tensor).numpy()[0]

print(f"\nInput: {len(feature_cols)} features")
print(f"Output predictions:")
print(f"  1-day:  {output[0]:+.4f} ({output[0]*100:+.2f}%)")
print(f"  5-day:  {output[1]:+.4f} ({output[1]*100:+.2f}%)")
print(f"  20-day: {output[2]:+.4f} ({output[2]*100:+.2f}%)")

if output[0] > 0:
    print(f"\n✅ Prediction: BUY (price expected to rise)")
else:
    print(f"\n⚠️  Prediction: SELL (price expected to fall)")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
print("\nNext steps:")
print("1. Update C++ feature extraction to provide all 42 features")
print("2. Add StandardScaler normalization in C++ (use same mean/std from Python)")
print("3. Test C++ ONNX inference with sample input")
print("4. Integrate with live trading engine")
print("="*80)
