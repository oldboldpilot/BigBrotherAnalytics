#!/usr/bin/env python3
"""
Generate Python Ground Truth for C++ Pipeline Validation

This script:
1. Loads data from DuckDB
2. Extracts features using Python
3. Normalizes using MinMaxNormalizer
4. Runs PyTorch model inference
5. Saves predictions as ground truth for C++ comparison

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-14
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import duckdb

# Add scripts/ml to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts' / 'ml'))
from minmax_normalizer import MinMaxNormalizer

# Import the trained model architecture
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts' / 'ml'))
from train_price_predictor_clean import PricePredictorClean

def extract_features_python(bars):
    """
    Extract features using Python implementation
    (Should match C++ FeatureExtractor logic)
    """
    # TODO: Implement Python feature extraction
    # For now, return placeholder
    return np.zeros(85, dtype=np.float32)

def main():
    print("=" * 70)
    print("Generating Python Ground Truth for C++ Validation")
    print("=" * 70)

    # Load model
    print("\nLoading trained model...")
    model = PricePredictorClean(input_size=85)
    model.load_state_dict(torch.load('models/price_predictor_85feat_best.pth'))
    model.eval()
    print("✓ Model loaded")

    # Load normalizer
    print("\nLoading normalizer...")
    normalizer = MinMaxNormalizer.load(Path('models/normalizer_85feat.json'))
    print("✓ Normalizer loaded")

    # Connect to DuckDB
    print("\nConnecting to DuckDB...")
    conn = duckdb.connect('data/market_data.duckdb', read_only=True)

    # Query data for multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    ground_truth = {
        'symbols': symbols,
        'predictions': []
    }

    print(f"\nGenerating predictions for {len(symbols)} symbols...")

    for symbol in symbols:
        query = f"""
        SELECT * FROM daily_bars
        WHERE symbol = '{symbol}'
        ORDER BY date DESC
        LIMIT 100
        """

        result = conn.execute(query).fetchdf()

        if len(result) == 0:
            print(f"  ⚠️  {symbol}: No data")
            continue

        # Extract features (placeholder - would use actual feature extraction)
        # In production, this should match the C++ feature extractor exactly
        features = extract_features_python(result)

        # Normalize
        features_normalized = normalizer.transform(features.reshape(1, -1))

        # Predict
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_normalized)
            predictions = model(features_tensor).numpy()[0]

        ground_truth['predictions'].append({
            'symbol': symbol,
            'date': result['date'].iloc[0],
            'predictions': predictions.tolist(),
            'raw_features': features.tolist(),
            'normalized_features': features_normalized[0].tolist()
        })

        print(f"  ✓ {symbol}: [{predictions[0]:.6f}, {predictions[1]:.6f}, {predictions[2]:.6f}]")

    conn.close()

    # Save ground truth
    output_path = Path('tests/python_ground_truth.json')
    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✓ Saved ground truth to {output_path}")
    print(f"  Samples: {len(ground_truth['predictions'])}")

    print("\n" + "=" * 70)
    print("Ground truth generation complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Build C++ test: cmake --build build --target test_full_cpp_pipeline")
    print("  2. Run validation: ./build/bin/test_full_cpp_pipeline")

if __name__ == '__main__':
    main()
