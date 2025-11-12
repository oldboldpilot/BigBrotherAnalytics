#!/usr/bin/env python3
"""
ML Prediction Service - Fast price prediction for trading engine

Usage from C++:
    echo '{"symbol": "SPY", "features": [close, open, high, ...]}' | python ml_prediction_service.py

Returns JSON:
    {
        "symbol": "SPY",
        "predictions": {
            "day_1_change": 0.0123,
            "day_5_change": 0.0456,
            "day_20_change": 0.0789
        },
        "signal": "BUY",  # or "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL"
        "confidence": 0.85,
        "timestamp": "2025-11-12T03:00:00"
    }
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

# Load model once at startup
MODEL_PATH = Path('models/price_predictor_best.pth')
SCALER = None
FEATURE_COLS = None
MODEL = None

def load_model():
    """Load model and scaler (called once at startup)"""
    global MODEL, SCALER, FEATURE_COLS

    if not MODEL_PATH.exists():
        return False

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location='cpu')

    # Get feature columns and scaler
    FEATURE_COLS = checkpoint['feature_cols']
    SCALER = checkpoint['scaler']

    # Recreate model architecture
    import torch.nn as nn

    class PricePredictor(nn.Module):
        def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=3, dropout=0.3):
            super(PricePredictor, self).__init__()

            layers = []
            prev_size = input_size

            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, output_size))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    # Create model and load weights
    MODEL = PricePredictor(input_size=len(FEATURE_COLS))
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    MODEL.eval()

    return True

def predict(features):
    """
    Run prediction on feature vector

    Args:
        features: dict with feature names as keys

    Returns:
        dict with predictions and signal
    """
    # Extract features in correct order
    feature_vector = np.array([features.get(col, 0.0) for col in FEATURE_COLS])

    # Normalize
    feature_vector = SCALER.transform(feature_vector.reshape(1, -1))

    # Convert to tensor
    input_tensor = torch.FloatTensor(feature_vector)

    # Run inference
    with torch.no_grad():
        predictions = MODEL(input_tensor).numpy()[0]

    # Predictions are: [1d_change, 5d_change, 20d_change]
    day_1_change = float(predictions[0])
    day_5_change = float(predictions[1])
    day_20_change = float(predictions[2])

    # Weighted signal (emphasize near-term predictions)
    weighted_change = day_1_change * 0.5 + day_5_change * 0.3 + day_20_change * 0.2

    # Determine signal
    if weighted_change > 0.05:
        signal = "STRONG_BUY"
        confidence = min(0.95, 0.6 + abs(weighted_change) * 5)
    elif weighted_change > 0.02:
        signal = "BUY"
        confidence = min(0.85, 0.6 + abs(weighted_change) * 5)
    elif weighted_change < -0.05:
        signal = "STRONG_SELL"
        confidence = min(0.95, 0.6 + abs(weighted_change) * 5)
    elif weighted_change < -0.02:
        signal = "SELL"
        confidence = min(0.85, 0.6 + abs(weighted_change) * 5)
    else:
        signal = "HOLD"
        confidence = 0.5

    return {
        "predictions": {
            "day_1_change": day_1_change,
            "day_5_change": day_5_change,
            "day_20_change": day_20_change,
            "weighted_change": weighted_change
        },
        "signal": signal,
        "confidence": confidence
    }

def main():
    """Main entry point for service"""
    # Load model once
    if not load_model():
        error = {"error": "Model not found", "model_path": str(MODEL_PATH)}
        print(json.dumps(error), file=sys.stderr)
        sys.exit(1)

    # Read JSON input from stdin
    try:
        input_data = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        error = {"error": f"Invalid JSON: {e}"}
        print(json.dumps(error), file=sys.stderr)
        sys.exit(1)

    symbol = input_data.get('symbol', 'UNKNOWN')
    features = input_data.get('features', {})

    # Run prediction
    try:
        result = predict(features)

        # Add metadata
        result['symbol'] = symbol
        result['timestamp'] = datetime.now().isoformat()

        # Output JSON
        print(json.dumps(result))
        sys.exit(0)

    except Exception as e:
        error = {"error": f"Prediction failed: {e}", "symbol": symbol}
        print(json.dumps(error), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
