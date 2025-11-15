#!/usr/bin/env python3
"""
Test that the trained model produces different predictions for different stocks
"""

import torch
import numpy as np
import pickle
from pathlib import Path

# Load model architecture
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_price_predictor_clean import PricePredictorClean

def test_prediction_differentiation():
    """Test that model produces different predictions for different stocks"""

    print("=" * 80)
    print("TESTING PREDICTION DIFFERENTIATION")
    print("=" * 80)

    # Load model
    model_path = Path('models/price_predictor_85feat_best.pth')
    scaler_path = Path('models/scaler_85feat.pkl')

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False

    if not scaler_path.exists():
        print(f"❌ Scaler not found: {scaler_path}")
        return False

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PricePredictorClean(input_size=85).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print(f"✓ Loaded model and scaler")
    print(f"✓ Device: {device}")
    print()

    # Create sample features for different stocks
    # Features: 58 base + 3 temporal + 20 price_diff + 4 autocorr = 85
    # Index 47 is symbol_encoded (SPY=0, QQQ=1, IWM=2)

    # SPY features (typical large-cap ETF)
    spy_features = np.random.randn(85).astype(np.float32)
    spy_features[0] = 670.0   # close
    spy_features[1] = 669.5   # open
    spy_features[5] = 52.8    # rsi_14
    spy_features[47] = 0.0    # symbol_encoded (SPY)

    # QQQ features (tech-heavy, higher volatility)
    qqq_features = np.random.randn(85).astype(np.float32)
    qqq_features[0] = 605.0   # close
    qqq_features[1] = 604.2   # open
    qqq_features[5] = 50.2    # rsi_14
    qqq_features[47] = 1.0    # symbol_encoded (QQQ)

    # IWM features (small-cap, different characteristics)
    iwm_features = np.random.randn(85).astype(np.float32)
    iwm_features[0] = 236.0   # close
    iwm_features[1] = 235.8   # open
    iwm_features[5] = 39.6    # rsi_14
    iwm_features[47] = 2.0    # symbol_encoded (IWM)

    # Stack features
    features = np.vstack([spy_features, qqq_features, iwm_features])

    # Normalize
    features_norm = scaler.transform(features)

    # Get predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(features_norm).to(device)
        predictions = model(X_tensor).cpu().numpy()

    # Display results
    symbols = ['SPY', 'QQQ', 'IWM']
    print("Predictions for different stocks:")
    print("-" * 80)
    print(f"{'Symbol':<10} {'1-day %':<12} {'5-day %':<12} {'20-day %':<12}")
    print("-" * 80)

    for i, symbol in enumerate(symbols):
        pred_1d = predictions[i, 0] * 100
        pred_5d = predictions[i, 1] * 100
        pred_20d = predictions[i, 2] * 100
        print(f"{symbol:<10} {pred_1d:>10.2f}% {pred_5d:>10.2f}% {pred_20d:>10.2f}%")

    print("-" * 80)

    # Check if predictions are different
    all_same = np.allclose(predictions[0], predictions[1], atol=1e-6) and \
               np.allclose(predictions[1], predictions[2], atol=1e-6)

    if all_same:
        print("❌ FAIL: All predictions are identical!")
        print("   Model is NOT differentiating between stocks")
        return False
    else:
        # Calculate differences
        diff_spy_qqq = np.abs(predictions[0] - predictions[1]).max()
        diff_qqq_iwm = np.abs(predictions[1] - predictions[2]).max()
        diff_spy_iwm = np.abs(predictions[0] - predictions[2]).max()

        max_diff = max(diff_spy_qqq, diff_qqq_iwm, diff_spy_iwm)

        print(f"✓ PASS: Predictions are different!")
        print(f"  Max difference: {max_diff*100:.4f}%")
        print(f"  SPY vs QQQ: {diff_spy_qqq*100:.4f}%")
        print(f"  QQQ vs IWM: {diff_qqq_iwm*100:.4f}%")
        print(f"  SPY vs IWM: {diff_spy_iwm*100:.4f}%")
        print()
        print("✓ Model is properly differentiating between stocks!")
        return True

if __name__ == '__main__':
    success = test_prediction_differentiation()
    sys.exit(0 if success else 1)
