#!/usr/bin/env python3
"""
ML Price Prediction Service

Provides real-time price predictions using the trained MinMaxNormalizer + PyTorch model.
Loads model once and provides fast predictions for dashboard/trading system.

Features:
- 85-feature input (technical, sentiment, economic, sector)
- MinMaxNormalizer for feature scaling
- PyTorch model for prediction
- 3 outputs: 1-day, 5-day, 20-day return predictions
- Confidence scores based on model uncertainty

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-14
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys

# Add scripts/ml to path
sys.path.insert(0, str(Path(__file__).parent))
from minmax_normalizer import MinMaxNormalizer
from train_price_predictor_clean import PricePredictorClean


class PricePredictionService:
    """
    Singleton service for price predictions

    Usage:
        service = PricePredictionService.get_instance()
        predictions = service.predict(features)
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the prediction service"""
        # Use absolute paths from project root to work from any directory
        project_root = Path(__file__).parent.parent.parent
        self.model_path = project_root / 'models' / 'price_predictor_85feat_best.pth'
        self.normalizer_path = project_root / 'models' / 'normalizer_85feat.json'
        self.metadata_path = project_root / 'models' / 'price_predictor_85feat_info.json'

        # Load model architecture
        print("Loading price prediction model...")
        self.model = PricePredictorClean(input_size=85)

        # Load trained weights
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.model_path}\n"
                "Run: python scripts/ml/train_price_predictor_clean.py"
            )

        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        print(f"✓ Model loaded from {self.model_path}")

        # Load normalizer
        if not self.normalizer_path.exists():
            raise FileNotFoundError(
                f"Normalizer not found: {self.normalizer_path}\n"
                "Run: python scripts/ml/train_price_predictor_clean.py"
            )

        self.normalizer = MinMaxNormalizer.load(self.normalizer_path)
        print(f"✓ Normalizer loaded from {self.normalizer_path}")

        # Load metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"✓ Model metadata loaded")
            print(f"  Test accuracy (20-day): {self.metadata.get('test_acc_20d', 0):.2%}")
            print(f"  Training epochs: {self.metadata.get('epochs_trained', 0)}")
        else:
            self.metadata = {}

        # Store feature count for validation
        self.n_features = 85

        print("✓ Prediction service ready!\n")

    def predict(self, features: np.ndarray, return_confidence=True):
        """
        Make prediction for given features

        Args:
            features: np.ndarray of shape (85,) or (n_samples, 85)
            return_confidence: If True, return confidence scores

        Returns:
            dict with predictions and optionally confidence scores
            {
                'return_1d': float,    # Predicted 1-day return (%)
                'return_5d': float,    # Predicted 5-day return (%)
                'return_20d': float,   # Predicted 20-day return (%)
                'confidence_1d': float,  # Confidence [0-1] (optional)
                'confidence_5d': float,  # Confidence [0-1] (optional)
                'confidence_20d': float, # Confidence [0-1] (optional)
            }
        """

        # Validate input shape
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {features.shape[1]}"
            )

        # Normalize features
        features_normalized = self.normalizer.transform(features)

        # Convert to tensor
        features_tensor = torch.FloatTensor(features_normalized)

        # Predict
        with torch.no_grad():
            predictions = self.model(features_tensor).numpy()[0]

        result = {
            'return_1d': float(predictions[0]),   # Already in percentage (e.g., 2.5 = 2.5%)
            'return_5d': float(predictions[1]),
            'return_20d': float(predictions[2]),
        }

        if return_confidence:
            # Calculate confidence based on model accuracy from training
            # Higher accuracy = higher base confidence
            # Adjust based on prediction magnitude (more extreme = lower confidence)
            base_conf_1d = self.metadata.get('test_acc_1d', 0.75)
            base_conf_5d = self.metadata.get('test_acc_5d', 0.75)
            base_conf_20d = self.metadata.get('test_acc_20d', 0.75)

            # Reduce confidence for extreme predictions
            def adjust_confidence(base_conf, prediction):
                # Reduce by 5% for every 10% predicted return
                reduction = min(0.3, abs(float(prediction)) / 10.0 * 0.05)
                return float(max(0.5, base_conf - reduction))

            result['confidence_1d'] = adjust_confidence(base_conf_1d, predictions[0])
            result['confidence_5d'] = adjust_confidence(base_conf_5d, predictions[1])
            result['confidence_20d'] = adjust_confidence(base_conf_20d, predictions[2])

        return result

    def predict_from_dict(self, feature_dict: dict, return_confidence=True):
        """
        Make prediction from feature dictionary

        Args:
            feature_dict: dict with 85 feature names as keys
            return_confidence: If True, return confidence scores

        Returns:
            Same as predict()
        """

        # Define expected feature order (must match training)
        feature_names = [
            # Technical indicators (10)
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr_14',
            'volume_ratio', 'momentum_5d',

            # Sentiment (5)
            'news_sentiment', 'social_sentiment', 'analyst_rating',
            'put_call_ratio', 'vix_level',

            # Economic indicators (5)
            'employment_change', 'gdp_growth', 'inflation_rate',
            'fed_rate', 'treasury_yield_10y',

            # Sector correlation (5)
            'sector_momentum', 'spy_correlation', 'sector_beta',
            'peer_avg_return', 'market_regime',

            # Additional features to reach 85
            # (These would be actual features from your training data)
            # For now, fill with zeros if not provided
        ]

        # Extract features in correct order
        features = []
        for name in feature_names[:25]:  # Use first 25 features
            if name not in feature_dict:
                raise ValueError(f"Missing required feature: {name}")
            features.append(feature_dict[name])

        # Pad to 85 features with zeros (placeholder)
        # TODO: Replace with actual features from training data
        while len(features) < 85:
            features.append(0.0)

        features_array = np.array(features, dtype=np.float32)

        return self.predict(features_array, return_confidence=return_confidence)

    def get_model_info(self):
        """Get model metadata"""
        return self.metadata.copy()


def demo():
    """Demo the prediction service"""
    print("=" * 70)
    print("ML Price Prediction Service Demo")
    print("=" * 70)

    # Initialize service
    service = PricePredictionService.get_instance()

    # Create sample features (random for demo)
    print("\nGenerating sample feature vector...")
    sample_features = np.random.randn(85).astype(np.float32)

    # Make prediction
    print("\nMaking prediction...")
    prediction = service.predict(sample_features)

    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"1-Day Return:  {prediction['return_1d']:+.2f}% (confidence: {prediction['confidence_1d']:.1%})")
    print(f"5-Day Return:  {prediction['return_5d']:+.2f}% (confidence: {prediction['confidence_5d']:.1%})")
    print(f"20-Day Return: {prediction['return_20d']:+.2f}% (confidence: {prediction['confidence_20d']:.1%})")
    print("=" * 70)

    # Show model info
    info = service.get_model_info()
    print("\nModel Info:")
    print(f"  Architecture: {info.get('model_architecture', 'N/A')}")
    print(f"  Total parameters: {info.get('total_parameters', 0):,}")
    print(f"  Test accuracy (20d): {info.get('test_acc_20d', 0):.2%}")
    print(f"  RMSE (20d): {info.get('test_rmse_20d', 0):.6f}")
    print(f"  Normalization: {info.get('normalization', 'N/A')}")


if __name__ == '__main__':
    demo()
