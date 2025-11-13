#!/usr/bin/env python3
"""
Validate 60-feature model and test inference

Usage:
    uv run python scripts/ml/validate_60feat_model.py
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import duckdb

print("=" * 80)
print("MODEL VALIDATION AND INFERENCE TEST")
print("=" * 80)

# Define model architecture
class PricePredictor(nn.Module):
    def __init__(self):
        super(PricePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(60, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.network(x)

# Load model
print("\n1. Loading trained model...")
model_path = Path('models/price_predictor_60feat_best.pth')
checkpoint = torch.load(model_path, weights_only=False)

model = PricePredictor()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"   Model loaded from: {model_path}")
print(f"   Trained epoch: {checkpoint['epoch'] + 1}")

# Load scaler parameters
print("\n2. Loading scaler parameters...")
with open('models/scaler_params.json', 'r') as f:
    scaler_params = json.load(f)

print(f"   Features: {scaler_params['n_features']}")
print(f"   Mean shape: {len(scaler_params['mean'])}")
print(f"   Scale shape: {len(scaler_params['scale'])}")

# Recreate scaler
scaler = StandardScaler()
scaler.mean_ = np.array(scaler_params['mean'])
scaler.scale_ = np.array(scaler_params['scale'])
scaler.var_ = np.array(scaler_params['var'])
scaler.n_features_in_ = scaler_params['n_features']

print("\n3. Loading test data...")
conn = duckdb.connect('data/custom_training_data.duckdb', read_only=True)
test_df = conn.execute("SELECT * FROM test").df()
conn.close()

print(f"   Test samples: {len(test_df)}")

# Prepare features
exclude_cols = ['Date', 'symbol', 'close', 'open', 'high', 'low', 'volume',
                'return_1d', 'return_5d', 'return_20d',
                'target_1d', 'target_5d', 'target_20d']
all_cols = test_df.columns.tolist()
feature_cols = [col for col in all_cols if col not in exclude_cols][:60]

X_test = test_df[feature_cols].values
y_test = test_df[['target_1d', 'target_5d', 'target_20d']].values

print(f"   Features: {len(feature_cols)}")

# Test inference pipeline
print("\n" + "=" * 80)
print("INFERENCE PIPELINE TEST")
print("=" * 80)

print("\nStep 1: Raw features (sample 0)")
sample_raw = X_test[0]
print(f"  Shape: {sample_raw.shape}")
print(f"  Range: [{sample_raw.min():.4f}, {sample_raw.max():.4f}]")
print(f"  First 5 values: {sample_raw[:5]}")

print("\nStep 2: Normalized features")
sample_normalized = scaler.transform(sample_raw.reshape(1, -1))
print(f"  Shape: {sample_normalized.shape}")
print(f"  Range: [{sample_normalized.min():.4f}, {sample_normalized.max():.4f}]")
print(f"  First 5 values: {sample_normalized[0, :5]}")

print("\nStep 3: Model inference")
sample_tensor = torch.FloatTensor(sample_normalized)
with torch.no_grad():
    prediction = model(sample_tensor).numpy()[0]

print(f"  Prediction: {prediction}")
print(f"  1-day: {prediction[0]*100:.2f}%")
print(f"  5-day: {prediction[1]*100:.2f}%")
print(f"  20-day: {prediction[2]*100:.2f}%")

print("\nStep 4: Actual values")
actual = y_test[0]
print(f"  Actual: {actual}")
print(f"  1-day: {actual[0]*100:.2f}%")
print(f"  5-day: {actual[1]*100:.2f}%")
print(f"  20-day: {actual[2]*100:.2f}%")

print("\nStep 5: Direction match")
pred_dir = np.sign(prediction)
actual_dir = np.sign(actual)
match = pred_dir == actual_dir
print(f"  1-day: {pred_dir[0]:+.0f} vs {actual_dir[0]:+.0f} -> {'MATCH' if match[0] else 'MISS'}")
print(f"  5-day: {pred_dir[1]:+.0f} vs {actual_dir[1]:+.0f} -> {'MATCH' if match[1] else 'MISS'}")
print(f"  20-day: {pred_dir[2]:+.0f} vs {actual_dir[2]:+.0f} -> {'MATCH' if match[2] else 'MISS'}")

# Batch inference test
print("\n" + "=" * 80)
print("BATCH INFERENCE TEST")
print("=" * 80)

X_test_normalized = scaler.transform(X_test)
X_test_tensor = torch.FloatTensor(X_test_normalized)

with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

# Calculate metrics
rmse_1d = np.sqrt(np.mean((predictions[:, 0] - y_test[:, 0])**2))
rmse_5d = np.sqrt(np.mean((predictions[:, 1] - y_test[:, 1])**2))
rmse_20d = np.sqrt(np.mean((predictions[:, 2] - y_test[:, 2])**2))

acc_1d = np.mean(np.sign(predictions[:, 0]) == np.sign(y_test[:, 0]))
acc_5d = np.mean(np.sign(predictions[:, 1]) == np.sign(y_test[:, 1]))
acc_20d = np.mean(np.sign(predictions[:, 2]) == np.sign(y_test[:, 2]))

print(f"\nTest Set Performance:")
print(f"  Samples: {len(predictions)}")
print(f"\n  RMSE:")
print(f"    1-day: {rmse_1d:.4f} ({rmse_1d*100:.2f}%)")
print(f"    5-day: {rmse_5d:.4f} ({rmse_5d*100:.2f}%)")
print(f"    20-day: {rmse_20d:.4f} ({rmse_20d*100:.2f}%)")
print(f"\n  Directional Accuracy:")
print(f"    1-day: {acc_1d*100:.1f}%")
print(f"    5-day: {acc_5d*100:.1f}%")
print(f"    20-day: {acc_20d*100:.1f}%")

# Check for extreme predictions
extreme_mask = np.abs(predictions) > 0.5  # >50% return
extreme_count = np.sum(extreme_mask)

if extreme_count > 0:
    print(f"\n  WARNING: {extreme_count} extreme predictions (>50% return)")
    extreme_indices = np.where(extreme_mask)
    for i in range(min(5, len(extreme_indices[0]))):
        idx = extreme_indices[0][i]
        col = extreme_indices[1][i]
        print(f"    Sample {idx}, output {col}: {predictions[idx, col]*100:.2f}%")
else:
    print(f"\n  All predictions within reasonable range (<50% return)")

# Load model info
print("\n" + "=" * 80)
print("MODEL INFORMATION")
print("=" * 80)

with open('models/price_predictor_60feat_info.json', 'r') as f:
    model_info = json.load(f)

print(f"\nTraining Date: {model_info['training_date']}")
print(f"Architecture: {model_info['model_architecture']}")
print(f"Parameters: {model_info['total_parameters']:,}")
print(f"Features: {model_info['features']}")
print(f"\nData:")
print(f"  Training samples: {model_info['train_samples']:,}")
print(f"  Validation samples: {model_info['val_samples']:,}")
print(f"  Test samples: {model_info['test_samples']:,}")
print(f"\nTraining:")
print(f"  Epochs: {model_info['epochs_trained']}/{model_info['total_epochs']}")
print(f"  Time: {model_info['training_time_minutes']:.1f} minutes")
print(f"  Best val loss: {model_info['best_val_loss']:.6f}")
print(f"\nData Quality:")
print(f"  Constant features: {model_info['constant_features']}")
print(f"  Zero features: {model_info['zero_features']}")

# Comparison with old model
print("\n" + "=" * 80)
print("COMPARISON: OLD VS NEW MODEL")
print("=" * 80)

old_accuracy = 0.53  # 51-56% range
new_accuracy = acc_1d

print(f"\nOLD MODEL (Buggy Training):")
print(f"  Features: 60 (but only 16 populated, 44 zeros)")
print(f"  Constant features: 17")
print(f"  1-day accuracy: 51-56%")
print(f"  Status: Poor feature diversity")

print(f"\nNEW MODEL (Properly Trained):")
print(f"  Features: 60 (properly collected)")
print(f"  Constant features: {model_info['constant_features']}")
print(f"  1-day accuracy: {new_accuracy*100:.1f}%")
print(f"  Status: {'Better feature diversity' if model_info['constant_features'] < 17 else 'Still has issues'}")

improvement = (new_accuracy - old_accuracy) / old_accuracy * 100
print(f"\nImprovement: {improvement:+.1f}%")

if new_accuracy >= 0.70:
    print("Target achieved: >70% accuracy")
elif new_accuracy > old_accuracy:
    print(f"Improvement shown, but below 70% target")
else:
    print("No improvement - investigate data quality")

# C++ Integration readiness
print("\n" + "=" * 80)
print("C++ INTEGRATION READINESS")
print("=" * 80)

weights_dir = Path('models/weights')
expected_files = [
    'network_0_weight.bin', 'network_0_bias.bin',
    'network_3_weight.bin', 'network_3_bias.bin',
    'network_6_weight.bin', 'network_6_bias.bin',
    'network_9_weight.bin', 'network_9_bias.bin',
    'network_12_weight.bin', 'network_12_bias.bin'
]

all_files_exist = True
for filename in expected_files:
    filepath = weights_dir / filename
    if filepath.exists():
        size = filepath.stat().st_size
        print(f"  {filename}: {size:,} bytes")
    else:
        print(f"  {filename}: MISSING")
        all_files_exist = False

scaler_exists = Path('models/scaler_params.json').exists()
print(f"\n  scaler_params.json: {'EXISTS' if scaler_exists else 'MISSING'}")

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

if all_files_exist and scaler_exists:
    print("\nSTATUS: READY FOR C++ INTEGRATION")
    print("\nChecklist:")
    print("  [X] Model trained with 60 features")
    print("  [X] Weights exported to binary format (10 files)")
    print("  [X] Scaler parameters saved to JSON")
    print("  [X] Inference pipeline validated")
    print(f"  [{'X' if new_accuracy >= 0.70 else ' '}] Accuracy target achieved (>70%)")
    print(f"  [{'X' if extreme_count == 0 else ' '}] No extreme predictions")

    if new_accuracy < 0.70:
        print("\nRECOMMENDATIONS:")
        if model_info['constant_features'] > 0:
            print(f"  - Remove {model_info['constant_features']} constant features")
        if model_info['zero_features'] > 0:
            print(f"  - Fix {model_info['zero_features']} zero features")
        print("  - Collect more diverse training data")
        print("  - Consider feature engineering improvements")
else:
    print("\nSTATUS: NOT READY")
    print("Missing required files for C++ integration")

print("\n" + "=" * 80)
