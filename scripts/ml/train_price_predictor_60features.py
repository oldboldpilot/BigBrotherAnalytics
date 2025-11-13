#!/usr/bin/env python3
"""
Price Predictor Neural Network - 60 Features, Properly Trained
Architecture: 60 -> 256 -> 128 -> 64 -> 32 -> 3 (58,947 parameters)

Target: >70% validation accuracy (up from previous 51-56%)

Usage:
    uv run python scripts/ml/train_price_predictor_60features.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("=" * 80)
print("PRICE PREDICTOR TRAINING - 60 FEATURES (PROPERLY COLLECTED)")
print("=" * 80)

# Check for GPU
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# Load training data from DuckDB (properly collected with all 60 features)
db_path = Path('data/custom_training_data.duckdb')

if not db_path.exists():
    print("ERROR: Training database not found!")
    print("Expected: data/custom_training_data.duckdb")
    sys.exit(1)

print("Loading data from DuckDB (custom_training_data.duckdb)...")
import duckdb
conn = duckdb.connect(str(db_path), read_only=True)

train_df = conn.execute("SELECT * FROM train").df()
val_df = conn.execute("SELECT * FROM validation").df()
test_df = conn.execute("SELECT * FROM test").df()

conn.close()

print(f"Training: {len(train_df):,} samples")
print(f"Validation: {len(val_df):,} samples")
print(f"Test: {len(test_df):,} samples")

# Define feature columns (exclude metadata and targets)
exclude_cols = ['Date', 'symbol', 'close', 'open', 'high', 'low', 'volume',
                'return_1d', 'return_5d', 'return_20d',
                'target_1d', 'target_5d', 'target_20d']
all_cols = train_df.columns.tolist()
feature_cols = [col for col in all_cols if col not in exclude_cols]

target_cols = ['target_1d', 'target_5d', 'target_20d']

print(f"\nFeatures: {len(feature_cols)} (Target: 60)")
print(f"First 10: {', '.join(feature_cols[:10])}")
print(f"Last 10: {', '.join(feature_cols[-10:])}")

# Verify we have exactly 60 features
if len(feature_cols) != 60:
    print(f"\nWARNING: Expected 60 features, found {len(feature_cols)}")
    if len(feature_cols) < 60:
        print("ERROR: Not enough features. Need 60 for the model architecture.")
        sys.exit(1)
    elif len(feature_cols) > 60:
        print(f"INFO: Truncating to first 60 features")
        feature_cols = feature_cols[:60]

# Feature analysis
print("\n" + "=" * 80)
print("FEATURE QUALITY ANALYSIS")
print("=" * 80)

X_train_raw = train_df[feature_cols].values
print(f"\nChecking for data quality issues...")

# Check for constant features
constant_features = []
zero_features = []
for i, col in enumerate(feature_cols):
    vals = X_train_raw[:, i]
    if np.std(vals) == 0:
        constant_features.append((col, vals[0]))
    if np.all(vals == 0):
        zero_features.append(col)

if constant_features:
    print(f"\nWARNING: {len(constant_features)} constant features (no variation):")
    for col, val in constant_features[:5]:
        print(f"  - {col}: {val}")
    if len(constant_features) > 5:
        print(f"  ... and {len(constant_features) - 5} more")

if zero_features:
    print(f"\nWARNING: {len(zero_features)} all-zero features:")
    for col in zero_features[:5]:
        print(f"  - {col}")
    if len(zero_features) > 5:
        print(f"  ... and {len(zero_features) - 5} more")

# Feature statistics
print(f"\nFeature Statistics:")
print(f"  Mean: {np.mean(X_train_raw):.4f}")
print(f"  Std: {np.std(X_train_raw):.4f}")
print(f"  Min: {np.min(X_train_raw):.4f}")
print(f"  Max: {np.max(X_train_raw):.4f}")
print(f"  NaN count: {np.isnan(X_train_raw).sum()}")

# Prepare data
X_train = train_df[feature_cols].values
y_train = train_df[target_cols].values

X_val = val_df[feature_cols].values
y_val = val_df[target_cols].values

X_test = test_df[feature_cols].values
y_test = test_df[target_cols].values

print(f"\nInput shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  X_val: {X_val.shape}")
print(f"  X_test: {X_test.shape}")

# Normalize features using StandardScaler
print("\nNormalizing features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler parameters for C++ inference
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'var': scaler.var_.tolist(),
    'n_features': len(feature_cols),
    'feature_names': feature_cols
}

models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

with open('models/scaler_params.json', 'w') as f:
    json.dump(scaler_params, f, indent=2)

print(f"Saved scaler parameters to models/scaler_params.json")
print(f"  Mean range: [{np.min(scaler.mean_):.4f}, {np.max(scaler.mean_):.4f}]")
print(f"  Scale range: [{np.min(scaler.scale_):.4f}, {np.max(scaler.scale_):.4f}]")

# Check for normalization issues
if np.any(np.abs(scaler.scale_) > 1000):
    print("\nWARNING: Some features have very large scale factors (>1000)")
if np.any(scaler.scale_ == 0):
    print("\nERROR: Some features have zero scale (constant features)")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train_scaled).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val_scaled).to(device)
y_val_t = torch.FloatTensor(y_val).to(device)
X_test_t = torch.FloatTensor(X_test_scaled).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

# Define model architecture (MUST MATCH EXISTING C++ IMPLEMENTATION)
# 60 -> 256 -> 128 -> 64 -> 32 -> 3 (58,947 parameters)
class PricePredictor(nn.Module):
    def __init__(self):
        super(PricePredictor, self).__init__()

        # Sequential layers matching C++ implementation
        # Layer indices: 0, 3, 6, 9, 12 (skipping ReLU layers)
        self.network = nn.Sequential(
            nn.Linear(60, 256),   # network_0
            nn.ReLU(),            # network_1
            nn.Linear(256, 128),  # network_3
            nn.ReLU(),            # network_4
            nn.Linear(128, 64),   # network_6
            nn.ReLU(),            # network_7
            nn.Linear(64, 32),    # network_9
            nn.ReLU(),            # network_10
            nn.Linear(32, 3)      # network_12
        )

    def forward(self, x):
        return self.network(x)

# Create model
model = PricePredictor().to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print("\n" + "=" * 80)
print("MODEL ARCHITECTURE")
print("=" * 80)
print(f"Input: 60 features")
print(f"Hidden layers: [256, 128, 64, 32]")
print(f"Output: 3 predictions (1d, 5d, 20d returns)")
print(f"Total parameters: {total_params:,} (Expected: 58,947)")
print()

# Verify parameter count
expected_params = (60*256 + 256) + (256*128 + 128) + (128*64 + 64) + (64*32 + 32) + (32*3 + 3)
print(f"Expected: {expected_params:,}")
print(f"Actual: {total_params:,}")
if total_params != expected_params:
    print("WARNING: Parameter count mismatch!")
else:
    print("Parameter count verified!")

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training configuration
print("\n" + "=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print(f"Optimizer: Adam (lr=0.001)")
print(f"Loss: MSELoss")
print(f"Batch size: 64")
print(f"Epochs: 100 (with early stopping)")
print(f"Early stopping patience: 10")
print(f"Learning rate scheduler: ReduceLROnPlateau")
print()

epochs = 100
batch_size = 64
best_val_loss = float('inf')
patience_counter = 0
max_patience = 10

train_losses = []
val_losses = []

start_time = time.time()

print("=" * 80)
print("TRAINING")
print("=" * 80)

for epoch in range(epochs):
    # Training
    model.train()
    epoch_loss = 0
    num_batches = 0

    for i in range(0, len(X_train_t), batch_size):
        batch_X = X_train_t[i:i+batch_size]
        batch_y = y_train_t[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_train_loss = epoch_loss / num_batches

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t).item()

    scheduler.step(val_loss)

    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)

    # Calculate RMSE and directional accuracy
    with torch.no_grad():
        val_preds = model(X_val_t).cpu().numpy()
        val_true = y_val_t.cpu().numpy()

        rmse_1d = np.sqrt(np.mean((val_preds[:, 0] - val_true[:, 0])**2))
        rmse_5d = np.sqrt(np.mean((val_preds[:, 1] - val_true[:, 1])**2))
        rmse_20d = np.sqrt(np.mean((val_preds[:, 2] - val_true[:, 2])**2))

        # Directional accuracy
        acc_1d = np.mean(np.sign(val_preds[:, 0]) == np.sign(val_true[:, 0]))
        acc_5d = np.mean(np.sign(val_preds[:, 1]) == np.sign(val_true[:, 1]))
        acc_20d = np.mean(np.sign(val_preds[:, 2]) == np.sign(val_true[:, 2]))

    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        elapsed = time.time() - start_time
        print(f"\nEpoch [{epoch+1}/{epochs}] - {elapsed:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  RMSE: 1d={rmse_1d:.4f} ({rmse_1d*100:.2f}%), 5d={rmse_5d:.4f}, 20d={rmse_20d:.4f}")
        print(f"  Directional Accuracy: 1d={acc_1d*100:.1f}%, 5d={acc_5d*100:.1f}%, 20d={acc_20d*100:.1f}%")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'scaler': scaler,
            'feature_cols': feature_cols,
        }, 'models/price_predictor_60feat_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

total_time = time.time() - start_time
print(f"\nTraining complete in {total_time/60:.1f} minutes")

# Load best model for final evaluation
print("\n" + "=" * 80)
print("FINAL EVALUATION (TEST SET)")
print("=" * 80)

checkpoint = torch.load('models/price_predictor_60feat_best.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Test set evaluation
model.eval()
with torch.no_grad():
    test_preds = model(X_test_t).cpu().numpy()
    test_true = y_test_t.cpu().numpy()

    rmse_1d = np.sqrt(np.mean((test_preds[:, 0] - test_true[:, 0])**2))
    rmse_5d = np.sqrt(np.mean((test_preds[:, 1] - test_true[:, 1])**2))
    rmse_20d = np.sqrt(np.mean((test_preds[:, 2] - test_true[:, 2])**2))

    print(f"RMSE:")
    print(f"  1-day: {rmse_1d:.4f} ({rmse_1d*100:.2f}%)")
    print(f"  5-day: {rmse_5d:.4f} ({rmse_5d*100:.2f}%)")
    print(f"  20-day: {rmse_20d:.4f} ({rmse_20d*100:.2f}%)")

# Calculate directional accuracy
def calculate_accuracy(preds, true):
    pred_direction = np.sign(preds)
    true_direction = np.sign(true)
    accuracy = np.mean(pred_direction == true_direction)
    return accuracy

acc_1d = calculate_accuracy(test_preds[:, 0], test_true[:, 0])
acc_5d = calculate_accuracy(test_preds[:, 1], test_true[:, 1])
acc_20d = calculate_accuracy(test_preds[:, 2], test_true[:, 2])

print(f"\nDirectional Accuracy (Trading Signal):")
print(f"  1-day: {acc_1d*100:.1f}% {'[PROFITABLE]' if acc_1d >= 0.70 else '[NEEDS IMPROVEMENT]'}")
print(f"  5-day: {acc_5d*100:.1f}% {'[PROFITABLE]' if acc_5d >= 0.70 else '[NEEDS IMPROVEMENT]'}")
print(f"  20-day: {acc_20d*100:.1f}% {'[PROFITABLE]' if acc_20d >= 0.70 else '[NEEDS IMPROVEMENT]'}")

# Sample predictions
print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS (First 10 test samples)")
print("=" * 80)
print(f"{'Predicted':<35} {'Actual':<35} {'Match':<10}")
print(f"{'1d':<10} {'5d':<10} {'20d':<10} {'1d':<10} {'5d':<10} {'20d':<10}")
print("-" * 80)

for i in range(min(10, len(test_preds))):
    pred = test_preds[i]
    true = test_true[i]
    match_1d = 'YES' if np.sign(pred[0]) == np.sign(true[0]) else 'NO'
    match_5d = 'YES' if np.sign(pred[1]) == np.sign(true[1]) else 'NO'
    match_20d = 'YES' if np.sign(pred[2]) == np.sign(true[2]) else 'NO'

    print(f"{pred[0]*100:>9.2f}% {pred[1]*100:>9.2f}% {pred[2]*100:>9.2f}% | "
          f"{true[0]*100:>9.2f}% {true[1]*100:>9.2f}% {true[2]*100:>9.2f}% | "
          f"{match_1d} {match_5d} {match_20d}")

# Check for extreme predictions
extreme_mask = (np.abs(test_preds) > 1.0)  # >100% return
if np.any(extreme_mask):
    print(f"\nWARNING: {np.sum(extreme_mask)} extreme predictions (>100% return)")
    extreme_indices = np.where(extreme_mask)
    for idx in extreme_indices[0][:5]:
        print(f"  Sample {idx}: {test_preds[idx] * 100}%")
else:
    print(f"\nNo extreme predictions detected (all predictions < 100%)")

# Export weights to binary format for C++ inference
print("\n" + "=" * 80)
print("EXPORTING WEIGHTS TO BINARY FORMAT")
print("=" * 80)

weights_dir = Path('models/weights')
weights_dir.mkdir(exist_ok=True)

# Get model state dict
state_dict = model.state_dict()

# Layer mapping (Sequential indices)
# In PyTorch Sequential: Linear layers are at even indices (0, 2, 4, 6, 8)
# In C++ code: Expected indices are (0, 3, 6, 9, 12) for consistency
# network.0 = Linear(60, 256)   -> network_0
# network.2 = Linear(256, 128)  -> network_3
# network.4 = Linear(128, 64)   -> network_6
# network.6 = Linear(64, 32)    -> network_9
# network.8 = Linear(32, 3)     -> network_12

layer_mapping = {
    'network.0': 'network_0',
    'network.2': 'network_3',
    'network.4': 'network_6',
    'network.6': 'network_9',
    'network.8': 'network_12'
}

exported_files = []

for pytorch_layer, cpp_layer in layer_mapping.items():
    # Export weights
    weight_key = f'{pytorch_layer}.weight'
    bias_key = f'{pytorch_layer}.bias'

    weight_data = state_dict[weight_key].cpu().numpy().astype(np.float32)
    bias_data = state_dict[bias_key].cpu().numpy().astype(np.float32)

    # Save as binary files
    weight_file = weights_dir / f'{cpp_layer}_weight.bin'
    bias_file = weights_dir / f'{cpp_layer}_bias.bin'

    weight_data.tofile(weight_file)
    bias_data.tofile(bias_file)

    exported_files.append(str(weight_file))
    exported_files.append(str(bias_file))

    print(f"{cpp_layer}:")
    print(f"  Weight: {weight_data.shape} -> {weight_file}")
    print(f"  Bias: {bias_data.shape} -> {bias_file}")

print(f"\nExported {len(exported_files)} weight files:")
for f in sorted(exported_files):
    file_size = Path(f).stat().st_size
    print(f"  {f} ({file_size:,} bytes)")

# Save training info
info = {
    'training_date': datetime.now().isoformat(),
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'test_samples': len(test_df),
    'features': len(feature_cols),
    'feature_names': feature_cols,
    'constant_features': len(constant_features),
    'zero_features': len(zero_features),
    'epochs_trained': checkpoint['epoch'] + 1,
    'total_epochs': epochs,
    'best_val_loss': float(best_val_loss),
    'test_rmse_1d': float(rmse_1d),
    'test_rmse_5d': float(rmse_5d),
    'test_rmse_20d': float(rmse_20d),
    'test_acc_1d': float(acc_1d),
    'test_acc_5d': float(acc_5d),
    'test_acc_20d': float(acc_20d),
    'target_accuracy': 0.70,
    'achieved_target': bool(acc_1d >= 0.70),
    'training_time_minutes': total_time / 60,
    'model_architecture': '60->256->128->64->32->3',
    'total_parameters': total_params,
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'batch_size': batch_size,
    'normalization': 'StandardScaler',
}

with open('models/price_predictor_60feat_info.json', 'w') as f:
    json.dump(info, f, indent=2)

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Model saved: models/price_predictor_60feat_best.pth")
print(f"Info saved: models/price_predictor_60feat_info.json")
print(f"Scaler saved: models/scaler_params.json")
print(f"Weights exported: models/weights/ (10 .bin files)")
print()
print(f"OLD MODEL: 51-56% accuracy (16 features populated, 44 zeros)")
print(f"NEW MODEL: {acc_1d*100:.1f}% accuracy (60 features, properly collected)")
print()

if acc_1d >= 0.70:
    print("STATUS: TARGET ACHIEVED (>70% accuracy)")
    print("Model is ready for C++ integration")
else:
    print(f"STATUS: IMPROVEMENT NEEDED ({acc_1d*100:.1f}% < 70% target)")
    print("Recommendations:")
    if constant_features:
        print(f"  - Remove {len(constant_features)} constant features")
    if zero_features:
        print(f"  - Fix {len(zero_features)} all-zero features")
    print("  - Collect more training data")
    print("  - Try different feature engineering")

print("=" * 80)
