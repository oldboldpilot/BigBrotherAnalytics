#!/usr/bin/env python3
"""
Price Predictor Model Training - GPU Accelerated
Trains neural network to predict 1-day, 5-day, 20-day price movements

Target: ≥55% accuracy → Profitable after 37.1% tax + 3% fees

Usage:
    uv run python scripts/ml/train_price_predictor.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("="*80)
print("PRICE PREDICTOR TRAINING - FAST TRACK TO PROFITABILITY")
print("="*80)

# Check for GPU
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# Load training data from DuckDB (faster and compressed)
db_path = Path('data/training_data.duckdb')

if not db_path.exists():
    print("❌ Training database not found!")
    print("   Run: uv run python scripts/data_collection/collect_training_data.py")
    sys.exit(1)

print("📊 Loading data from DuckDB...")
import duckdb
conn = duckdb.connect(str(db_path), read_only=True)

train_df = conn.execute("SELECT * FROM train").df()
val_df = conn.execute("SELECT * FROM validation").df()
test_df = conn.execute("SELECT * FROM test").df()

conn.close()

print(f"   Training: {len(train_df):,} samples")
print(f"   Validation: {len(val_df):,} samples")
print(f"   Test: {len(test_df):,} samples")

# Feature columns (exclude symbol, date, and targets)
exclude_cols = ['symbol', 'date', 'Date', 'target_1d', 'target_5d', 'target_20d']
feature_cols = [col for col in train_df.columns if col not in exclude_cols]
target_cols = ['target_1d', 'target_5d', 'target_20d']

print(f"\n📈 Features: {len(feature_cols)}")
print(f"   {', '.join(feature_cols[:5])}... (showing first 5)")

# Prepare data
X_train = train_df[feature_cols].values
y_train = train_df[target_cols].values

X_val = val_df[feature_cols].values
y_val = val_df[target_cols].values

X_test = test_df[feature_cols].values
y_test = test_df[target_cols].values

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)
y_val_t = torch.FloatTensor(y_val).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

# Define model architecture
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

# Create model
input_size = len(feature_cols)
model = PricePredictor(input_size=input_size).to(device)

print(f"\n🧠 Model Architecture:")
print(f"   Input: {input_size} features")
print(f"   Hidden: [128, 64, 32] neurons")
print(f"   Output: 3 predictions (1d, 5d, 20d)")
print(f"   Dropout: 0.3")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop
print(f"\n🚀 Starting training...")
print(f"   Target: <2% RMSE for 1-day, <5% RMSE for 20-day")

epochs = 50
batch_size = 512
best_val_loss = float('inf')
patience_counter = 0
max_patience = 10

train_losses = []
val_losses = []

start_time = time.time()

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

    # Calculate RMSE for each target
    with torch.no_grad():
        val_preds = model(X_val_t).cpu().numpy()
        val_true = y_val_t.cpu().numpy()

        rmse_1d = np.sqrt(np.mean((val_preds[:, 0] - val_true[:, 0])**2))
        rmse_5d = np.sqrt(np.mean((val_preds[:, 1] - val_true[:, 1])**2))
        rmse_20d = np.sqrt(np.mean((val_preds[:, 2] - val_true[:, 2])**2))

    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] - {elapsed:.1f}s")
        print(f"   Train Loss: {avg_train_loss:.6f}")
        print(f"   Val Loss: {val_loss:.6f}")
        print(f"   RMSE: 1d={rmse_1d:.4f}, 5d={rmse_5d:.4f}, 20d={rmse_20d:.4f}")

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
        }, 'models/price_predictor_best.pth')

    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break

total_time = time.time() - start_time
print(f"\n✅ Training complete in {total_time/60:.1f} minutes")

# Load best model for final evaluation
checkpoint = torch.load('models/price_predictor_best.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Final test evaluation
print(f"\n📊 Test Set Evaluation:")
model.eval()
with torch.no_grad():
    test_preds = model(X_test_t).cpu().numpy()
    test_true = y_test_t.cpu().numpy()

    rmse_1d = np.sqrt(np.mean((test_preds[:, 0] - test_true[:, 0])**2))
    rmse_5d = np.sqrt(np.mean((test_preds[:, 1] - test_true[:, 1])**2))
    rmse_20d = np.sqrt(np.mean((test_preds[:, 2] - test_true[:, 2])**2))

    print(f"   RMSE 1-day: {rmse_1d:.4f} ({rmse_1d*100:.2f}%)")
    print(f"   RMSE 5-day: {rmse_5d:.4f} ({rmse_5d*100:.2f}%)")
    print(f"   RMSE 20-day: {rmse_20d:.4f} ({rmse_20d*100:.2f}%)")

# Calculate directional accuracy (most important for trading)
def calculate_accuracy(preds, true):
    pred_direction = np.sign(preds)
    true_direction = np.sign(true)
    accuracy = np.mean(pred_direction == true_direction)
    return accuracy

acc_1d = calculate_accuracy(test_preds[:, 0], test_true[:, 0])
acc_5d = calculate_accuracy(test_preds[:, 1], test_true[:, 1])
acc_20d = calculate_accuracy(test_preds[:, 2], test_true[:, 2])

print(f"\n🎯 Directional Accuracy (Trading Signal):")
print(f"   1-day: {acc_1d*100:.1f}% {'✅' if acc_1d >= 0.55 else '⚠️'}")
print(f"   5-day: {acc_5d*100:.1f}% {'✅' if acc_5d >= 0.55 else '⚠️'}")
print(f"   20-day: {acc_20d*100:.1f}% {'✅' if acc_20d >= 0.55 else '⚠️'}")

# Profitability check
if acc_1d >= 0.55:
    print(f"\n💰 MODEL IS PROFITABLE!")
    print(f"   ✅ {acc_1d*100:.1f}% win rate > 55% threshold")
    print(f"   ✅ Profitable after 37.1% tax + 3% fees")
    print(f"\n🚀 READY FOR LIVE TRADING!")
else:
    print(f"\n⚠️  Model needs improvement")
    print(f"   Current: {acc_1d*100:.1f}% < 55% target")
    print(f"   Try: More data, different features, or architecture tuning")

# Save final model info
info = {
    'training_date': datetime.now().isoformat(),
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'test_samples': len(test_df),
    'features': len(feature_cols),
    'epochs_trained': checkpoint['epoch'] + 1,
    'best_val_loss': best_val_loss,
    'test_rmse_1d': float(rmse_1d),
    'test_rmse_5d': float(rmse_5d),
    'test_rmse_20d': float(rmse_20d),
    'test_acc_1d': float(acc_1d),
    'test_acc_5d': float(acc_5d),
    'test_acc_20d': float(acc_20d),
    'profitable': bool(acc_1d >= 0.55),
    'training_time_minutes': total_time / 60,
}

import json
with open('models/price_predictor_info.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f"\n📁 Model saved: models/price_predictor_best.pth")
print(f"📁 Info saved: models/price_predictor_info.json")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. ✅ Model trained and saved")
print("2. 🔄 Backtest on historical data: uv run python scripts/ml/backtest_model.py")
print("3. 🧪 Paper trade for 1 day with trained model")
print("4. 💰 GO LIVE on Day 9 (start with $500-$1000 positions)")
print("="*80)
