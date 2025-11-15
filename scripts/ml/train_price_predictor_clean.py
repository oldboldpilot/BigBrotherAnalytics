#!/usr/bin/env python3
"""
Train Price Predictor on Clean Dataset (85 features)

Uses the clean training dataset with:
- 58 base features (no constant/zero features)
- 3 temporal features (year, month, day)
- 20 first-order differences (price_diff_1d through 20d)
- 4 autocorrelation features (lags 1, 5, 10, 20)

Total: 85 input features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import duckdb
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import time

# Add scripts/ml to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from minmax_normalizer import MinMaxNormalizer

# Model architecture
class PricePredictorClean(nn.Module):
    """Neural network for price prediction with 85 input features"""
    def __init__(self, input_size=85):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Output: return_1d, return_5d, return_20d
        )

    def forward(self, x):
        return self.network(x)

def load_clean_data():
    """Load clean training data from DuckDB"""
    print("=" * 80)
    print("LOADING CLEAN DATASET")
    print("=" * 80)

    conn = duckdb.connect('data/clean_training_data.duckdb', read_only=True)

    # Load train/val/test splits
    df_train = conn.execute("SELECT * FROM train").fetchdf()
    df_val = conn.execute("SELECT * FROM validation").fetchdf()
    df_test = conn.execute("SELECT * FROM test").fetchdf()

    print(f"Train: {len(df_train):,} samples")
    print(f"Val:   {len(df_val):,} samples")
    print(f"Test:  {len(df_test):,} samples")

    conn.close()

    # Identify feature columns (exclude Date, symbol, and targets)
    exclude_cols = ['Date', 'symbol', 'return_1d', 'return_5d', 'return_20d']
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]

    print(f"\nFeatures: {len(feature_cols)}")

    # Extract features and targets
    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train[['return_1d', 'return_5d', 'return_20d']].values.astype(np.float32)

    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val[['return_1d', 'return_5d', 'return_20d']].values.astype(np.float32)

    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = df_test[['return_1d', 'return_5d', 'return_20d']].values.astype(np.float32)

    # Check for NaN/Inf
    print("\nData quality check:")
    print(f"  Train NaN: {np.isnan(X_train).sum()}, Inf: {np.isinf(X_train).sum()}")
    print(f"  Val NaN:   {np.isnan(X_val).sum()}, Inf: {np.isinf(X_val).sum()}")
    print(f"  Test NaN:  {np.isnan(X_test).sum()}, Inf: {np.isinf(X_test).sum()}")

    # Normalize features with MinMaxNormalizer (matches C++ implementation)
    print("\nNormalizing features with MinMaxNormalizer...")
    normalizer = MinMaxNormalizer()
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)

    # Save normalizer in JSON format (C++ compatible)
    normalizer_path = Path('models/normalizer_85feat.json')
    normalizer.save(normalizer_path)
    print(f"✓ Saved normalizer to {normalizer_path}")

    # Also export as C++ header for easy integration
    header_path = Path('src/ml/normalizer_params_85feat.hpp')
    normalizer.export_cpp_header(header_path, template_size=85)
    print(f"✓ Exported C++ header to {header_path}")

    # Print normalization statistics
    print(f"\nNormalization statistics:")
    print(f"  Feature min: [{normalizer.min_[0]:.3f}, ..., {normalizer.min_[-1]:.3f}]")
    print(f"  Feature max: [{normalizer.max_[0]:.3f}, ..., {normalizer.max_[-1]:.3f}]")
    print(f"  Output range: [0.0, 1.0]")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols

def train_model(model, train_loader, val_loader, optimizer, criterion,
                num_epochs=100, patience=10, target_accuracy=0.70, device='cpu'):
    """Train the model with early stopping"""

    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    print(f"Epochs: {num_epochs}")
    print(f"Patience: {patience} (early stopping)")
    print(f"Target accuracy: {target_accuracy:.1%}")
    print(f"Device: {device}")

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/price_predictor_85feat_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (best: {best_epoch})")
                break

    training_time = time.time() - start_time

    # Load best model
    model.load_state_dict(torch.load('models/price_predictor_85feat_best.pth'))

    return best_val_loss, best_epoch, epoch + 1, training_time

def evaluate_model(model, X_test, y_test, device='cpu'):
    """Evaluate model on test set"""

    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        predictions = model(X_tensor).cpu().numpy()

    # Compute RMSE for each target
    rmse_1d = np.sqrt(np.mean((predictions[:, 0] - y_test[:, 0]) ** 2))
    rmse_5d = np.sqrt(np.mean((predictions[:, 1] - y_test[:, 1]) ** 2))
    rmse_20d = np.sqrt(np.mean((predictions[:, 2] - y_test[:, 2]) ** 2))

    # Compute directional accuracy (sign match)
    acc_1d = np.mean((predictions[:, 0] * y_test[:, 0]) > 0)
    acc_5d = np.mean((predictions[:, 1] * y_test[:, 1]) > 0)
    acc_20d = np.mean((predictions[:, 2] * y_test[:, 2]) > 0)

    print(f"\nTest RMSE:")
    print(f"  1-day:  {rmse_1d:.6f}")
    print(f"  5-day:  {rmse_5d:.6f}")
    print(f"  20-day: {rmse_20d:.6f}")

    print(f"\nTest Directional Accuracy:")
    print(f"  1-day:  {acc_1d:.2%}")
    print(f"  5-day:  {acc_5d:.2%}")
    print(f"  20-day: {acc_20d:.2%}")

    return {
        'rmse_1d': rmse_1d,
        'rmse_5d': rmse_5d,
        'rmse_20d': rmse_20d,
        'acc_1d': acc_1d,
        'acc_5d': acc_5d,
        'acc_20d': acc_20d
    }

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n" + "=" * 80)
    print("DEVICE CONFIGURATION")
    print("=" * 80)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols = load_clean_data()

    # Create data loaders
    batch_size = 64
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    input_size = X_train.shape[1]
    print(f"\n" + "=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    print(f"Input size: {input_size}")

    model = PricePredictorClean(input_size=input_size).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Architecture: {input_size}->256->128->64->32->3")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    best_val_loss, best_epoch, total_epochs, training_time = train_model(
        model, train_loader, val_loader, optimizer, criterion,
        num_epochs=2000, patience=2000, target_accuracy=0.70, device=device
    )

    # Evaluate
    test_metrics = evaluate_model(model, X_test, y_test, device=device)

    # Save metadata (convert NumPy types to Python types for JSON serialization)
    metadata = {
        'training_date': datetime.now().isoformat(),
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'test_samples': int(len(X_test)),
        'features': int(input_size),
        'feature_names': feature_cols,
        'constant_features': 0,
        'zero_features': 0,
        'data_quality': 'clean',
        'epochs_trained': int(best_epoch),
        'total_epochs': int(total_epochs),
        'best_val_loss': float(best_val_loss),
        'test_rmse_1d': float(test_metrics['rmse_1d']),
        'test_rmse_5d': float(test_metrics['rmse_5d']),
        'test_rmse_20d': float(test_metrics['rmse_20d']),
        'test_acc_1d': float(test_metrics['acc_1d']),
        'test_acc_5d': float(test_metrics['acc_5d']),
        'test_acc_20d': float(test_metrics['acc_20d']),
        'target_accuracy': 0.70,
        'achieved_target': bool(test_metrics['acc_20d'] >= 0.70),
        'training_time_minutes': float(training_time / 60),
        'model_architecture': f"{input_size}->256->128->64->32->3",
        'total_parameters': int(total_params),
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'batch_size': int(batch_size),
        'normalization': 'MinMaxNormalizer'
    }

    metadata_path = Path('models/price_predictor_85feat_info.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Best epoch: {best_epoch}/{total_epochs}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"\nTest Results:")
    print(f"  1-day accuracy:  {test_metrics['acc_1d']:.2%}")
    print(f"  5-day accuracy:  {test_metrics['acc_5d']:.2%}")
    print(f"  20-day accuracy: {test_metrics['acc_20d']:.2%}")

    if test_metrics['acc_20d'] >= 0.70:
        print(f"\n✅ TARGET ACHIEVED! 20-day accuracy: {test_metrics['acc_20d']:.2%} >= 70%")
    else:
        print(f"\n⚠️  Target not reached. 20-day accuracy: {test_metrics['acc_20d']:.2%} < 70%")

    print(f"\nModel saved: models/price_predictor_85feat_best.pth")
    print(f"Metadata saved: {metadata_path}")
    print(f"Normalizer saved: models/normalizer_85feat.json")
    print(f"C++ header saved: src/ml/normalizer_params_85feat.hpp")

    print("\nNext steps:")
    print("  1. Export weights: python scripts/ml/export_cpp_model_weights.py")
    print("  2. Run benchmark: ./build/bin/benchmark_all_ml_engines")
    print("  3. Verify parity: ./build/bin/test_cpp_python_parity")

if __name__ == '__main__':
    main()
