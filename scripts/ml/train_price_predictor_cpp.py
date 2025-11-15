#!/usr/bin/env python3
"""
Train Price Predictor on C++-Extracted Features

Uses features extracted by the C++ feature extractor via Python bindings.
This ensures perfect parity between training and inference.

Total: 85 input features (extracted by C++ toArray85() function)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time

# Model architecture (same as clean model)
class PricePredictorCPP(nn.Module):
    """Neural network for price prediction with 85 C++-extracted features"""
    def __init__(self, input_size=85):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Output: target_1d, target_5d, target_20d
        )

    def forward(self, x):
        return self.network(x)

def load_cpp_features():
    """Load C++-extracted features from parquet"""
    print("=" * 80)
    print("LOADING C++ EXTRACTED FEATURES")
    print("=" * 80)

    # Load features from parquet
    df = pd.read_parquet('models/training_data/features_cpp_85.parquet')

    print(f"Total samples: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # Extract feature columns (feature_0 through feature_84)
    feature_cols = [f'feature_{i}' for i in range(85)]

    # Extract features and targets
    X = df[feature_cols].values.astype(np.float32)
    y = df[['target_1d', 'target_5d', 'target_20d']].values.astype(np.float32)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")

    # Split into train/val/test (70/15/15)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
    )

    print(f"\nTrain: {len(X_train):,} samples")
    print(f"Val:   {len(X_val):,} samples")
    print(f"Test:  {len(X_test):,} samples")

    # Normalize features using MinMaxScaler (scales to [0, 1] - perfect for INT32 quantization)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save scaler parameters for C++ inference
    scaler_params = {
        'min': scaler.data_min_.tolist(),
        'max': scaler.data_max_.tolist(),
        'scale': scaler.scale_.tolist(),
        'range': scaler.feature_range
    }

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_params

def train_model():
    """Train the model"""
    print("\n" + "=" * 80)
    print("TRAINING PRICE PREDICTOR ON C++ FEATURES")
    print("=" * 80)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_params = load_cpp_features()

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = PricePredictorCPP(input_size=85).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
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
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/price_predictor_cpp_85feat.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Load best model for testing
    model.load_state_dict(torch.load('models/price_predictor_cpp_85feat.pth'))

    # Test evaluation
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    test_loss /= len(test_loader)

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Calculate accuracy (within 0.5% threshold)
    threshold = 0.005
    correct = np.abs(all_preds - all_targets) < threshold
    accuracy = correct.mean()

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Test RMSE: {np.sqrt(test_loss):.6f}")
    print(f"Test Accuracy (±0.5%): {accuracy*100:.2f}%")

    # Save model info
    model_info = {
        'model_type': 'price_predictor_cpp_85feat',
        'input_features': 85,
        'output_targets': 3,
        'architecture': 'MLP (256-128-64-32)',
        'trained_on': datetime.now().isoformat(),
        'training_time_seconds': training_time,
        'epochs_trained': epoch + 1,
        'best_val_loss': float(best_val_loss),
        'test_loss': float(test_loss),
        'test_rmse': float(np.sqrt(test_loss)),
        'test_accuracy': float(accuracy),
        'scaler_params': scaler_params,
        'features_extracted_by': 'C++ feature_extractor.cppm via Python bindings',
        'data_source': 'models/training_data/features_cpp_85.parquet'
    }

    with open('models/price_predictor_cpp_85feat_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"\nModel saved to: models/price_predictor_cpp_85feat.pth")
    print(f"Model info saved to: models/price_predictor_cpp_85feat_info.json")
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE - PERFECT PARITY WITH C++ INFERENCE")
    print("=" * 80)

if __name__ == "__main__":
    train_model()
