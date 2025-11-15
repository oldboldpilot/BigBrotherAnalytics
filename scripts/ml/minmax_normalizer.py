"""
MinMaxNormalizer - Python implementation matching C++ Normalizer

Provides dataset-level min/max normalization to [0, 1] range.
Designed to match the C++ Normalizer module exactly for perfect parity.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-14
Architecture: "omo oko ni e" - composable transformations
"""

import numpy as np
from typing import Optional
import json
from pathlib import Path


class MinMaxNormalizer:
    """
    Dataset-level min/max normalization to [0, 1]

    Matches C++ bigbrother::ml::Normalizer<N> exactly:
    - Learn min/max from training dataset once
    - Apply consistently to all samples (train/val/test/inference)
    - transform: raw → [0, 1]
    - inverse: [0, 1] → raw

    Formula:
        transform:  normalized[i] = (x[i] - min[i]) / (max[i] - min[i])
        inverse:    x[i] = normalized[i] * (max[i] - min[i]) + min[i]
    """

    def __init__(self):
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.range_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None

    def fit(self, X: np.ndarray) -> 'MinMaxNormalizer':
        """
        Learn min/max from training dataset

        Args:
            X: Training data (n_samples, n_features)

        Returns:
            self (for chaining)
        """
        if len(X) == 0:
            raise ValueError("Cannot fit on empty dataset")

        # Learn global min/max across all samples
        self.min_ = np.min(X, axis=0).astype(np.float32)
        self.max_ = np.max(X, axis=0).astype(np.float32)

        # Precompute range
        self.range_ = self.max_ - self.min_

        # Avoid division by zero for constant features
        # (matches C++ behavior: range[i] = 1.0 if range[i] < 1e-8)
        constant_features = self.range_ < 1e-8
        self.range_[constant_features] = 1.0

        self.n_features_ = X.shape[1]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform: raw → [0, 1]

        Args:
            X: Raw data (n_samples, n_features)

        Returns:
            Normalized data in [0, 1]
        """
        if self.min_ is None or self.max_ is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}"
            )

        # normalized = (x - min) / range
        normalized = (X - self.min_) / self.range_

        # Clamp to [0, 1] (matches C++ behavior)
        normalized = np.clip(normalized, 0.0, 1.0)

        return normalized.astype(np.float32)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse: [0, 1] → raw

        Args:
            X: Normalized data in [0, 1]

        Returns:
            Original scale data
        """
        if self.min_ is None or self.max_ is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}"
            )

        # original = normalized * range + min
        return (X * self.range_ + self.min_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit() then transform()"""
        return self.fit(X).transform(X)

    def save(self, path: Path) -> None:
        """
        Save normalizer parameters to JSON (C++ compatible)

        Format:
            {
                "n_features": 85,
                "min": [0.1, 0.2, ...],
                "max": [1.0, 2.0, ...],
                "range": [0.9, 1.8, ...]
            }
        """
        if self.min_ is None or self.max_ is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        params = {
            "n_features": int(self.n_features_),
            "min": self.min_.tolist(),
            "max": self.max_.tolist(),
            "range": self.range_.tolist(),
        }

        with open(path, 'w') as f:
            json.dump(params, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'MinMaxNormalizer':
        """Load normalizer parameters from JSON"""
        with open(path, 'r') as f:
            params = json.load(f)

        normalizer = cls()
        normalizer.n_features_ = params["n_features"]
        normalizer.min_ = np.array(params["min"], dtype=np.float32)
        normalizer.max_ = np.array(params["max"], dtype=np.float32)
        normalizer.range_ = np.array(params["range"], dtype=np.float32)

        return normalizer

    def export_cpp_header(self, path: Path, template_size: int) -> None:
        """
        Export as C++ header with constexpr arrays

        Generates code like:
            static constexpr std::array<float, 85> FEATURE_MIN = {...};
            static constexpr std::array<float, 85> FEATURE_MAX = {...};
        """
        if self.min_ is None or self.max_ is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        with open(path, 'w') as f:
            f.write(f"// Auto-generated normalizer parameters\n")
            f.write(f"// Generated: {np.datetime64('now')}\n")
            f.write(f"// Features: {self.n_features_}\n\n")

            f.write(f"static constexpr std::array<float, {template_size}> FEATURE_MIN = {{\n    ")
            f.write(",\n    ".join(f"{val:.8f}f" for val in self.min_))
            f.write("\n};\n\n")

            f.write(f"static constexpr std::array<float, {template_size}> FEATURE_MAX = {{\n    ")
            f.write(",\n    ".join(f"{val:.8f}f" for val in self.max_))
            f.write("\n};\n")

    def get_params(self) -> dict:
        """Get normalizer parameters (sklearn-compatible API)"""
        return {
            "n_features": self.n_features_,
            "min": self.min_,
            "max": self.max_,
            "range": self.range_,
        }

    def __repr__(self) -> str:
        if self.n_features_ is None:
            return "MinMaxNormalizer(not fitted)"
        return f"MinMaxNormalizer(n_features={self.n_features_})"
