# C++ Single Source of Truth Template

**Author:** Olumuyiwa Oluwasanmi
**Date:** 2025-11-14
**Purpose:** Template for implementing C++ Single Source of Truth architecture for data/feature/quantization operations

---

## Overview

This template provides a step-by-step guide for implementing the C++ Single Source of Truth pattern, ensuring perfect parity between training and inference for ML systems.

**Core Principle:** ALL data extraction, feature extraction, and quantization operations MUST be implemented in C++ with Python bindings for training. NO Python-only implementations allowed.

---

## When to Use This Template

Use this template when you need to:
- Extract data from databases or APIs for ML training and inference
- Calculate features (technical indicators, Greeks, lags, etc.)
- Implement quantization/dequantization logic
- Preprocess data (normalization, scaling)
- Implement ANY operation used in BOTH training AND inference

**Do NOT use for:**
- Model training code (use Python/PyTorch)
- Hyperparameter tuning
- Visualization and plotting
- Exploratory data analysis (EDA)
- Operations used ONLY in training (never in inference)

---

## Step 1: Implement C++23 Module

**File:** `src/<component>/<feature_name>.cppm`

```cpp
/**
 * BigBrotherAnalytics - <Feature Name>
 *
 * <Brief description of feature>
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: <YYYY-MM-DD>
 *
 * Following C++ Core Guidelines:
 * - Trailing return type syntax throughout
 * - std::expected for error handling
 * - [[nodiscard]] for all getters
 */

// Global module fragment (standard library only)
module;

#include <vector>
#include <array>
#include <span>
#include <expected>
#include <chrono>

// Module declaration
export module bigbrother.<component>.<feature_name>;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;

// Exported interface
export namespace bigbrother::<component> {

/**
 * <Feature Name> class
 *
 * <Detailed description of what this feature does>
 *
 * Thread-Safety: <describe thread-safety guarantees>
 * Performance: <describe performance characteristics>
 */
class <FeatureName> {
public:
    // Constructor
    <FeatureName>() = default;

    /**
     * Extract/calculate feature
     *
     * @param input Input data
     * @return Feature values or error
     *
     * Performance: <time complexity>
     * Thread-Safety: <guarantees>
     */
    [[nodiscard]] auto calculateFeature(
        std::span<double const> input
    ) const -> std::expected<std::vector<float>, Error>;

    /**
     * Additional feature calculation methods
     */
    [[nodiscard]] auto someOtherMethod(/* params */) const -> ReturnType;

private:
    // Private helper methods
    auto helperMethod(/* params */) const -> ReturnType;
};

} // namespace bigbrother::<component>
```

---

## Step 2: Create Python Binding

**File:** `src/python_bindings/<feature_name>_bindings.cpp`

```cpp
/**
 * BigBrotherAnalytics - <Feature Name> Python Bindings
 *
 * Exposes C++ <FeatureName> to Python for training use
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: <YYYY-MM-DD>
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/chrono.h>

// Import C++ module
import bigbrother.<component>.<feature_name>;

namespace py = pybind11;
using namespace bigbrother::<component>;

PYBIND11_MODULE(<feature_name>_cpp, m) {
    m.doc() = "C++ implementation of <FeatureName> with perfect training/inference parity";

    // Export main class
    py::class_<<FeatureName>>(m, "<FeatureName>")
        .def(py::init<>())
        .def("calculate_feature", &<FeatureName>::calculateFeature,
             py::arg("input"),
             "Calculate feature from input data")
        .def("some_other_method", &<FeatureName>::someOtherMethod,
             py::arg("param1"),
             py::arg("param2"),
             "Description of method");

    // Export error types if needed
    // py::class_<Error>(m, "Error")
    //     .def_readonly("code", &Error::code)
    //     .def_readonly("message", &Error::message);
}
```

---

## Step 3: Add to CMakeLists.txt

**File:** `CMakeLists.txt` (add to existing file)

```cmake
# ============================================================================
# <Feature Name> Python Binding
# ============================================================================

# Find pybind11 (should already be found at top of file)
# find_package(pybind11 REQUIRED)

# Build Python binding
pybind11_add_module(<feature_name>_py
    src/python_bindings/<feature_name>_bindings.cpp
)

# Link against C++ module
target_link_libraries(<feature_name>_py PRIVATE
    <component>_module  # The C++ module library
)

# Set output properties
set_target_properties(<feature_name>_py PROPERTIES
    OUTPUT_NAME "<feature_name>_cpp"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
)

# Install to Python directory
install(TARGETS <feature_name>_py
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/python
)
```

---

## Step 4: Build and Verify

```bash
# 1. Configure CMake with Ninja (required for C++23 modules)
cmake -G Ninja -B build

# 2. Build C++ module
ninja -C build <component>_module

# 3. Build Python binding
ninja -C build <feature_name>_py

# 4. Verify output
ls -lh python/<feature_name>_cpp*.so

# Expected: <feature_name>_cpp.cpython-313-x86_64-linux-gnu.so (or similar)
```

---

## Step 5: Create Parity Test

**File:** `tests/test_<feature_name>_parity.py`

```python
#!/usr/bin/env python3
"""
Test parity between C++ and training usage

Author: Olumuyiwa Oluwasanmi
Date: <YYYY-MM-DD>
"""
import sys
sys.path.insert(0, 'python')
from <feature_name>_cpp import <FeatureName>
import numpy as np

def test_basic_parity():
    """Verify basic feature calculation works"""
    feature = <FeatureName>()

    # Test data
    input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Calculate feature
    result = feature.calculate_feature(input_data)

    # Verify result
    assert len(result) > 0, "Result should not be empty"
    assert isinstance(result[0], float), "Result should be floats"

    print("✅ Basic parity verified")

def test_consistency():
    """Verify feature calculation is deterministic"""
    feature = <FeatureName>()

    input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Calculate twice
    result1 = feature.calculate_feature(input_data)
    result2 = feature.calculate_feature(input_data)

    # Verify identical
    assert np.allclose(result1, result2), "Results should be identical"

    print("✅ Consistency verified")

def test_edge_cases():
    """Verify handling of edge cases"""
    feature = <FeatureName>()

    # Empty input
    try:
        result = feature.calculate_feature(np.array([]))
        # Should handle gracefully or raise appropriate error
        print("✅ Empty input handled")
    except Exception as e:
        print(f"✅ Empty input error: {e}")

    # Single value
    result = feature.calculate_feature(np.array([1.0]))
    assert result is not None
    print("✅ Single value handled")

if __name__ == "__main__":
    test_basic_parity()
    test_consistency()
    test_edge_cases()
    print("\n✅ ALL PARITY TESTS PASSED")
```

**Run parity tests:**
```bash
PYTHONPATH=python:$PYTHONPATH uv run python tests/test_<feature_name>_parity.py
```

---

## Step 6: Use in Training Pipeline

**File:** `scripts/ml/train_with_<feature_name>.py`

```python
#!/usr/bin/env python3
"""
Training script using C++ Single Source of Truth

Author: Olumuyiwa Oluwasanmi
Date: <YYYY-MM-DD>
"""
import sys
sys.path.insert(0, 'python')

# Import C++ modules via pybind11
from <feature_name>_cpp import <FeatureName>
import pandas as pd
import numpy as np

def generate_training_data():
    """Generate training features using C++ implementation"""
    feature = <FeatureName>()

    # Load raw data
    df = pd.read_parquet('data/raw_data.parquet')

    all_features = []

    for idx in range(len(df)):
        # Extract feature using C++
        input_data = df['input_column'][idx]
        features = feature.calculate_feature(input_data)

        all_features.append({
            'sample_id': idx,
            **{f'feature_{i}': features[i] for i in range(len(features))}
        })

    # Save to training database
    df_features = pd.DataFrame(all_features)
    df_features.to_parquet('models/training_data/features_<feature_name>.parquet')
    print(f"✅ Saved {len(df_features)} samples")

if __name__ == "__main__":
    generate_training_data()
```

---

## Step 7: Use in C++ Inference

**File:** `src/main.cpp` (or relevant trading engine file)

```cpp
import bigbrother.<component>.<feature_name>;

auto main() -> int {
    using namespace bigbrother::<component>;

    // Initialize feature extractor (same as training!)
    <FeatureName> feature;

    // Main trading loop
    while (trading_active) {
        // Fetch live data
        auto input_data = fetch_live_data();

        // Extract features (IDENTICAL to training)
        auto features = feature.calculateFeature(input_data);

        if (!features) {
            logger->error("Feature extraction failed: {}", features.error().message);
            continue;
        }

        // Use features for prediction
        auto prediction = predictor->predict(*features);

        // Perfect parity guaranteed - same code path as training
        if (prediction && meets_criteria(prediction)) {
            execute_trade(prediction);
        }
    }
}
```

---

## Step 8: Document

**Update these files:**

1. **`FEATURE_EXTRACTION_ARCHITECTURE.md`** - Add section describing your feature
2. **`docs/CODING_STANDARDS.md`** - Reference your feature as example
3. **`.ai/claude.md`** - Add to implementation examples
4. **`copilot-instructions.md`** - Add to implementation examples

---

## Checklist

Before considering the implementation complete:

- [ ] C++23 module implemented in `src/<component>/<feature_name>.cppm`
- [ ] Python binding created in `src/python_bindings/<feature_name>_bindings.cpp`
- [ ] CMakeLists.txt updated to build binding
- [ ] Binding builds successfully: `ninja -C build <feature_name>_py`
- [ ] Parity test created in `tests/test_<feature_name>_parity.py`
- [ ] Parity test passes: `uv run python tests/test_<feature_name>_parity.py`
- [ ] Training script updated to use C++ binding
- [ ] C++ inference code updated to use same module
- [ ] Documentation updated (4 files listed above)
- [ ] Deprecated Python-only code removed (if any)
- [ ] Code review completed
- [ ] CI/CD pipeline passes

---

## Common Pitfalls

**1. Forgetting to import C++ module in binding:**
```cpp
// ❌ WRONG - Missing import
#include <pybind11/pybind11.h>
PYBIND11_MODULE(my_module, m) {
    // Error: FeatureName not defined
}

// ✅ CORRECT - Import first
#include <pybind11/pybind11.h>
import bigbrother.component.feature_name;  // Add this!
PYBIND11_MODULE(my_module, m) {
    // Now FeatureName is available
}
```

**2. Incorrect CMake output name:**
```cmake
# ❌ WRONG - Python won't find it
set_target_properties(my_feature_py PROPERTIES
    OUTPUT_NAME "my_feature"  # Missing _cpp suffix
)

# ✅ CORRECT - Must end with _cpp
set_target_properties(my_feature_py PROPERTIES
    OUTPUT_NAME "my_feature_cpp"
)
```

**3. Not adding python directory to PYTHONPATH:**
```bash
# ❌ WRONG - Module not found
python scripts/train.py

# ✅ CORRECT - Add python dir
PYTHONPATH=python:$PYTHONPATH uv run python scripts/train.py
```

**4. Python-only feature calculation:**
```python
# ❌ WRONG - Duplicating C++ logic in Python
def calculate_feature(data):
    # This will drift from C++ implementation!
    return data * 2.0 + 1.0

# ✅ CORRECT - Use C++ binding
from feature_name_cpp import FeatureName
feature = FeatureName()
result = feature.calculate_feature(data)
```

---

## Reference Implementation

**See existing implementations:**
- **Data Loading:** `src/ml/data_loader.cppm` + `src/python_bindings/data_loader_bindings.cpp`
- **Feature Extraction:** `src/market_intelligence/feature_extractor.cppm` + `src/python_bindings/feature_extractor_bindings.cpp`
- **Training Pipeline:** `scripts/ml/prepare_features_cpp.py`
- **Parity Tests:** `tests/test_feature_parity.py`

---

## Support

For questions or issues:
1. Check `FEATURE_EXTRACTION_ARCHITECTURE.md` for detailed architecture
2. Check `docs/CODING_STANDARDS.md` for coding standards
3. Check `.ai/claude.md` or `copilot-instructions.md` for AI agent guidance
4. Review existing implementations listed above

---

**Author:** Olumuyiwa Oluwasanmi
**Last Updated:** 2025-11-14
**Version:** 1.0.0
