#!/bin/bash
# Script to make risk management classes with mutexes movable for pybind11

FILES=(
    "src/risk_management/correlation_analyzer.cppm"
    "src/risk_management/performance_metrics.cppm"
    "src/risk_management/risk_manager.cppm"
    "src/risk_management/stress_testing.cppm"
    "src/risk_management/var_calculator.cppm"
)

for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Skipping $file (not found)"
        continue
    fi

    echo "Processing $file..."

    # Find the class name
    classname=$(grep -oP 'class \K\w+(?=.*\{)' "$file" | head -1)
    echo "  Found class: $classname"

    # Check if it has a mutex
    if ! grep -q "mutable std::mutex" "$file"; then
        echo "  No mutex found, skipping"
        continue
    fi

    # Check if it already has move constructor
    if grep -q "($classname&&" "$file"; then
        echo "  Already has move constructor, skipping"
        continue
    fi

    echo "  Adding move semantics..."

    # This is complex sed surgery - we'll handle it manually for each file
done

echo "Done. Please verify changes and rebuild."
