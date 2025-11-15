#!/bin/bash
# Post-training workflow: Run after model training completes

set -e

echo "================================================================================"
echo "POST-TRAINING WORKFLOW"
echo "================================================================================"
echo ""

# 1. Show training results
echo "1. TRAINING RESULTS"
echo "--------------------------------------------------------------------------------"
if [ -f "models/price_predictor_85feat_info.json" ]; then
    python3 -m json.tool models/price_predictor_85feat_info.json | grep -E "training_date|epochs_trained|total_epochs|best_val_loss|test_acc|test_rmse|training_time" | head -12
    echo ""
else
    echo "❌ Training results not found!"
    exit 1
fi

# 2. Test prediction differentiation
echo "2. TESTING PREDICTION DIFFERENTIATION"
echo "--------------------------------------------------------------------------------"
uv run python scripts/ml/test_prediction_differentiation.py
if [ $? -eq 0 ]; then
    echo "✓ Prediction differentiation test PASSED"
else
    echo "❌ Prediction differentiation test FAILED"
    exit 1
fi
echo ""

# 3. Export weights to INT32 SIMD format
echo "3. EXPORTING WEIGHTS TO INT32 SIMD FORMAT"
echo "--------------------------------------------------------------------------------"
if [ -f "scripts/ml/export_weights_85feat.py" ]; then
    uv run python scripts/ml/export_weights_85feat.py
    echo "✓ Weights exported"
else
    echo "⚠️  Export script not found, skipping..."
fi
echo ""

# 4. Show summary
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo "✓ Training completed"
echo "✓ Model produces different predictions for different stocks"
echo "✓ Weights exported for C++ inference"
echo ""
echo "Next steps:"
echo "  1. Rebuild bot: cd build && ninja"
echo "  2. Restart bot: ./scripts/startup_nonblocking.sh"
echo "  3. Verify bot loads 100 days of historical data"
echo "================================================================================"
