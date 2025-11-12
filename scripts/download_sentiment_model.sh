#!/usr/bin/env bash
# Download Sentiment Model Data
#
# This script downloads the pre-trained sentiment analysis model files
# that are too large to store in GitHub.

set -euo pipefail

MODELS_DIR="models/sentiment"
MODEL_URL="${SENTIMENT_MODEL_URL:-https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main}"

echo "========================================="
echo "Downloading Sentiment Model Files"
echo "========================================="

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

echo ""
echo "📁 Target directory: $MODELS_DIR"
echo "🌐 Source: $MODEL_URL"
echo ""

# Download model files
download_file() {
    local filename="$1"
    local url="$2"
    local output_path="$MODELS_DIR/$filename"

    if [ -f "$output_path" ]; then
        echo "✅ $filename already exists ($(du -h "$output_path" | cut -f1))"
        return 0
    fi

    echo "⬇️  Downloading $filename..."

    if command -v curl &> /dev/null; then
        curl -L -o "$output_path" "$url/$filename"
    elif command -v wget &> /dev/null; then
        wget -O "$output_path" "$url/$filename"
    else
        echo "❌ Error: Neither curl nor wget found. Please install one of them."
        exit 1
    fi

    echo "✅ Downloaded $filename ($(du -h "$output_path" | cut -f1))"
}

# Download each model file
# Note: Update these URLs to match your actual model hosting location
download_file "sentiment_model.onnx" "$MODEL_URL"
download_file "sentiment_model.onnx.data" "$MODEL_URL"
download_file "config.json" "$MODEL_URL"

echo ""
echo "========================================="
echo "✅ Sentiment Model Download Complete!"
echo "========================================="
echo ""
echo "Model files are located in: $MODELS_DIR"
echo "Total size: $(du -sh $MODELS_DIR | cut -f1)"
echo ""
echo "These files are ignored by git (.gitignore)"
echo "Re-run this script after cloning the repository."
