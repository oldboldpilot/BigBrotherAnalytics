#!/usr/bin/env python3
"""
BigBrotherAnalytics - Export DistilRoBERTa Sentiment Model to ONNX

Exports the DistilRoBERTa-financial sentiment model to ONNX format for
fast C++ inference with ONNX Runtime.

Model: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
Performance: 98.23% accuracy on financial news sentiment
Architecture: 6 layers, 768 dimensions, 82M parameters

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def export_model_to_onnx(
    model_name: str = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    output_dir: str = "models/sentiment",
    max_seq_length: int = 512,
) -> None:
    """
    Export DistilRoBERTa sentiment model to ONNX format.

    Args:
        model_name: HuggingFace model identifier
        output_dir: Directory to save ONNX model
        max_seq_length: Maximum sequence length for tokenization
    """
    print("=" * 70)
    print("DistilRoBERTa Sentiment Model - ONNX Export")
    print("=" * 70)
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_name}")
    print(f"Output Directory: {output_path.absolute()}")
    print(f"Max Sequence Length: {max_seq_length}")
    print()

    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Save tokenizer
    tokenizer_path = output_path / "tokenizer"
    print(f"Saving tokenizer to: {tokenizer_path}")
    tokenizer.save_pretrained(tokenizer_path)

    # Create dummy input for export
    print("Creating dummy input for ONNX export...")
    dummy_text = "The company reported strong earnings growth this quarter."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
    )

    # Export to ONNX
    onnx_path = output_path / "sentiment_model.onnx"
    print(f"Exporting model to ONNX: {onnx_path}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size"},
            },
            opset_version=17,  # ONNX opset 17 for best compatibility
            do_constant_folding=True,
        )

    print()
    print("✓ ONNX export complete!")
    print()

    # Verify ONNX model
    print("Verifying ONNX model...")
    import onnx

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verification passed!")
    print()

    # Test inference with ONNX Runtime
    print("Testing ONNX Runtime inference...")
    import onnxruntime as ort

    # Try CUDA provider first, fall back to CPU
    providers = ["CPUExecutionProvider"]
    if torch.cuda.is_available():
        providers.insert(0, "CUDAExecutionProvider")
        print(f"CUDA available: Using GPU acceleration")
    else:
        print(f"CUDA not available: Using CPU")

    session = ort.InferenceSession(str(onnx_path), providers=providers)

    # Run inference
    ort_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    }
    ort_outputs = session.run(None, ort_inputs)
    logits = ort_outputs[0]

    # Convert logits to sentiment
    import numpy as np

    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()[0]
    labels = ["negative", "neutral", "positive"]
    predicted_class = np.argmax(probs)
    predicted_label = labels[predicted_class]
    confidence = probs[predicted_class]

    print(f"Test Input: '{dummy_text}'")
    print(f"Predicted Sentiment: {predicted_label} (confidence: {confidence:.4f})")
    print(f"Probabilities: negative={probs[0]:.4f}, neutral={probs[1]:.4f}, positive={probs[2]:.4f}")
    print()

    # Save model info
    info_path = output_path / "model_info.txt"
    with open(info_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Architecture: DistilRoBERTa (6 layers, 768 dimensions, 82M parameters)\n")
        f.write(f"Accuracy: 98.23% on financial news sentiment\n")
        f.write(f"Max Sequence Length: {max_seq_length}\n")
        f.write(f"ONNX Opset Version: 17\n")
        f.write(f"Vocabulary Size: {tokenizer.vocab_size}\n")
        f.write(f"Labels: {labels}\n")
        f.write(f"\nInput Names: input_ids, attention_mask\n")
        f.write(f"Output Names: logits\n")
        f.write(f"\nLabel Mapping:\n")
        f.write(f"  0: negative\n")
        f.write(f"  1: neutral\n")
        f.write(f"  2: positive\n")
        f.write(f"\nSentiment Score Conversion:\n")
        f.write(f"  negative: -1.0 to -0.33\n")
        f.write(f"  neutral:  -0.33 to 0.33\n")
        f.write(f"  positive:  0.33 to 1.0\n")

    print(f"Model info saved to: {info_path}")
    print()

    # Display model size
    model_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print("=" * 70)
    print("Export Summary")
    print("=" * 70)
    print(f"ONNX Model: {onnx_path}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Model Info: {info_path}")
    print()
    print("✓ Ready for C++ integration with ONNX Runtime!")
    print()
    print("Next Steps:")
    print("  1. Implement WordPiece tokenizer in C++")
    print("  2. Create ml_sentiment_analyzer.cppm module")
    print("  3. Integrate with news_ingestion system")
    print("=" * 70)


if __name__ == "__main__":
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    models_dir = project_root / "models" / "sentiment"

    print(f"Project Root: {project_root}")
    print()

    export_model_to_onnx(
        model_name="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        output_dir=str(models_dir),
        max_seq_length=512,
    )
