#!/usr/bin/env python3
"""
Export Sentiment Model to ONNX Format

Exports the DistilRoBERTa financial sentiment model to ONNX format
for deployment in C++ applications using ONNX Runtime.

This is an optional step for Phase 2 (ONNX C++ integration).
Phase 1 uses the model directly in Python.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
Phase 5+: ML Sentiment Integration

Usage:
    python3 scripts/ml/export_sentiment_model.py
    python3 scripts/ml/export_sentiment_model.py --output models/sentiment.onnx
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def export_to_onnx(
    model_name: str,
    output_path: str,
    opset_version: int = 17,
    use_dynamo: bool = True
) -> None:
    """
    Export transformer model to ONNX format

    Args:
        model_name: Hugging Face model ID
        output_path: Output .onnx file path
        opset_version: ONNX opset version (default: 17 for latest features)
        use_dynamo: Use new PyTorch 2.x dynamo exporter (recommended)
    """
    logger.info(f"Exporting model: {model_name}")
    logger.info(f"Output path: {output_path}")

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Set model to eval mode
    model.eval()

    # Create dummy input for tracing
    dummy_text = "Company reports strong quarterly earnings"
    logger.info(f"Creating dummy input: '{dummy_text}'")

    dummy_inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Extract input_ids and attention_mask
    input_ids = dummy_inputs['input_ids']
    attention_mask = dummy_inputs['attention_mask']

    logger.info(f"Input shape: {input_ids.shape}")

    # Export to ONNX
    logger.info("Exporting to ONNX format...")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    try:
        if use_dynamo:
            # New PyTorch 2.x exporter (recommended)
            logger.info("Using torch.onnx.export with dynamo=True (PyTorch 2.x)")

            torch.onnx.export(
                model,
                args=(input_ids, attention_mask),
                f=output_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size'}
                },
                do_constant_folding=True,
                opset_version=opset_version,
                dynamo=True  # Use new exporter
            )
        else:
            # Legacy exporter (fallback)
            logger.info("Using legacy torch.onnx.export")

            torch.onnx.export(
                model,
                args=(input_ids, attention_mask),
                f=output_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size'}
                },
                do_constant_folding=True,
                opset_version=opset_version
            )

        logger.info(f"✓ Model exported successfully to {output_path}")

        # Print file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        logger.info(f"✓ ONNX model size: {file_size:.2f} MB")

    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        raise


def verify_onnx_model(model_path: str) -> None:
    """
    Verify exported ONNX model can be loaded

    Args:
        model_path: Path to .onnx file
    """
    try:
        import onnx
        import onnxruntime as ort

        logger.info("\nVerifying ONNX model...")

        # Load and check ONNX model
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✓ ONNX model is valid")

        # Test ONNX Runtime inference
        logger.info("Testing ONNX Runtime inference...")
        session = ort.InferenceSession(model_path)

        # Get input/output info
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]

        logger.info(f"  Input names: {input_names}")
        logger.info(f"  Output names: {output_names}")

        # Create dummy input
        import numpy as np
        dummy_input_ids = np.random.randint(0, 1000, size=(1, 128), dtype=np.int64)
        dummy_attention_mask = np.ones((1, 128), dtype=np.int64)

        # Run inference
        outputs = session.run(
            output_names,
            {
                'input_ids': dummy_input_ids,
                'attention_mask': dummy_attention_mask
            }
        )

        logger.info(f"✓ ONNX Runtime inference successful")
        logger.info(f"  Output shape: {outputs[0].shape}")

    except ImportError as e:
        logger.warning(f"Cannot verify ONNX model - missing dependencies: {e}")
        logger.warning("Install with: pip install onnx onnxruntime")
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
        raise


def export_tokenizer_config(model_name: str, output_dir: str) -> None:
    """
    Export tokenizer configuration for C++ implementation

    Args:
        model_name: Hugging Face model ID
        output_dir: Directory to save tokenizer config
    """
    logger.info("\nExporting tokenizer configuration...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save tokenizer files
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"✓ Tokenizer saved to {output_dir}")
    logger.info(f"  Files: vocab.json, merges.txt, tokenizer_config.json")

    # Print important tokenizer info
    logger.info("\nTokenizer Information:")
    logger.info(f"  Vocab size: {tokenizer.vocab_size}")
    logger.info(f"  Max length: {tokenizer.model_max_length}")
    logger.info(f"  Padding side: {tokenizer.padding_side}")
    logger.info(f"  Truncation side: {tokenizer.truncation_side}")
    logger.info(f"  Special tokens: {tokenizer.all_special_tokens}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Export sentiment model to ONNX format"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/sentiment_distilroberta.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="models/tokenizer",
        help="Directory to save tokenizer config"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--no-dynamo",
        action="store_true",
        help="Disable dynamo exporter (use legacy)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX model verification"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("ONNX Model Export for Financial Sentiment Analysis")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"Output: {args.output}")
    print(f"Opset version: {args.opset}")
    print(f"Use dynamo: {not args.no_dynamo}")

    # Export to ONNX
    try:
        export_to_onnx(
            model_name=args.model,
            output_path=args.output,
            opset_version=args.opset,
            use_dynamo=not args.no_dynamo
        )

        # Verify ONNX model
        if not args.no_verify:
            verify_onnx_model(args.output)

        # Export tokenizer config
        export_tokenizer_config(args.model, args.tokenizer_dir)

        print("\n" + "=" * 70)
        print("EXPORT COMPLETE")
        print("=" * 70)
        print(f"\nONNX model: {args.output}")
        print(f"Tokenizer config: {args.tokenizer_dir}")
        print("\nNext steps:")
        print("  1. Verify model with: python3 scripts/ml/test_onnx_inference.py")
        print("  2. Integrate into C++ using ONNX Runtime C++ API")
        print("  3. See docs/ML_SENTIMENT_ANALYSIS_PROPOSAL.md for details")
        print("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"\nExport failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
