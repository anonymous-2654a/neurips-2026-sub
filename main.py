#!/usr/bin/env python3
"""
OmniGuard Main Inference Script

This script provides simple inference capabilities for safety evaluation
on BeaverTails and VLGuard datasets.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import OmniGuardEvaluator
from src.safety_datasets import BeaverTails, VLGuard
from src.utils import LLAMA_GUARD_3_CATEGORY
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OmniGuard Safety Evaluation - Main Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference on BeaverTails dataset
  python main.py --dataset beavertails --model-path /path/to/model --num-samples 100
  
  # Inference on VLGuard dataset
  python main.py --dataset vlguard --model-path /path/to/model --num-samples 50
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["beavertails", "vlguard"],
        help="Dataset to run inference on (beavertails or vlguard)"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the OmniGuard model or HuggingFace model ID"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all samples)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="30k_test",
        help="Dataset split for BeaverTails (30k_test, 30k_train, 330k_test, 330k_train)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results (optional)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu, default: auto-detect)"
    )
    
    return parser.parse_args()


def inference_beavertails(model_path, split="30k_test", num_samples=None, output_file=None, device=None):
    """
    Run inference on BeaverTails dataset.
    
    Args:
        model_path: Path to the model
        split: Dataset split to use
        num_samples: Number of samples to evaluate (None = all)
        output_file: Path to save results
        device: Device to run on
    """
    print("=" * 70)
    print("OmniGuard Inference - BeaverTails Dataset")
    print("=" * 70)
    
    # Initialize evaluator
    print(f"\n[1/3] Loading model from: {model_path}")
    evaluator = OmniGuardEvaluator(model_id=model_path, device=device)
    print("✓ Model loaded successfully")
    
    # Load dataset
    print(f"\n[2/3] Loading BeaverTails dataset (split: {split})")
    dataset = BeaverTails(split=split)
    total_samples = len(dataset)
    num_eval = num_samples if num_samples else total_samples
    print(f"✓ Dataset loaded: {total_samples} samples")
    print(f"✓ Will process: {num_eval} samples")
    
    # Run inference
    print(f"\n[3/3] Running inference...")
    results = []
    
    for i, sample in enumerate(tqdm(dataset, total=num_eval, desc="Processing")):
        if num_samples and i >= num_samples:
            break
        
        # Perform evaluation
        result = evaluator.evaluate(
            prompt_1=sample["prompt_1"],
            prompt_2=sample["prompt_2"],
            categories=LLAMA_GUARD_3_CATEGORY
        )
        
        # Store result
        results.append({
            "id": sample["id"],
            "prompt": sample["prompt_1"][:100],
            "response": sample["prompt_2"][:100],
            "result": result
        })
    
    # Print results
    print(f"\n[4/4] Results Summary")
    print("-" * 70)
    print(f"Total Processed: {len(results)}")
    print("-" * 70)
    
    # Save to file if specified
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_file}")
    
    print("\n✓ Inference completed successfully!\n")
    return results


def inference_vlguard(model_path, num_samples=None, output_file=None, device=None):
    """
    Run inference on VLGuard dataset.
    
    Args:
        model_path: Path to the model
        num_samples: Number of samples to evaluate (None = all)
        output_file: Path to save results
        device: Device to run on
    """
    print("=" * 70)
    print("OmniGuard Inference - VLGuard Dataset")
    print("=" * 70)
    
    # Initialize evaluator
    print(f"\n[1/3] Loading model from: {model_path}")
    evaluator = OmniGuardEvaluator(model_id=model_path, device=device)
    print("✓ Model loaded successfully")
    
    # Load dataset
    print(f"\n[2/3] Loading VLGuard dataset (with images)")
    print("Note: This will download the dataset if not already cached")
    dataset = VLGuard()
    total_samples = len(dataset)
    num_eval = num_samples if num_samples else total_samples
    print(f"✓ Dataset loaded: {total_samples} samples")
    print(f"✓ Will process: {num_eval} samples")
    
    # Run inference
    print(f"\n[3/3] Running inference...")
    results = []
    
    for i, sample in enumerate(tqdm(dataset, total=num_eval, desc="Processing")):
        if num_samples and i >= num_samples:
            break
        
        # Perform evaluation
        result = evaluator.evaluate(
            prompt_1=sample["prompt_1"],
            prompt_2=sample["prompt_2"],
            categories=LLAMA_GUARD_3_CATEGORY,
            media=sample.get("media"),
            media_type=sample.get("media_type")
        )
        
        # Store result
        results.append({
            "id": sample["id"],
            "media": sample.get("media"),
            "media_type": sample.get("media_type"),
            "result": result
        })
    
    # Print results
    print(f"\n[4/4] Results Summary")
    print("-" * 70)
    print(f"Total Processed: {len(results)}")
    print("-" * 70)
    
    # Save to file if specified
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_file}")
    
    print("\n✓ Inference completed successfully!\n")
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        if args.dataset == "beavertails":
            inference_beavertails(
                model_path=args.model_path,
                split=args.split,
                num_samples=args.num_samples,
                output_file=args.output,
                device=args.device
            )
        elif args.dataset == "vlguard":
            inference_vlguard(
                model_path=args.model_path,
                num_samples=args.num_samples,
                output_file=args.output,
                device=args.device
            )
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
