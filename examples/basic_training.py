#!/usr/bin/env python3
"""
Basic Training Example for Masked Critique Fine-tuning

This script demonstrates how to train a model using the Masked Critique Fine-tuning approach.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import btsft
sys.path.append(str(Path(__file__).parent.parent))

from btsft.func.training import train


def main():
    """Run basic training example."""
    
    print("🚀 Starting Masked Critique Fine-tuning Training Example")
    print("=" * 60)
    
    # Training configuration
    config = {
        "model_name": "deepscaler_fullft_all_updated",
        "checkpoint": "agentica-org/DeepScaleR-1.5B-Preview",
        "tokenizer_name": "agentica-org/DeepScaleR-1.5B-Preview",
        
        # Dataset configuration
        "dataset_train": "data/sanitized_data_v4.csv",
        
        # Training parameters
        "epochs": 3,
        "batch_size": 1,
        "accumulation_iter": 8,  # Effective batch size = 8
        "lr": 5e-5,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "max_length": 16384,
        
        # Masked Critique configuration
        "threshold": 0.2,        # Masking threshold for critique tokens
        "bf_beta": 0.05,        # Reward function influence weight
        
        # Output directories
        "logging_dir": "./logs/example_training",
        "output_dir": "./results/example_training",
        
        # Training settings
        "save_steps": 128,
        "device": "cuda",
        "num_workers": 4,
        "seed": 42,
    }
    
    print("📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print("\n🔧 Starting training...")
    
    try:
        # Start training
        train(**config)
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        print("💡 Make sure you have:")
        print("   - GPU with CUDA support")
        print("   - Sufficient memory (at least 24GB VRAM)")
        print("   - All dependencies installed")
        print("   - Data file available at data/sanitized_data_v4.csv")


if __name__ == "__main__":
    main()
