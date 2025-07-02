#!/usr/bin/env python3
"""
Quick Start Script for Research Paper Analysis
Uses a very small model for fast training and evaluation.
"""

import os
import sys
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the pipeline with optimized settings for quick results."""
    
    print("🚀 Quick Start: Research Paper Analysis Pipeline")
    print("=" * 60)
    print("Using ultra-optimized settings for fast training:")
    print("• Model: distilbert-base-uncased (66M parameters)")
    print("• Epochs: 1")
    print("• Batch Size: 2")
    print("• Max Length: 128")
    print("• Device: CPU (forced)")
    print("=" * 60)
    
    # Set environment variables to force CPU usage
    env = os.environ.copy()
    env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = ''
    
    # Command with ultra-optimized settings
    cmd = [
        sys.executable, "train_pipeline.py",
        "--model", "distilbert-base-uncased",
        "--epochs", "1",
        "--batch-size", "2",
        "--max-length", "128",
        "--lora-r", "2",
        "--learning-rate", "2e-5"
    ]
    
    try:
        print("Starting pipeline...")
        result = subprocess.run(cmd, check=True, capture_output=False, env=env)
        print("\n✅ Pipeline completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main() 