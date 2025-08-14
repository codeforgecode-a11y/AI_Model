#!/usr/bin/env python3
"""
CPU Training with Configurable Detailed Logging
Provides real-time visibility into training progress
"""

import os
import sys
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Training with different logging configurations"""
    parser = argparse.ArgumentParser(description="CPU training with detailed logging options")
    parser.add_argument("--logging-mode", choices=["detailed", "moderate", "minimal"],
                       default="detailed", help="Logging verbosity level")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for testing")

    args = parser.parse_args()

    # Configure logging frequency based on mode
    if args.logging_mode == "detailed":
        logging_steps = 5   # Log every 5 steps (very detailed)
        print("🔍 Detailed logging mode: Every 5 steps")
    elif args.logging_mode == "moderate":
        logging_steps = 25  # Log every 25 steps (moderate)
        print("📊 Moderate logging mode: Every 25 steps")
    else:  # minimal
        logging_steps = 100 # Log every 100 steps (minimal)
        print("📈 Minimal logging mode: Every 100 steps")

    # Import and run training
    from train_model import main as train_main

    # Override sys.argv to pass arguments to train_model
    original_argv = sys.argv
    sys.argv = [
        "train_model.py",
        "--logging-steps", str(logging_steps),
        "--batch-size", "1",
        "--gradient-accumulation", "32",
        "--epochs", "1" if args.max_samples else "3",
    ]

    if args.max_samples:
        sys.argv.extend(["--max-samples", str(args.max_samples)])

    try:
        result = train_main()
        return result
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    print("🖥️ CPU Training with Detailed Progress Logging")
    print("=" * 50)
    exit(main())