#!/usr/bin/env python3
"""
Simple training script for multi-dataset model
Usage: python train_model.py
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_dataset_trainer import MultiDatasetTrainer, TrainingConfig, DatasetConfig

def setup_logging():
    """Setup detailed logging for training progress monitoring"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler('training_progress.log'),
            logging.StreamHandler()
        ]
    )

def check_datasets(max_samples=None):
    """Check if datasets exist and return valid configurations"""
    configs = []

    # Check conversational datasets
    conv_paths = [
        ("alpaca", "Datasets/Conversational_Datasets/alpaca_dataset/train.jsonl"),
        ("vicuna", "Datasets/Conversational_Datasets/vicuna_dataset/train.jsonl"),
        ("wizardlm", "Datasets/Conversational_Datasets/wizardlm_dataset/train.jsonl")
    ]

    for name, path in conv_paths:
        if os.path.exists(path):
            sample_limit = max_samples if max_samples else 5000  # Full dataset unless limited
            configs.append(DatasetConfig(
                name=f"conversational_{name}",
                path=path,
                weight=0.4,
                max_samples=sample_limit
            ))
            print(f"✓ Found conversational dataset: {name}")

    # Check programming datasets
    prog_paths = [
        ("codealpaca", "Datasets/Programming_DataSets/codealpaca/data/code_alpaca_2k.json"),
        ("humaneval", "Datasets/Programming_DataSets/human-eval/data/example_problem.jsonl")
    ]

    for name, path in prog_paths:
        if os.path.exists(path):
            sample_limit = max_samples if max_samples else 2000
            configs.append(DatasetConfig(
                name=f"programming_{name}",
                path=path,
                weight=0.4,
                max_samples=sample_limit
            ))
            print(f"✓ Found programming dataset: {name}")

    # Check cybersecurity datasets
    cyber_path = "Datasets/CyberSecurity_DataSets/nvdcve/nvdcve"
    if os.path.exists(cyber_path) and os.listdir(cyber_path):
        sample_limit = max_samples if max_samples else 1000
        configs.append(DatasetConfig(
            name="cybersecurity_cve",
            path=cyber_path,
            weight=0.2,
            max_samples=sample_limit
        ))
        print("✓ Found cybersecurity dataset: CVE data")

    return configs

def main():
    """Main CPU training function with detailed progress logging"""
    parser = argparse.ArgumentParser(description="Train multi-dataset model on CPU with detailed logging")
    parser.add_argument("--model", default="distilgpt2", help="Base model name (CPU-optimized)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (CPU-optimized)")
    parser.add_argument("--output-dir", default="./trained_multi_model_cpu", help="Output directory")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length (CPU-optimized)")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--gradient-accumulation", type=int, default=32, help="Gradient accumulation steps")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per dataset for testing")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log progress every N steps")

    args = parser.parse_args()

    # Setup detailed logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting CPU training with detailed progress monitoring")
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation}")
    logger.info(f"Logging frequency: every {args.logging_steps} steps")
    logger.info(f"Output directory: {args.output_dir}")

    # Check available datasets
    logger.info("Checking available datasets...")
    dataset_configs = check_datasets(args.max_samples)

    if not dataset_configs:
        logger.error("No valid datasets found! Please check your dataset paths.")
        return 1

    logger.info(f"Found {len(dataset_configs)} valid datasets")

    # Detailed logging configuration
    config = TrainingConfig(
        model_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        warmup_steps=50,
        save_steps=250,   # Frequent saves for long training
        eval_steps=125,   # Regular evaluation
        logging_steps=args.logging_steps,  # Configurable logging frequency
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation,
        fp16=False,
        dataloader_num_workers=2,  # Some parallel loading for efficiency
        cpu_only=True,
        memory_efficient=True,
        checkpoint_resume=args.resume
    )

    try:
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = MultiDatasetTrainer(config)

        # Add datasets
        for dataset_config in dataset_configs:
            logger.info(f"Loading dataset: {dataset_config.name}")
            trainer.add_dataset(dataset_config)

        # Start training
        logger.info("🎯 Starting CPU training with detailed progress monitoring...")
        logger.info("📊 Progress will be logged every {} steps".format(args.logging_steps))
        logger.info("💾 Checkpoints will be saved every {} steps".format(config.save_steps))
        logger.info("📈 Evaluation will run every {} steps".format(config.eval_steps))
        logger.info("⏱️ Estimated training time: 24-48 hours")
        logger.info("🔄 Training can be interrupted with Ctrl+C and resumed later")

        trainer.train()

        logger.info("🎉 Training completed successfully!")
        logger.info(f"💾 Model saved to: {args.output_dir}")

        return 0

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())