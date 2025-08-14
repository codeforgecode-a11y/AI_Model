#!/usr/bin/env python3
"""
Maximum Performance CPU Training Script
All monitoring and visualization disabled for maximum resource utilization
"""

import os
import sys
import warnings

# Disable all warnings for performance
warnings.filterwarnings("ignore")

# Disable transformers logging
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_dataset_trainer import MultiDatasetTrainer, TrainingConfig, DatasetConfig

def main():
    """Maximum performance training"""
    print("CPU Performance Training")
    print("All monitoring disabled")

    # Check datasets
    configs = []

    # Conversational
    conv_paths = [
        ("alpaca", "Datasets/Conversational_Datasets/alpaca_dataset/train.jsonl"),
        ("vicuna", "Datasets/Conversational_Datasets/vicuna_dataset/train.jsonl"),
        ("wizardlm", "Datasets/Conversational_Datasets/wizardlm_dataset/train.jsonl")
    ]

    for name, path in conv_paths:
        if os.path.exists(path):
            configs.append(DatasetConfig(
                name=f"conv_{name}",
                path=path,
                weight=0.4,
                max_samples=None  # Use full dataset
            ))

    # Programming
    prog_paths = [
        ("codealpaca", "Datasets/Programming_DataSets/codealpaca/data/code_alpaca_2k.json"),
        ("humaneval", "Datasets/Programming_DataSets/human-eval/data/example_problem.jsonl")
    ]

    for name, path in prog_paths:
        if os.path.exists(path):
            configs.append(DatasetConfig(
                name=f"prog_{name}",
                path=path,
                weight=0.4,
                max_samples=None
            ))

    # Cybersecurity
    cyber_path = "Datasets/CyberSecurity_DataSets/nvdcve/nvdcve"
    if os.path.exists(cyber_path) and os.listdir(cyber_path):
        configs.append(DatasetConfig(
            name="cyber_cve",
            path=cyber_path,
            weight=0.2,
            max_samples=None
        ))

    if not configs:
        print("No datasets found!")
        return 1

    print(f"Datasets: {len(configs)}")

    # Maximum performance configuration
    config = TrainingConfig(
        model_name="distilgpt2",
        max_length=256,
        batch_size=1,
        learning_rate=3e-5,
        num_epochs=3,
        warmup_steps=50,
        save_steps=500,
        eval_steps=1000,
        output_dir="./cpu_performance_model",
        gradient_accumulation_steps=32,
        fp16=False,
        dataloader_num_workers=0,
        cpu_only=True,
        memory_efficient=True,
        checkpoint_resume=True
    )

    try:
        # Initialize trainer
        trainer = MultiDatasetTrainer(config)

        # Add datasets
        for dataset_config in configs:
            trainer.add_dataset(dataset_config)

        # Start training
        print("Training started...")
        trainer.train()
        print("Training completed!")

        return 0

    except Exception as e:
        print(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())