#!/usr/bin/env python3
"""
Test script to verify the training setup
"""

import os
import sys
import json
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

def check_dependencies():
    """Check if required packages are installed"""
    logger = logging.getLogger(__name__)

    required_packages = [
        'torch',
        'transformers',
        'numpy',
        'tqdm'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} is missing")

    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Install with: pip install -r ml_training_requirements.txt")
        return False

    return True

def check_datasets():
    """Check if datasets are available"""
    logger = logging.getLogger(__name__)

    dataset_paths = [
        ("Conversational - Alpaca", "Datasets/Conversational_Datasets/alpaca_dataset/train.jsonl"),
        ("Conversational - Vicuna", "Datasets/Conversational_Datasets/vicuna_dataset/train.jsonl"),
        ("Conversational - WizardLM", "Datasets/Conversational_Datasets/wizardlm_dataset/train.jsonl"),
        ("Programming - CodeAlpaca", "Datasets/Programming_DataSets/codealpaca/data/code_alpaca_2k.json"),
        ("Programming - HumanEval", "Datasets/Programming_DataSets/human-eval/data/example_problem.jsonl"),
        ("Cybersecurity - CVE", "Datasets/CyberSecurity_DataSets/nvdcve/nvdcve")
    ]

    available_datasets = []

    for name, path in dataset_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                # Check if directory has JSON files
                json_files = list(Path(path).glob("*.json"))
                if json_files:
                    logger.info(f"✓ {name}: {len(json_files)} files")
                    available_datasets.append((name, path))
                else:
                    logger.warning(f"⚠ {name}: Directory exists but no JSON files found")
            else:
                # Check file size
                size = os.path.getsize(path)
                logger.info(f"✓ {name}: {size/1024/1024:.1f} MB")
                available_datasets.append((name, path))
        else:
            logger.warning(f"✗ {name}: Not found at {path}")

    if not available_datasets:
        logger.error("No datasets found! Please check your dataset paths.")
        return False

    logger.info(f"Found {len(available_datasets)} available datasets")
    return True

def check_sample_data():
    """Check sample data format"""
    logger = logging.getLogger(__name__)

    # Check conversational data format
    alpaca_path = "Datasets/Conversational_Datasets/alpaca_dataset/train.jsonl"
    if os.path.exists(alpaca_path):
        try:
            with open(alpaca_path, 'r') as f:
                first_line = f.readline()
                sample = json.loads(first_line)

                required_fields = ['instruction', 'output']
                if all(field in sample for field in required_fields):
                    logger.info("✓ Conversational data format is correct")
                else:
                    logger.warning(f"⚠ Conversational data missing fields: {required_fields}")
        except Exception as e:
            logger.error(f"✗ Error reading conversational data: {e}")

    # Check programming data format
    code_path = "Datasets/Programming_DataSets/codealpaca/data/code_alpaca_2k.json"
    if os.path.exists(code_path):
        try:
            with open(code_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    sample = data[0]
                    if 'instruction' in sample and 'output' in sample:
                        logger.info("✓ Programming data format is correct")
                    else:
                        logger.warning("⚠ Programming data format may be incorrect")
        except Exception as e:
            logger.error(f"✗ Error reading programming data: {e}")

    # Check cybersecurity data format
    cve_path = "Datasets/CyberSecurity_DataSets/nvdcve/nvdcve"
    if os.path.exists(cve_path):
        json_files = list(Path(cve_path).glob("*.json"))
        if json_files:
            try:
                with open(json_files[0], 'r') as f:
                    data = json.load(f)
                    if 'cve' in data:
                        logger.info("✓ Cybersecurity data format is correct")
                    else:
                        logger.warning("⚠ Cybersecurity data format may be incorrect")
            except Exception as e:
                logger.error(f"✗ Error reading cybersecurity data: {e}")

def check_gpu():
    """Check GPU availability"""
    logger = logging.getLogger(__name__)

    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

            logger.info(f"✓ GPU available: {gpu_name}")
            logger.info(f"✓ GPU memory: {gpu_memory:.1f} GB")
            logger.info(f"✓ GPU count: {gpu_count}")

            if gpu_memory < 8:
                logger.warning("⚠ GPU memory < 8GB. Consider reducing batch size.")

            return True
        else:
            logger.warning("⚠ No GPU available. Training will use CPU (very slow).")
            return False
    except ImportError:
        logger.error("✗ PyTorch not installed")
        return False

def main():
    """Main test function"""
    logger = setup_logging()

    logger.info("=== Multi-Dataset Training Setup Test ===")

    # Check dependencies
    logger.info("\n1. Checking dependencies...")
    deps_ok = check_dependencies()

    # Check GPU
    logger.info("\n2. Checking GPU...")
    gpu_ok = check_gpu()

    # Check datasets
    logger.info("\n3. Checking datasets...")
    datasets_ok = check_datasets()

    # Check data format
    logger.info("\n4. Checking data formats...")
    check_sample_data()

    # Summary
    logger.info("\n=== Summary ===")

    if deps_ok:
        logger.info("✓ Dependencies: OK")
    else:
        logger.error("✗ Dependencies: Missing packages")

    if gpu_ok:
        logger.info("✓ GPU: Available")
    else:
        logger.warning("⚠ GPU: Not available or insufficient memory")

    if datasets_ok:
        logger.info("✓ Datasets: Found")
    else:
        logger.error("✗ Datasets: Missing or incomplete")

    if deps_ok and datasets_ok:
        logger.info("\n🎉 Setup looks good! You can start training with:")
        logger.info("   python train_model.py")

        if not gpu_ok:
            logger.info("\n⚠ Note: Training without GPU will be very slow.")
            logger.info("   Consider using Google Colab or a cloud GPU instance.")
    else:
        logger.error("\n❌ Setup incomplete. Please fix the issues above.")

    return deps_ok and datasets_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)