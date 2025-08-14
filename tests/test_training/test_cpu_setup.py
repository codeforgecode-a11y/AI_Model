#!/usr/bin/env python3
"""
CPU Training Setup Test
Verifies CPU-optimized training setup and provides recommendations
"""

import os
import sys
import torch
import psutil
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

def test_cpu_setup():
    """Test CPU training setup"""
    logger = setup_logging()

    logger.info("🖥️  CPU Training Setup Test")
    logger.info("=" * 50)

    # Test PyTorch CPU setup
    logger.info("1. Testing PyTorch CPU setup...")
    try:
        # Check PyTorch version
        logger.info(f"✓ PyTorch version: {torch.__version__}")

        # Check CPU availability
        if torch.cuda.is_available():
            logger.warning("⚠ CUDA is available but we'll use CPU only")
        else:
            logger.info("✓ CUDA not available - perfect for CPU training")

        # Test CPU tensor operations
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)
        z = torch.mm(x, y)
        logger.info("✓ CPU tensor operations working")

        # Test CPU threading
        logger.info(f"✓ PyTorch threads: {torch.get_num_threads()}")

    except Exception as e:
        logger.error(f"✗ PyTorch CPU test failed: {e}")
        return False

    # Test system resources
    logger.info("\n2. Testing system resources...")
    try:
        # CPU information
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        logger.info(f"✓ CPU cores: {cpu_count} logical cores")
        if cpu_freq:
            logger.info(f"✓ CPU frequency: {cpu_freq.current:.0f} MHz")

        # Memory information
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        logger.info(f"✓ Total RAM: {memory_gb:.1f} GB")
        logger.info(f"✓ Available RAM: {memory.available / (1024**3):.1f} GB")

        if memory_gb < 8:
            logger.warning("⚠ Less than 8GB RAM - consider reducing batch size")
        elif memory_gb >= 16:
            logger.info("✓ Sufficient RAM for training")
        else:
            logger.info("✓ Adequate RAM for training")

    except Exception as e:
        logger.error(f"✗ System resource test failed: {e}")
        return False

    # Test dataset availability
    logger.info("\n3. Testing dataset availability...")
    dataset_paths = [
        ("Conversational - Alpaca", "Datasets/Conversational_Datasets/alpaca_dataset/train.jsonl"),
        ("Programming - CodeAlpaca", "Datasets/Programming_DataSets/codealpaca/data/code_alpaca_2k.json"),
        ("Cybersecurity - CVE", "Datasets/CyberSecurity_DataSets/nvdcve/nvdcve")
    ]

    available_datasets = 0
    for name, path in dataset_paths:
        if os.path.exists(path):
            logger.info(f"✓ {name}: Found")
            available_datasets += 1
        else:
            logger.warning(f"⚠ {name}: Not found at {path}")

    if available_datasets == 0:
        logger.error("✗ No datasets found!")
        return False
    elif available_datasets < 3:
        logger.warning(f"⚠ Only {available_datasets}/3 datasets found")
    else:
        logger.info("✓ All datasets found")

    # Test model loading
    logger.info("\n4. Testing model loading...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = "distilgpt2"
        logger.info(f"Loading {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Force to CPU
        model = model.to('cpu')

        # Test tokenization
        text = "Hello, this is a test."
        tokens = tokenizer(text, return_tensors='pt')

        # Test forward pass
        with torch.no_grad():
            outputs = model(**tokens)

        logger.info("✓ Model loading and inference test passed")

        # Memory usage after model loading
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"✓ Memory usage with model: {memory_mb:.1f} MB")

    except Exception as e:
        logger.error(f"✗ Model loading test failed: {e}")
        return False

    # Provide recommendations
    logger.info("\n5. Recommendations for your system:")

    if cpu_count >= 4:
        logger.info("✓ Good CPU thread count for training")
    else:
        logger.warning("⚠ Limited CPU threads - training will be slower")

    if memory_gb >= 16:
        logger.info("✓ Excellent RAM for CPU training")
        logger.info("  Recommended: Use default settings")
    elif memory_gb >= 8:
        logger.info("✓ Adequate RAM for CPU training")
        logger.info("  Recommended: Use default settings, monitor memory")
    else:
        logger.warning("⚠ Limited RAM")
        logger.info("  Recommended: --batch-size 1 --max-length 128")

    # Training time estimates
    logger.info("\n6. Training time estimates:")
    if available_datasets >= 3:
        logger.info("  Full training (all datasets): 24-48 hours")
        logger.info("  Test run (1000 samples): 2-4 hours")
    else:
        logger.info("  Partial training: 8-24 hours")

    logger.info("\n🎉 CPU setup test completed!")
    logger.info("\nNext steps:")
    logger.info("1. Install dependencies: pip install -r cpu_training_requirements.txt")
    logger.info("2. Start test run: python train_model.py --max-samples 1000 --epochs 1")
    logger.info("3. Monitor training: python monitor_cpu_training.py")

    return True

def main():
    """Main test function"""
    success = test_cpu_setup()

    if success:
        print("\n✅ CPU training setup is ready!")
        return 0
    else:
        print("\n❌ CPU training setup has issues. Please fix them before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())