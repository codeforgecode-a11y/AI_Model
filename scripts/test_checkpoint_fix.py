#!/usr/bin/env python3
"""
Test script to verify checkpoint saving fix
"""

import os
import sys
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_checkpoint_fix():
    """Test the checkpoint saving fix"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        from multi_dataset_trainer import CPUOptimizedTrainer, TrainingConfig
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

        logger.info("Testing checkpoint saving fix...")

        # Create minimal setup
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create dummy dataset
        class DummyDataset:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                text = f"Test sample {idx}"
                encoding = tokenizer(text, truncation=True, padding='max_length',
                                   max_length=32, return_tensors='pt')
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                    'labels': encoding['input_ids'].squeeze()
                }

        # Create training arguments
        training_args = TrainingArguments(
            output_dir="./test_checkpoint_fix",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_steps=5,
            logging_steps=2,
            save_total_limit=2,
        )

        # Create trainer
        trainer = CPUOptimizedTrainer(
            model=model,
            args=training_args,
            train_dataset=DummyDataset(),
            tokenizer=tokenizer,
        )

        logger.info("✓ CPUOptimizedTrainer created successfully")

        # Test checkpoint saving directly
        try:
            test_metrics = {"eval_loss": 2.5, "eval_accuracy": 0.8}
            result = trainer._save_checkpoint(model, None, test_metrics)
            logger.info("✓ Checkpoint saving test successful")
            return True
        except Exception as e:
            logger.error(f"✗ Checkpoint saving test failed: {e}")
            return False

    except Exception as e:
        logger.error(f"✗ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = test_checkpoint_fix()
    if success:
        print("\n✅ Checkpoint saving fix test passed!")
        sys.exit(0)
    else:
        print("\n❌ Checkpoint saving fix test failed!")
        sys.exit(1)