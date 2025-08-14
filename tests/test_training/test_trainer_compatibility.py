#!/usr/bin/env python3
"""
Test script to verify Trainer method compatibility
"""

import sys
import inspect
import logging
import torch

def test_trainer_compatibility():
    """Test Trainer method signatures for compatibility"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        from transformers import Trainer, TrainingArguments
        logger.info("✓ Successfully imported Trainer and TrainingArguments")

        # Check training_step method signature
        trainer_signature = inspect.signature(Trainer.training_step)
        params = list(trainer_signature.parameters.keys())

        logger.info(f"Trainer.training_step parameters: {params}")

        # Expected parameters in different versions
        if len(params) == 2:  # model, inputs
            logger.info("✓ Found 2-parameter training_step (older version)")
        elif len(params) == 3:  # model, inputs, num_items_in_batch
            logger.info("✓ Found 3-parameter training_step (newer version)")
        else:
            logger.warning(f"⚠ Unexpected number of parameters: {len(params)}")

        # Test creating a minimal trainer
        training_args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_steps=100,
        )

        # Create a dummy model and dataset for testing
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create dummy dataset
        class DummyDataset:
            def __init__(self, tokenizer, size=10):
                self.tokenizer = tokenizer
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                text = f"This is test sample {idx}"
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=64,
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                    'labels': encoding['input_ids'].squeeze()
                }

        dummy_dataset = DummyDataset(tokenizer)

        # Test creating trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dummy_dataset,
            tokenizer=tokenizer,
        )

        logger.info("✓ Successfully created Trainer instance")

        # Test training_step method call
        sample_data = dummy_dataset[0]
        inputs = {k: v.unsqueeze(0) for k, v in sample_data.items()}

        # Test the method signature
        try:
            if len(params) == 2:
                loss = trainer.training_step(model, inputs)
            else:
                loss = trainer.training_step(model, inputs, 1)
            logger.info("✓ training_step method call successful")
        except Exception as e:
            logger.error(f"✗ training_step method call failed: {e}")
            return False

        return True

    except Exception as e:
        logger.error(f"✗ Error testing Trainer compatibility: {e}")
        return False

if __name__ == "__main__":
    success = test_trainer_compatibility()
    if success:
        print("\n✅ Trainer compatibility test passed!")
        sys.exit(0)
    else:
        print("\n❌ Trainer compatibility test failed!")
        sys.exit(1)