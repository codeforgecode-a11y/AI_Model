#!/usr/bin/env python3
"""
Test script to verify Trainer method signatures for compatibility
"""

import sys
import inspect
import logging

def test_trainer_method_signatures():
    """Test Trainer method signatures for compatibility"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        from transformers import Trainer, TrainingArguments
        logger.info("✓ Successfully imported Trainer and TrainingArguments")

        # Check training_step method signature
        training_step_signature = inspect.signature(Trainer.training_step)
        training_step_params = list(training_step_signature.parameters.keys())

        logger.info(f"Trainer.training_step parameters: {training_step_params}")

        # Check log method signature
        log_signature = inspect.signature(Trainer.log)
        log_params = list(log_signature.parameters.keys())

        logger.info(f"Trainer.log parameters: {log_params}")

        # Check _save_checkpoint method signature
        save_checkpoint_signature = inspect.signature(Trainer._save_checkpoint)
        save_checkpoint_params = list(save_checkpoint_signature.parameters.keys())

        logger.info(f"Trainer._save_checkpoint parameters: {save_checkpoint_params}")

        # Expected parameters in different versions
        if len(training_step_params) == 2:  # model, inputs
            logger.info("✓ Found 2-parameter training_step (older version)")
        elif len(training_step_params) == 3:  # model, inputs, num_items_in_batch
            logger.info("✓ Found 3-parameter training_step (newer version)")
        else:
            logger.warning(f"⚠ Unexpected training_step parameters: {len(training_step_params)}")

        if len(log_params) == 1:  # logs
            logger.info("✓ Found 1-parameter log method (older version)")
        elif len(log_params) == 2:  # logs, start_time
            logger.info("✓ Found 2-parameter log method (newer version)")
        else:
            logger.warning(f"⚠ Unexpected log parameters: {len(log_params)}")

        if len(save_checkpoint_params) == 2:  # model, trial
            logger.info("✓ Found 2-parameter _save_checkpoint (newer version)")
        elif len(save_checkpoint_params) == 3:  # model, trial, metrics
            logger.info("✓ Found 3-parameter _save_checkpoint (older version)")
        else:
            logger.warning(f"⚠ Unexpected _save_checkpoint parameters: {len(save_checkpoint_params)}")

        # Test creating a minimal trainer
        training_args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_steps=100,
            logging_steps=10,
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

        # Test the log method signature
        try:
            test_logs = {"test_loss": 1.0, "test_metric": 0.5}

            if len(log_params) == 1:
                trainer.log(test_logs)
            else:
                import time
                trainer.log(test_logs, time.time())
            logger.info("✓ log method call successful")
        except Exception as e:
            logger.error(f"✗ log method call failed: {e}")
            return False

        # Test the _save_checkpoint method signature
        try:
            test_metrics = {"eval_loss": 2.5}

            if len(save_checkpoint_params) == 2:
                # Newer version: only model and trial
                result = trainer._save_checkpoint(model, None)
            else:
                # Older version: model, trial, and metrics
                result = trainer._save_checkpoint(model, None, test_metrics)
            logger.info("✓ _save_checkpoint method call successful")
        except Exception as e:
            logger.error(f"✗ _save_checkpoint method call failed: {e}")
            return False

        return True

    except Exception as e:
        logger.error(f"✗ Error testing Trainer compatibility: {e}")
        return False

if __name__ == "__main__":
    success = test_trainer_method_signatures()
    if success:
        print("\n✅ Trainer method signature compatibility test passed!")
        sys.exit(0)
    else:
        print("\n❌ Trainer method signature compatibility test failed!")
        sys.exit(1)