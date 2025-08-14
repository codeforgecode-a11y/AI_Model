#!/usr/bin/env python3
"""
Test script to verify TrainingArguments compatibility
"""

import sys
import inspect
import logging

def test_training_args():
    """Test TrainingArguments parameter compatibility"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        from transformers import TrainingArguments
        logger.info("✓ Successfully imported TrainingArguments")

        # Check available parameters
        signature = inspect.signature(TrainingArguments.__init__)
        params = list(signature.parameters.keys())

        logger.info(f"Available parameters: {len(params)}")

        # Check for evaluation strategy parameters
        if "eval_strategy" in params:
            logger.info("✓ Found 'eval_strategy' parameter (newer version)")
            eval_param = "eval_strategy"
        elif "evaluation_strategy" in params:
            logger.info("✓ Found 'evaluation_strategy' parameter (older version)")
            eval_param = "evaluation_strategy"
        else:
            logger.error("✗ No evaluation strategy parameter found!")
            return False

        # Test creating TrainingArguments with minimal parameters
        test_args = {
            "output_dir": "./test_output",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "save_steps": 100,
            eval_param: "steps",
            "eval_steps": 50,
        }

        # Add optional parameters if they exist
        optional_params = {
            "save_strategy": "steps",
            "fp16": False,
            "dataloader_num_workers": 1,
            "remove_unused_columns": False,
            "report_to": None,
        }

        for key, value in optional_params.items():
            if key in params:
                test_args[key] = value
                logger.info(f"✓ Added optional parameter: {key}")
            else:
                logger.info(f"⚠ Skipping unsupported parameter: {key}")

        # Try to create TrainingArguments
        training_args = TrainingArguments(**test_args)
        logger.info("✓ Successfully created TrainingArguments")

        return True

    except Exception as e:
        logger.error(f"✗ Error testing TrainingArguments: {e}")
        return False

if __name__ == "__main__":
    success = test_training_args()
    if success:
        print("\n✅ TrainingArguments compatibility test passed!")
        sys.exit(0)
    else:
        print("\n❌ TrainingArguments compatibility test failed!")
        sys.exit(1)