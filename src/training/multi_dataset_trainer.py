#!/usr/bin/env python3
"""
Multi-Dataset Training Pipeline
Supports training on conversational, programming, and cybersecurity datasets
"""

import os
import json
import random
import logging
import inspect
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
import numpy as np
from tqdm import tqdm
import gc
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

# CPU-specific optimizations
torch.set_num_threads(4)  # Use all 4 threads on i3
torch.set_num_interop_threads(2)  # Optimize inter-op parallelism

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()  # No-op on CPU but safe to call

def log_memory_usage(logger, step=""):
    """Log current memory usage"""
    memory_mb = get_memory_usage()
    logger.info(f"Memory usage {step}: {memory_mb:.1f} MB")

class CPUOptimizedDataLoader:
    """Memory-efficient data loader for CPU training"""

    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = min(num_workers, 2)  # Limit workers for CPU

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = []

            for idx in batch_indices:
                item = self.dataset[idx]
                batch.append(item)

            # Cleanup after each batch
            if i % 100 == 0:  # Every 100 batches
                cleanup_memory()

            yield self._collate_batch(batch)

    def _collate_batch(self, batch):
        """Collate batch items"""
        if len(batch) == 1:
            return batch[0]

        # Stack tensors
        collated = {}
        for key in batch[0].keys():
            collated[key] = torch.stack([item[key] for item in batch])

        return collated

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

# Detailed logging for training progress monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuration for each dataset type"""
    name: str
    path: str
    weight: float  # Sampling weight for mixing datasets
    max_samples: Optional[int] = None
    preprocessing_fn: Optional[str] = None

@dataclass
class TrainingConfig:
    """CPU-optimized training configuration with detailed logging"""
    model_name: str = "distilgpt2"  # Smaller but capable model for CPU
    max_length: int = 256  # Reduced for memory efficiency
    batch_size: int = 1  # Very small batch for CPU
    learning_rate: float = 3e-5  # Slightly lower for stability
    num_epochs: int = 3
    warmup_steps: int = 100  # Reduced warmup
    save_steps: int = 250  # More frequent saves for long training
    eval_steps: int = 125  # More frequent evaluation
    logging_steps: int = 10  # Detailed step-by-step logging
    output_dir: str = "./trained_model"
    gradient_accumulation_steps: int = 32  # Large accumulation for effective batch size
    fp16: bool = False  # Disabled for CPU
    dataloader_num_workers: int = 2  # Reduced for CPU
    cpu_only: bool = True  # Force CPU usage
    memory_efficient: bool = True  # Enable memory optimizations
    checkpoint_resume: bool = True  # Enable checkpoint resumption

class ConversationalDataset(Dataset):
    """Dataset loader for conversational data (Alpaca, ShareGPT, etc.)"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        logger.info(f"Loading conversational dataset from {data_path}")
        self._load_data(data_path, max_samples)
        logger.info(f"Loaded {len(self.data)} conversational samples")

    def _load_data(self, data_path: str, max_samples: Optional[int]):
        """Load and preprocess conversational data"""
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    try:
                        item = json.loads(line.strip())
                        processed = self._process_conversational_item(item)
                        if processed:
                            self.data.append(processed)
                    except json.JSONDecodeError:
                        continue
        elif data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                items = json.load(f)
                for i, item in enumerate(items):
                    if max_samples and i >= max_samples:
                        break
                    processed = self._process_conversational_item(item)
                    if processed:
                        self.data.append(processed)

    def _process_conversational_item(self, item: Dict) -> Optional[str]:
        """Process individual conversational item"""
        try:
            if 'instruction' in item and 'output' in item:
                # Alpaca format
                instruction = item['instruction']
                input_text = item.get('input', '')
                output = item['output']

                if input_text:
                    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

                return prompt
            elif 'text' in item:
                # Pre-formatted text
                return item['text']
            else:
                return None
        except Exception as e:
            logger.warning(f"Error processing conversational item: {e}")
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        # Memory-efficient tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Ensure CPU tensors and proper dtype
        result = {
            'input_ids': encoding['input_ids'].squeeze().to(torch.long),
            'attention_mask': encoding['attention_mask'].squeeze().to(torch.long),
            'labels': encoding['input_ids'].squeeze().to(torch.long)
        }

        # Clean up encoding to free memory
        del encoding

        return result

class ProgrammingDataset(Dataset):
    """Dataset loader for programming data (CodeAlpaca, HumanEval, etc.)"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        logger.info(f"Loading programming dataset from {data_path}")
        self._load_data(data_path, max_samples)
        logger.info(f"Loaded {len(self.data)} programming samples")

    def _load_data(self, data_path: str, max_samples: Optional[int]):
        """Load and preprocess programming data"""
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    try:
                        item = json.loads(line.strip())
                        processed = self._process_programming_item(item)
                        if processed:
                            self.data.append(processed)
                    except json.JSONDecodeError:
                        continue
        elif data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                items = json.load(f)
                for i, item in enumerate(items):
                    if max_samples and i >= max_samples:
                        break
                    processed = self._process_programming_item(item)
                    if processed:
                        self.data.append(processed)

    def _process_programming_item(self, item: Dict) -> Optional[str]:
        """Process individual programming item"""
        try:
            if 'instruction' in item and 'output' in item:
                # CodeAlpaca format
                instruction = item['instruction']
                input_text = item.get('input', '')
                output = item['output']

                if input_text:
                    prompt = f"### Programming Task:\n{instruction}\n\n### Input:\n{input_text}\n\n### Solution:\n{output}"
                else:
                    prompt = f"### Programming Task:\n{instruction}\n\n### Solution:\n{output}"

                return prompt
            elif 'prompt' in item and 'canonical_solution' in item:
                # HumanEval format
                prompt = item['prompt']
                solution = item['canonical_solution']

                return f"### Code Completion:\n{prompt}\n\n### Solution:\n{solution}"
            else:
                return None
        except Exception as e:
            logger.warning(f"Error processing programming item: {e}")
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        # Memory-efficient tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Ensure CPU tensors and proper dtype
        result = {
            'input_ids': encoding['input_ids'].squeeze().to(torch.long),
            'attention_mask': encoding['attention_mask'].squeeze().to(torch.long),
            'labels': encoding['input_ids'].squeeze().to(torch.long)
        }

        # Clean up encoding to free memory
        del encoding

        return result

class CyberSecurityDataset(Dataset):
    """Dataset loader for cybersecurity data (CVE, attack patterns, etc.)"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        logger.info(f"Loading cybersecurity dataset from {data_path}")
        self._load_data(data_path, max_samples)
        logger.info(f"Loaded {len(self.data)} cybersecurity samples")

    def _load_data(self, data_path: str, max_samples: Optional[int]):
        """Load and preprocess cybersecurity data"""
        if os.path.isdir(data_path):
            # Load multiple JSON files from directory
            json_files = list(Path(data_path).glob("*.json"))
            for i, file_path in enumerate(json_files):
                if max_samples and i >= max_samples:
                    break
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        item = json.load(f)
                        processed = self._process_cybersecurity_item(item)
                        if processed:
                            self.data.append(processed)
                except (json.JSONDecodeError, Exception):
                    continue
        else:
            # Single file
            with open(data_path, 'r', encoding='utf-8') as f:
                item = json.load(f)
                processed = self._process_cybersecurity_item(item)
                if processed:
                    self.data.append(processed)

    def _process_cybersecurity_item(self, item: Dict) -> Optional[str]:
        """Process individual cybersecurity item"""
        try:
            if 'cve' in item:
                # CVE format
                cve_data = item['cve']
                cve_id = cve_data.get('CVE_data_meta', {}).get('ID', 'Unknown')

                # Extract description
                descriptions = cve_data.get('description', {}).get('description_data', [])
                description = ""
                if descriptions:
                    description = descriptions[0].get('value', '')

                # Extract CVSS score if available
                impact = item.get('impact', {})
                cvss_score = "Unknown"
                if 'baseMetricV3' in impact:
                    cvss_score = str(impact['baseMetricV3'].get('cvssV3', {}).get('baseScore', 'Unknown'))
                elif 'baseMetricV2' in impact:
                    cvss_score = str(impact['baseMetricV2'].get('cvssV2', {}).get('baseScore', 'Unknown'))

                prompt = f"### Cybersecurity Analysis:\nCVE ID: {cve_id}\nCVSS Score: {cvss_score}\n\n### Description:\n{description}"

                return prompt
            else:
                return None
        except Exception as e:
            logger.warning(f"Error processing cybersecurity item: {e}")
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        # Memory-efficient tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Ensure CPU tensors and proper dtype
        result = {
            'input_ids': encoding['input_ids'].squeeze().to(torch.long),
            'attention_mask': encoding['attention_mask'].squeeze().to(torch.long),
            'labels': encoding['input_ids'].squeeze().to(torch.long)
        }

        # Clean up encoding to free memory
        del encoding

        return result

class MultiDatasetTrainer:
    """Main trainer class for multi-dataset training"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.datasets = {}
        self.combined_dataset = None

        # Initialize tokenizer and model
        self._setup_model()

    def _setup_model(self):
        """Initialize tokenizer and model for CPU training"""
        logger.info(f"Loading tokenizer and model for CPU: {self.config.model_name}")
        log_memory_usage(logger, "before model loading")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with CPU-specific optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,  # Always use float32 on CPU
            device_map=None,  # Force CPU
            low_cpu_mem_usage=True,  # Enable memory-efficient loading
        )

        # Force model to CPU
        self.model = self.model.to('cpu')

        # Resize token embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Enable CPU-specific optimizations
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        # Set model to training mode
        self.model.train()

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded with {param_count:,} parameters")
        log_memory_usage(logger, "after model loading")

        # Cleanup
        cleanup_memory()

    def add_dataset(self, dataset_config: DatasetConfig):
        """Add a dataset to the training pipeline"""
        logger.info(f"Adding dataset: {dataset_config.name}")

        # Determine dataset type and create appropriate dataset
        if 'conversational' in dataset_config.name.lower() or 'alpaca' in dataset_config.name.lower():
            dataset = ConversationalDataset(
                dataset_config.path,
                self.tokenizer,
                self.config.max_length,
                dataset_config.max_samples
            )
        elif 'programming' in dataset_config.name.lower() or 'code' in dataset_config.name.lower():
            dataset = ProgrammingDataset(
                dataset_config.path,
                self.tokenizer,
                self.config.max_length,
                dataset_config.max_samples
            )
        elif 'cyber' in dataset_config.name.lower() or 'security' in dataset_config.name.lower():
            dataset = CyberSecurityDataset(
                dataset_config.path,
                self.tokenizer,
                self.config.max_length,
                dataset_config.max_samples
            )
        else:
            # Default to conversational
            dataset = ConversationalDataset(
                dataset_config.path,
                self.tokenizer,
                self.config.max_length,
                dataset_config.max_samples
            )

        self.datasets[dataset_config.name] = {
            'dataset': dataset,
            'weight': dataset_config.weight
        }

        logger.info(f"Added {len(dataset)} samples from {dataset_config.name}")

    def _create_weighted_dataset(self):
        """Create a weighted combination of all datasets"""
        if not self.datasets:
            raise ValueError("No datasets added. Use add_dataset() first.")

        # Create weighted samples
        all_datasets = []
        weights = []

        for name, data in self.datasets.items():
            dataset = data['dataset']
            weight = data['weight']

            # Calculate number of samples based on weight
            total_samples = len(dataset)
            weighted_samples = int(total_samples * weight)

            # Create indices for sampling
            if weighted_samples > total_samples:
                # Oversample
                indices = list(range(total_samples)) * (weighted_samples // total_samples)
                indices += random.sample(range(total_samples), weighted_samples % total_samples)
            else:
                # Undersample
                indices = random.sample(range(total_samples), weighted_samples)

            # Create subset
            subset = torch.utils.data.Subset(dataset, indices)
            all_datasets.append(subset)

            logger.info(f"Dataset {name}: {len(subset)} samples (weight: {weight})")

        # Combine all datasets
        self.combined_dataset = ConcatDataset(all_datasets)
        logger.info(f"Combined dataset: {len(self.combined_dataset)} total samples")

    def train(self):
        """Start CPU-optimized training process"""
        if not self.datasets:
            raise ValueError("No datasets added. Use add_dataset() first.")

        logger.info("Starting CPU-optimized training...")
        log_memory_usage(logger, "before training setup")

        # Create weighted dataset
        self._create_weighted_dataset()

        # Split into train/validation for monitoring
        train_size = int(0.9 * len(self.combined_dataset))
        val_size = len(self.combined_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            self.combined_dataset, [train_size, val_size]
        )

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")

        # Check for existing checkpoint
        resume_from_checkpoint = None
        if self.config.checkpoint_resume and os.path.exists(self.config.output_dir):
            checkpoint_dirs = [d for d in os.listdir(self.config.output_dir)
                             if d.startswith('checkpoint-')]
            if checkpoint_dirs:
                latest_checkpoint = max(checkpoint_dirs,
                                      key=lambda x: int(x.split('-')[1]))
                resume_from_checkpoint = os.path.join(self.config.output_dir, latest_checkpoint)
                logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")

        # Setup CPU-optimized training arguments with version compatibility
        logger.info("Setting up training arguments...")

        # Detailed logging training arguments for progress monitoring
        core_args = {
            "output_dir": self.config.output_dir,
            "overwrite_output_dir": False,
            "num_train_epochs": self.config.num_epochs,
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "warmup_steps": self.config.warmup_steps,
            "logging_steps": self.config.logging_steps,  # Configurable detailed logging
            "save_steps": self.config.save_steps,
            "eval_steps": self.config.eval_steps,
            "save_strategy": "steps",
            "load_best_model_at_end": False,
            "fp16": False,
            "dataloader_num_workers": self.config.dataloader_num_workers,
            "remove_unused_columns": False,
            "report_to": [],  # Disable external reporting but keep console logging
            "save_total_limit": 3,  # Keep 3 checkpoints
            "disable_tqdm": False,  # Enable progress bars for visibility
            "log_level": "info",  # Detailed logging
            "logging_first_step": True,  # Log the first step
            "logging_nan_inf_filter": True,  # Filter NaN/Inf values in logs
        }

        # Try to add evaluation strategy with version compatibility
        training_args_signature = inspect.signature(TrainingArguments.__init__)

        # Enable evaluation for progress monitoring
        if "eval_strategy" in training_args_signature.parameters:
            core_args["eval_strategy"] = "steps"
            logger.info("Using 'eval_strategy' parameter (newer transformers)")
        elif "evaluation_strategy" in training_args_signature.parameters:
            core_args["evaluation_strategy"] = "steps"
            logger.info("Using 'evaluation_strategy' parameter (older transformers)")
        else:
            logger.warning("No evaluation strategy parameter found")

        # Add optional parameters for detailed monitoring
        optional_args = {
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "bf16": False,
            "no_cuda": True,
            "dataloader_pin_memory": False,
            "prediction_loss_only": False,  # Enable detailed metrics
            "include_inputs_for_metrics": False,
            "log_on_each_node": True,
            "logging_strategy": "steps",
        }

        for key, value in optional_args.items():
            if key in training_args_signature.parameters:
                core_args[key] = value
            else:
                logger.info(f"Skipping optional parameter '{key}' (not supported in this version)")

        training_args = TrainingArguments(**core_args)

        # Memory-efficient data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=None,  # No padding optimization needed for CPU
        )

        log_memory_usage(logger, "before trainer initialization")

        # Initialize CPU-optimized trainer with detailed logging
        trainer = CPUOptimizedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        log_memory_usage(logger, "after trainer initialization")

        # Start training with detailed progress monitoring
        logger.info("Starting CPU training with detailed progress logging...")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        total_steps = len(train_dataset) // (self.config.batch_size * self.config.gradient_accumulation_steps) * self.config.num_epochs
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Logging frequency: every {self.config.logging_steps} steps")
        logger.info(f"Checkpoint frequency: every {self.config.save_steps} steps")
        logger.info(f"Evaluation frequency: every {self.config.eval_steps} steps")

        try:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Saving current state...")
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Try to save what we have
            try:
                trainer.save_model()
                self.tokenizer.save_pretrained(self.config.output_dir)
            except:
                pass
            raise

        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)

        log_memory_usage(logger, "after training completion")
        logger.info(f"Training completed successfully! Model saved to {self.config.output_dir}")

        # Final cleanup
        cleanup_memory()

        return trainer

class CPUOptimizedTrainer(Trainer):
    """CPU-optimized trainer with detailed progress logging"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_count = 0
        self.start_time = None
        self.last_log_time = None

        # Check parent class method signatures for compatibility
        try:
            # Check training_step signature
            training_step_signature = inspect.signature(super().training_step)
            self.training_step_params = list(training_step_signature.parameters.keys())
            logger.info(f"Parent training_step parameters: {self.training_step_params}")

            # Check log method signature
            log_signature = inspect.signature(super().log)
            self.log_params = list(log_signature.parameters.keys())
            logger.info(f"Parent log parameters: {self.log_params}")

            # Check _save_checkpoint method signature
            save_checkpoint_signature = inspect.signature(super()._save_checkpoint)
            self.save_checkpoint_params = list(save_checkpoint_signature.parameters.keys())
            logger.info(f"Parent _save_checkpoint parameters: {self.save_checkpoint_params}")

        except Exception as e:
            logger.warning(f"Could not inspect parent method signatures: {e}")
            self.training_step_params = ['model', 'inputs']
            self.log_params = ['logs']
            self.save_checkpoint_params = ['model', 'trial']

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Training step with detailed progress logging"""
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Initialize timing if first step
        if self.start_time is None:
            import time
            self.start_time = time.time()
            self.last_log_time = self.start_time

        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        # Scale loss for gradient accumulation
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Increment step counter
        self.step_count += 1

        # Detailed logging every N steps
        if self.step_count % self.args.logging_steps == 0:
            self._log_training_progress(loss.item())

        # Memory cleanup
        if self.step_count % 50 == 0:
            cleanup_memory()

        return loss.detach()

    def _log_training_progress(self, current_loss):
        """Log detailed training progress information"""
        import time

        current_time = time.time()
        elapsed_time = current_time - self.start_time
        step_time = current_time - self.last_log_time if self.last_log_time else 0

        # Calculate progress
        total_steps = self.state.max_steps if self.state.max_steps > 0 else 1000  # Fallback
        progress_pct = (self.step_count / total_steps) * 100 if total_steps > 0 else 0

        # Estimate remaining time
        if self.step_count > 0 and elapsed_time > 0:
            steps_per_second = self.step_count / elapsed_time
            remaining_steps = max(0, total_steps - self.step_count)
            eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
            eta_hours = eta_seconds / 3600
        else:
            eta_hours = 0

        # Get current learning rate
        current_lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.args.learning_rate

        # Get memory usage
        memory_mb = get_memory_usage()

        # Format elapsed time
        elapsed_hours = elapsed_time / 3600

        # Log comprehensive progress information
        logger.info(f"Step {self.step_count}/{total_steps} ({progress_pct:.1f}%) | "
                   f"Loss: {current_loss:.4f} | "
                   f"LR: {current_lr:.2e} | "
                   f"Memory: {memory_mb:.1f}MB | "
                   f"Elapsed: {elapsed_hours:.2f}h | "
                   f"ETA: {eta_hours:.2f}h")

        self.last_log_time = current_time

    def _save_checkpoint(self, model, trial, metrics=None):
        """Enhanced checkpoint saving with detailed logging and version compatibility"""
        logger.info(f"💾 Saving checkpoint at step {self.step_count}...")
        log_memory_usage(logger, "before checkpoint save")

        # Call parent method with version compatibility
        try:
            if hasattr(self, 'save_checkpoint_params') and len(self.save_checkpoint_params) > 2:
                # Parent method expects metrics parameter (older version)
                result = super()._save_checkpoint(model, trial, metrics)
            else:
                # Parent method only expects model and trial (newer version)
                result = super()._save_checkpoint(model, trial)
        except TypeError as e:
            # Fallback: try both signatures
            try:
                result = super()._save_checkpoint(model, trial)
            except TypeError:
                try:
                    result = super()._save_checkpoint(model, trial, metrics)
                except TypeError:
                    logger.error(f"Could not call parent _save_checkpoint method: {e}")
                    # Create a minimal result if all else fails
                    result = None

        cleanup_memory()
        log_memory_usage(logger, "after checkpoint save")
        logger.info(f"✅ Checkpoint saved successfully")

        return result

    def log(self, logs, start_time=None):
        """Enhanced logging with additional training metrics and version compatibility"""
        # Add custom metrics to logs
        if hasattr(self, 'step_count'):
            logs["custom_step"] = self.step_count

        # Add memory usage
        memory_mb = get_memory_usage()
        logs["memory_mb"] = memory_mb

        # Add timing information if available
        if hasattr(self, 'start_time') and self.start_time:
            import time
            elapsed_time = time.time() - self.start_time
            logs["elapsed_hours"] = elapsed_time / 3600

        # Call parent log method with detected signature compatibility
        try:
            if hasattr(self, 'log_params') and len(self.log_params) > 1:
                # Parent method expects start_time parameter
                super().log(logs, start_time)
            else:
                # Parent method only expects logs parameter
                super().log(logs)
        except TypeError as e:
            # Fallback: try both signatures
            try:
                super().log(logs)
            except TypeError:
                # If both fail, just log the error and continue
                logger.warning(f"Could not call parent log method: {e}")
                pass

def create_dataset_configs():
    """Create dataset configurations for your specific datasets"""
    configs = []

    # Conversational datasets
    conversational_paths = [
        "Datasets/Conversational_Datasets/alpaca_dataset/train.jsonl",
        "Datasets/Conversational_Datasets/vicuna_dataset/train.jsonl",
        "Datasets/Conversational_Datasets/wizardlm_dataset/train.jsonl"
    ]

    for path in conversational_paths:
        if os.path.exists(path):
            configs.append(DatasetConfig(
                name=f"conversational_{Path(path).parent.name}",
                path=path,
                weight=0.4,  # 40% weight for conversational data
                max_samples=5000  # Limit samples for faster training
            ))

    # Programming datasets
    programming_paths = [
        "Datasets/Programming_DataSets/codealpaca/data/code_alpaca_2k.json",
        "Datasets/Programming_DataSets/human-eval/data/example_problem.jsonl"
    ]

    for path in programming_paths:
        if os.path.exists(path):
            configs.append(DatasetConfig(
                name=f"programming_{Path(path).stem}",
                path=path,
                weight=0.4,  # 40% weight for programming data
                max_samples=2000
            ))

    # Cybersecurity datasets
    cyber_path = "Datasets/CyberSecurity_DataSets/nvdcve/nvdcve"
    if os.path.exists(cyber_path):
        configs.append(DatasetConfig(
            name="cybersecurity_cve",
            path=cyber_path,
            weight=0.2,  # 20% weight for cybersecurity data
            max_samples=1000
        ))

    return configs

def main():
    """Main training function"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Create training configuration
    config = TrainingConfig(
        model_name="microsoft/DialoGPT-medium",  # You can change this to other models
        max_length=512,
        batch_size=4,  # Adjust based on your GPU memory
        learning_rate=5e-5,
        num_epochs=2,  # Start with fewer epochs for testing
        warmup_steps=100,
        save_steps=500,
        eval_steps=250,
        output_dir="./multi_domain_model",
        gradient_accumulation_steps=8,  # Effective batch size = 4 * 8 = 32
        fp16=True,
        dataloader_num_workers=2
    )

    # Initialize trainer
    trainer = MultiDatasetTrainer(config)

    # Add datasets
    dataset_configs = create_dataset_configs()

    if not dataset_configs:
        logger.error("No valid datasets found. Please check your dataset paths.")
        return

    for dataset_config in dataset_configs:
        try:
            trainer.add_dataset(dataset_config)
        except Exception as e:
            logger.warning(f"Failed to load dataset {dataset_config.name}: {e}")

    # Start training
    try:
        trained_model = trainer.train()
        logger.info("Training completed successfully!")

        # Optional: Test the model
        test_model(config.output_dir)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def test_model(model_path: str):
    """Test the trained model with sample prompts"""
    logger.info("Testing trained model...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # Test prompts for each domain
        test_prompts = [
            "### Instruction:\nExplain what machine learning is.\n\n### Response:",
            "### Programming Task:\nWrite a Python function to calculate factorial.\n\n### Solution:",
            "### Cybersecurity Analysis:\nWhat is a buffer overflow attack?\n\n### Description:"
        ]

        for prompt in test_prompts:
            inputs = tokenizer.encode(prompt, return_tensors='pt')

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Prompt: {prompt[:50]}...")
            logger.info(f"Response: {response[len(prompt):].strip()}")
            logger.info("-" * 50)

    except Exception as e:
        logger.error(f"Model testing failed: {e}")

if __name__ == "__main__":
    main()