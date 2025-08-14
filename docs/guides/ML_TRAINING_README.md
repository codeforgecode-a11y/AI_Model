# Multi-Dataset Training Pipeline

A comprehensive training system for fine-tuning language models on multiple dataset types: conversational, programming, and cybersecurity data.

## Overview

This training pipeline allows you to:
- Train a single model on multiple dataset types simultaneously
- Handle different data formats (JSONL, JSON) automatically
- Configure dataset mixing ratios and sampling weights
- Monitor training progress and evaluate model performance
- Support various base models (GPT-2, DialoGPT, CodeT5, etc.)

## Dataset Structure

Your datasets are organized as follows:

```
Datasets/
├── Conversational_Datasets/
│   ├── alpaca_dataset/train.jsonl
│   ├── vicuna_dataset/train.jsonl
│   └── wizardlm_dataset/train.jsonl
├── Programming_DataSets/
│   ├── codealpaca/data/code_alpaca_2k.json
│   └── human-eval/data/example_problem.jsonl
└── CyberSecurity_DataSets/
    └── nvdcve/nvdcve/*.json
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r ml_training_requirements.txt
```

### 2. Basic Training

```bash
python train_model.py
```

### 3. Custom Training

```bash
python train_model.py \
    --model "microsoft/DialoGPT-medium" \
    --epochs 3 \
    --batch-size 4 \
    --output-dir "./my_trained_model" \
    --learning-rate 5e-5
```

## Configuration

### Command Line Options

- `--model`: Base model name (default: "microsoft/DialoGPT-medium")
- `--epochs`: Number of training epochs (default: 2)
- `--batch-size`: Training batch size (default: 2)
- `--output-dir`: Output directory for trained model (default: "./trained_multi_model")
- `--max-length`: Maximum sequence length (default: 512)
- `--learning-rate`: Learning rate (default: 5e-5)

### YAML Configuration

You can also use the `training_config.yaml` file for more detailed configuration:

```yaml
model:
  name: "microsoft/DialoGPT-medium"
  max_length: 512

training:
  batch_size: 4
  learning_rate: 5e-5
  num_epochs: 3
  # ... more options
```

## Dataset Types and Processing

### 1. Conversational Datasets
- **Format**: JSONL with `instruction`, `input`, `output` fields
- **Processing**: Formats as instruction-following prompts
- **Example**:
  ```json
  {
    "instruction": "Explain machine learning",
    "input": "",
    "output": "Machine learning is..."
  }
  ```

### 2. Programming Datasets
- **Format**: JSON/JSONL with programming tasks and solutions
- **Processing**: Formats as code completion tasks
- **Example**:
  ```json
  {
    "instruction": "Write a function to reverse a string",
    "output": "def reverse_string(s): return s[::-1]"
  }
  ```

### 3. Cybersecurity Datasets
- **Format**: JSON files with CVE data
- **Processing**: Extracts vulnerability information and CVSS scores
- **Example**: CVE entries with descriptions and severity scores

## Model Architecture

The pipeline supports various transformer models:

- **DialoGPT**: Good for conversational tasks
- **CodeT5**: Optimized for code generation
- **GPT-2**: General-purpose language model
- **Custom models**: Any HuggingFace compatible model

## Training Features

### Multi-Dataset Mixing
- Configurable sampling weights for each dataset type
- Automatic data balancing and shuffling
- Support for different dataset sizes

### Memory Optimization
- Gradient accumulation for large effective batch sizes
- Mixed precision training (FP16)
- Efficient data loading with multiple workers

### Monitoring and Evaluation
- Real-time training metrics
- Validation loss tracking
- Automatic model checkpointing
- Post-training evaluation on sample prompts

## Hardware Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (RTX 3070 or better)
- **RAM**: 16GB system memory
- **Storage**: 50GB free space

### Recommended Requirements
- **GPU**: 24GB VRAM (RTX 4090 or A100)
- **RAM**: 32GB system memory
- **Storage**: 100GB SSD

## Training Time Estimates

| Dataset Size | GPU | Epochs | Estimated Time |
|-------------|-----|--------|----------------|
| Small (5K samples) | RTX 3070 | 2 | 2-3 hours |
| Medium (20K samples) | RTX 4090 | 3 | 4-6 hours |
| Large (50K samples) | A100 | 3 | 8-12 hours |

## Output Structure

After training, you'll have:

```
trained_multi_model/
├── config.json              # Model configuration
├── pytorch_model.bin         # Model weights
├── tokenizer.json           # Tokenizer
├── tokenizer_config.json    # Tokenizer config
├── special_tokens_map.json  # Special tokens
└── training_args.bin        # Training arguments
```

## Usage Examples

### Basic Training
```python
from multi_dataset_trainer import MultiDatasetTrainer, TrainingConfig, DatasetConfig

# Create configuration
config = TrainingConfig(
    model_name="microsoft/DialoGPT-medium",
    batch_size=4,
    num_epochs=2
)

# Initialize trainer
trainer = MultiDatasetTrainer(config)

# Add datasets
trainer.add_dataset(DatasetConfig(
    name="conversational_alpaca",
    path="Datasets/Conversational_Datasets/alpaca_dataset/train.jsonl",
    weight=0.4
))

# Start training
trainer.train()
```

### Advanced Configuration
```python
config = TrainingConfig(
    model_name="microsoft/DialoGPT-medium",
    max_length=512,
    batch_size=2,
    learning_rate=3e-5,
    num_epochs=3,
    warmup_steps=200,
    save_steps=500,
    eval_steps=250,
    gradient_accumulation_steps=8,
    fp16=True
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 1`
   - Increase gradient accumulation: `gradient_accumulation_steps=16`
   - Use smaller model or reduce max_length

2. **Dataset Loading Errors**
   - Check file paths and permissions
   - Verify JSON/JSONL format
   - Ensure datasets contain required fields

3. **Slow Training**
   - Enable FP16: `fp16=True`
   - Increase batch size if memory allows
   - Use more workers: `dataloader_num_workers=4`

### Performance Optimization

1. **Memory Optimization**
   ```python
   config.gradient_accumulation_steps = 8  # Larger effective batch
   config.fp16 = True                      # Half precision
   config.dataloader_num_workers = 4       # Parallel loading
   ```

2. **Speed Optimization**
   ```python
   config.max_length = 256                 # Shorter sequences
   config.batch_size = 8                   # Larger batches
   ```

## Model Evaluation

The pipeline includes automatic evaluation with sample prompts:

```python
# Test prompts for each domain
test_prompts = [
    "### Instruction:\nExplain artificial intelligence.\n\n### Response:",
    "### Programming Task:\nWrite a sorting algorithm.\n\n### Solution:",
    "### Cybersecurity Analysis:\nWhat is a DDoS attack?\n\n### Description:"
]
```

## Next Steps

After training, you can:

1. **Deploy the model** using the HuggingFace Transformers library
2. **Fine-tune further** on specific tasks
3. **Evaluate performance** on held-out test sets
4. **Integrate** into applications or APIs

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review training logs in `training.log`
3. Verify dataset formats and paths
4. Monitor GPU memory usage during training
```