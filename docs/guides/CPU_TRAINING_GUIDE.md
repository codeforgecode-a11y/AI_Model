# CPU-Only Training Guide for Multi-Dataset Pipeline

**Optimized for Intel Core i3 7th Generation with Limited Hardware**

## 🎯 Overview

This guide provides a complete CPU-optimized training pipeline that maintains full model capabilities and complete dataset sizes while being efficient on limited hardware. The optimizations focus on memory management, CPU utilization, and training stability for extended training periods.

## 🖥️ Hardware Specifications

**Your System:**
- CPU: Intel Core i3 7th generation (dual-core, 4 threads)
- GPU: Intel integrated graphics (no CUDA support)
- RAM: Limited (optimization assumes 8-16GB)
- Storage: Standard HDD/SSD

**Optimizations Applied:**
- CPU-only PyTorch operations
- Multi-threaded CPU utilization (4 threads)
- Aggressive memory management
- Frequent checkpointing for long training sessions

## 📊 Training Estimates

### **Realistic Time Expectations:**

| Dataset Size | Effective Batch Size | Estimated Time |
|-------------|---------------------|----------------|
| Full datasets (~60K samples) | 32 (1×32 accumulation) | **24-48 hours** |
| Reduced datasets (~20K samples) | 32 | **8-16 hours** |
| Test run (~5K samples) | 32 | **2-4 hours** |

### **Training Configuration:**
- **Model**: DistilGPT-2 (82M parameters)
- **Sequence Length**: 256 tokens (reduced for memory)
- **Batch Size**: 1 (with 32-step gradient accumulation)
- **Effective Batch Size**: 32
- **Learning Rate**: 3e-5 (optimized for CPU training)
- **Epochs**: 3

## 🚀 Quick Start

### 1. Install CPU-Optimized Dependencies

```bash
# Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r cpu_training_requirements.txt
```

### 2. Start Training

```bash
# Detailed progress monitoring (recommended for long training)
python train_model.py

# Custom logging frequency
python train_model.py --logging-steps 10

# Test run with detailed logging
python train_model.py --max-samples 1000 --logging-steps 5

# Different logging modes
python train_with_logging.py --logging-mode detailed    # Every 5 steps
python train_with_logging.py --logging-mode moderate    # Every 25 steps
python train_with_logging.py --logging-mode minimal     # Every 100 steps
```

### 3. Progress Monitoring Options

Choose your preferred level of detail:

**Detailed Logging (Every 5-10 steps):**
```bash
python train_model.py --logging-steps 5
```

**Moderate Logging (Every 25-50 steps):**
```bash
python train_model.py --logging-steps 25
```

**Minimal Logging (Every 100+ steps):**
```bash
python train_model.py --logging-steps 100
```

## 🔧 CPU-Specific Optimizations

### **Memory Management:**
- **Gradient Accumulation**: Effective batch size of 32 with minimal memory
- **Frequent Cleanup**: Memory cleanup every 50 steps
- **Checkpoint Limiting**: Keep only 3 most recent checkpoints
- **Tensor Optimization**: CPU-specific tensor operations

### **CPU Utilization:**
- **Thread Optimization**: Uses all 4 available threads
- **Inter-op Parallelism**: Optimized for dual-core CPU
- **Memory-Efficient Loading**: Reduced data loader workers
- **Gradient Checkpointing**: Trades compute for memory

### **Training Stability:**
- **Frequent Saves**: Checkpoint every 100 steps
- **Resume Capability**: Automatic checkpoint resumption
- **Error Handling**: Graceful handling of interruptions
- **Memory Monitoring**: Real-time memory usage tracking

## 📁 File Structure

```
CPU Training Files:
├── multi_dataset_trainer.py      # CPU-optimized trainer
├── train_model.py                # CPU training script
├── monitor_cpu_training.py       # Training monitor
├── cpu_training_config.yaml      # CPU configuration
├── cpu_training_requirements.txt # CPU dependencies
└── CPU_TRAINING_GUIDE.md         # This guide

Generated During Training:
├── cpu_trained_model/            # Model output
│   ├── checkpoint-100/           # Training checkpoints
│   ├── checkpoint-200/
│   └── ...
├── cpu_training.log              # Training logs
└── training.log                  # Detailed logs
```

## 🎛️ Configuration Options

### **Command Line Options:**

```bash
python train_model.py [OPTIONS]

Options:
  --model TEXT                 Base model (default: distilgpt2)
  --epochs INTEGER            Training epochs (default: 3)
  --batch-size INTEGER        Batch size (default: 1)
  --gradient-accumulation INT Gradient accumulation steps (default: 32)
  --max-samples INTEGER       Limit samples per dataset (for testing)
  --resume                    Resume from checkpoint
  --output-dir TEXT           Output directory
  --max-length INTEGER        Max sequence length (default: 256)
  --learning-rate FLOAT       Learning rate (default: 3e-5)
```

### **Dataset Weights (Maintained):**
- **Conversational**: 40% (Alpaca, Vicuna, WizardLM)
- **Programming**: 40% (CodeAlpaca, HumanEval)
- **Cybersecurity**: 20% (CVE data)

## 🔄 Training Process

### **Phase 1: Initialization (5-10 minutes)**
- Model loading and CPU optimization
- Dataset preprocessing and tokenization
- Memory allocation and thread setup

### **Phase 2: Training (24-48 hours)**
- Step-by-step gradient accumulation
- Frequent checkpointing and memory cleanup
- Progress monitoring and logging

### **Phase 3: Completion**
- Final model saving
- Training statistics summary
- Model validation

## 📈 Monitoring and Progress

### **Real-time Progress Logging:**

The training now provides detailed step-by-step progress information:

**Example Detailed Log Output:**
```
15:23:45 - INFO - Step 50/5625 (0.9%) | Loss: 3.2847 | LR: 2.95e-05 | Memory: 2847.3MB | Elapsed: 0.15h | ETA: 16.2h
15:23:52 - INFO - Step 60/5625 (1.1%) | Loss: 3.1923 | LR: 2.98e-05 | Memory: 2851.7MB | Elapsed: 0.16h | ETA: 15.8h
15:23:59 - INFO - 💾 Saving checkpoint at step 60...
15:24:03 - INFO - ✅ Checkpoint saved successfully
15:24:06 - INFO - Step 70/5625 (1.2%) | Loss: 3.1456 | LR: 3.00e-05 | Memory: 2849.1MB | Elapsed: 0.17h | ETA: 15.5h
```

**Log Information Includes:**
- Current step and total steps with percentage
- Current loss value
- Learning rate at each step
- Memory usage in MB
- Time elapsed and estimated time remaining
- Checkpoint saving notifications

### **Log Files:**
- `cpu_training.log`: Training progress and metrics
- `training.log`: Detailed training information

## 🛠️ Troubleshooting

### **Common Issues:**

1. **TrainingArguments Parameter Error**
   ```bash
   # Test parameter compatibility first
   python test_training_args.py

   # If issues persist, check transformers version
   pip install transformers>=4.30.0
   ```

2. **Trainer Method Signature Error**
   ```bash
   # Test trainer method compatibility
   python test_trainer_methods.py

   # The CPUOptimizedTrainer automatically handles version differences
   ```

3. **Logging Method Signature Error**
   ```bash
   # Test all trainer method signatures
   python test_trainer_methods.py

   # The enhanced logging automatically adapts to your transformers version
   ```

4. **Checkpoint Saving Method Signature Error**
   ```bash
   # Test checkpoint saving compatibility
   python test_checkpoint_fix.py

   # The checkpoint saving automatically handles version differences
   ```

2. **Out of Memory Error**
   ```bash
   # Reduce batch size or sequence length
   python train_model.py --batch-size 1 --max-length 128
   ```

3. **Training Too Slow**
   ```bash
   # Test with limited samples first
   python train_model.py --max-samples 500 --epochs 1
   ```

4. **Process Interrupted**
   ```bash
   # Resume from checkpoint
   python train_model.py --resume
   ```

5. **High Memory Usage**
   - Close other applications
   - Monitor with `python monitor_cpu_training.py`
   - Reduce `max_length` if needed

### **Performance Optimization:**

1. **System Preparation:**
   ```bash
   # Close unnecessary applications
   # Ensure adequate swap space
   # Monitor system temperature
   ```

2. **Training Optimization:**
   ```bash
   # Start with test run
   python train_model.py --max-samples 1000 --epochs 1

   # If successful, run full training
   python train_model.py
   ```

## 🎯 Expected Results

### **Model Capabilities After Training:**
- **Conversational AI**: General Q&A and instruction following
- **Code Generation**: Python, JavaScript, and other programming tasks
- **Cybersecurity Analysis**: Vulnerability assessment and security concepts

### **Performance Expectations:**
- **Quality**: Comparable to GPU-trained models
- **Speed**: Significantly slower inference on CPU
- **Memory**: ~3GB RAM during training, ~1GB for inference

### **Training Metrics:**
- **Final Loss**: Expected ~2.5-3.0 (varies by dataset)
- **Perplexity**: Expected ~15-25
- **Convergence**: Typically after 2-3 epochs

## 🔄 Resuming Training

Training can be interrupted and resumed:

```bash
# Training automatically saves checkpoints every 100 steps
# To resume after interruption:
python train_model.py --resume

# Or specify output directory:
python train_model.py --resume --output-dir "./my_cpu_model"
```

## 📝 Next Steps After Training

1. **Test the Model:**
   ```bash
   python test_trained_model.py --model-path "./cpu_trained_model"
   ```

2. **Deploy for Inference:**
   - Use the trained model with HuggingFace Transformers
   - Implement in your applications
   - Consider quantization for faster inference

3. **Further Fine-tuning:**
   - Fine-tune on specific tasks
   - Adjust for domain-specific applications

## ⚠️ Important Notes

- **Training Time**: 24-48 hours is normal for CPU training
- **System Load**: Expect high CPU usage (80-90%) during training
- **Memory Usage**: Monitor RAM usage, expect 2-4GB
- **Interruption Safety**: Training can be safely interrupted and resumed
- **Quality**: Full model quality is maintained despite CPU training

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review log files for error messages
3. Start with a test run using `--max-samples 500`
4. Monitor system resources during training

The CPU-optimized pipeline maintains full model capabilities while being efficient on your hardware constraints.