#!/usr/bin/env python3
"""
Training Verification Script
Checks if the model is actually learning and improving
"""

import os
import sys
import torch
import json
from datetime import datetime

def check_training_progress():
    """Check if training is progressing correctly"""
    print("🔍 Training Verification Report")
    print("=" * 50)

    # 1. Check if training is running
    check_training_process()

    # 2. Analyze loss progression
    analyze_loss_progression()

    # 3. Check checkpoint files
    check_checkpoints()

    # 4. Test model responses (if checkpoints exist)
    test_model_responses()

    # 5. Monitor system resources
    check_system_resources()

def check_training_process():
    """Check if training process is active"""
    print("\n1. 📊 Training Process Status")

    # Check for training log file
    if os.path.exists("training_progress.log"):
        print("✓ Training log file found")

        # Get last few lines
        with open("training_progress.log", "r") as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                print(f"✓ Last log entry: {last_line}")

                # Check if recent (within last hour)
                if "Step" in last_line:
                    print("✓ Training appears to be active")
                else:
                    print("⚠ Training may have stopped")
            else:
                print("⚠ Log file is empty")
    else:
        print("❌ No training log file found")

def analyze_loss_progression():
    """Analyze loss values to see if model is learning"""
    print("\n2. 📈 Loss Progression Analysis")

    if not os.path.exists("training_progress.log"):
        print("❌ No training log to analyze")
        return

    losses = []
    steps = []

    try:
        with open("training_progress.log", "r") as f:
            for line in f:
                if "Loss:" in line and "Step" in line:
                    # Extract step and loss
                    parts = line.split("|")
                    for part in parts:
                        if "Step" in part:
                            step = int(part.split("/")[0].split()[-1])
                            steps.append(step)
                        elif "Loss:" in part:
                            loss = float(part.split(":")[1].strip())
                            losses.append(loss)

        if len(losses) >= 2:
            print(f"✓ Found {len(losses)} loss values")
            print(f"✓ Initial loss: {losses[0]:.4f}")
            print(f"✓ Latest loss: {losses[-1]:.4f}")

            # Check if loss is decreasing
            if losses[-1] < losses[0]:
                improvement = ((losses[0] - losses[-1]) / losses[0]) * 100
                print(f"✅ Loss improved by {improvement:.1f}% - Model is learning!")
            else:
                print("⚠ Loss hasn't improved - may need more time or adjustment")

            # Show recent trend
            if len(losses) >= 10:
                recent_losses = losses[-10:]
                if recent_losses[-1] < recent_losses[0]:
                    print("✅ Recent trend: Loss is decreasing")
                else:
                    print("⚠ Recent trend: Loss is not decreasing")
        else:
            print("⚠ Not enough loss data to analyze")

    except Exception as e:
        print(f"❌ Error analyzing loss: {e}")

def check_checkpoints():
    """Check if model checkpoints are being saved"""
    print("\n3. 💾 Checkpoint Analysis")

    checkpoint_dirs = ["./trained_multi_model_cpu", "./cpu_trained_model", "./trained_model"]

    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            print(f"✓ Found checkpoint directory: {checkpoint_dir}")

            # List checkpoint files
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
            if checkpoints:
                print(f"✓ Found {len(checkpoints)} checkpoints")
                latest = max(checkpoints, key=lambda x: int(x.split('-')[1]))
                print(f"✓ Latest checkpoint: {latest}")

                # Check checkpoint size
                checkpoint_path = os.path.join(checkpoint_dir, latest)
                if os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
                    size = os.path.getsize(os.path.join(checkpoint_path, "pytorch_model.bin"))
                    print(f"✓ Model file size: {size / 1024 / 1024:.1f} MB")
                    return checkpoint_dir, latest
            else:
                print("⚠ No checkpoints found in directory")

    print("❌ No checkpoint directories found")
    return None, None

def test_model_responses():
    """Test model responses to see if it's learning"""
    print("\n4. 🧠 Model Response Testing")

    checkpoint_dir, latest_checkpoint = check_checkpoints()

    if not checkpoint_dir or not latest_checkpoint:
        print("❌ No checkpoints available for testing")
        return

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"✓ Loading model from: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # Test prompts for different domains
        test_prompts = [
            "### Instruction:\nWhat is machine learning?\n\n### Response:",
            "### Programming Task:\nWrite a Python function to add two numbers.\n\n### Solution:",
            "### Cybersecurity Analysis:\nWhat is a firewall?\n\n### Description:"
        ]

        print("✓ Testing model responses:")

        for i, prompt in enumerate(test_prompts):
            inputs = tokenizer.encode(prompt, return_tensors='pt')

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()

            print(f"\n  Test {i+1}:")
            print(f"  Prompt: {prompt[:50]}...")
            print(f"  Response: {generated_text[:100]}...")

            # Basic quality check
            if len(generated_text) > 10 and not generated_text.startswith("###"):
                print("  ✅ Model generating reasonable responses")
            else:
                print("  ⚠ Model responses may need more training")

    except Exception as e:
        print(f"❌ Error testing model: {e}")

def check_system_resources():
    """Check system resource usage"""
    print("\n5. 🖥️ System Resource Usage")

    try:
        import psutil

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"✓ CPU usage: {cpu_percent}%")

        if cpu_percent > 80:
            print("✅ High CPU usage - training is active")
        elif cpu_percent > 50:
            print("✓ Moderate CPU usage - training may be active")
        else:
            print("⚠ Low CPU usage - training may not be running")

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        print(f"✓ Memory usage: {memory_percent}%")

        # Check for Python processes
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'train' in cmdline.lower():
                        python_processes.append(proc.info['pid'])
            except:
                pass

        if python_processes:
            print(f"✅ Found {len(python_processes)} training processes running")
        else:
            print("⚠ No training processes found")

    except ImportError:
        print("⚠ psutil not available for system monitoring")
    except Exception as e:
        print(f"❌ Error checking system resources: {e}")

def main():
    """Main verification function"""
    check_training_progress()

    print("\n" + "=" * 50)
    print("📋 Summary:")
    print("- Check loss progression: Should decrease over time")
    print("- Monitor checkpoints: Should be created regularly")
    print("- Test model responses: Should improve with training")
    print("- Watch system resources: High CPU usage indicates active training")
    print("\n💡 Tip: Run this script periodically to monitor training progress")

if __name__ == "__main__":
    main()