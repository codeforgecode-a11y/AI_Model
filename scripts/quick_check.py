#!/usr/bin/env python3
"""
Quick Training Check - Fast verification if training is working
"""

import os
import subprocess

def quick_training_check():
    """Quick check if training is working"""
    print("⚡ Quick Training Check")
    print("=" * 30)

    # 1. Check if training process is running
    print("1. 🔍 Process Check:")
    try:
        result = subprocess.run(['pgrep', '-f', 'train'], capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Training process is running")
        else:
            print("❌ No training process found")
    except:
        print("⚠ Could not check processes")

    # 2. Check recent log activity
    print("\n2. 📝 Log Activity:")
    if os.path.exists("training_progress.log"):
        try:
            # Get file modification time
            import time
            mod_time = os.path.getmtime("training_progress.log")
            current_time = time.time()
            minutes_ago = (current_time - mod_time) / 60

            if minutes_ago < 5:
                print(f"✅ Log updated {minutes_ago:.1f} minutes ago")
            else:
                print(f"⚠ Log last updated {minutes_ago:.1f} minutes ago")

            # Show last line
            with open("training_progress.log", "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if "Step" in last_line:
                        print(f"✅ Latest: {last_line}")
                    else:
                        print(f"⚠ Latest: {last_line}")
        except Exception as e:
            print(f"❌ Error reading log: {e}")
    else:
        print("❌ No training log found")

    # 3. Check CPU usage
    print("\n3. 🖥️ CPU Usage:")
    try:
        result = subprocess.run(['top', '-bn1'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'python' in line.lower() and ('train' in line.lower() or 'cpu' in line.lower()):
                print(f"✅ Python process: {line.strip()}")
                break
        else:
            print("⚠ No Python training process visible in top")
    except:
        print("⚠ Could not check CPU usage")

    # 4. Quick loss check
    print("\n4. 📉 Loss Trend:")
    if os.path.exists("training_progress.log"):
        try:
            losses = []
            with open("training_progress.log", "r") as f:
                for line in f:
                    if "Loss:" in line:
                        try:
                            loss_part = [p for p in line.split("|") if "Loss:" in p][0]
                            loss = float(loss_part.split(":")[1].strip())
                            losses.append(loss)
                        except:
                            continue

            if len(losses) >= 2:
                print(f"✅ First loss: {losses[0]:.4f}")
                print(f"✅ Latest loss: {losses[-1]:.4f}")
                if losses[-1] < losses[0]:
                    print("✅ Loss is decreasing - Model is learning!")
                else:
                    print("⚠ Loss not decreasing yet")
            else:
                print("⚠ Not enough loss data")
        except Exception as e:
            print(f"❌ Error checking loss: {e}")

    print("\n" + "=" * 30)
    print("💡 For detailed analysis, run: python verify_training.py")

if __name__ == "__main__":
    quick_training_check()