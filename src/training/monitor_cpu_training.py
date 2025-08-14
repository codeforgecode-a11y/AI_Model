#!/usr/bin/env python3
"""
CPU Training Monitor
Monitors CPU training progress, memory usage, and provides estimates
"""

import os
import time
import psutil
import json
import argparse
from datetime import datetime, timedelta
import threading
from pathlib import Path

class CPUTrainingMonitor:
    """Monitor CPU training progress and system resources"""

    def __init__(self, output_dir="./cpu_trained_model", log_file="cpu_training.log"):
        self.output_dir = output_dir
        self.log_file = log_file
        self.start_time = None
        self.monitoring = False
        self.stats = {
            'start_time': None,
            'current_step': 0,
            'total_steps': 0,
            'current_epoch': 0,
            'total_epochs': 3,
            'memory_usage': [],
            'cpu_usage': [],
            'estimated_completion': None
        }

    def start_monitoring(self):
        """Start monitoring training progress"""
        self.monitoring = True
        self.start_time = datetime.now()
        self.stats['start_time'] = self.start_time.isoformat()

        print("🖥️  CPU Training Monitor Started")
        print(f"📁 Monitoring directory: {self.output_dir}")
        print(f"📝 Log file: {self.log_file}")
        print("=" * 60)

        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()

        return monitor_thread

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        print("\n🛑 Monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._update_stats()
                self._display_status()
                time.sleep(30)  # Update every 30 seconds
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(60)

    def _update_stats(self):
        """Update training statistics"""
        # Get system stats
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()

        self.stats['memory_usage'].append(memory_mb)
        self.stats['cpu_usage'].append(cpu_percent)

        # Keep only last 100 readings
        if len(self.stats['memory_usage']) > 100:
            self.stats['memory_usage'] = self.stats['memory_usage'][-100:]
            self.stats['cpu_usage'] = self.stats['cpu_usage'][-100:]

        # Try to parse training progress from logs
        self._parse_training_progress()

        # Estimate completion time
        self._estimate_completion()

    def _parse_training_progress(self):
        """Parse training progress from log file"""
        if not os.path.exists(self.log_file):
            return

        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()

            # Look for training step information
            for line in reversed(lines[-50:]):  # Check last 50 lines
                if 'step' in line.lower() and '/' in line:
                    # Try to extract step information
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'step' in part.lower() and i + 1 < len(parts):
                            try:
                                step_info = parts[i + 1]
                                if '/' in step_info:
                                    current, total = step_info.split('/')
                                    self.stats['current_step'] = int(current)
                                    self.stats['total_steps'] = int(total)
                                    break
                            except:
                                continue

                # Look for epoch information
                if 'epoch' in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'epoch' in part.lower() and i + 1 < len(parts):
                            try:
                                epoch_info = parts[i + 1]
                                if '/' in epoch_info:
                                    current, total = epoch_info.split('/')
                                    self.stats['current_epoch'] = int(current)
                                    self.stats['total_epochs'] = int(total)
                                    break
                            except:
                                continue
        except Exception as e:
            pass  # Ignore parsing errors

    def _estimate_completion(self):
        """Estimate training completion time"""
        if self.stats['current_step'] > 0 and self.stats['total_steps'] > 0:
            elapsed = datetime.now() - self.start_time
            progress = self.stats['current_step'] / self.stats['total_steps']

            if progress > 0:
                total_estimated = elapsed / progress
                remaining = total_estimated - elapsed
                completion_time = datetime.now() + remaining
                self.stats['estimated_completion'] = completion_time.isoformat()

    def _display_status(self):
        """Display current training status"""
        os.system('clear' if os.name == 'posix' else 'cls')

        print("🖥️  CPU Training Monitor")
        print("=" * 60)

        # Training progress
        if self.stats['total_steps'] > 0:
            progress = (self.stats['current_step'] / self.stats['total_steps']) * 100
            print(f"📊 Progress: {progress:.1f}% ({self.stats['current_step']}/{self.stats['total_steps']} steps)")
        else:
            print("📊 Progress: Initializing...")

        if self.stats['total_epochs'] > 0:
            print(f"🔄 Epoch: {self.stats['current_epoch']}/{self.stats['total_epochs']}")

        # Time information
        elapsed = datetime.now() - self.start_time
        print(f"⏱️  Elapsed: {str(elapsed).split('.')[0]}")

        if self.stats['estimated_completion']:
            completion = datetime.fromisoformat(self.stats['estimated_completion'])
            remaining = completion - datetime.now()
            if remaining.total_seconds() > 0:
                print(f"⏳ Estimated remaining: {str(remaining).split('.')[0]}")
                print(f"🎯 Estimated completion: {completion.strftime('%Y-%m-%d %H:%M:%S')}")

        # System resources
        if self.stats['memory_usage']:
            current_memory = self.stats['memory_usage'][-1]
            avg_memory = sum(self.stats['memory_usage']) / len(self.stats['memory_usage'])
            max_memory = max(self.stats['memory_usage'])

            print(f"💾 Memory: {current_memory:.1f} MB (avg: {avg_memory:.1f} MB, max: {max_memory:.1f} MB)")

        if self.stats['cpu_usage']:
            current_cpu = self.stats['cpu_usage'][-1]
            avg_cpu = sum(self.stats['cpu_usage']) / len(self.stats['cpu_usage'])
            print(f"🔥 CPU: {current_cpu:.1f}% (avg: {avg_cpu:.1f}%)")

        # Checkpoints
        if os.path.exists(self.output_dir):
            checkpoints = [d for d in os.listdir(self.output_dir) if d.startswith('checkpoint-')]
            if checkpoints:
                latest = max(checkpoints, key=lambda x: int(x.split('-')[1]))
                print(f"💾 Latest checkpoint: {latest}")

        print("=" * 60)
        print("Press Ctrl+C to stop monitoring")

def main():
    """Main monitoring function"""
    parser = argparse.ArgumentParser(description="Monitor CPU training progress")
    parser.add_argument("--output-dir", default="./cpu_trained_model", help="Training output directory")
    parser.add_argument("--log-file", default="cpu_training.log", help="Training log file")

    args = parser.parse_args()

    monitor = CPUTrainingMonitor(args.output_dir, args.log_file)

    try:
        monitor_thread = monitor.start_monitoring()

        # Keep main thread alive
        while monitor.monitoring:
            time.sleep(1)

    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("\nMonitoring stopped by user")

if __name__ == "__main__":
    main()