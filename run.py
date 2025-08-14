#!/usr/bin/env python3
"""
AugmentCode Launcher
Simple launcher script for the AI assistant applications.
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Main launcher function."""
    if len(sys.argv) < 2:
        print("AugmentCode AI Assistant Launcher")
        print("=" * 40)
        print("Usage: python run.py <application>")
        print()
        print("Available applications:")
        print("  text    - Text-based chat companion")
        print("  voice   - Voice-based companion")
        print("  train   - Model training interface")
        print()
        print("Examples:")
        print("  python run.py text")
        print("  python run.py voice")
        print("  python run.py train")
        return

    app = sys.argv[1].lower()
    
    # Map applications to their file paths
    apps = {
        'text': 'src/companions/text_chat_companion.py',
        'voice': 'src/companions/voice_companion.py',
        'train': 'src/training/train_model.py'
    }
    
    if app not in apps:
        print(f"Error: Unknown application '{app}'")
        print(f"Available applications: {', '.join(apps.keys())}")
        return
    
    script_path = Path(apps[app])
    if not script_path.exists():
        print(f"Error: Application file not found: {script_path}")
        return
    
    print(f"Starting {app} application...")
    
    # Pass any additional arguments to the application
    cmd = [sys.executable, str(script_path)] + sys.argv[2:]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print(f"\n{app.title()} application stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"Error running {app} application: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
