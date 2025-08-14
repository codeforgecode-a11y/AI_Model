#!/usr/bin/env python3
"""
Setup script for Offline AI Voice Companion
Handles installation of dependencies and system requirements
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors."""
    print(f"🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_ollama():
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run("ollama --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            
            # Check if Ollama service is running
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    print(f"✅ Ollama is running with {len(models)} models")
                    for model in models:
                        print(f"   - {model['name']}")
                    return True
                else:
                    print("⚠️  Ollama is installed but not running")
                    return False
            except:
                print("⚠️  Ollama is installed but not accessible")
                return False
        else:
            print("❌ Ollama is not installed")
            return False
    except:
        print("❌ Ollama is not installed")
        return False

def install_system_dependencies():
    """Install system-level dependencies based on OS."""
    system = platform.system().lower()

    print(f"📦 System detected: {system}")
    print("ℹ️  Text-based chat companion requires no special system dependencies.")
    print("✅ All required dependencies will be installed via pip.")

def install_python_dependencies():
    """Install Python dependencies."""
    print("🐍 Installing Python dependencies...")
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if Path("requirements.txt").exists():
        success = run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                            "Installing Python packages")
        if not success:
            print("⚠️  Some packages failed to install. Trying individual installation...")
            
            # Try installing packages individually
            packages = [
                "ollama>=0.1.7",
                "requests>=2.31.0",
                "python-dotenv>=1.0.0",
                "colorama>=0.4.6"
            ]
            
            for package in packages:
                run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}")
    else:
        print("❌ requirements.txt not found")

def download_ollama_model():
    """Download a recommended Ollama model optimized for i3 7th gen systems."""
    print("🤖 Setting up Ollama model (optimized for 8GB RAM)...")

    if not check_ollama():
        print("❌ Ollama is not running. Please start Ollama first:")
        print("   Linux/macOS: ollama serve")
        print("   Windows: Start Ollama from the application")
        return False

    # Prioritize lightweight models for i3 systems
    models_to_try = [
        ("llama3.2:1b", "1.3GB - Best for i3 systems"),
        ("phi3:mini", "2.3GB - Alternative lightweight model"),
        ("gemma2:2b", "1.6GB - Google's efficient model"),
        ("qwen2:1.5b", "0.9GB - Very lightweight option")
    ]

    print("💡 Recommended models for your i3 7th gen system:")
    for model, description in models_to_try:
        print(f"   - {model}: {description}")

    for model, description in models_to_try:
        print(f"\n🔄 Trying to download {model} ({description})...")
        if run_command(f"ollama pull {model}", f"Downloading {model}"):
            print(f"✅ Successfully downloaded {model}")
            print(f"💡 This model is optimized for your 8GB RAM system")
            return True
        else:
            print(f"⚠️  Failed to download {model}, trying next...")

    print("❌ Failed to download any model. Please manually run:")
    print("   ollama pull llama3.2:1b  # Recommended for your system")
    return False

def test_installation():
    """Test if all components are working."""
    print("\n🧪 Testing installation...")

    # Test core imports
    try:
        import ollama
        response = ollama.list()
        print(f"✅ Ollama connection successful, {len(response['models'])} models available")
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")

    try:
        import requests
        print("✅ Requests library import successful")
    except ImportError as e:
        print(f"❌ Requests import failed: {e}")

    try:
        import colorama
        print("✅ Colorama import successful")
    except ImportError as e:
        print(f"❌ Colorama import failed: {e}")

def main():
    """Main setup function."""
    print("🚀 Setting up Text-Based AI Chat Companion")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install system dependencies
    install_system_dependencies()
    
    # Install Python dependencies
    install_python_dependencies()
    
    # Check Ollama and download model
    if check_ollama():
        download_ollama_model()
    else:
        print("\n⚠️  Ollama setup required:")
        print("1. Install Ollama from https://ollama.ai")
        print("2. Start Ollama service")
        print("3. Run: ollama pull llama3.2")
    
    # Test installation
    test_installation()
    
    print("\n🎉 Setup complete!")
    print("To start the applications, run:")
    print("   python src/companions/text_chat_companion.py  # Text-based chat")
    print("   python src/companions/voice_companion.py      # Voice-based chat")

if __name__ == "__main__":
    main()
