# Performance Optimization Guide for i3 7th Gen Systems

This guide provides specific optimizations for running the Voice Companion on systems with:
- Intel i3 7th generation processor (dual-core)
- 8GB RAM
- Integrated graphics (no dedicated GPU)

## 🎯 Optimized Configuration

The system has been pre-configured with these optimizations:

### Whisper Model Settings
- **Model**: `tiny` (39MB vs 142MB for base)
- **Device**: CPU-only (no GPU acceleration)
- **FP16**: Disabled for CPU compatibility
- **Language**: Fixed to English (avoids detection overhead)

### Ollama Model Settings
- **Recommended Model**: `llama3.2:1b` (1.3GB RAM usage)
- **Context Window**: Reduced to 2048 tokens
- **Thread Limit**: 2 threads (matches dual-core CPU)
- **Response Length**: Limited to 100 tokens for faster generation

### Memory Optimizations
- **Conversation History**: Limited to 10 exchanges
- **Audio Buffer**: Reduced to 2048 bytes
- **Model Loading**: Whisper kept in memory for faster inference

## 📊 Expected Performance

### Processing Times (Approximate)
- **Audio Recording**: Real-time
- **Speech-to-Text**: 2-5 seconds for 5-second audio
- **LLM Response**: 5-15 seconds depending on complexity
- **Text-to-Speech**: 1-2 seconds

### Memory Usage
- **Base System**: ~2GB
- **Whisper Model**: ~200MB
- **Ollama Model**: ~1.5GB
- **Total Peak**: ~4GB (well within 8GB limit)

## 🚀 Installation Steps for Your System

### 1. Install Dependencies
```bash
# Activate virtual environment
source env/bin/activate

# Install optimized requirements
pip install -r requirements.txt
```

### 2. Download Optimal Ollama Model
```bash
# Start Ollama service
ollama serve

# In another terminal, download the 1B model
ollama pull llama3.2:1b
```

### 3. Test System Performance
```bash
python test_components.py
```

## ⚡ Performance Tips

### System-Level Optimizations

1. **Close Unnecessary Applications**
   ```bash
   # Check memory usage
   free -h
   
   # Close browser tabs and other apps
   ```

2. **Set CPU Governor to Performance** (Linux)
   ```bash
   sudo cpupower frequency-set -g performance
   ```

3. **Disable Swap if Possible** (if you have enough RAM)
   ```bash
   sudo swapoff -a  # Temporary
   ```

### Application-Level Optimizations

1. **Use Shorter Audio Clips**
   - Keep voice inputs under 10 seconds
   - Pause between thoughts for better processing

2. **Optimize Conversation Flow**
   - Ask concise questions
   - Avoid very long conversations (history gets cleared automatically)

3. **Monitor Resource Usage**
   ```bash
   # Monitor CPU and memory
   htop
   
   # Monitor during voice companion usage
   watch -n 1 'free -h && echo "---" && ps aux | grep -E "(python|ollama)" | head -5'
   ```

## 🔧 Troubleshooting Performance Issues

### If Whisper is Too Slow
```python
# In config.py, try even smaller model
WHISPER_CONFIG = {
    "model_size": "tiny.en",  # English-only version is faster
    # ... other settings
}
```

### If Ollama is Too Slow
```bash
# Try an even smaller model
ollama pull qwen2:1.5b  # 0.9GB model

# Or reduce context further in config.py
OLLAMA_CONFIG = {
    "num_ctx": 1024,  # Reduce from 2048
    # ... other settings
}
```

### If Audio Processing Lags
```python
# In config.py, reduce audio quality
AUDIO_CONFIG = {
    "sample_rate": 8000,  # Reduce from 16000
    "chunk_size": 512,    # Reduce from 1024
    # ... other settings
}
```

## 📈 Monitoring Performance

### Real-time Monitoring Script
```bash
# Create a monitoring script
cat > monitor_performance.sh << 'EOF'
#!/bin/bash
echo "Voice Companion Performance Monitor"
echo "=================================="
while true; do
    clear
    echo "$(date)"
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
    echo ""
    echo "Memory Usage:"
    free -h | grep "Mem:"
    echo ""
    echo "Voice Companion Processes:"
    ps aux | grep -E "(python|ollama)" | grep -v grep
    echo ""
    sleep 2
done
EOF

chmod +x monitor_performance.sh
./monitor_performance.sh
```

## 🎛️ Advanced Optimizations

### 1. Compile Whisper with Optimizations
```bash
# Install with CPU optimizations
pip uninstall openai-whisper
pip install openai-whisper --no-cache-dir --force-reinstall
```

### 2. Use Quantized Models
```bash
# Try quantized Ollama models (if available)
ollama pull llama3.2:1b-q4_0  # 4-bit quantization
```

### 3. Optimize Python Runtime
```bash
# Use Python with optimizations
export PYTHONOPTIMIZE=1
python voice_companion.py
```

## 🔍 Benchmarking Your System

Run this benchmark to test your system's capabilities:

```python
# benchmark.py
import time
import whisper
import ollama

def benchmark_whisper():
    print("Benchmarking Whisper...")
    model = whisper.load_model("tiny", device="cpu")
    
    # Test with 5-second silence
    import numpy as np
    audio = np.zeros(16000 * 5, dtype=np.float32)
    
    start = time.time()
    result = model.transcribe(audio, fp16=False)
    end = time.time()
    
    print(f"Whisper processing time: {end - start:.2f} seconds")

def benchmark_ollama():
    print("Benchmarking Ollama...")
    
    start = time.time()
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": "Say hello"}],
        options={"num_predict": 10}
    )
    end = time.time()
    
    print(f"Ollama response time: {end - start:.2f} seconds")
    print(f"Response: {response['message']['content']}")

if __name__ == "__main__":
    benchmark_whisper()
    benchmark_ollama()
```

## 📋 Recommended Workflow

1. **Start Ollama**: `ollama serve`
2. **Monitor Resources**: Run monitoring script in background
3. **Start Voice Companion**: `python voice_companion.py`
4. **Keep Sessions Short**: 5-10 exchanges max
5. **Restart if Slow**: Memory can accumulate over time

## 🎯 Expected Results

With these optimizations, your i3 7th gen system should provide:
- ✅ Responsive voice recognition (2-5 seconds)
- ✅ Reasonable AI responses (5-15 seconds)
- ✅ Stable performance for 30+ minute sessions
- ✅ Memory usage under 4GB total

If performance is still unsatisfactory, consider upgrading to a system with:
- Quad-core CPU (i5 or better)
- 16GB RAM
- Dedicated GPU for AI acceleration
