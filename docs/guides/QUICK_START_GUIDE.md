# Quick Start Guide - Qwen2.5-Coder-Tools Integration

Get up and running with the enhanced Text Chat Companion featuring dual-model architecture in just a few minutes!

## Prerequisites

Before you begin, ensure you have:
- Python 3.8 or higher
- Ollama installed and running
- At least 8GB of available RAM (for the 7B model)
- Stable internet connection for initial model download

## Step 1: Install Ollama

If you haven't installed Ollama yet:

### Linux/macOS
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Windows
Download and install from [ollama.ai](https://ollama.ai)

### Start Ollama Service
```bash
ollama serve
```

## Step 2: Install Required Models

### Install the General Model (if not already installed)
```bash
ollama pull llama3.2:1b
```

### Install the Qwen Coding Model
```bash
ollama pull hhao/qwen2.5-coder-tools:7b
```

**Note**: The Qwen model is approximately 4.7GB and may take some time to download depending on your internet connection.

### Verify Installation
```bash
ollama list
```

You should see both models listed:
```
NAME                           ID              SIZE      MODIFIED
hhao/qwen2.5-coder-tools:7b   abc123def456    4.7 GB    2 minutes ago
llama3.2:1b                   def456ghi789    1.3 GB    1 hour ago
```

## Step 3: Install Python Dependencies

Create a virtual environment (recommended):
```bash
python -m venv qwen_env
source qwen_env/bin/activate  # On Windows: qwen_env\Scripts\activate
```

Install required packages:
```bash
pip install ollama colorama sqlite3
```

## Step 4: Download Integration Files

Ensure you have all the required files in your working directory:
- `text_chat_companion.py` (enhanced main file)
- `model_selector.py`
- `qwen_coder_interface.py`
- `code_analyzer.py`
- `memory_context_analyzer.py`
- `Memory.py` (existing memory system)
- `enhanced_llm.py` (existing enhanced LLM)

## Step 5: Test the Integration

### Quick Test
```bash
python test_qwen_integration.py
```

This will run basic tests to ensure everything is working correctly.

### End-to-End Test
```bash
python test_end_to_end_integration.py
```

This performs comprehensive testing including actual model responses.

## Step 6: Start Using the Enhanced Companion

### Basic Usage
```bash
python text_chat_companion.py
```

### Example Interactions

#### Coding Query
```
You: How do I write a Python function to calculate factorial?
Assistant: [Using Qwen2.5-Coder-Tools] Here's a Python function to calculate factorial:

def factorial(n):
    """Calculate factorial of a non-negative integer."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

# Alternative iterative approach:
def factorial_iterative(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Usage examples:
print(factorial(5))  # Output: 120
print(factorial_iterative(5))  # Output: 120
```

#### General Conversation
```
You: What's the weather like today?
Assistant: [Using Llama3.2:1b] I don't have access to real-time weather data, but I'd be happy to help you find weather information! You can check the weather by:

1. Looking at weather apps on your phone
2. Visiting weather websites like weather.com or weather.gov
3. Asking voice assistants like Siri or Google Assistant
4. Checking local news stations

Is there anything else I can help you with?
```

#### Context-Aware Switching
```
You: How do I write a Python function?
Assistant: [Using Qwen2.5-Coder-Tools] Here's how to write a basic Python function...

You: Can you optimize this code?
Assistant: [Using Qwen2.5-Coder-Tools] Certainly! Based on our previous discussion about Python functions, here are some optimization techniques...
```

## Step 7: Monitor Performance

### View Model Statistics
While the companion is running, type:
```
/stats
```

This shows:
- Model usage patterns
- Response times
- Confidence scores
- Switching frequency

### View Memory Status
Type:
```
/memory
```

This displays:
- Memory system status
- Context analysis
- Topic continuity
- Recommendations

## Troubleshooting

### Common Issues and Solutions

#### 1. "Model not found" Error
**Problem**: Qwen model not available
**Solution**:
```bash
ollama pull hhao/qwen2.5-coder-tools:7b
ollama list  # Verify installation
```

#### 2. Slow Response Times
**Problem**: Model responses are taking too long
**Solutions**:
- Ensure sufficient RAM (8GB+ recommended)
- Close other resource-intensive applications
- Check if Ollama service is running properly
- Consider using a smaller model for testing

#### 3. Incorrect Model Selection
**Problem**: System choosing wrong model for queries
**Solutions**:
- Check confidence threshold in configuration
- Review coding keywords in your query
- Use explicit model selection if needed
- Check context from previous interactions

#### 4. Memory System Issues
**Problem**: Context not being maintained properly
**Solutions**:
- Verify SQLite database permissions
- Check memory configuration settings
- Clear memory database if corrupted
- Restart the companion

### Debug Mode
Enable detailed logging:
```bash
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from text_chat_companion import TextChatCompanion
companion = TextChatCompanion()
"
```

## Configuration Customization

### Adjust Model Selection Sensitivity
Edit the configuration in `text_chat_companion.py`:

```python
Config.MODEL_SELECTION_CONFIG = {
    'confidence_threshold': 0.7,  # Higher = more selective
    'coding_keywords_weight': 0.5,  # Increase for more coding detection
    'code_pattern_weight': 0.3,
    'context_weight': 0.2
}
```

### Memory System Tuning
```python
Config.MEMORY_CONFIG = {
    'context_window': 15,  # More context history
    'relevance_threshold': 0.2,  # Lower = more inclusive
    'topic_continuity_weight': 0.4
}
```

## Performance Tips

### Optimize for Your Use Case

#### For Coding-Heavy Workflows
- Increase `coding_keywords_weight` to 0.6
- Lower `confidence_threshold` to 0.5
- Increase `context_window` to 20

#### For General Conversation
- Decrease `coding_keywords_weight` to 0.2
- Increase `confidence_threshold` to 0.8
- Focus on conversational context

#### For Mixed Usage
- Keep default settings
- Monitor `/stats` to adjust based on usage patterns
- Use explicit model hints when needed

## Next Steps

### Explore Advanced Features
1. **Custom Prompting**: Modify prompts in `qwen_coder_interface.py`
2. **Memory Analysis**: Use `memory_context_analyzer.py` for insights
3. **Code Analysis**: Leverage `code_analyzer.py` for language detection
4. **Model Statistics**: Monitor performance with built-in analytics

### Integration with Development Tools
- Set up as a coding assistant in your IDE
- Create custom scripts for specific workflows
- Integrate with version control systems
- Build automated code review processes

### Community and Support
- Check the README for detailed documentation
- Review API documentation for advanced usage
- Run test suites to verify functionality
- Contribute improvements and bug fixes

## Success Indicators

You'll know the integration is working correctly when:
- ✅ Both models appear in `ollama list`
- ✅ Test scripts pass without errors
- ✅ Coding queries automatically use Qwen model
- ✅ General queries use Llama model
- ✅ Context is maintained across interactions
- ✅ Model statistics show appropriate usage patterns

## Getting Help

If you encounter issues:
1. Check this troubleshooting section
2. Review the detailed README documentation
3. Run the test suites to identify specific problems
4. Check Ollama service status and logs
5. Verify model availability and configuration

Happy coding with your enhanced AI companion! 🚀
