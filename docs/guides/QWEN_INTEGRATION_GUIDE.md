# Qwen2.5-Coder-Tools Integration Guide

## Overview

This guide documents the integration of the Qwen2.5-Coder-Tools 7B model into the existing voice companion system. The integration provides specialized coding assistance while maintaining compatibility with the three-component memory architecture and TTS/STT functionality.

## Architecture

### Dual-Model System

The system now supports intelligent model selection between:

1. **General Model** (`llama3.2:1b`) - For general conversation
2. **Coding Model** (`hhao/qwen2.5-coder-tools:7b`) - For programming assistance
3. **Fallback Model** - Basic LLM interface for error recovery

### Key Components

#### 1. QwenCoderInterface (`qwen_coder_interface.py`)
- Specialized interface for the Qwen2.5-Coder-Tools model
- Handles coding-specific prompts and responses
- Includes code analysis and language detection
- Optimized configuration for coding tasks

#### 2. ModelSelector (`model_selector.py`)
- Intelligent query classification
- Model selection based on content analysis
- Performance tracking and fallback logic
- Confidence scoring for model decisions

#### 3. Enhanced TextChatCompanion (`text_chat_companion.py`)
- Updated to support multiple models
- Intelligent routing of queries
- Enhanced memory integration
- New `models` command for statistics

## Features

### Automatic Model Selection

The system automatically detects coding-related queries and routes them to the appropriate model:

**Coding Queries** → Qwen2.5-Coder-Tools
- Function/class definitions
- Debugging requests
- Code explanations
- Programming concepts
- Language-specific syntax

**General Queries** → llama3.2:1b
- Casual conversation
- General knowledge
- Non-technical topics

### Code Analysis

The `CodeAnalyzer` class provides:
- Programming language detection
- Coding query classification
- Code block extraction
- Pattern recognition for various languages

### Task-Specific Prompts

The Qwen interface creates specialized prompts for different coding tasks:
- **Debug**: Error analysis and fixes
- **Implement**: Code creation and development
- **Explain**: Concept clarification
- **Review**: Code quality assessment

## Configuration

### Model Settings

```python
OLLAMA_CONFIG = {
    "general_model": {
        "model_name": "llama3.2:1b",
        "temperature": 0.7,
        "max_tokens": 500,
        "timeout": 30
    },
    "coding_model": {
        "model_name": "hhao/qwen2.5-coder-tools:7b",
        "temperature": 0.3,  # Lower for precision
        "max_tokens": 800,   # Longer for code
        "timeout": 45,       # More time for complex tasks
        "num_ctx": 8192      # Larger context
    }
}
```

### Selection Thresholds

```python
MODEL_SELECTION_CONFIG = {
    "coding_threshold": 0.3,      # Min confidence for coding model
    "fallback_threshold": 3.0,    # Max response time before fallback
    "max_retries": 2,
    "prefer_specialized": True
}
```

## Usage

### New Commands

- `models` - Show model statistics and availability
- `help` - Updated to reflect dual-model features

### Example Interactions

**Coding Query:**
```
You: How do I write a Python function to sort a list?
Assistant: [Uses Qwen2.5-Coder-Tools for specialized response]
```

**General Query:**
```
You: What's the weather like today?
Assistant: [Uses llama3.2:1b for general conversation]
```

## Memory Integration

The three-component memory system works seamlessly with both models:

### Context Memory
- Maintains conversation history across model switches
- Preserves topic continuity

### Database Memory
- Stores interactions with model metadata
- Tracks model performance and usage

### Learning Memory
- Learns from coding and general interactions
- Improves model selection over time

## Performance Monitoring

The system tracks:
- Response times for each model
- Success rates and error handling
- Model selection accuracy
- Fallback frequency

Access statistics with the `models` command:
```
🤖 Model Statistics

General Model:
  Status: Available
  Success Rate: 95.2%
  Avg Response Time: 0.85s
  Recent Responses: 15

Coding Model:
  Status: Available
  Success Rate: 92.8%
  Avg Response Time: 1.45s
  Recent Responses: 8
```

## Installation and Setup

### Prerequisites

1. Ollama installed and running
2. Existing voice companion system
3. Python dependencies installed

### Installation Steps

1. **Download Qwen Model:**
   ```bash
   ollama pull hhao/qwen2.5-coder-tools:7b
   ```

2. **Verify Installation:**
   ```bash
   python test_qwen_integration.py
   ```

3. **Start Enhanced Companion:**
   ```bash
   python text_chat_companion.py
   ```

## Testing

The integration includes comprehensive tests:

- **Import Tests**: Verify all modules load correctly
- **Code Analyzer Tests**: Validate query classification
- **Model Selector Tests**: Check intelligent routing
- **Qwen Interface Tests**: Confirm model availability
- **Memory Integration Tests**: Verify data persistence
- **Full Integration Tests**: End-to-end validation

Run tests with:
```bash
python test_qwen_integration.py
```

## Troubleshooting

### Common Issues

1. **Qwen Model Not Available**
   - Ensure model is downloaded: `ollama list`
   - Check Ollama service is running
   - System falls back to general model

2. **Slow Response Times**
   - Monitor with `models` command
   - Adjust timeout settings in configuration
   - Check system resources

3. **Incorrect Model Selection**
   - Review query classification logic
   - Adjust confidence thresholds
   - Check coding keyword detection

### Fallback Behavior

The system gracefully handles failures:
1. Qwen model unavailable → Use general model
2. General model fails → Use basic LLM interface
3. All models fail → Display error message

## Future Enhancements

Potential improvements:
- Additional specialized models (e.g., for specific languages)
- Enhanced code analysis with AST parsing
- Integration with code execution environments
- Voice-to-code functionality
- Real-time collaboration features

## Support

For issues or questions:
1. Check the test results for diagnostic information
2. Review logs for error details
3. Verify model availability with `ollama list`
4. Use the `models` command for system status

The integration maintains backward compatibility while adding powerful coding assistance capabilities to your voice companion system.
