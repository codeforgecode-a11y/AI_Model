# API Documentation - Qwen2.5-Coder-Tools Integration

This document provides detailed API documentation for all components of the Qwen2.5-Coder-Tools integration.

## Table of Contents
1. [Model Selector API](#model-selector-api)
2. [Qwen Coder Interface API](#qwen-coder-interface-api)
3. [Code Analyzer API](#code-analyzer-api)
4. [Memory Context Analyzer API](#memory-context-analyzer-api)
5. [Enhanced Text Chat Companion API](#enhanced-text-chat-companion-api)

## Model Selector API

### Class: `ModelSelector`

Intelligent model selection based on query analysis and conversation context.

#### Constructor
```python
ModelSelector(config: Dict[str, Any] = None)
```

**Parameters:**
- `config` (Dict[str, Any], optional): Configuration dictionary for model selection parameters

#### Methods

##### `select_model(query: str, context: Dict[str, Any] = None) -> Tuple[ModelType, float]`

Selects the appropriate model based on query analysis.

**Parameters:**
- `query` (str): User input query to analyze
- `context` (Dict[str, Any], optional): Additional context information

**Returns:**
- `Tuple[ModelType, float]`: Selected model type and confidence score

**Example:**
```python
selector = ModelSelector()
model_type, confidence = selector.select_model("How do I write a Python function?")
# Returns: (ModelType.CODING, 0.85)
```

##### `analyze_query(query: str) -> Dict[str, Any]`

Performs detailed analysis of the input query.

**Parameters:**
- `query` (str): Query to analyze

**Returns:**
- `Dict[str, Any]`: Analysis results including language detection, patterns, and scores

## Qwen Coder Interface API

### Class: `QwenCoderInterface`

Specialized interface for interacting with the Qwen2.5-Coder-Tools model.

#### Constructor
```python
QwenCoderInterface(model_name: str = "hhao/qwen2.5-coder-tools:7b")
```

**Parameters:**
- `model_name` (str): Name of the Qwen model to use

#### Methods

##### `generate_response(prompt: str, context: str = "", **kwargs) -> Dict[str, Any]`

Generates a response using the Qwen coding model.

**Parameters:**
- `prompt` (str): User prompt/query
- `context` (str, optional): Conversation context
- `**kwargs`: Additional parameters for model generation

**Returns:**
- `Dict[str, Any]`: Response dictionary with content, metadata, and performance metrics

**Example:**
```python
interface = QwenCoderInterface()
response = interface.generate_response(
    prompt="Write a Python function to sort a list",
    context="We're working on data processing algorithms"
)
```

##### `validate_response(response: str, query: str) -> Dict[str, Any]`

Validates the quality and relevance of a generated response.

**Parameters:**
- `response` (str): Generated response to validate
- `query` (str): Original query

**Returns:**
- `Dict[str, Any]`: Validation results with quality scores and metrics

## Code Analyzer API

### Class: `CodeAnalyzer`

Analyzes code content and programming-related queries.

#### Constructor
```python
CodeAnalyzer()
```

#### Methods

##### `detect_language(code: str) -> str`

Detects the programming language of the given code.

**Parameters:**
- `code` (str): Code snippet to analyze

**Returns:**
- `str`: Detected programming language

**Example:**
```python
analyzer = CodeAnalyzer()
language = analyzer.detect_language("def hello(): print('world')")
# Returns: "python"
```

##### `analyze_code_patterns(text: str) -> Dict[str, Any]`

Analyzes text for code patterns and programming constructs.

**Parameters:**
- `text` (str): Text to analyze for code patterns

**Returns:**
- `Dict[str, Any]`: Analysis results with detected patterns and confidence scores

##### `classify_task_type(query: str) -> str`

Classifies the type of programming task based on the query.

**Parameters:**
- `query` (str): Query to classify

**Returns:**
- `str`: Task type (e.g., "implement", "debug", "explain", "optimize")

## Memory Context Analyzer API

### Class: `MemoryContextAnalyzer`

Analyzes conversation context for enhanced memory integration.

#### Constructor
```python
MemoryContextAnalyzer()
```

#### Methods

##### `analyze_conversation_context(interactions: List[Dict[str, Any]], current_query: str) -> Dict[str, Any]`

Analyzes conversation context to provide enhanced memory insights.

**Parameters:**
- `interactions` (List[Dict[str, Any]]): List of recent interactions
- `current_query` (str): Current user query

**Returns:**
- `Dict[str, Any]`: Context analysis results with recommendations

**Example:**
```python
analyzer = MemoryContextAnalyzer()
context = analyzer.analyze_conversation_context(
    interactions=conversation_history,
    current_query="How do I handle exceptions?"
)
```

##### `calculate_coding_score(interactions: List[Dict[str, Any]]) -> float`

Calculates how coding-focused the recent conversation has been.

**Parameters:**
- `interactions` (List[Dict[str, Any]]): List of interactions to analyze

**Returns:**
- `float`: Coding score between 0.0 and 1.0

##### `extract_relevant_context(interactions: List[Dict[str, Any]], current_query: str, context_type: str) -> List[Dict[str, Any]]`

Extracts most relevant context for the current query.

**Parameters:**
- `interactions` (List[Dict[str, Any]]): Available interactions
- `current_query` (str): Current query
- `context_type` (str): Type of context ("coding" or "general")

**Returns:**
- `List[Dict[str, Any]]`: Most relevant interactions

## Enhanced Text Chat Companion API

### Class: `TextChatCompanion`

Main interface for the enhanced chat companion with dual-model support.

#### Constructor
```python
TextChatCompanion(internet_mode: str = "auto")
```

**Parameters:**
- `internet_mode` (str): Internet connectivity mode ("auto", "online", "offline")

#### Methods

##### `process_user_input(user_input: str) -> str`

Processes user input and generates an appropriate response.

**Parameters:**
- `user_input` (str): User's input message

**Returns:**
- `str`: Generated response

**Example:**
```python
companion = TextChatCompanion()
response = companion.process_user_input("How do I write a Python function?")
```

##### `show_model_stats() -> None`

Displays comprehensive model usage statistics.

**Example:**
```python
companion.show_model_stats()
# Outputs detailed statistics about model usage, performance, etc.
```

##### `show_memory_status() -> None`

Displays enhanced memory system status with context analysis.

**Example:**
```python
companion.show_memory_status()
# Outputs memory status, context analysis, and recommendations
```

##### `get_model_recommendations() -> List[str]`

Gets recommendations for optimal model usage based on conversation patterns.

**Returns:**
- `List[str]`: List of recommendations

## Configuration Classes

### Class: `ModelType`

Enumeration of available model types.

```python
class ModelType(Enum):
    GENERAL = "general"
    CODING = "coding"
```

### Configuration Dictionaries

#### Model Selection Configuration
```python
MODEL_SELECTION_CONFIG = {
    'coding_model': 'hhao/qwen2.5-coder-tools:7b',
    'general_model': 'llama3.2:1b',
    'confidence_threshold': 0.6,
    'coding_keywords_weight': 0.4,
    'code_pattern_weight': 0.3,
    'context_weight': 0.3,
    'max_context_length': 4000,
    'response_timeout': 30
}
```

#### Memory Configuration
```python
MEMORY_CONFIG = {
    'context_window': 10,
    'relevance_threshold': 0.3,
    'topic_continuity_weight': 0.3,
    'temporal_decay': 3600
}
```

## Error Handling

### Common Exceptions

#### `ModelNotAvailableError`
Raised when a required model is not available in Ollama.

#### `InvalidQueryError`
Raised when a query cannot be processed due to invalid format or content.

#### `ContextAnalysisError`
Raised when context analysis fails due to insufficient data or processing errors.

### Error Response Format
```python
{
    'error': True,
    'error_type': 'ModelNotAvailableError',
    'message': 'Qwen model not available',
    'fallback_used': True,
    'fallback_model': 'llama3.2:1b'
}
```

## Response Formats

### Standard Response Format
```python
{
    'content': 'Generated response text',
    'model_used': 'hhao/qwen2.5-coder-tools:7b',
    'model_type': 'coding',
    'confidence': 0.85,
    'response_time': 2.34,
    'metadata': {
        'language_detected': 'python',
        'task_type': 'implement',
        'validation_score': 0.92
    }
}
```

### Context Analysis Response Format
```python
{
    'context_type': 'coding',
    'coding_score': 0.78,
    'topic_continuity': {
        'score': 0.85,
        'current_topic': 'python_functions',
        'topic_changes': 1
    },
    'model_usage_pattern': {
        'dominant_model': 'coding',
        'model_switches': 2
    },
    'recommendations': [
        'Continue with coding context',
        'Maintain technical depth'
    ]
}
```

## Usage Examples

### Basic Integration
```python
from text_chat_companion import TextChatCompanion

# Initialize the companion
companion = TextChatCompanion(internet_mode="offline")

# Process coding query
response = companion.process_user_input("Write a Python function to calculate fibonacci numbers")

# Check model statistics
companion.show_model_stats()

# View memory status
companion.show_memory_status()
```

### Advanced Usage with Custom Configuration
```python
from model_selector import ModelSelector, ModelType
from qwen_coder_interface import QwenCoderInterface

# Custom configuration
config = {
    'confidence_threshold': 0.7,
    'coding_keywords_weight': 0.5
}

# Initialize components
selector = ModelSelector(config)
qwen_interface = QwenCoderInterface()

# Manual model selection and response generation
model_type, confidence = selector.select_model("Debug this Python code")
if model_type == ModelType.CODING:
    response = qwen_interface.generate_response("Debug this Python code")
```

This API documentation provides comprehensive information for developers working with the Qwen2.5-Coder-Tools integration. For additional examples and use cases, refer to the test files and README documentation.
