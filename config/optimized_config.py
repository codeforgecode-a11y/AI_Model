"""
Optimized Configuration for Enhanced Text Chat Companion Performance

This configuration focuses on:
1. Result Quality: Better AI responses, context understanding, accuracy
2. Text Processing: Optimized text input/output handling and response generation
"""

import os
from pathlib import Path

# === Base directories ===
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# === ENHANCED OLLAMA SETTINGS FOR BETTER RESULT QUALITY ===
OLLAMA_CONFIG = {
    # Model selection for better responses
    "model_name": "llama3.2:1b",    # Lightweight model optimized for text chat
    "base_url": "http://localhost:11434",

    # Response quality parameters
    "temperature": 0.7,             # Balanced creativity vs consistency
    "top_k": 40,                    # Top-k sampling for better quality
    "top_p": 0.9,                   # Nucleus sampling for coherent responses
    "repeat_penalty": 1.1,          # Prevent repetition
    "presence_penalty": 0.0,        # Encourage topic diversity
    "frequency_penalty": 0.0,       # Reduce word repetition

    # Context and length settings
    "num_ctx": 4096,                # Larger context window for better understanding
    "num_predict": 500,             # Longer responses for more complete text answers
    "max_tokens": 500,              # Maximum response length for text

    # Performance settings
    "timeout": 30,                  # Reasonable timeout for text responses
    "num_thread": 4,                # CPU threads for processing
    "num_gpu": 0,                   # Use CPU for consistency

    # Advanced generation settings
    "mirostat": 0,                  # Mirostat sampling (0=disabled, 1=v1, 2=v2)
    "mirostat_eta": 0.1,           # Mirostat learning rate
    "mirostat_tau": 5.0,           # Mirostat target entropy
    "penalize_newline": False,      # Allow natural paragraph breaks in text
    "stop": ["Human:", "Assistant:", "User:"],  # Stop sequences
}

# === TEXT INPUT SETTINGS FOR BETTER USER EXPERIENCE ===
TEXT_INPUT_CONFIG = {
    "enabled": True,                # Enable text input mode
    "max_length": 2000,             # Maximum characters per input (increased for text mode)
    "min_length": 1,                # Minimum characters per input
    "strip_whitespace": True,       # Remove leading/trailing whitespace
    "allow_empty": False,           # Don't allow empty input
    "encoding": "utf-8",            # Text encoding
    "prompt": "You: ",              # Input prompt
    "multiline": False,             # Single line input for simplicity
    "timeout": 300,                 # Input timeout in seconds (5 minutes)

    # Text processing
    "normalize_unicode": True,      # Normalize unicode characters
    "filter_control_chars": True,   # Remove control characters
    "max_consecutive_spaces": 2,    # Limit consecutive spaces
}

# === ENHANCED PERFORMANCE SETTINGS ===
PERFORMANCE_CONFIG = {
    # Memory management
    "max_conversation_history": 50,     # Increased for better context in text mode
    "context_window_size": 10,          # Recent context for responses
    "memory_cleanup_interval": 100,     # Clean up memory every N interactions

    # Text processing
    "response_timeout": 30,             # Timeout for text responses
    "text_processing_threads": 2,       # Parallel text processing
    "input_validation_timeout": 5,      # Input validation timeout

    # Response optimization
    "enable_response_caching": True,    # Cache similar responses
    "cache_size": 100,                  # Number of cached responses (increased for text)
    "similarity_threshold": 0.85,       # Threshold for cache hits
    "enable_context_compression": True, # Compress old context
    "compression_ratio": 0.5,           # Context compression ratio

    # Text-specific optimizations
    "max_input_processing_time": 10,    # Maximum time to process input
    "enable_parallel_processing": True, # Enable parallel processing where possible
    "batch_size": 1,                    # Process one message at a time for responsiveness
}

# === ENHANCED RESULT QUALITY SETTINGS ===
RESULT_QUALITY_CONFIG = {
    # Context enhancement
    "enable_context_memory": True,      # Use conversation context
    "context_relevance_threshold": 0.7, # Relevance threshold for context
    "max_context_tokens": 2000,         # Maximum context tokens (increased for text)

    # Response improvement
    "enable_response_validation": True,  # Validate response quality
    "min_response_length": 10,          # Minimum response length
    "max_response_length": 1000,        # Maximum response length (increased for text)
    "enable_fact_checking": False,      # Basic fact checking (requires internet)
    "enable_sentiment_analysis": True,  # Analyze response sentiment

    # Query understanding
    "enable_intent_detection": True,    # Detect user intent
    "enable_entity_extraction": True,   # Extract entities from queries
    "enable_query_expansion": True,     # Expand queries for better understanding

    # Response formatting
    "enable_response_formatting": True, # Format responses for clarity
    "remove_redundancy": True,          # Remove redundant information
    "improve_coherence": True,          # Improve response coherence
    "add_examples": True,               # Add examples when helpful in text mode
    "enable_markdown_formatting": False, # Enable markdown in responses
}

# === TEXT PROCESSING ENHANCEMENT SETTINGS ===
TEXT_PROCESSING_CONFIG = {
    # Input processing
    "enable_spell_checking": True,      # Basic spell checking for user input
    "enable_grammar_checking": False,   # Grammar checking (resource intensive)
    "auto_correct_typos": True,         # Auto-correct common typos
    "normalize_whitespace": True,       # Normalize whitespace in input

    # Output processing
    "enable_text_formatting": True,    # Format output text for readability
    "preserve_user_formatting": True,  # Preserve user's text formatting
    "smart_punctuation": True,         # Smart punctuation handling
    "paragraph_breaks": True,          # Add appropriate paragraph breaks

    # Language processing
    "detect_language": False,          # Auto-detect input language
    "default_language": "en",          # Default language
    "enable_translation": False,       # Enable translation features
}

# === INTERNET AND LEARNING SETTINGS ===
INTERNET_CONFIG = {
    "enabled": True,
    "default_mode": "hybrid",
    "fallback_to_offline": True,
    "connection_timeout": 8,
    "request_timeout": 25,
    "max_retries": 2,
    "user_agent": "TextChatCompanion/2.0 (Enhanced AI Assistant)",
    "rate_limit_delay": 0.5,
    "enable_caching": True,             # Cache internet responses
    "cache_duration": 3600,             # Cache duration in seconds
}

# === WEB SEARCH OPTIMIZATION ===
WEB_SEARCH_CONFIG = {
    "enabled": True,
    "engine": "duckduckgo",
    "max_results": 3,                   # Fewer results for faster processing
    "snippet_length": 150,              # Shorter snippets for better focus
    "search_timeout": 10,               # Faster timeout
    "safe_search": True,
    "region": "us-en",
    "enable_result_ranking": True,      # Rank results by relevance
    "filter_duplicates": True,          # Remove duplicate results
}

# === REAL-TIME DATA OPTIMIZATION ===
REALTIME_DATA_CONFIG = {
    "weather": {
        "enabled": True,
        "api_provider": "openweathermap",
        "api_key": "",
        "default_location": "auto",
        "units": "metric",
        "timeout": 8,
        "cache_duration": 600,          # Cache weather for 10 minutes
    },
    "news": {
        "enabled": True,
        "api_provider": "newsapi",
        "api_key": "",
        "country": "us",
        "category": "general",
        "max_articles": 3,              # Fewer articles for faster processing
        "timeout": 12,
        "cache_duration": 1800,         # Cache news for 30 minutes
    },
    "time": {
        "enabled": True,
        "timezone_api": "worldtimeapi",
        "timeout": 5,
        "cache_duration": 3600,         # Cache time data for 1 hour
    }
}

# === LOGGING OPTIMIZATION ===
LOGGING_CONFIG = {
    "level": "INFO",
    "file": LOGS_DIR / "voice_companion_optimized.log",
    "max_size": 20 * 1024 * 1024,      # 20MB for more detailed logs
    "backup_count": 5,
    "enable_performance_logging": True, # Log performance metrics
    "log_audio_stats": True,           # Log audio processing statistics
    "log_response_times": True,        # Log response generation times
}

# === TEXT INPUT SETTINGS ===
TEXT_INPUT_CONFIG = {
    "enabled": True,
    "max_length": 500,                  # Reasonable limit for better processing
    "min_length": 1,
    "strip_whitespace": True,
    "allow_empty": False,
    "encoding": "utf-8",
    "prompt": "💬 Enter your message: ",
    "multiline": False,
    "timeout": 300,
    "enable_autocomplete": False,       # Autocomplete suggestions
    "enable_history": True,             # Input history
    "history_size": 20,                 # Number of inputs to remember
}

# === INPUT MODE SETTINGS ===
INPUT_MODE_CONFIG = {
    "default_mode": "voice",
    "allow_mode_switching": True,
    "text_only_mode": False,
    "voice_only_mode": False,
    "show_mode_menu": True,
    "auto_detect_mode": True,           # Auto-detect best input mode
    "fallback_to_text": True,           # Fallback to text if voice fails
}

# === WAKE WORD OPTIMIZATION ===
WAKE_WORD_CONFIG = {
    "enabled": True,
    "word": "hey jarvis",
    "sensitivity": 0.3,                 # Balanced sensitivity
    "timeout": 5.0,                     # Wake word timeout
    "confirmation_required": False,     # Require confirmation after wake word
    "enable_continuous_listening": True, # Continuous wake word detection
}
