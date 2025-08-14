"""
Configuration settings for the Voice Companion
"""

import os
from pathlib import Path


# === Base directories ===
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# === Ollama settings ===
OLLAMA_CONFIG = {
    "model_name": "llama3.2:1b",  # Lightweight model for text-based chat
    "base_url": "http://localhost:11434",
    "temperature": 0.7,           # Good balance for conversational responses
    "max_tokens": 500,            # Increased for fuller text responses
    "timeout": 30,
    "num_ctx": 2048,
    "num_predict": 500
}

# Text input settings
TEXT_INPUT_CONFIG = {
    "enabled": True,              # Enable text input mode
    "max_length": 1000,           # Maximum characters per input
    "min_length": 1,              # Minimum characters per input
    "strip_whitespace": True,     # Remove leading/trailing whitespace
    "allow_empty": False,         # Allow empty input (just press Enter)
    "encoding": "utf-8",          # Text encoding
    "prompt": "You: ",            # Input prompt
    "multiline": False,           # Allow multiline input
    "timeout": 300                # Input timeout in seconds (5 minutes)
}

# Internet learning settings
INTERNET_CONFIG = {
    "enabled": True,              # Enable internet learning capabilities
    "default_mode": "hybrid",     # Default mode: "offline", "online", "hybrid"
    "fallback_to_offline": True,  # Fallback to offline mode if internet fails
    "connection_timeout": 10,     # Connection timeout in seconds
    "request_timeout": 30,        # Request timeout in seconds
    "max_retries": 3,             # Maximum retry attempts
    "user_agent": "TextChatCompanion/1.0 (Educational AI Assistant)",
    "rate_limit_delay": 1.0,      # Delay between requests in seconds
}

# Web search settings
WEB_SEARCH_CONFIG = {
    "enabled": True,              # Enable web search functionality
    "engine": "duckduckgo",       # Search engine: "duckduckgo", "google" (requires API key)
    "max_results": 5,             # Maximum search results to process
    "snippet_length": 200,        # Maximum length of text snippets
    "search_timeout": 15,         # Search timeout in seconds
    "safe_search": True,          # Enable safe search
    "region": "us-en",            # Search region/language
}

# Real-time data settings
REALTIME_DATA_CONFIG = {
    "weather": {
        "enabled": True,
        "api_provider": "openweathermap",  # Weather API provider
        "api_key": "",                     # API key (user must provide)
        "default_location": "auto",        # Default location or "auto" for IP-based
        "units": "metric",                 # "metric", "imperial", "kelvin"
        "timeout": 10
    },
    "news": {
        "enabled": True,
        "api_provider": "newsapi",         # News API provider
        "api_key": "",                     # API key (user must provide)
        "country": "us",                   # Country code for news
        "category": "general",             # News category
        "max_articles": 5,                 # Maximum articles to fetch
        "timeout": 15
    },
    "time": {
        "enabled": True,
        "timezone_api": "worldtimeapi",    # Timezone API provider
        "timeout": 5
    }
}

# Logging settings
LOGGING_CONFIG = {
    "level": "INFO",
    "file": LOGS_DIR / "voice_companion.log",
    "max_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 3
}

# === Performance settings ===
PERFORMANCE_CONFIG = {
    "max_conversation_history": 8, 
    "audio_buffer_size": 2048,     
    "transcription_timeout": 12,   # Slightly shorter to reduce lag
    "response_timeout": 50,        
    "whisper_threads": 2,          
    "enable_vad": True,
    "min_audio_length": 0.8        # Respond faster to short prompts
}