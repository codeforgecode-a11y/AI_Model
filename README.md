# Text-Based AI Chat Companion

A completely text-based AI chat companion that runs locally on your machine. Built with privacy and security in mind, featuring intelligent conversation with memory and optional internet enhancement.

## Features

### Core Functionality
- 💬 **Text Chat Interface**: Clean, intuitive text-based conversation
- 🤖 **Local LLM**: Integrates with Ollama for intelligent responses
- 🧠 **Memory System**: Remembers conversation context and learns from interactions
- 🌐 **Internet Enhancement**: Optional web search and real-time data integration
- 🔄 **Hybrid Mode**: Combine offline AI with internet when needed
- 📴 **Offline Mode**: Full functionality without internet connection

### Advanced Features
- 🎯 **Context Awareness**: Maintains conversation history and topic understanding
- 🔍 **Web Search**: Search for up-to-date information online
- 🌤️ **Real-time Data**: Get current weather, time, and news
- 📊 **Learning System**: Adapts to user preferences and conversation patterns
- 🗃️ **Persistent Memory**: Stores conversations and knowledge in local database
- ⚡ **Response Quality**: Enhanced prompt engineering and validation

### System Features
- 🔒 **Privacy First**: All data stored locally, optional internet features
- 🌍 **Cross-Platform**: Works on Windows, macOS, and Linux
- 🎯 **Modular Design**: Easy to customize and extend
- ⚡ **Optimized**: Lightweight design for efficient performance

## Architecture

```
Text Input → Input Validation → Enhanced LLM → Text Output
                                      ↓
Memory System ← Context Management → Internet Enhancement (Optional)
     ↓                                        ↓
Database Storage                    Web Search & Real-time Data

Enhanced LLM Pipeline:
User Query → Context Analysis → Memory Retrieval → Internet Learning (Optional) → Ollama LLM → Response

Memory System Components:
- Context Memory (conversation history)
- Database Memory (persistent storage)
- Learning Memory (pattern recognition)

Internet Enhancement Components:
- Web Search (DuckDuckGo API)
- Real-time Weather Data
- Current Time/Date Information
- News Headlines
- Connectivity Management & Fallback
```

## Prerequisites

### 1. Ollama Installation

First, install Ollama from [https://ollama.ai](https://ollama.ai):

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download and install from the official website.

### 2. Start Ollama Service

```bash
# Linux/macOS
ollama serve

# Windows: Start Ollama from the application menu
```

### 3. Download a Model

```bash
# Lightweight model (1.3GB)
ollama pull llama3.2:1b

# Standard model (4.7GB) - better quality
ollama pull llama3.2

# Alternative lightweight models
ollama pull phi3:mini
ollama pull gemma2:2b
```

## Installation

### Automatic Setup

1. Clone or download this project
2. Navigate to the project directory
3. Run the setup script:

```bash
# Make sure you're in a virtual environment
python -m venv env
source env/bin/activate  # Linux/macOS
# or
env\Scripts\activate     # Windows

# Run setup
python setup.py
```

### Manual Installation

If the automatic setup fails, install dependencies manually:

#### System Dependencies

No special system dependencies are required for the text-based chat companion. All dependencies are Python packages that will be installed via pip.

#### Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Text-based chat companion
python src/companions/text_chat_companion.py

# Voice-based companion
python src/companions/voice_companion.py
```

### Chat Interface

The application provides a clean text-based chat interface:

1. **Start**: Run the application to begin chatting
2. **Type**: Enter your message at the "You: " prompt
3. **Submit**: Press Enter to send your message
4. **Response**: The AI will process and respond with text
5. **Continue**: Keep chatting naturally

### Internet Modes

The application supports different internet modes:

1. **Offline Mode** - Use only local AI (faster, completely private)
2. **Online Mode** - Use internet for current information (requires connection)
3. **Hybrid Mode** - Combine local AI with internet when needed (recommended)

### Available Commands

Type these commands during conversation:

- `help` - Show help information and available commands
- `memory` - Display memory system status and conversation statistics
- `clear` - Clear conversation history (preserves preferences)
- `quit` or `exit` - Exit the application gracefully

### Features in Action

**Context Memory**: The AI remembers your conversation and maintains context across messages.

**Internet Enhancement**: When enabled, the AI can search the web and provide current information about weather, time, and other topics.

**Learning System**: The AI learns from your interactions and adapts to your preferences over time.

### Internet Learning Features

When internet mode is enabled, the AI can:

**Automatic Context Detection:**
- Weather queries: "What's the weather like?" → Gets current weather data
- Time queries: "What time is it?" → Gets current time and date
- News queries: "What's in the news?" → Gets latest headlines
- General questions: "What is..." → Searches web for current information

**Manual Requests:**
- Ask about current events, recent developments, or breaking news
- Request weather updates for any location
- Get real-time information that supplements the local AI's knowledge
- Search for recent information on any topic

## Configuration

Edit files in the `config/` directory to customize:

- **Ollama Model**: Switch between different LLM models
- **Text Settings**: Adjust input validation and processing options
- **Internet Settings**: Configure web search and real-time data features
- **Memory Settings**: Modify conversation history and learning parameters

## Troubleshooting

### Common Issues

#### 1. Ollama Connection Issues

Make sure Ollama is running:
```bash
# Check if Ollama is running
ollama list

# Start Ollama if not running
ollama serve
```

#### 2. Model Not Found

If you get a "model not found" error:
```bash
# List available models
ollama list

# Pull the required model
ollama pull llama3.2:1b
```

#### 3. Memory System Issues

If you encounter database errors:
- Check write permissions in the data/Memory/Database directory
- Delete the database file to reset: `rm data/Memory/Database/text_chat_memory.db`
- Restart the application to recreate the database

#### 4. Internet Features Not Working

- Check your internet connection
- Verify firewall settings allow outbound connections
- Try switching to offline mode if internet features are not needed

### Performance Optimization

#### For Better Response Quality
- Use larger Ollama models: `llama3.2:3b` or `llama3.2:8b`
- Increase context window size in configuration
- Enable response validation and caching

#### For Better Speed
- Use smaller models: `llama3.2:1b` (default)
- Reduce conversation history length
- Disable internet features if not needed
- Use CPU optimization settings

## Advanced Features

### Memory System

The application includes a sophisticated memory system:

- **Context Memory**: Maintains conversation flow and topic understanding
- **Database Memory**: Stores conversations persistently across sessions
- **Learning Memory**: Adapts to user preferences and improves over time

### Internet Enhancement

Optional internet features provide:

- **Web Search**: Real-time information from DuckDuckGo
- **Weather Data**: Current weather information
- **Time Services**: Accurate time and date information
- **Smart Routing**: Automatic detection of when internet data is needed

### Customization

The modular design allows easy customization:

- **LLM Models**: Switch between different Ollama models
- **Memory Settings**: Adjust conversation history and learning parameters
- **Internet Settings**: Configure web search and data sources
- **Text Processing**: Customize input validation and formatting

## File Structure

```
Linux_Dev_Dir/
├── README.md                          # Main project README
├── setup.py                          # Installation script
├── requirements.txt                   # Core dependencies
├── .gitignore                         # Git ignore file
│
├── src/                              # Source code
│   ├── companions/                    # AI companion applications
│   │   ├── text_chat_companion.py    # Text-based chat companion
│   │   └── voice_companion.py        # Voice-based companion
│   ├── core/                         # Core system components
│   │   ├── enhanced_llm.py           # Enhanced LLM interface
│   │   ├── model_selector.py         # Model selection logic
│   │   └── session_manager.py        # Session management
│   ├── interfaces/                   # External service interfaces
│   │   ├── qwen_coder_interface.py   # Qwen coding model interface
│   │   └── web_search_integration.py # Web search integration
│   ├── systems/                      # Specialized systems
│   │   ├── intelligent_knowledge_system.py
│   │   ├── technical_guidance_system.py
│   │   └── privacy_security_manager.py
│   ├── database/                     # Database management
│   │   ├── enhanced_database_manager.py
│   │   └── enhanced_memory_system.py
│   └── training/                     # ML training components
│       ├── multi_dataset_trainer.py
│       └── train_model.py
│
├── config/                           # Configuration files
│   ├── config.py                     # Basic configuration
│   ├── optimized_config.py           # Advanced configuration
│   └── *.yaml                        # YAML configuration files
│
├── tests/                           # Test files
│   ├── test_companions/             # Tests for companion apps
│   ├── test_core/                   # Tests for core components
│   ├── test_systems/                # Tests for specialized systems
│   └── test_training/               # Tests for training components
│
├── docs/                            # Documentation
│   ├── guides/                      # User guides
│   ├── api/                         # API documentation
│   ├── examples/                    # Usage examples
│   └── reports/                     # Technical reports
│
├── data/                            # Data and datasets
│   ├── Datasets/                    # Training datasets
│   ├── Memory/                      # Memory system
│   │   ├── context.py              # Context management
│   │   ├── database.py             # Database interface
│   │   └── Database/               # SQLite database files
│   └── models/                      # Trained models
│
├── scripts/                         # Utility scripts
├── logs/                           # Log files
├── cache/                          # Cache files
└── build/                          # Build artifacts
```

## Contributing

Feel free to contribute improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Use responsibly and respect privacy.

## Privacy & Security

- **Local Processing**: Everything runs on your machine
- **Optional Internet**: Internet features are optional and clearly marked
- **No Data Collection**: Your conversations stay on your machine
- **Open Source**: All code is transparent and auditable
- **Secure Storage**: Local SQLite database for conversation history

## Support

For issues and questions:
1. Check the troubleshooting section in `docs/guides/TROUBLESHOOTING_GUIDE.md`
2. Review the logs in `logs/` directory
3. Test Ollama connection separately
4. Ensure all dependencies are properly installed
5. Check Memory system permissions and database access in `data/Memory/Database/`
