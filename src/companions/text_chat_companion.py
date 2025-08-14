#!/usr/bin/env python3
"""
AugmentCode - Advanced Private AI Assistant
A comprehensive private, offline AI assistant specialized in:
- Software development, debugging, and automation
- Authorized cybersecurity research and penetration testing
- Technical guidance with step-by-step explanations
- Privacy-focused local processing with persistent memory
- Tool-assisted workflows and development optimization

Core Features:
- Dual-model architecture (General + Specialized Coding)
- Three-component memory system (Context + Database + Learning)
- Enhanced technical guidance for authorized security research
- Persistent conversation memory with privacy controls
- Optimized handling for coding tasks and debugging workflows
- Minimal disclaimers with direct, structured technical answers
"""

import os
import os.path
import sys
import time
import logging
import signal
import json
import urllib.request
import urllib.parse
import urllib.error
import socket
import textwrap
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from hashlib import sha256

# Third-party imports
import ollama
import requests
from colorama import init, Fore, Style

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import memory system, enhanced LLM, and new model interfaces
try:
    from data.Memory import MemorySystem
    from src.core.enhanced_llm import EnhancedLLMInterface
    from src.interfaces.qwen_coder_interface import QwenCoderInterface
    from src.core.model_selector import ModelSelector, ModelType
    from src.systems.memory_context_analyzer import MemoryContextAnalyzer
    from src.systems.technical_guidance_system import TechnicalGuidanceSystem, GuidanceType, SecurityContext
    from src.database.enhanced_memory_system import EnhancedMemorySystem, PrivacyManager
    from src.systems.intelligent_knowledge_system import IntelligentKnowledgeSystem
    from src.interfaces.web_search_integration import create_web_search_function
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are available:")
    print("- Memory system (Memory/)")
    print("- enhanced_llm.py")
    print("- qwen_coder_interface.py")
    print("- model_selector.py")
    print("- memory_context_analyzer.py")
    print("- technical_guidance_system.py")
    print("- enhanced_memory_system.py")
    sys.exit(1)

# Initialize colorama for cross-platform colored output
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/text_chat_companion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Configuration settings for the text chat companion."""
    
    # LLM Configuration - Multi-Model Support
    OLLAMA_CONFIG = {
        "general_model": {
            "model_name": "llama3.2:1b",
            "base_url": "http://localhost:11434",
            "temperature": 0.7,
            "max_tokens": 500,
            "timeout": 30,
            "num_ctx": 2048,
            "num_predict": 500
        },
        "coding_model": {
            "model_name": "hhao/qwen2.5-coder-tools:7b",
            "base_url": "http://localhost:11434",
            "temperature": 0.3,  # Lower temperature for more precise code
            "max_tokens": 800,   # Longer responses for code explanations
            "timeout": 45,       # More time for complex coding tasks
            "num_ctx": 8192,     # Larger context for code analysis
            "num_predict": 500
        },
        "fallback_model": {
            "model_name": "llama3.2:1b",
            "base_url": "http://localhost:11434",
            "temperature": 0.7,
            "max_tokens": 300,
            "timeout": 20,
            "num_ctx": 2048,
            "num_predict": 300
        }
    }
    
    # Internet Configuration
    INTERNET_CONFIG = {
        "enabled": True,
        "default_mode": "hybrid",
        "fallback_to_offline": True,
        "connection_timeout": 10,
        "request_timeout": 30,
        "max_retries": 3,
        "user_agent": "TextChatCompanion/1.0 (Educational AI Assistant)",
        "rate_limit_delay": 1.0,
    }
    
    # Web Search Configuration
    WEB_SEARCH_CONFIG = {
        "enabled": True,
        "engine": "duckduckgo",
        "max_results": 5,
        "snippet_length": 200,
        "search_timeout": 15,
        "safe_search": True,
        "region": "us-en",
    }
    
    # Text Input Configuration
    TEXT_INPUT_CONFIG = {
        "enabled": True,
        "max_length": 1000,
        "min_length": 1,
        "strip_whitespace": True,
        "allow_empty": False,
        "encoding": "utf-8",
        "prompt": "You: ",
        "multiline": False,
        "timeout": 300
    }

    # Display Configuration
    DISPLAY_CONFIG = {
        "enabled": True,
        "max_line_width": 80,           # Maximum characters per line for wrapping
        "indent_size": 4,               # Indentation for wrapped lines
        "show_metadata": False,         # Show response metadata by default
        "show_timestamps": False,       # Show timestamps with responses
        "show_quality_scores": False,   # Show response quality scores
        "show_processing_time": False,  # Show processing time
        "word_wrap": True,              # Enable word wrapping
        "preserve_formatting": True,    # Preserve original text formatting
        "use_colors": True,             # Use colored output
        "accessibility_mode": False,    # Enhanced accessibility features
        "response_separator": "─" * 50, # Visual separator between responses
        "show_separator": True,         # Show visual separators
        "compact_mode": False,          # Compact display mode
        "verbose_mode": False,          # Verbose mode with extra information
    }

    # Privacy Configuration - Enhanced for Private Assistant
    PRIVACY_CONFIG = {
        "enabled": True,
        "disable_response_logging": False,  # Disable logging of responses
        "show_privacy_indicator": True,     # Show privacy status
        "anonymize_logs": False,            # Anonymize sensitive data in logs
        "session_only_mode": False,         # Don't persist conversations
        "auto_clear_after_hours": 0,        # Auto-clear conversations (0=disabled)
        "encrypt_local_storage": True,      # Encrypt local database (enabled by default)
        "privacy_level": "high",            # standard, high, maximum
        "data_retention_days": 0,           # Days to keep data (0=forever)
        "allow_metadata_collection": True,  # Allow collection of metadata
        "offline_only_mode": False,         # Allow hybrid online/offline operation
        "secure_memory_mode": True,         # Enhanced memory security
        "sanitize_sensitive_data": True,    # Auto-sanitize sensitive information
        "local_processing_only": False,     # Allow external services when beneficial
    }

    # Technical Guidance Configuration
    TECHNICAL_GUIDANCE_CONFIG = {
        "enabled": True,
        "cybersecurity_mode": True,         # Enable cybersecurity research guidance
        "penetration_testing_mode": True,   # Enable pen testing assistance
        "step_by_step_explanations": True,  # Provide detailed step-by-step guidance
        "include_code_examples": True,      # Include practical code examples
        "tool_integration": True,           # Enable tool-assisted workflows
        "minimal_disclaimers": True,        # Reduce unnecessary disclaimers
        "direct_technical_answers": True,   # Provide direct, structured answers
        "authorized_context_assumed": True, # Assume proper authorization for security tasks
        "educational_focus": True,          # Focus on educational value
        "practical_examples": True,         # Include practical, executable examples
    }

    # Workflow Assistance Configuration
    WORKFLOW_CONFIG = {
        "enabled": True,
        "development_workflows": True,      # Software development assistance
        "debugging_workflows": True,        # Debugging and troubleshooting
        "security_workflows": True,         # Security testing workflows
        "automation_workflows": True,       # Automation and scripting
        "documentation_workflows": True,    # Documentation generation
        "code_review_workflows": True,      # Code review assistance
        "deployment_workflows": True,       # Deployment and DevOps
        "learning_workflows": True,         # Learning and skill development
    }

    # Intelligent Knowledge Acquisition Configuration
    KNOWLEDGE_ACQUISITION_CONFIG = {
        "enabled": True,
        "auto_web_search": True,            # Automatically trigger web search for unknown topics
        "search_confidence_threshold": 0.3, # Minimum confidence to trigger search
        "max_search_results": 5,            # Maximum number of search results to evaluate
        "cache_duration_hours": 1,          # How long to cache search results
        "store_learned_info": True,         # Store learned information in memory
        "source_attribution": True,        # Include source references in responses
        "quality_threshold": 0.6,           # Minimum quality score for sources
        "max_sources_per_response": 3,      # Maximum sources to include in response
        "enable_real_time_updates": True,   # Enable real-time information updates
        "learning_persistence": True,       # Persist learned information across sessions
    }

    # Model Selection Configuration
    MODEL_SELECTION_CONFIG = {
        "enabled": True,
        "coding_threshold": 0.3,      # Minimum confidence for coding model
        "fallback_threshold": 3.0,    # Max response time before fallback
        "max_retries": 2,
        "prefer_specialized": True,   # Prefer specialized models when available
        "auto_detect_language": True, # Auto-detect programming languages
        "cache_model_decisions": True # Cache model selection decisions
    }


class InternetConnectivity:
    """Handles internet connectivity checks and management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_connected = False
        self.last_check = 0
        self.check_interval = 30  # Check every 30 seconds
    
    def check_connection(self) -> bool:
        """Check if internet connection is available."""
        current_time = time.time()
        
        # Use cached result if recent
        if current_time - self.last_check < self.check_interval:
            return self.is_connected
        
        try:
            # Try to connect to a reliable server
            socket.create_connection(("8.8.8.8", 53), timeout=self.config["connection_timeout"])
            self.is_connected = True
            logger.debug("Internet connection available")
        except (socket.error, socket.timeout):
            self.is_connected = False
            logger.debug("No internet connection")
        
        self.last_check = current_time
        return self.is_connected


class WebSearchEngine:
    """Handles web search functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TextChatCompanion/1.0 (Educational AI Assistant)'
        })
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform web search using DuckDuckGo.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        if not self.config["enabled"]:
            return []
        
        try:
            # Simple DuckDuckGo search implementation
            search_url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = self.session.get(
                search_url, 
                params=params, 
                timeout=self.config["search_timeout"]
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Process results
            for item in data.get('RelatedTopics', [])[:self.config["max_results"]]:
                if 'Text' in item and 'FirstURL' in item:
                    results.append({
                        'title': item.get('Text', '')[:100],
                        'url': item.get('FirstURL', ''),
                        'snippet': item.get('Text', '')[:self.config["snippet_length"]]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []


class RealTimeDataFetcher:
    """Fetches real-time data like weather, news, etc."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = requests.Session()
    
    def get_current_time(self) -> str:
        """Get current time information."""
        try:
            from datetime import datetime
            now = datetime.now()
            return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        except Exception as e:
            logger.error(f"Failed to get time: {e}")
            return "Unable to get current time"
    
    def get_weather(self, location: str = "auto") -> str:
        """Get weather information (placeholder implementation)."""
        # This would integrate with a weather API in a real implementation
        return f"Weather information for {location} is not available in this demo version."
    
    def get_news(self, category: str = "general") -> str:
        """Get news information (placeholder implementation)."""
        # This would integrate with a news API in a real implementation
        return f"News in {category} category is not available in this demo version."


class LLMInterface:
    """Interface for Large Language Model interactions."""
    
    def __init__(self, model_name: str, internet_mode: str = "hybrid"):
        self.model_name = model_name
        self.internet_mode = internet_mode
        self.internet = InternetConnectivity(Config.INTERNET_CONFIG)
        self.web_search = WebSearchEngine(Config.WEB_SEARCH_CONFIG)
        self.data_fetcher = RealTimeDataFetcher({})
        
        # Test Ollama connection
        self._test_connection()
        
        logger.info(f"LLM Interface initialized with model: {model_name}")
    
    def _test_connection(self) -> None:
        """Test connection to Ollama."""
        try:
            response = ollama.list()
            models = response.get('models', [])

            # Handle different response formats
            if models and isinstance(models[0], dict):
                if 'name' in models[0]:
                    available_models = [model['name'] for model in models]
                elif 'model' in models[0]:
                    available_models = [model['model'] for model in models]
                else:
                    # Fallback: use the first key that looks like a model name
                    available_models = [str(model.get(list(model.keys())[0], '')) for model in models]
            else:
                available_models = [str(model) for model in models]

            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                if available_models:
                    self.model_name = available_models[0]
                    logger.info(f"Using available model: {self.model_name}")
                else:
                    raise Exception("No Ollama models available")

        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            raise
    
    def generate_response(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """
        Generate AI response to user input.
        
        Args:
            user_input: User's input text
            context: Additional context from memory system
            
        Returns:
            AI response text
        """
        try:
            # Prepare the prompt with context
            prompt = self._prepare_prompt(user_input, context)
            
            # Generate response using Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': Config.OLLAMA_CONFIG['temperature'],
                    'num_predict': Config.OLLAMA_CONFIG['num_predict'],
                    'num_ctx': Config.OLLAMA_CONFIG['num_ctx']
                }
            )
            
            ai_response = response['message']['content'].strip()
            
            # Enhance response with internet data if needed and available
            if self.internet_mode in ["hybrid", "online"] and self.internet.check_connection():
                ai_response = self._enhance_with_internet_data(user_input, ai_response)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    def _prepare_prompt(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Prepare the prompt with context information."""
        prompt_parts = []
        
        # Add context if available
        if context and context.get('context_memory'):
            recent_history = context['context_memory'].get('conversation_history', [])
            if recent_history:
                prompt_parts.append("Recent conversation context:")
                for interaction in recent_history[-3:]:  # Last 3 interactions
                    if interaction.get('user_input'):
                        prompt_parts.append(f"User: {interaction['user_input']}")
                    if interaction.get('ai_response'):
                        prompt_parts.append(f"Assistant: {interaction['ai_response']}")
                prompt_parts.append("")
        
        # Add current user input
        prompt_parts.append(f"Current user input: {user_input}")
        prompt_parts.append("")
        prompt_parts.append("Please provide a helpful, accurate, and conversational response.")
        
        return "\n".join(prompt_parts)
    
    def _enhance_with_internet_data(self, user_input: str, ai_response: str) -> str:
        """Enhance response with internet data if relevant."""
        # Simple keyword detection for when to use internet data
        time_keywords = ['time', 'date', 'today', 'now', 'current']
        weather_keywords = ['weather', 'temperature', 'forecast', 'rain', 'sunny']
        search_keywords = ['search', 'find', 'look up', 'what is', 'who is']
        
        user_lower = user_input.lower()
        
        try:
            # Add current time if requested
            if any(keyword in user_lower for keyword in time_keywords):
                time_info = self.data_fetcher.get_current_time()
                ai_response += f"\n\n{time_info}"
            
            # Add weather if requested
            elif any(keyword in user_lower for keyword in weather_keywords):
                weather_info = self.data_fetcher.get_weather()
                ai_response += f"\n\n{weather_info}"
            
            # Add search results if it seems like a search query
            elif any(keyword in user_lower for keyword in search_keywords):
                search_results = self.web_search.search(user_input)
                if search_results:
                    ai_response += "\n\nHere are some relevant search results:"
                    for i, result in enumerate(search_results[:3], 1):
                        ai_response += f"\n{i}. {result['title']}: {result['snippet']}"
            
        except Exception as e:
            logger.error(f"Failed to enhance response with internet data: {e}")
        
        return ai_response


class AugmentCodeAssistant:
    """
    AugmentCode - Advanced Private AI Assistant

    Main class for the comprehensive private assistant with enhanced capabilities for:
    - Technical guidance and cybersecurity research
    - Persistent memory with privacy controls
    - Tool-assisted workflows
    - Direct, structured technical answers
    """

    def __init__(self, ollama_model: str = "llama3.2:1b", internet_mode: str = "offline"):
        """
        Initialize AugmentCode private assistant with enhanced capabilities.

        Args:
            ollama_model: Primary Ollama model name to use
            internet_mode: Internet mode (offline recommended for privacy)
        """
        logger.info("Initializing AugmentCode Private Assistant...")

        try:
            # Initialize Enhanced Memory System with privacy controls
            self.enhanced_memory = EnhancedMemorySystem(
                db_path="data/Memory/Database/augmentcode_memory.db",
                privacy_config=Config.PRIVACY_CONFIG,
                technical_config=Config.TECHNICAL_GUIDANCE_CONFIG
            )

            # Initialize base Memory System for compatibility
            self.memory = MemorySystem(
                db_path="data/Memory/Database/augmentcode_memory.db",
                max_context_history=100,  # Increased for better context
                context_window=20         # Larger window for technical discussions
            )

            # Set internet mode based on configuration and user preference
            if Config.PRIVACY_CONFIG.get('offline_only_mode', False):
                self.internet_mode = "offline"
                logger.info("🔒 Forced offline mode for enhanced privacy")
            else:
                self.internet_mode = internet_mode
                logger.info(f"🌐 Internet mode set to: {self.internet_mode}")

            # Initialize Technical Guidance System
            self.technical_guidance = TechnicalGuidanceSystem(
                config=Config.TECHNICAL_GUIDANCE_CONFIG
            )

            # Initialize Intelligent Knowledge System
            self.knowledge_system = IntelligentKnowledgeSystem(
                config=Config.KNOWLEDGE_ACQUISITION_CONFIG
            )

            # Set up web search integration if internet mode allows
            if self.internet_mode in ["online", "hybrid"]:
                web_search_config = {
                    'search_engine': 'duckduckgo',  # Default to DuckDuckGo for privacy
                    'default_engine': 'duckduckgo'
                }
                web_search_function = create_web_search_function(web_search_config)
                self.knowledge_system.web_integrator.set_web_search_function(web_search_function)
                logger.info("🌐 Web search integration enabled")

            # Initialize Model Selector with enhanced configuration
            self.model_selector = ModelSelector(Config.MODEL_SELECTION_CONFIG)

            # Initialize Memory Context Analyzer
            self.memory_context_analyzer = MemoryContextAnalyzer()

            # Initialize Enhanced LLM Interface (General Model)
            self.llm = EnhancedLLMInterface(
                model_name=ollama_model,
                config=Config.OLLAMA_CONFIG["general_model"]
            )

            # Initialize Qwen Coder Interface (Coding Model)
            try:
                self.qwen_coder = QwenCoderInterface(
                    model_name=Config.OLLAMA_CONFIG["coding_model"]["model_name"],
                    config=Config.OLLAMA_CONFIG["coding_model"]
                )
                self.model_selector.update_model_status(ModelType.CODING, True)
                logger.info("✅ Qwen Coder interface initialized")
            except Exception as e:
                logger.warning(f"Qwen Coder interface failed to initialize: {e}")
                self.qwen_coder = None
                self.model_selector.update_model_status(ModelType.CODING, False)

            # Initialize basic LLM interface for fallback
            self.basic_llm = LLMInterface(ollama_model, self.internet_mode)

            self.internet_mode = internet_mode
            self.is_running = False

            # Initialize display and privacy settings
            self.display_config = Config.DISPLAY_CONFIG.copy()
            self.privacy_config = Config.PRIVACY_CONFIG.copy()

            # Initialize display state
            self.last_response_time = 0
            self.response_count = 0

            logger.info("✅ Text Chat Companion initialized successfully!")
            logger.info(f"💡 Primary model: '{ollama_model}'")
            logger.info(f"🔧 Coding model: {'Available' if self.qwen_coder else 'Not available'}")
            logger.info(f"🌐 Internet mode: {internet_mode}")

            # Show privacy status if enabled
            if self.privacy_config["show_privacy_indicator"]:
                privacy_level = self.privacy_config["privacy_level"]
                logger.info(f"🔒 Privacy level: {privacy_level}")
                if self.privacy_config["disable_response_logging"]:
                    logger.info("🔇 Response logging disabled for privacy")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def get_text_input(self) -> Optional[str]:
        """Get text input from user."""
        try:
            config = Config.TEXT_INPUT_CONFIG
            prompt = config["prompt"]

            # Get user input
            user_input = input(prompt).strip()

            # Validate input
            if not user_input and not config["allow_empty"]:
                return None

            if len(user_input) > config["max_length"]:
                print(f"{Fore.YELLOW}⚠️  Input too long. Maximum {config['max_length']} characters.{Style.RESET_ALL}")
                return None

            if len(user_input) < config["min_length"]:
                return None

            return user_input

        except KeyboardInterrupt:
            return "quit"
        except Exception as e:
            logger.error(f"Text input failed: {e}")
            return None

    def process_user_input(self, user_input: str) -> Optional[str]:
        """
        Process user input with enhanced technical guidance and privacy controls.

        Args:
            user_input: User's text input

        Returns:
            Tuple of (AI response, metadata, processing_time) or error response
        """
        try:
            start_time = time.time()

            # Get base context for compatibility first
            context = self.memory.get_response_context()

            # Check for technical guidance request
            guidance_type, guidance_confidence, security_context = self.technical_guidance.analyze_query_type(user_input)

            # Check if web search is needed for unknown or current information
            knowledge_result = None
            if Config.KNOWLEDGE_ACQUISITION_CONFIG.get('enabled', True):
                knowledge_result = self.knowledge_system.process_query(user_input, context)
                if knowledge_result.get('web_search_performed') and knowledge_result.get('search_results', {}).get('success'):
                    logger.info(f"Web search performed for query: {user_input}")

            # Get enhanced context from memory systems
            enhanced_context = self.enhanced_memory.get_enhanced_context(
                user_input, guidance_type.value if guidance_confidence > 0.3 else None
            )

            # Extract clean conversation history for context
            context_memory = context.get('context_memory', {})
            conversation_history = context_memory.get('conversation_history', [])
            current_topic = context_memory.get('current_topic')

            # Analyze conversation context using enhanced analyzer
            context_analysis = self.memory_context_analyzer.analyze_conversation_context(
                conversation_history, user_input
            )

            # Format context as clean conversation history with enhanced insights
            context_str = ""
            if conversation_history:
                # Use relevant context from analyzer
                relevant_interactions = context_analysis.get('relevant_context', conversation_history[-3:])

                context_str = f"Context: {context_analysis['context_summary']}\n"
                context_str += "Recent relevant conversation:\n"

                for interaction in relevant_interactions:
                    if interaction.get('user_input'):
                        context_str += f"User: {interaction['user_input']}\n"
                    if interaction.get('ai_response'):
                        context_str += f"Assistant: {interaction['ai_response']}\n"

            if current_topic:
                context_str += f"\nCurrent topic: {current_topic}"

            # Add context type information for model selection
            context_type = context_analysis.get('context_type', 'general')
            if context_type == 'coding':
                context_str += f"\nContext: Technical/coding discussion (confidence: {context_analysis['coding_score']:.2f})"

            # Select appropriate model
            selected_model, confidence = self.model_selector.select_model(user_input, context)
            logger.info(f"Selected model: {selected_model.value} (confidence: {confidence:.2f})")

            ai_response = None
            metadata = {}
            model_used = "unknown"

            # Check if this should use technical guidance system
            # Use technical guidance for high-confidence technical queries OR coding requests
            should_use_technical_guidance = (
                (guidance_confidence > 0.3 and
                 guidance_type in [GuidanceType.CYBERSECURITY, GuidanceType.PENETRATION_TESTING,
                                 GuidanceType.DEVELOPMENT, GuidanceType.DEBUGGING]) or
                selected_model == ModelType.CODING or  # Always use for coding queries
                self.technical_guidance._is_code_generation_request(user_input)  # Explicit code requests
            )

            if (should_use_technical_guidance and
                Config.TECHNICAL_GUIDANCE_CONFIG.get('enabled', True)):

                # Use technical guidance system for specialized responses
                try:
                    technical_response, tech_metadata = self.technical_guidance.generate_technical_guidance(
                        user_input, context_str.strip() if context_str.strip() else None
                    )

                    # Enhance with AI model response if needed
                    if selected_model == ModelType.CODING and self.qwen_coder:
                        ai_enhancement, ai_metadata = self.qwen_coder.generate_coding_response(
                            user_input, context_str.strip() if context_str.strip() else None
                        )
                        if ai_enhancement:
                            technical_response += f"\n\n## AI Enhancement:\n{ai_enhancement}"
                            tech_metadata.update(ai_metadata)

                    # Enhance with web-sourced information if available
                    if knowledge_result and knowledge_result.get('web_search_performed'):
                        web_info = knowledge_result.get('synthesized_info', {})
                        if web_info.get('content'):
                            technical_response += f"\n\n## Current Information from Web Sources:\n{web_info['content']}"
                            if knowledge_result.get('sources'):
                                technical_response += f"\n\n**Sources:**\n"
                                for i, source in enumerate(knowledge_result['sources'], 1):
                                    technical_response += f"{i}. [{source.get('title', 'Unknown')}]({source.get('url', '#')})\n"
                            tech_metadata['web_enhanced'] = True
                            tech_metadata['web_confidence'] = web_info.get('confidence', 0.0)

                    ai_response = technical_response
                    metadata = tech_metadata
                    model_used = "technical_guidance"

                except Exception as e:
                    logger.warning(f"Technical guidance failed: {e}, falling back to standard models")
                    ai_response = None

            # Standard model selection if technical guidance not used or failed
            if ai_response is None:
                try:
                    if selected_model == ModelType.CODING and self.qwen_coder:
                        # Use Qwen Coder for coding queries
                        ai_response, metadata = self.qwen_coder.generate_coding_response(
                            user_input,
                            context_str.strip() if context_str.strip() else None
                        )
                        model_used = "qwen_coder"

                        # If Qwen couldn't handle it, fall back to general model
                        if ai_response is None:
                            logger.info("Qwen coder returned None, falling back to general model")
                            selected_model = ModelType.GENERAL

                    if ai_response is None and selected_model == ModelType.GENERAL:
                        # Use Enhanced LLM for general queries
                        ai_response, metadata = self.llm.generate_response(
                            user_input,
                            additional_context=context_str.strip() if context_str.strip() else None
                        )
                        model_used = "enhanced_llm"
                        logger.info(f"Enhanced LLM response quality: {metadata.get('validation_score', 'N/A')}")

                    if ai_response is None:
                        # Final fallback to basic LLM
                        logger.warning("All specialized models failed, using basic LLM fallback")
                        ai_response = self.basic_llm.generate_response(user_input, context)
                        model_used = "basic_llm"
                        metadata = {}

                    # Enhance any response with web information if available and not already enhanced
                    if (knowledge_result and knowledge_result.get('web_search_performed') and
                        not metadata.get('web_enhanced', False)):
                        web_info = knowledge_result.get('synthesized_info', {})
                        if web_info.get('content') and ai_response:
                            ai_response += f"\n\n## Additional Current Information:\n{web_info['content']}"
                            if knowledge_result.get('sources'):
                                ai_response += f"\n\n**Sources:**\n"
                                for i, source in enumerate(knowledge_result['sources'], 1):
                                    ai_response += f"{i}. [{source.get('title', 'Unknown')}]({source.get('url', '#')})\n"
                            metadata['web_enhanced'] = True
                            metadata['web_confidence'] = web_info.get('confidence', 0.0)

                except Exception as e:
                    logger.warning(f"Selected model failed: {e}, using fallback")
                    ai_response = self.basic_llm.generate_response(user_input, context)
                    model_used = "basic_llm_fallback"
                    metadata = {}

            # Record performance metrics
            response_time = time.time() - start_time
            success = ai_response is not None and len(ai_response.strip()) > 0
            self.model_selector.record_performance(selected_model, response_time, success)

            # Enhanced metadata for memory system
            enhanced_metadata = {
                'response_quality': metadata.get('validation_score', 0.0),
                'model_used': model_used,
                'model_type': selected_model.value,
                'model_confidence': confidence,
                'response_time': response_time,
                'internet_mode': self.internet_mode,
                'timestamp': time.time(),
                'is_coding_query': selected_model == ModelType.CODING,
                'task_type': metadata.get('task_type', 'general'),
                'language_detected': metadata.get('language_detected'),
                'context_type': 'coding' if selected_model == ModelType.CODING else 'general',
                'guidance_type': guidance_type.value if guidance_confidence > 0.3 else None,
                'guidance_confidence': guidance_confidence,
                'security_context': security_context.value if guidance_confidence > 0.3 else None,
                'technical_guidance_used': model_used == "technical_guidance",
                'tools_recommended': metadata.get('tools_recommended', False),
                'workflow_included': metadata.get('workflow_included', False),
                'web_search_performed': knowledge_result.get('web_search_performed', False) if knowledge_result else False,
                'web_enhanced': metadata.get('web_enhanced', False),
                'web_confidence': metadata.get('web_confidence', 0.0),
                'knowledge_sources': len(knowledge_result.get('sources', [])) if knowledge_result else 0
            }

            # Store interaction in both memory systems
            # Base memory for compatibility
            self.memory.add_interaction(
                user_input=user_input,
                ai_response=ai_response,
                metadata=enhanced_metadata
            )

            # Enhanced memory with privacy controls and technical knowledge storage
            enhanced_storage_result = self.enhanced_memory.add_interaction(
                user_input=user_input,
                ai_response=ai_response,
                metadata=enhanced_metadata,
                guidance_type=guidance_type.value if guidance_confidence > 0.3 else None
            )

            # Return response with metadata for enhanced display
            return ai_response, enhanced_metadata, response_time

        except Exception as e:
            logger.error(f"Input processing failed: {e}")
            error_response = "I apologize, but I encountered an error processing your input. Please try again."
            return error_response, {}, 0.0

    def display_response(self, response: str, metadata: Dict[str, Any] = None, processing_time: float = 0.0) -> None:
        """
        Enhanced display method for AI responses with improved formatting and privacy controls.

        Args:
            response: The AI response text to display
            metadata: Optional metadata about the response (quality scores, etc.)
            processing_time: Time taken to generate the response
        """
        try:
            # Validate response
            if not self._validate_response(response):
                return

            # Check privacy settings
            if self.privacy_config.get("disable_response_logging", False):
                # Don't log the response content if privacy is enabled
                logger.info("Response displayed (content not logged for privacy)")
            else:
                logger.debug(f"Displaying response: {response[:100]}...")

            # Update response tracking
            self.response_count += 1
            self.last_response_time = time.time()

            # Format and display the response
            self._display_formatted_response(response, metadata, processing_time)

        except Exception as e:
            logger.error(f"Response display failed: {e}")
            # Fallback to simple display
            self._display_fallback_response(response)

    def _validate_response(self, response: str) -> bool:
        """Validate response before display."""
        if not response:
            print(f"{Fore.RED}❌ Empty response received{Style.RESET_ALL}")
            return False

        if len(response.strip()) == 0:
            print(f"{Fore.RED}❌ Response contains only whitespace{Style.RESET_ALL}")
            return False

        # Check for extremely long responses
        max_length = 10000  # Reasonable limit
        if len(response) > max_length:
            logger.warning(f"Response is very long ({len(response)} chars), truncating for display")
            return True

        # Check for potentially problematic content
        if response.count('\n') > 100:  # Too many line breaks
            logger.warning("Response has excessive line breaks")

        return True

    def _display_formatted_response(self, response: str, metadata: Dict[str, Any] = None, processing_time: float = 0.0) -> None:
        """Display response with enhanced formatting."""
        config = self.display_config

        # Show separator if enabled
        if config.get("show_separator", True) and not config.get("compact_mode", False):
            separator = config.get("response_separator", "─" * 50)
            if config.get("use_colors", True):
                print(f"{Fore.CYAN}{separator}{Style.RESET_ALL}")
            else:
                print(separator)

        # Show timestamp if enabled
        if config.get("show_timestamps", False):
            timestamp = datetime.now().strftime("%H:%M:%S")
            if config.get("use_colors", True):
                print(f"{Fore.YELLOW}[{timestamp}]{Style.RESET_ALL}")
            else:
                print(f"[{timestamp}]")

        # Show privacy indicator if enabled
        if self.privacy_config.get("show_privacy_indicator", True):
            privacy_level = self.privacy_config.get("privacy_level", "standard")
            privacy_icon = "🔒" if privacy_level == "high" else "🔓" if privacy_level == "standard" else "🔐"
            if config.get("use_colors", True):
                print(f"{Fore.BLUE}{privacy_icon} Privacy: {privacy_level}{Style.RESET_ALL}")
            else:
                print(f"{privacy_icon} Privacy: {privacy_level}")

        # Format the main response
        formatted_response = self._format_response_text(response)

        # Display the response with AI label
        if config.get("use_colors", True):
            print(f"\n{Fore.GREEN}🤖 AI:{Style.RESET_ALL}")
        else:
            print(f"\n🤖 AI:")

        # Display the formatted response
        print(formatted_response)

        # Show metadata if enabled
        if config.get("verbose_mode", False) or config.get("show_metadata", False):
            self._display_metadata(metadata, processing_time)

        # Add spacing after response
        if not config.get("compact_mode", False):
            print()

    def _format_response_text(self, response: str) -> str:
        """Format response text with word wrapping and proper indentation."""
        config = self.display_config

        if not config.get("word_wrap", True):
            return response

        max_width = config.get("max_line_width", 80)
        indent_size = config.get("indent_size", 4)
        preserve_formatting = config.get("preserve_formatting", True)

        # Handle different types of content
        lines = response.split('\n')
        formatted_lines = []

        for line in lines:
            if not line.strip():
                # Preserve empty lines
                formatted_lines.append("")
                continue

            # Check if line looks like code or formatted content
            if preserve_formatting and (
                line.startswith('    ') or  # Already indented
                line.startswith('\t') or   # Tab indented
                '```' in line or           # Code blocks
                line.startswith('- ') or   # List items
                line.startswith('* ') or   # List items
                re.match(r'^\d+\.', line)  # Numbered lists
            ):
                # Don't wrap pre-formatted content
                formatted_lines.append(line)
            else:
                # Wrap regular text
                wrapped = textwrap.fill(
                    line,
                    width=max_width,
                    subsequent_indent=' ' * indent_size,
                    break_long_words=False,
                    break_on_hyphens=False
                )
                formatted_lines.append(wrapped)

        return '\n'.join(formatted_lines)

    def _display_metadata(self, metadata: Dict[str, Any] = None, processing_time: float = 0.0) -> None:
        """Display response metadata if available."""
        config = self.display_config

        if not metadata and processing_time == 0.0:
            return

        metadata_lines = []

        # Processing time
        if config.get("show_processing_time", False) and processing_time > 0:
            metadata_lines.append(f"⏱️  Processing time: {processing_time:.2f}s")

        # Quality score
        if config.get("show_quality_scores", False) and metadata:
            quality_score = metadata.get("validation_score", metadata.get("quality_score"))
            if quality_score is not None:
                metadata_lines.append(f"📊 Quality score: {quality_score:.3f}")

        # Response count
        if config.get("verbose_mode", False):
            metadata_lines.append(f"💬 Response #{self.response_count}")

        # Additional metadata
        if metadata and config.get("verbose_mode", False):
            for key, value in metadata.items():
                if key not in ["validation_score", "quality_score"] and value is not None:
                    metadata_lines.append(f"📋 {key}: {value}")

        # Display metadata
        if metadata_lines:
            if config.get("use_colors", True):
                print(f"{Fore.CYAN}📈 Response Info:{Style.RESET_ALL}")
            else:
                print("📈 Response Info:")

            for line in metadata_lines:
                if config.get("use_colors", True):
                    print(f"  {Fore.CYAN}{line}{Style.RESET_ALL}")
                else:
                    print(f"  {line}")

    def _display_fallback_response(self, response: str) -> None:
        """Fallback display method for when enhanced display fails."""
        try:
            print(f"\n🤖 AI: {response}\n")
        except Exception as e:
            # Last resort - basic print
            print(f"\nAI: {response}\n")
            logger.error(f"Even fallback display failed: {e}")

    def toggle_verbose_mode(self) -> None:
        """Toggle verbose mode for detailed response information."""
        self.display_config["verbose_mode"] = not self.display_config["verbose_mode"]
        status = "enabled" if self.display_config["verbose_mode"] else "disabled"
        print(f"🔧 Verbose mode {status}")

    def toggle_privacy_mode(self) -> None:
        """Toggle enhanced privacy mode."""
        current = self.privacy_config["disable_response_logging"]
        self.privacy_config["disable_response_logging"] = not current
        status = "enabled" if not current else "disabled"
        print(f"🔒 Privacy mode {status}")
        if not current:
            print("   Response content will not be logged")
        else:
            print("   Response content logging resumed")

    def show_help(self) -> None:
        """Show comprehensive help information for AugmentCode."""
        help_text = f"""
{Fore.CYAN}📖 AugmentCode - Advanced Private AI Assistant{Style.RESET_ALL}

{Fore.GREEN}🔒 Privacy-First Design:{Style.RESET_ALL}
  • All processing happens locally on your machine
  • Persistent memory with encryption and privacy controls
  • Intelligent web search for current information when needed
  • Conversation history stored securely in local database

{Fore.YELLOW}💻 Core Commands:{Style.RESET_ALL}
  • Type your message and press Enter to chat
  • 'help' - Show this help message
  • 'memory' - Show memory system status and context analysis
  • 'models' - Show model statistics and availability
  • 'guidance' - Show technical guidance statistics
  • 'knowledge' - Show intelligent knowledge acquisition statistics
  • 'privacy' - Show privacy settings and toggle privacy mode
  • 'clear' - Clear conversation history
  • 'verbose' - Toggle verbose mode (show response metadata)
  • 'export' - Export knowledge and conversation data
  • 'quit' or 'exit' - Exit the application

{Fore.CYAN}🛡️ Technical Capabilities:{Style.RESET_ALL}
  • Software development and debugging assistance
  • Authorized cybersecurity research and penetration testing
  • Step-by-step technical explanations with code examples
  • Tool recommendations and workflow guidance
  • Direct, structured answers with minimal disclaimers

{Fore.MAGENTA}🤖 AI Models:{Style.RESET_ALL}
  • General Model: Conversational AI for general queries
  • Coding Model: Specialized programming and technical assistance
  • Technical Guidance: Expert system for cybersecurity and development
  • Intelligent model selection based on query analysis

{Fore.YELLOW}💡 Usage Tips:{Style.RESET_ALL}
  • Ask technical questions for specialized guidance
  • Mention specific tools or technologies for targeted help
  • Request step-by-step explanations for complex procedures
  • The AI learns from your interactions and adapts over time
  • All security guidance assumes proper authorization
        """
        print(help_text)

    def show_memory_status(self) -> None:
        """Show memory system status with enhanced context analysis."""
        try:
            status = self.memory.get_system_status()
            print(f"\n{Fore.CYAN}🧠 Memory System Status{Style.RESET_ALL}")
            print(f"Session ID: {status['session_id']}")
            print(f"Conversation turns: {status['context_memory_status']['conversation_turns']}")
            print(f"Current topic: {status['context_memory_status']['current_topic'] or 'None'}")
            print(f"Database interactions: {status['database_memory_status']['total_interactions']}")

            # Show enhanced context analysis
            context = self.memory.get_response_context()
            context_memory = context.get('context_memory', {})
            conversation_history = context_memory.get('conversation_history', [])

            if conversation_history:
                context_analysis = self.memory_context_analyzer.analyze_conversation_context(
                    conversation_history, ""
                )

                print(f"\n{Fore.YELLOW}📊 Context Analysis:{Style.RESET_ALL}")
                print(f"Context Type: {context_analysis['context_type']}")
                print(f"Coding Score: {context_analysis['coding_score']:.2f}")
                print(f"Topic Continuity: {context_analysis['topic_continuity']['score']:.2f}")
                print(f"Dominant Model: {context_analysis['model_usage_pattern']['dominant_model']}")
                print(f"Context Summary: {context_analysis['context_summary']}")

                if context_analysis['recommendations']:
                    print(f"\n{Fore.GREEN}💡 Recommendations:{Style.RESET_ALL}")
                    for rec in context_analysis['recommendations']:
                        print(f"  • {rec}")

            print()
        except Exception as e:
            logger.error(f"Failed to show memory status: {e}")
            print(f"{Fore.RED}❌ Failed to retrieve memory status{Style.RESET_ALL}")

    def show_model_stats(self) -> None:
        """Show model statistics and availability."""
        try:
            stats = self.model_selector.get_model_stats()
            print(f"\n{Fore.CYAN}🤖 Model Statistics{Style.RESET_ALL}")

            for model_name, stat in stats.items():
                status_color = Fore.GREEN if stat['available'] else Fore.RED
                status_text = "Available" if stat['available'] else "Unavailable"

                print(f"\n{Fore.YELLOW}{model_name.title()} Model:{Style.RESET_ALL}")
                print(f"  Status: {status_color}{status_text}{Style.RESET_ALL}")
                print(f"  Success Rate: {stat['success_rate']:.1%}")
                print(f"  Avg Response Time: {stat['avg_response_time']:.2f}s")
                print(f"  Recent Responses: {stat['recent_responses']}")

            # Show current model configuration
            print(f"\n{Fore.YELLOW}Configuration:{Style.RESET_ALL}")
            print(f"  Coding Threshold: {self.model_selector.selection_config['coding_threshold']}")
            print(f"  Fallback Threshold: {self.model_selector.selection_config['fallback_threshold']}s")
            print(f"  Prefer Specialized: {self.model_selector.selection_config['prefer_specialized']}")
            print()

        except Exception as e:
            logger.error(f"Failed to show model stats: {e}")
            print(f"{Fore.RED}❌ Failed to retrieve model statistics{Style.RESET_ALL}")

    def show_guidance_stats(self) -> None:
        """Show technical guidance system statistics."""
        try:
            stats = self.technical_guidance.get_guidance_statistics()
            print(f"\n{Fore.CYAN}🛡️ Technical Guidance Statistics{Style.RESET_ALL}")
            print(f"Total queries processed: {stats.get('total_queries', 0)}")

            if stats.get('guidance_type_distribution'):
                print(f"\n{Fore.YELLOW}Query Type Distribution:{Style.RESET_ALL}")
                for gtype, count in stats['guidance_type_distribution'].items():
                    print(f"  • {gtype.replace('_', ' ').title()}: {count}")

            if stats.get('most_common_type'):
                print(f"\nMost common guidance type: {stats['most_common_type'].replace('_', ' ').title()}")

            if stats.get('average_response_length'):
                print(f"Average response length: {stats['average_response_length']:.0f} characters")

            print()
        except Exception as e:
            logger.error(f"Failed to show guidance stats: {e}")
            print(f"{Fore.RED}❌ Failed to retrieve guidance statistics{Style.RESET_ALL}")

    def show_knowledge_stats(self) -> None:
        """Show intelligent knowledge acquisition statistics."""
        try:
            stats = self.knowledge_system.get_learning_statistics()
            print(f"\n{Fore.CYAN}🧠 Knowledge Acquisition Statistics{Style.RESET_ALL}")
            print(f"Total queries processed: {stats.get('total_queries_processed', 0)}")
            print(f"Web searches performed: {stats.get('web_searches_performed', 0)}")
            print(f"Knowledge cache size: {stats.get('knowledge_cache_size', 0)}")
            print(f"Average confidence: {stats.get('average_confidence', 0):.2f}")

            recent_learning = stats.get('recent_learning', [])
            if recent_learning:
                print(f"\n{Fore.YELLOW}Recent Learning Activity:{Style.RESET_ALL}")
                for entry in recent_learning[-5:]:  # Show last 5
                    timestamp = entry.get('timestamp', 'Unknown')
                    query = entry.get('query', 'Unknown')[:50]
                    confidence = entry.get('confidence', 0)
                    sources = entry.get('source_count', 0)
                    print(f"  • {timestamp[:19]}: '{query}...' (confidence: {confidence:.2f}, sources: {sources})")

            print()
        except Exception as e:
            logger.error(f"Failed to show knowledge stats: {e}")
            print(f"{Fore.RED}❌ Failed to retrieve knowledge statistics{Style.RESET_ALL}")

    def show_privacy_status(self) -> None:
        """Show current privacy settings and controls."""
        try:
            privacy_config = Config.PRIVACY_CONFIG
            print(f"\n{Fore.CYAN}🔒 Privacy Status{Style.RESET_ALL}")
            print(f"Privacy Level: {privacy_config.get('privacy_level', 'standard').title()}")
            print(f"Offline Only Mode: {'✅ Enabled' if privacy_config.get('offline_only_mode') else '❌ Disabled'}")
            print(f"Local Storage Encryption: {'✅ Enabled' if privacy_config.get('encrypt_local_storage') else '❌ Disabled'}")
            print(f"Response Logging: {'❌ Disabled' if privacy_config.get('disable_response_logging') else '✅ Enabled'}")
            print(f"Sensitive Data Sanitization: {'✅ Enabled' if privacy_config.get('sanitize_sensitive_data') else '❌ Disabled'}")
            print(f"Session Only Mode: {'✅ Enabled' if privacy_config.get('session_only_mode') else '❌ Disabled'}")
            print()
        except Exception as e:
            logger.error(f"Failed to show privacy status: {e}")
            print(f"{Fore.RED}❌ Failed to retrieve privacy status{Style.RESET_ALL}")

    def export_knowledge(self) -> None:
        """Export knowledge and conversation data."""
        try:
            # Export from enhanced memory system
            knowledge_data = self.enhanced_memory.export_knowledge()

            # Save to file
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = f"augmentcode_export_{timestamp}.json"

            import json
            with open(export_file, 'w') as f:
                json.dump(knowledge_data, f, indent=2)

            print(f"{Fore.GREEN}✅ Knowledge exported to {export_file}{Style.RESET_ALL}")
            print(f"Export includes technical patterns, conversation data, and privacy settings")

        except Exception as e:
            logger.error(f"Failed to export knowledge: {e}")
            print(f"{Fore.RED}❌ Failed to export knowledge{Style.RESET_ALL}")

    def clear_conversation(self) -> None:
        """Clear conversation history with privacy preservation options."""
        try:
            # Clear both memory systems
            self.memory.clear_session()
            self.enhanced_memory.clear_session(preserve_technical_knowledge=True)

            print(f"{Fore.GREEN}✅ Conversation history cleared{Style.RESET_ALL}")
            print(f"Technical knowledge patterns preserved for future sessions")
        except Exception as e:
            logger.error(f"Failed to clear conversation: {e}")
            print(f"{Fore.RED}❌ Failed to clear conversation history{Style.RESET_ALL}")

    def run(self) -> None:
        """Main conversation loop."""
        self.is_running = True

        # Welcome message
        print(f"\n{Fore.CYAN}🤖 AugmentCode - Advanced Private AI Assistant Started{Style.RESET_ALL}")
        print(f"{Fore.GREEN}💬 Your privacy-focused technical companion is ready{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 Type 'help' for commands, 'quit' to exit{Style.RESET_ALL}")
        print(f"{Fore.BLUE}🔒 Privacy mode: {Config.PRIVACY_CONFIG.get('privacy_level', 'standard')}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}🌐 Internet mode: {self.internet_mode}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}🛡️ Technical guidance: {'Enabled' if Config.TECHNICAL_GUIDANCE_CONFIG.get('enabled') else 'Disabled'}{Style.RESET_ALL}\n")

        try:
            while self.is_running:
                # Get user input
                user_input = self.get_text_input()

                if not user_input:
                    continue

                # Handle special commands
                user_input_lower = user_input.lower().strip()

                if user_input_lower in ['quit', 'exit', 'bye']:
                    print(f"{Fore.CYAN}👋 Goodbye! Thanks for using AugmentCode!{Style.RESET_ALL}")
                    break
                elif user_input_lower == 'help':
                    self.show_help()
                    continue
                elif user_input_lower == 'memory':
                    self.show_memory_status()
                    continue
                elif user_input_lower == 'models':
                    self.show_model_stats()
                    continue
                elif user_input_lower == 'guidance':
                    self.show_guidance_stats()
                    continue
                elif user_input_lower == 'knowledge':
                    self.show_knowledge_stats()
                    continue
                elif user_input_lower == 'privacy':
                    self.show_privacy_status()
                    continue
                elif user_input_lower == 'export':
                    self.export_knowledge()
                    continue
                elif user_input_lower == 'clear':
                    self.clear_conversation()
                    continue
                elif user_input_lower == 'verbose':
                    self.toggle_verbose_mode()
                    continue

                # Process user input and generate response
                result = self.process_user_input(user_input)

                if isinstance(result, tuple) and len(result) == 3:
                    ai_response, metadata, processing_time = result
                    if ai_response:
                        self.display_response(ai_response, metadata, processing_time)
                    else:
                        print(f"{Fore.RED}❌ Failed to generate response. Please try again.{Style.RESET_ALL}")
                else:
                    # Fallback for old format
                    ai_response = result
                    if ai_response:
                        self.display_response(ai_response)
                    else:
                        print(f"{Fore.RED}❌ Failed to generate response. Please try again.{Style.RESET_ALL}")

        except KeyboardInterrupt:
            print(f"\n{Fore.CYAN}👋 Goodbye! Thanks for chatting!{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            print(f"{Fore.RED}❌ An error occurred: {e}{Style.RESET_ALL}")
        finally:
            self.is_running = False


def signal_handler(_signum, _frame):
    """Handle interrupt signals gracefully."""
    print(f"\n{Fore.CYAN}👋 Received interrupt signal. Goodbye!{Style.RESET_ALL}")
    sys.exit(0)


def main():
    """Main entry point for AugmentCode Private Assistant."""
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print(f"{Fore.CYAN}🤖 Starting AugmentCode - Advanced Private AI Assistant...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}⏳ Initializing AI models, memory systems, and technical guidance...{Style.RESET_ALL}")

        # Create AugmentCode assistant with hybrid online/offline capability
        assistant = AugmentCodeAssistant(
            ollama_model="llama3.2:1b",  # Lightweight model for efficiency
            internet_mode="hybrid"       # Hybrid mode for optimal functionality
        )

        # Start the conversation loop
        assistant.run()

    except Exception as e:
        logger.error(f"AugmentCode failed to start: {e}")
        print(f"{Fore.RED}❌ Failed to start AugmentCode: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 Make sure Ollama is running and required models are available{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 Check that all required Python packages are installed{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
