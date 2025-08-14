#!/usr/bin/env python3
"""
Enhanced LLM Interface for Better Result Quality

Significant improvements for AI response quality:
1. Advanced prompt engineering and context management
2. Response validation and quality scoring
3. Intelligent context compression and relevance filtering
4. Multi-turn conversation optimization
5. Error detection and correction
6. Response formatting and coherence improvement
"""

import ollama
import logging
import time
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
from collections import deque

logger = logging.getLogger(__name__)


class PromptEngineer:
    """Advanced prompt engineering for better LLM responses."""
    
    def __init__(self):
        """Initialize prompt engineer."""
        self.system_prompts = {
            'default': """You are a helpful, accurate, and concise AI assistant. Provide clear, relevant responses that directly address the user's question. Be informative but not verbose. If you're unsure about something, say so rather than guessing.""",
            
            'conversational': """You are a friendly AI companion engaged in natural conversation. Respond in a warm, helpful manner while being accurate and informative. Keep responses conversational but focused. Remember the context of our ongoing discussion.""",
            
            'technical': """You are a knowledgeable technical assistant. Provide accurate, detailed technical information while keeping explanations clear and accessible. Use examples when helpful. If a question is outside your knowledge, be honest about limitations.""",
            
            'creative': """You are a creative and imaginative AI assistant. Help with creative tasks while maintaining accuracy for factual information. Be engaging and inspiring while staying grounded in reality.""",
            
            'analytical': """You are an analytical AI assistant focused on logical reasoning and problem-solving. Break down complex problems, provide structured analysis, and offer clear, evidence-based conclusions."""
        }
        
        self.response_guidelines = [
            "Be accurate and truthful",
            "Stay relevant to the user's question",
            "Be concise but complete",
            "Use clear, simple language",
            "Provide examples when helpful",
            "Acknowledge uncertainty when appropriate"
        ]
    
    def create_enhanced_prompt(self, user_input: str, context: List[Dict[str, str]] = None,
                              prompt_type: str = 'conversational',
                              additional_context: str = None) -> List[Dict[str, str]]:
        """
        Create an enhanced prompt with context and guidelines.
        
        Args:
            user_input: User's input
            context: Previous conversation context
            prompt_type: Type of prompt to use
            additional_context: Additional context information
            
        Returns:
            Formatted messages for the LLM
        """
        messages = []
        
        # Add system prompt
        system_prompt = self.system_prompts.get(prompt_type, self.system_prompts['default'])
        
        if additional_context:
            system_prompt += f"\n\nAdditional context: {additional_context}"
        
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add conversation context
        if context:
            # Limit context to prevent token overflow
            recent_context = context[-6:]  # Last 6 exchanges
            for msg in recent_context:
                messages.append(msg)
        
        # Add current user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        return messages
    
    def detect_prompt_type(self, user_input: str, context: List[Dict[str, str]] = None) -> str:
        """
        Detect the best prompt type based on user input and context.
        
        Args:
            user_input: User's input
            context: Conversation context
            
        Returns:
            Recommended prompt type
        """
        input_lower = user_input.lower()
        
        # Technical indicators
        technical_keywords = ['code', 'programming', 'algorithm', 'function', 'api', 'database',
                             'server', 'network', 'software', 'hardware', 'debug', 'error']
        if any(keyword in input_lower for keyword in technical_keywords):
            return 'technical'
        
        # Creative indicators
        creative_keywords = ['story', 'poem', 'creative', 'imagine', 'design', 'art',
                           'write', 'compose', 'brainstorm', 'idea']
        if any(keyword in input_lower for keyword in creative_keywords):
            return 'creative'
        
        # Analytical indicators
        analytical_keywords = ['analyze', 'compare', 'evaluate', 'pros and cons', 'advantages',
                              'disadvantages', 'problem', 'solution', 'strategy', 'plan']
        if any(keyword in input_lower for keyword in analytical_keywords):
            return 'analytical'
        
        # Default to conversational
        return 'conversational'


class ResponseValidator:
    """Validates and scores LLM responses for quality."""
    
    def __init__(self):
        """Initialize response validator."""
        self.quality_metrics = {
            'relevance': 0.3,
            'coherence': 0.25,
            'completeness': 0.2,
            'accuracy': 0.15,
            'clarity': 0.1
        }
        
        self.red_flags = [
            'I apologize, but I cannot',
            'I don\'t have access to',
            'I cannot browse the internet',
            'As an AI language model',
            'I don\'t have real-time',
            'I cannot provide medical advice',
            'I cannot provide legal advice'
        ]
    
    def validate_response(self, response: str, user_input: str,
                         context: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Validate and score a response.
        
        Args:
            response: LLM response to validate
            user_input: Original user input
            context: Conversation context
            
        Returns:
            Validation results with scores and suggestions
        """
        validation_result = {
            'overall_score': 0.0,
            'scores': {},
            'issues': [],
            'suggestions': [],
            'is_acceptable': True
        }
        
        # Calculate individual scores
        validation_result['scores']['relevance'] = self._score_relevance(response, user_input)
        validation_result['scores']['coherence'] = self._score_coherence(response)
        validation_result['scores']['completeness'] = self._score_completeness(response, user_input)
        validation_result['scores']['accuracy'] = self._score_accuracy(response)
        validation_result['scores']['clarity'] = self._score_clarity(response)
        
        # Calculate overall score
        overall_score = 0.0
        for metric, weight in self.quality_metrics.items():
            overall_score += validation_result['scores'][metric] * weight
        
        validation_result['overall_score'] = overall_score
        
        # Check for issues
        validation_result['issues'] = self._detect_issues(response)
        
        # Generate suggestions
        validation_result['suggestions'] = self._generate_suggestions(validation_result)
        
        # Determine if response is acceptable
        validation_result['is_acceptable'] = (
            overall_score >= 0.6 and
            len(validation_result['issues']) <= 2
        )
        
        return validation_result
    
    def _score_relevance(self, response: str, user_input: str) -> float:
        """Score response relevance to user input."""
        # Simple keyword overlap scoring
        user_words = set(user_input.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        user_words -= stop_words
        response_words -= stop_words
        
        if not user_words:
            return 0.5
        
        overlap = len(user_words.intersection(response_words))
        relevance_score = min(overlap / len(user_words), 1.0)
        
        # Boost score if response directly addresses question words
        question_words = {'what', 'how', 'when', 'where', 'why', 'who', 'which'}
        if any(word in user_input.lower() for word in question_words):
            if len(response) > 20:  # Substantial response to question
                relevance_score = min(relevance_score + 0.2, 1.0)
        
        return relevance_score
    
    def _score_coherence(self, response: str) -> float:
        """Score response coherence and flow."""
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return 0.8  # Single sentence is coherent by default
        
        coherence_score = 0.8  # Base score
        
        # Check for repetition
        unique_sentences = set(sentences)
        if len(unique_sentences) < len(sentences):
            coherence_score -= 0.2
        
        # Check for contradictions (simple heuristic)
        contradiction_pairs = [
            ('yes', 'no'), ('true', 'false'), ('can', 'cannot'),
            ('will', 'will not'), ('is', 'is not')
        ]
        
        response_lower = response.lower()
        for word1, word2 in contradiction_pairs:
            if word1 in response_lower and word2 in response_lower:
                coherence_score -= 0.1
        
        return max(coherence_score, 0.0)
    
    def _score_completeness(self, response: str, user_input: str) -> float:
        """Score response completeness."""
        # Basic length-based scoring
        response_length = len(response.split())
        
        if response_length < 5:
            return 0.2  # Too short
        elif response_length < 15:
            return 0.6  # Somewhat complete
        elif response_length < 50:
            return 0.9  # Good length
        else:
            return 0.7  # Might be too verbose
    
    def _score_accuracy(self, response: str) -> float:
        """Score response accuracy (basic checks)."""
        accuracy_score = 0.8  # Base score
        
        # Check for uncertainty indicators (good for accuracy)
        uncertainty_indicators = ['might', 'could', 'possibly', 'perhaps', 'likely', 'probably']
        if any(indicator in response.lower() for indicator in uncertainty_indicators):
            accuracy_score += 0.1
        
        # Check for overconfident statements
        overconfident_indicators = ['definitely', 'certainly', 'absolutely', 'always', 'never']
        overconfident_count = sum(1 for indicator in overconfident_indicators 
                                 if indicator in response.lower())
        if overconfident_count > 2:
            accuracy_score -= 0.2
        
        return min(max(accuracy_score, 0.0), 1.0)
    
    def _score_clarity(self, response: str) -> float:
        """Score response clarity."""
        clarity_score = 0.8  # Base score
        
        # Check average sentence length
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            
            if avg_sentence_length > 25:  # Too long
                clarity_score -= 0.2
            elif avg_sentence_length < 5:  # Too short
                clarity_score -= 0.1
        
        # Check for complex words (simple heuristic)
        words = response.split()
        complex_words = [word for word in words if len(word) > 12]
        if len(complex_words) > len(words) * 0.1:  # More than 10% complex words
            clarity_score -= 0.1
        
        return max(clarity_score, 0.0)
    
    def _detect_issues(self, response: str) -> List[str]:
        """Detect common issues in responses."""
        issues = []
        
        # Check for red flag phrases
        for red_flag in self.red_flags:
            if red_flag.lower() in response.lower():
                issues.append(f"Contains limitation phrase: '{red_flag}'")
        
        # Check for very short responses
        if len(response.split()) < 5:
            issues.append("Response is too short")
        
        # Check for very long responses
        if len(response.split()) > 200:
            issues.append("Response might be too verbose")
        
        # Check for repetition
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip().lower() for s in sentences if s.strip()]
        if len(sentences) != len(set(sentences)):
            issues.append("Contains repetitive content")
        
        return issues
    
    def _generate_suggestions(self, validation_result: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions based on validation results."""
        suggestions = []
        scores = validation_result['scores']
        
        if scores['relevance'] < 0.6:
            suggestions.append("Improve relevance to user's question")
        
        if scores['coherence'] < 0.6:
            suggestions.append("Improve logical flow and coherence")
        
        if scores['completeness'] < 0.6:
            suggestions.append("Provide more complete information")
        
        if scores['clarity'] < 0.6:
            suggestions.append("Simplify language and improve clarity")
        
        return suggestions


class ContextManager:
    """Manages conversation context intelligently."""
    
    def __init__(self, max_context_length: int = 4000):
        """Initialize context manager."""
        self.max_context_length = max_context_length
        self.conversation_history = deque(maxlen=50)
        self.context_cache = {}
        
    def add_exchange(self, user_input: str, assistant_response: str, metadata: Dict[str, Any] = None):
        """Add a conversation exchange to history."""
        exchange = {
            'user': user_input,
            'assistant': assistant_response,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.conversation_history.append(exchange)
    
    def get_relevant_context(self, current_input: str, max_exchanges: int = 6) -> List[Dict[str, str]]:
        """
        Get relevant context for current input.
        
        Args:
            current_input: Current user input
            max_exchanges: Maximum number of exchanges to include
            
        Returns:
            Relevant context messages
        """
        if not self.conversation_history:
            return []
        
        # Get recent exchanges
        recent_exchanges = list(self.conversation_history)[-max_exchanges:]
        
        # Convert to message format
        messages = []
        for exchange in recent_exchanges:
            messages.append({
                "role": "user",
                "content": exchange['user']
            })
            messages.append({
                "role": "assistant",
                "content": exchange['assistant']
            })
        
        # Check total token length (rough estimate)
        total_length = sum(len(msg['content'].split()) for msg in messages)
        
        # Compress if too long
        if total_length > self.max_context_length // 4:  # Rough token estimate
            messages = self._compress_context(messages)
        
        return messages
    
    def _compress_context(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Compress context to fit within limits."""
        # Simple compression: keep most recent messages
        target_length = self.max_context_length // 6
        current_length = 0
        compressed_messages = []
        
        # Add messages from most recent backwards
        for message in reversed(messages):
            msg_length = len(message['content'].split())
            if current_length + msg_length <= target_length:
                compressed_messages.insert(0, message)
                current_length += msg_length
            else:
                break
        
        return compressed_messages


class EnhancedLLMInterface:
    """Enhanced LLM interface with improved result quality."""
    
    def __init__(self, model_name: str = "llama3.2:3b", config: Dict[str, Any] = None):
        """
        Initialize enhanced LLM interface.
        
        Args:
            model_name: Ollama model name
            config: Configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
        
        # Initialize components
        self.prompt_engineer = PromptEngineer()
        self.response_validator = ResponseValidator()
        self.context_manager = ContextManager()
        
        # Response cache for similar queries
        self.response_cache = {}
        self.cache_enabled = self.config.get('enable_response_caching', True)
        
        # Test Ollama connection
        self._test_connection()
        
        logger.info(f"✅ Enhanced LLM interface initialized with model: {model_name}")
    
    def _test_connection(self):
        """Test connection to Ollama."""
        try:
            response = ollama.list()
            available_models = []
            
            if hasattr(response, 'models'):
                models_list = response.models
            elif isinstance(response, dict) and 'models' in response:
                models_list = response['models']
            else:
                models_list = response
            
            for model in models_list:
                if hasattr(model, 'model'):
                    available_models.append(model.model)
                elif isinstance(model, dict):
                    model_name = model.get('name', model.get('model', ''))
                    if model_name:
                        available_models.append(model_name)
            
            if not any(self.model_name in model for model in available_models):
                logger.warning(f"Model {self.model_name} not found in available models: {available_models}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise
    
    def generate_response(self, user_input: str, additional_context: str = None,
                         max_retries: int = 2) -> Tuple[str, Dict[str, Any]]:
        """
        Generate enhanced response with quality validation.
        
        Args:
            user_input: User's input
            additional_context: Additional context information
            max_retries: Maximum retry attempts for poor responses
            
        Returns:
            Tuple of (response, metadata)
        """
        start_time = time.time()
        metadata = {
            'processing_time': 0,
            'retries': 0,
            'validation_score': 0.0,
            'cache_hit': False,
            'prompt_type': 'conversational'
        }
        
        # Check cache first
        if self.cache_enabled:
            cache_key = self._generate_cache_key(user_input)
            if cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key]
                metadata['cache_hit'] = True
                metadata['processing_time'] = time.time() - start_time
                return cached_response['response'], metadata
        
        best_response = ""
        best_score = 0.0
        
        for attempt in range(max_retries + 1):
            try:
                # Detect optimal prompt type
                prompt_type = self.prompt_engineer.detect_prompt_type(
                    user_input, 
                    self.context_manager.get_relevant_context(user_input)
                )
                metadata['prompt_type'] = prompt_type
                
                # Create enhanced prompt
                messages = self.prompt_engineer.create_enhanced_prompt(
                    user_input,
                    self.context_manager.get_relevant_context(user_input),
                    prompt_type,
                    additional_context
                )
                
                # Generate response
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    options={
                        "temperature": self.config.get('temperature', 0.7),
                        "top_k": self.config.get('top_k', 40),
                        "top_p": self.config.get('top_p', 0.9),
                        "repeat_penalty": self.config.get('repeat_penalty', 1.1),
                        "num_predict": self.config.get('num_predict', 150),
                        "num_ctx": self.config.get('num_ctx', 4096),
                    }
                )
                
                response_text = response['message']['content'].strip()
                
                # Validate response
                validation_result = self.response_validator.validate_response(
                    response_text, user_input, messages
                )
                
                # Keep best response
                if validation_result['overall_score'] > best_score:
                    best_response = response_text
                    best_score = validation_result['overall_score']
                    metadata['validation_score'] = best_score
                    metadata['validation_issues'] = validation_result['issues']
                    metadata['validation_suggestions'] = validation_result['suggestions']
                
                # If response is good enough, use it
                if validation_result['is_acceptable']:
                    break
                
                metadata['retries'] = attempt + 1
                
                if attempt < max_retries:
                    logger.debug(f"Response quality low ({best_score:.2f}), retrying...")
                
            except Exception as e:
                logger.error(f"LLM generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    best_response = "I apologize, but I'm having trouble generating a response right now. Please try again."
                    metadata['error'] = str(e)
        
        # Post-process response
        final_response = self._post_process_response(best_response)
        
        # Add to context
        self.context_manager.add_exchange(user_input, final_response, metadata)
        
        # Cache response
        if self.cache_enabled and best_score > 0.7:
            cache_key = self._generate_cache_key(user_input)
            self.response_cache[cache_key] = {
                'response': final_response,
                'timestamp': datetime.now(),
                'score': best_score
            }
            
            # Limit cache size
            if len(self.response_cache) > 100:
                oldest_key = min(self.response_cache.keys(), 
                               key=lambda k: self.response_cache[k]['timestamp'])
                del self.response_cache[oldest_key]
        
        metadata['processing_time'] = time.time() - start_time
        
        logger.debug(f"Generated response (score: {best_score:.2f}, "
                    f"time: {metadata['processing_time']:.2f}s): '{final_response[:50]}...'")
        
        return final_response, metadata
    
    def _generate_cache_key(self, user_input: str) -> str:
        """Generate cache key for user input."""
        # Normalize input for caching
        normalized = re.sub(r'\s+', ' ', user_input.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _post_process_response(self, response: str) -> str:
        """Post-process response for better quality."""
        # Remove excessive whitespace
        processed = re.sub(r'\s+', ' ', response).strip()
        
        # Ensure proper sentence ending
        if processed and not processed.endswith(('.', '!', '?')):
            processed += '.'
        
        # Remove redundant phrases
        redundant_phrases = [
            "As an AI language model, ",
            "I'm an AI assistant and ",
            "As an artificial intelligence, "
        ]
        
        for phrase in redundant_phrases:
            processed = processed.replace(phrase, "")
        
        return processed
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation history."""
        history = list(self.context_manager.conversation_history)
        
        return {
            'total_exchanges': len(history),
            'recent_topics': self._extract_recent_topics(history[-5:]),
            'avg_response_length': self._calculate_avg_response_length(history),
            'conversation_start': history[0]['timestamp'] if history else None
        }
    
    def _extract_recent_topics(self, recent_history: List[Dict[str, Any]]) -> List[str]:
        """Extract topics from recent conversation."""
        topics = []
        for exchange in recent_history:
            # Simple keyword extraction
            user_input = exchange['user'].lower()
            words = user_input.split()
            
            # Look for topic indicators
            topic_words = [word for word in words if len(word) > 4 and word.isalpha()]
            topics.extend(topic_words[:2])  # Take first 2 significant words
        
        return list(set(topics))[:5]  # Return unique topics, max 5
    
    def _calculate_avg_response_length(self, history: List[Dict[str, Any]]) -> float:
        """Calculate average response length."""
        if not history:
            return 0.0
        
        total_words = sum(len(exchange['assistant'].split()) for exchange in history)
        return total_words / len(history)
    
    def clear_cache(self):
        """Clear response cache."""
        self.response_cache.clear()
        logger.info("Response cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.response_cache),
            'cache_enabled': self.cache_enabled,
            'oldest_entry': min(self.response_cache.values(), 
                              key=lambda x: x['timestamp'])['timestamp'] if self.response_cache else None
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize enhanced LLM
    config = {
        'temperature': 0.7,
        'top_k': 40,
        'top_p': 0.9,
        'num_predict': 150,
        'enable_response_caching': True
    }
    
    try:
        llm = EnhancedLLMInterface("llama3.2:3b", config)
        
        # Test queries
        test_queries = [
            "What is artificial intelligence?",
            "How do I write a Python function?",
            "Tell me a creative story about a robot.",
            "What are the pros and cons of renewable energy?"
        ]
        
        print("Enhanced LLM interface initialized successfully!")
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response, metadata = llm.generate_response(query)
            print(f"Response: {response}")
            print(f"Metadata: {metadata}")
        
        print(f"\nConversation summary: {llm.get_conversation_summary()}")
        print(f"Cache stats: {llm.get_cache_stats()}")
        
    except Exception as e:
        print(f"Failed to initialize enhanced LLM: {e}")
        print("Make sure Ollama is running and models are available.")
