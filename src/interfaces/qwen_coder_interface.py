#!/usr/bin/env python3
"""
Qwen2.5-Coder-Tools Interface for Enhanced Coding Assistance

This module provides a specialized interface for the Qwen2.5-Coder-Tools model,
designed specifically for coding tasks, debugging, and programming assistance.
It integrates with the existing voice companion system while maintaining
compatibility with the three-component memory architecture.
"""

import ollama
import logging
import time
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """Analyzes code snippets and determines appropriate assistance type."""
    
    def __init__(self):
        self.programming_keywords = {
            'python': ['def', 'class', 'import', 'from', 'if __name__', 'print(', 'return'],
            'javascript': ['function', 'const', 'let', 'var', 'console.log', '=>', 'require('],
            'java': ['public class', 'private', 'public', 'static void main', 'System.out'],
            'cpp': ['#include', 'int main', 'std::', 'cout', 'cin', 'namespace'],
            'rust': ['fn main', 'let mut', 'println!', 'use std::', 'impl', 'struct'],
            'go': ['package main', 'func main', 'import', 'fmt.Print', 'var', 'type'],
        }
        
        self.coding_patterns = [
            r'def\s+\w+\s*\(',  # Python function definition
            r'class\s+\w+\s*[:\(]',  # Class definition
            r'function\s+\w+\s*\(',  # JavaScript function
            r'public\s+class\s+\w+',  # Java class
            r'#include\s*<\w+>',  # C++ include
            r'fn\s+\w+\s*\(',  # Rust function
            r'package\s+\w+',  # Go package
        ]
    
    def detect_language(self, text: str) -> Optional[str]:
        """Detect programming language from code snippet."""
        text_lower = text.lower()
        
        for language, keywords in self.programming_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score >= 2:  # Require at least 2 keyword matches
                return language
        
        return None
    
    def is_coding_query(self, text: str) -> bool:
        """Determine if the query is coding-related."""
        coding_indicators = [
            'code', 'function', 'class', 'variable', 'algorithm', 'debug',
            'error', 'exception', 'syntax', 'compile', 'programming',
            'script', 'method', 'library', 'framework', 'api', 'bug',
            'implement', 'refactor', 'optimize', 'test', 'unit test'
        ]
        
        text_lower = text.lower()
        
        # Check for coding keywords
        if any(indicator in text_lower for indicator in coding_indicators):
            return True
        
        # Check for code patterns
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.coding_patterns):
            return True
        
        # Check for file extensions
        if re.search(r'\.\w{1,4}\b', text):  # File extensions like .py, .js, etc.
            return True
        
        return False
    
    def extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from text."""
        # Match code blocks with triple backticks
        code_blocks = re.findall(r'```[\w]*\n(.*?)\n```', text, re.DOTALL)
        
        # Match inline code with single backticks
        inline_code = re.findall(r'`([^`]+)`', text)
        
        return code_blocks + inline_code


class QwenCoderInterface:
    """Enhanced interface for Qwen2.5-Coder-Tools model."""
    
    def __init__(self, model_name: str = "hhao/qwen2.5-coder-tools:7b", config: Dict[str, Any] = None):
        """
        Initialize Qwen Coder interface.
        
        Args:
            model_name: Qwen model name
            config: Configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
        self.code_analyzer = CodeAnalyzer()
        
        # Coding-specific configuration
        self.coding_config = {
            'temperature': 0.3,  # Lower temperature for more precise code
            'top_k': 20,
            'top_p': 0.8,
            'repeat_penalty': 1.1,
            'num_predict': 500,  # Longer responses for code explanations
            'num_ctx': 8192,     # Larger context for code analysis
        }
        
        # Test model availability
        self._test_model_availability()
        
        logger.info(f"✅ Qwen Coder interface initialized with model: {model_name}")
    
    def _test_model_availability(self) -> None:
        """Test if the Qwen model is available."""
        try:
            response = ollama.list()
            models = response.get('models', [])
            
            # Extract model names
            if models and isinstance(models[0], dict):
                if 'name' in models[0]:
                    available_models = [model['name'] for model in models]
                elif 'model' in models[0]:
                    available_models = [model['model'] for model in models]
                else:
                    available_models = [str(model.get(list(model.keys())[0], '')) for model in models]
            else:
                available_models = [str(model) for model in models]
            
            if self.model_name not in available_models:
                logger.warning(f"Qwen model {self.model_name} not found. Available models: {available_models}")
                # Don't raise exception, allow fallback to other models
            else:
                logger.info(f"Qwen model {self.model_name} is available")
                
        except Exception as e:
            logger.error(f"Failed to check Qwen model availability: {e}")
    
    def create_coding_prompt(self, user_input: str, context: str = None, 
                           task_type: str = "general") -> List[Dict[str, str]]:
        """
        Create specialized prompts for coding tasks.
        
        Args:
            user_input: User's coding query
            context: Previous conversation context
            task_type: Type of coding task (debug, implement, explain, review)
            
        Returns:
            Formatted messages for the model
        """
        system_prompts = {
            "general": """You are an expert programming assistant powered by Qwen2.5-Coder-Tools. 
You excel at:
- Writing clean, efficient, and well-documented code
- Debugging and fixing code issues
- Explaining complex programming concepts
- Code review and optimization suggestions
- Best practices and design patterns

Always provide clear explanations with your code and consider edge cases.""",
            
            "debug": """You are a debugging specialist. When analyzing code:
1. Identify the specific issue or error
2. Explain why the error occurs
3. Provide a corrected version
4. Suggest preventive measures
5. Include test cases if relevant""",
            
            "implement": """You are a code implementation expert. When writing code:
1. Break down the problem into smaller components
2. Write clean, readable, and efficient code
3. Include proper error handling
4. Add comprehensive comments
5. Provide usage examples""",
            
            "explain": """You are a programming educator. When explaining code:
1. Break down complex concepts into simple terms
2. Use analogies when helpful
3. Provide step-by-step explanations
4. Include visual representations if applicable
5. Suggest related concepts to explore""",
            
            "review": """You are a code reviewer. When reviewing code:
1. Check for bugs and potential issues
2. Evaluate code quality and readability
3. Suggest improvements and optimizations
4. Verify adherence to best practices
5. Recommend refactoring opportunities"""
        }
        
        messages = []
        
        # Add system prompt
        system_prompt = system_prompts.get(task_type, system_prompts["general"])
        if context:
            system_prompt += f"\n\nPrevious context:\n{context}"
        
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add user message
        messages.append({
            "role": "user", 
            "content": user_input
        })
        
        return messages
    
    def determine_task_type(self, user_input: str) -> str:
        """Determine the type of coding task based on user input."""
        text_lower = user_input.lower()
        
        if any(word in text_lower for word in ['debug', 'error', 'fix', 'bug', 'wrong', 'issue']):
            return "debug"
        elif any(word in text_lower for word in ['implement', 'write', 'create', 'build', 'develop']):
            return "implement"
        elif any(word in text_lower for word in ['explain', 'how does', 'what is', 'understand', 'clarify']):
            return "explain"
        elif any(word in text_lower for word in ['review', 'check', 'improve', 'optimize', 'refactor']):
            return "review"
        else:
            return "general"
    
    def generate_coding_response(self, user_input: str, context: str = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate coding-specific response using Qwen model.
        
        Args:
            user_input: User's coding query
            context: Previous conversation context
            
        Returns:
            Tuple of (response, metadata)
        """
        start_time = time.time()
        metadata = {
            'processing_time': 0,
            'model_used': self.model_name,
            'task_type': 'coding',
            'language_detected': None,
            'is_coding_query': False
        }
        
        try:
            # Analyze the query
            is_coding = self.code_analyzer.is_coding_query(user_input)
            metadata['is_coding_query'] = is_coding
            
            if not is_coding:
                # Return None to indicate this should be handled by general model
                return None, metadata
            
            # Detect programming language
            detected_language = self.code_analyzer.detect_language(user_input)
            metadata['language_detected'] = detected_language
            
            # Determine task type
            task_type = self.determine_task_type(user_input)
            metadata['task_type'] = task_type
            
            # Create specialized prompt
            messages = self.create_coding_prompt(user_input, context, task_type)
            
            # Generate response with coding-optimized settings
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options=self.coding_config
            )
            
            response_text = response['message']['content'].strip()
            
            # Post-process response for better formatting
            response_text = self._format_code_response(response_text)
            
            metadata['processing_time'] = time.time() - start_time
            
            return response_text, metadata
            
        except Exception as e:
            logger.error(f"Qwen coding response generation failed: {e}")
            metadata['processing_time'] = time.time() - start_time
            metadata['error'] = str(e)
            return None, metadata
    
    def _format_code_response(self, response: str) -> str:
        """Format code response for better readability."""
        # Ensure code blocks are properly formatted
        # Add language hints to code blocks if missing
        formatted_response = response
        
        # Add newlines before and after code blocks for better readability
        formatted_response = re.sub(r'```(\w+)?\n', r'\n```\1\n', formatted_response)
        formatted_response = re.sub(r'\n```\n', r'\n```\n\n', formatted_response)
        
        return formatted_response.strip()


def test_qwen_interface():
    """Test function for Qwen interface."""
    try:
        qwen = QwenCoderInterface()
        
        # Test coding query detection
        test_queries = [
            "How do I write a Python function to sort a list?",
            "Debug this JavaScript code: function test() { console.log('hello' }",
            "What's the weather like today?",  # Non-coding query
            "Explain how recursion works in programming",
            "Review this code for potential issues: def factorial(n): return n * factorial(n-1)"
        ]
        
        print("Testing Qwen Coder Interface...")
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response, metadata = qwen.generate_coding_response(query)
            
            if response:
                print(f"Response: {response[:100]}...")
                print(f"Metadata: {metadata}")
            else:
                print("Non-coding query - would be handled by general model")
                
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_qwen_interface()
