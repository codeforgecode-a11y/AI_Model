#!/usr/bin/env python3
"""
Intelligent Model Selection Manager

This module manages the selection between different AI models based on query type,
context, and user preferences. It coordinates between the general conversation model
(llama3.2:1b) and the specialized coding model (Qwen2.5-Coder-Tools).
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enumeration of available model types."""
    GENERAL = "general"
    CODING = "coding"
    FALLBACK = "fallback"


class QueryClassifier:
    """Classifies user queries to determine appropriate model."""
    
    def __init__(self):
        self.coding_keywords = {
            'direct': [
                'code', 'function', 'class', 'variable', 'algorithm', 'debug',
                'programming', 'script', 'method', 'library', 'framework',
                'api', 'bug', 'syntax', 'compile', 'implement', 'refactor'
            ],
            'languages': [
                'python', 'javascript', 'java', 'cpp', 'c++', 'rust', 'go',
                'html', 'css', 'sql', 'bash', 'shell', 'php', 'ruby',
                'typescript', 'kotlin', 'swift', 'dart', 'scala'
            ],
            'tools': [
                'git', 'docker', 'kubernetes', 'npm', 'pip', 'maven',
                'gradle', 'webpack', 'babel', 'eslint', 'pytest'
            ],
            'concepts': [
                'recursion', 'iteration', 'oop', 'inheritance', 'polymorphism',
                'encapsulation', 'abstraction', 'design pattern', 'mvc',
                'rest api', 'database', 'sql query', 'json', 'xml'
            ]
        }
        
        self.general_keywords = [
            'weather', 'news', 'time', 'date', 'hello', 'hi', 'how are you',
            'tell me about', 'explain', 'what is', 'who is', 'where is',
            'when', 'why', 'story', 'joke', 'recipe', 'travel', 'health'
        ]
    
    def classify_query(self, query: str) -> Tuple[ModelType, float]:
        """
        Classify query and return model type with confidence score.
        
        Args:
            query: User's input query
            
        Returns:
            Tuple of (ModelType, confidence_score)
        """
        query_lower = query.lower()
        
        # Calculate coding score
        coding_score = 0
        total_coding_keywords = sum(len(keywords) for keywords in self.coding_keywords.values())
        
        for category, keywords in self.coding_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            # Weight different categories
            weight = {'direct': 3, 'languages': 2, 'tools': 2, 'concepts': 1}.get(category, 1)
            coding_score += matches * weight
        
        # Normalize coding score
        coding_confidence = min(coding_score / 10.0, 1.0)  # Cap at 1.0
        
        # Calculate general conversation score
        general_score = sum(1 for keyword in self.general_keywords if keyword in query_lower)
        general_confidence = min(general_score / 5.0, 1.0)  # Cap at 1.0
        
        # Check for code patterns (higher weight)
        code_patterns = [
            r'def\s+\w+', r'function\s+\w+', r'class\s+\w+', r'import\s+\w+',
            r'#include', r'package\s+\w+', r'fn\s+\w+', r'var\s+\w+',
            r'let\s+\w+', r'const\s+\w+', r'public\s+class'
        ]
        
        import re
        if any(re.search(pattern, query, re.IGNORECASE) for pattern in code_patterns):
            coding_confidence += 0.5
        
        # Check for file extensions
        if re.search(r'\.\w{1,4}\b', query):
            coding_confidence += 0.3
        
        # Determine model type
        if coding_confidence > 0.3:
            return ModelType.CODING, min(coding_confidence, 1.0)
        elif general_confidence > 0.2:
            return ModelType.GENERAL, min(general_confidence, 1.0)
        else:
            # Default to general for ambiguous queries
            return ModelType.GENERAL, 0.5


class ModelSelector:
    """Manages model selection and coordination."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize model selector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.classifier = QueryClassifier()
        
        # Model availability tracking
        self.model_status = {
            ModelType.GENERAL: True,  # Assume general model is available
            ModelType.CODING: False,  # Will be updated when Qwen is available
            ModelType.FALLBACK: True  # Basic LLM interface
        }
        
        # Performance tracking
        self.model_performance = {
            ModelType.GENERAL: {'response_time': [], 'success_rate': 1.0},
            ModelType.CODING: {'response_time': [], 'success_rate': 1.0},
            ModelType.FALLBACK: {'response_time': [], 'success_rate': 1.0}
        }
        
        # Selection preferences
        self.selection_config = {
            'coding_threshold': 0.3,  # Minimum confidence for coding model
            'fallback_threshold': 2.0,  # Max response time before fallback
            'max_retries': 2,
            'prefer_specialized': True  # Prefer specialized models when available
        }
        
        logger.info("Model selector initialized")
    
    def update_model_status(self, model_type: ModelType, available: bool):
        """Update model availability status."""
        self.model_status[model_type] = available
        logger.info(f"Model {model_type.value} status updated: {'available' if available else 'unavailable'}")
    
    def select_model(self, query: str, context: Dict[str, Any] = None) -> Tuple[ModelType, float]:
        """
        Select the most appropriate model for the query.
        
        Args:
            query: User's input query
            context: Additional context information
            
        Returns:
            Tuple of (selected_model_type, confidence)
        """
        # Classify the query
        suggested_model, confidence = self.classifier.classify_query(query)
        
        # Check model availability and performance
        if suggested_model == ModelType.CODING:
            if not self.model_status[ModelType.CODING]:
                logger.info("Coding model not available, falling back to general model")
                return ModelType.GENERAL, confidence * 0.8
            
            # Check if confidence meets threshold
            if confidence < self.selection_config['coding_threshold']:
                logger.info(f"Coding confidence {confidence:.2f} below threshold, using general model")
                return ModelType.GENERAL, confidence
        
        # Check performance history
        if suggested_model in self.model_performance:
            perf = self.model_performance[suggested_model]
            if perf['success_rate'] < 0.7:  # If success rate is low
                logger.warning(f"Model {suggested_model.value} has low success rate, considering fallback")
                if suggested_model == ModelType.CODING and self.model_status[ModelType.GENERAL]:
                    return ModelType.GENERAL, confidence * 0.9
        
        return suggested_model, confidence
    
    def record_performance(self, model_type: ModelType, response_time: float, success: bool):
        """Record model performance metrics."""
        if model_type in self.model_performance:
            perf = self.model_performance[model_type]
            
            # Update response time (keep last 10 measurements)
            perf['response_time'].append(response_time)
            if len(perf['response_time']) > 10:
                perf['response_time'].pop(0)
            
            # Update success rate (exponential moving average)
            alpha = 0.1  # Learning rate
            perf['success_rate'] = (1 - alpha) * perf['success_rate'] + alpha * (1.0 if success else 0.0)
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get current model statistics."""
        stats = {}
        
        for model_type, perf in self.model_performance.items():
            avg_response_time = sum(perf['response_time']) / len(perf['response_time']) if perf['response_time'] else 0
            
            stats[model_type.value] = {
                'available': self.model_status[model_type],
                'success_rate': perf['success_rate'],
                'avg_response_time': avg_response_time,
                'recent_responses': len(perf['response_time'])
            }
        
        return stats
    
    def should_fallback(self, model_type: ModelType, response_time: float = None) -> bool:
        """Determine if we should fallback to a different model."""
        # Check availability
        if not self.model_status[model_type]:
            return True
        
        # Check response time
        if response_time and response_time > self.selection_config['fallback_threshold']:
            return True
        
        # Check success rate
        if model_type in self.model_performance:
            if self.model_performance[model_type]['success_rate'] < 0.5:
                return True
        
        return False
    
    def get_fallback_model(self, original_model: ModelType) -> ModelType:
        """Get appropriate fallback model."""
        if original_model == ModelType.CODING:
            if self.model_status[ModelType.GENERAL]:
                return ModelType.GENERAL
            else:
                return ModelType.FALLBACK
        elif original_model == ModelType.GENERAL:
            return ModelType.FALLBACK
        else:
            return ModelType.FALLBACK


def test_model_selector():
    """Test function for model selector."""
    selector = ModelSelector()
    
    # Test queries
    test_queries = [
        "How do I write a Python function?",
        "What's the weather like today?",
        "Debug this code: def test(): print('hello'",
        "Tell me a joke",
        "Explain recursion in programming",
        "How are you doing?",
        "Create a REST API in Node.js",
        "What time is it?"
    ]
    
    print("Testing Model Selector...")
    
    for query in test_queries:
        model_type, confidence = selector.select_model(query)
        print(f"Query: {query}")
        print(f"Selected Model: {model_type.value}, Confidence: {confidence:.2f}\n")
    
    # Test performance recording
    selector.record_performance(ModelType.CODING, 1.5, True)
    selector.record_performance(ModelType.GENERAL, 0.8, True)
    
    print("Model Statistics:")
    stats = selector.get_model_stats()
    for model, stat in stats.items():
        print(f"{model}: {stat}")


if __name__ == "__main__":
    test_model_selector()
