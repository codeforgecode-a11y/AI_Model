#!/usr/bin/env python3
"""
Memory Context Analyzer for Enhanced Memory Integration

This module provides intelligent context analysis for the memory system,
enabling better distinction between coding and general conversation contexts.
It enhances the three-component memory architecture with context-aware features.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class MemoryContextAnalyzer:
    """Analyzes conversation context for enhanced memory integration."""
    
    def __init__(self):
        """Initialize the memory context analyzer."""
        self.coding_keywords = {
            'languages': ['python', 'javascript', 'java', 'cpp', 'rust', 'go', 'html', 'css'],
            'concepts': ['function', 'class', 'variable', 'algorithm', 'debug', 'error'],
            'tools': ['git', 'docker', 'npm', 'pip', 'ide', 'compiler'],
            'patterns': ['loop', 'recursion', 'inheritance', 'polymorphism', 'api']
        }
        
        self.context_weights = {
            'recent_interactions': 0.4,
            'topic_continuity': 0.3,
            'model_consistency': 0.2,
            'time_proximity': 0.1
        }
    
    def analyze_conversation_context(self, interactions: List[Dict[str, Any]], 
                                   current_query: str) -> Dict[str, Any]:
        """
        Analyze conversation context to provide enhanced memory insights.
        
        Args:
            interactions: List of recent interactions
            current_query: Current user query
            
        Returns:
            Context analysis results
        """
        if not interactions:
            return self._create_empty_context()
        
        # Analyze interaction patterns
        coding_score = self._calculate_coding_score(interactions)
        topic_continuity = self._analyze_topic_continuity(interactions)
        model_usage_pattern = self._analyze_model_usage(interactions)
        temporal_context = self._analyze_temporal_context(interactions)
        
        # Determine dominant context type
        context_type = 'coding' if coding_score > 0.5 else 'general'
        
        # Extract relevant context for current query
        relevant_context = self._extract_relevant_context(
            interactions, current_query, context_type
        )
        
        return {
            'context_type': context_type,
            'coding_score': coding_score,
            'topic_continuity': topic_continuity,
            'model_usage_pattern': model_usage_pattern,
            'temporal_context': temporal_context,
            'relevant_context': relevant_context,
            'context_summary': self._generate_context_summary(interactions, context_type),
            'recommendations': self._generate_recommendations(
                context_type, coding_score, topic_continuity
            )
        }
    
    def _calculate_coding_score(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate how coding-focused the recent conversation has been."""
        if not interactions:
            return 0.0
        
        coding_interactions = 0
        total_weight = 0
        
        for i, interaction in enumerate(interactions[-10:]):  # Last 10 interactions
            weight = 1.0 / (i + 1)  # More recent interactions have higher weight
            total_weight += weight
            
            metadata = interaction.get('metadata', {})
            if metadata.get('is_coding_query', False):
                coding_interactions += weight
        
        return coding_interactions / total_weight if total_weight > 0 else 0.0
    
    def _analyze_topic_continuity(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze topic continuity in the conversation."""
        if len(interactions) < 2:
            return {'score': 0.0, 'current_topic': None, 'topic_changes': 0}
        
        topics = []
        topic_changes = 0
        
        for interaction in interactions[-5:]:  # Last 5 interactions
            metadata = interaction.get('metadata', {})
            task_type = metadata.get('task_type', 'general')
            context_type = metadata.get('context_type', 'general')
            
            current_topic = f"{context_type}_{task_type}"
            
            if topics and topics[-1] != current_topic:
                topic_changes += 1
            
            topics.append(current_topic)
        
        # Calculate continuity score (lower changes = higher continuity)
        continuity_score = max(0.0, 1.0 - (topic_changes / len(topics)))
        
        return {
            'score': continuity_score,
            'current_topic': topics[-1] if topics else None,
            'topic_changes': topic_changes,
            'topic_sequence': topics
        }
    
    def _analyze_model_usage(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze model usage patterns."""
        model_counts = {}
        model_sequence = []
        
        for interaction in interactions[-10:]:
            metadata = interaction.get('metadata', {})
            model_used = metadata.get('model_used', 'unknown')
            model_type = metadata.get('model_type', 'general')
            
            model_counts[model_type] = model_counts.get(model_type, 0) + 1
            model_sequence.append(model_type)
        
        dominant_model = max(model_counts.items(), key=lambda x: x[1])[0] if model_counts else 'general'
        
        return {
            'model_counts': model_counts,
            'dominant_model': dominant_model,
            'model_sequence': model_sequence,
            'model_switches': len(set(model_sequence)) - 1 if model_sequence else 0
        }
    
    def _analyze_temporal_context(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in the conversation."""
        if not interactions:
            return {'session_duration': 0, 'interaction_frequency': 0, 'recent_activity': False}
        
        timestamps = []
        for interaction in interactions:
            metadata = interaction.get('metadata', {})
            timestamp = metadata.get('timestamp', time.time())
            timestamps.append(timestamp)
        
        if len(timestamps) < 2:
            return {'session_duration': 0, 'interaction_frequency': 0, 'recent_activity': True}
        
        session_duration = timestamps[-1] - timestamps[0]
        interaction_frequency = len(timestamps) / max(session_duration / 60, 1)  # interactions per minute
        recent_activity = (time.time() - timestamps[-1]) < 300  # Active in last 5 minutes
        
        return {
            'session_duration': session_duration,
            'interaction_frequency': interaction_frequency,
            'recent_activity': recent_activity,
            'last_interaction_time': timestamps[-1]
        }
    
    def _extract_relevant_context(self, interactions: List[Dict[str, Any]], 
                                current_query: str, context_type: str) -> List[Dict[str, Any]]:
        """Extract most relevant context for the current query."""
        relevant_interactions = []
        
        # Score interactions based on relevance
        for interaction in interactions[-5:]:  # Consider last 5 interactions
            relevance_score = self._calculate_relevance_score(
                interaction, current_query, context_type
            )
            
            if relevance_score > 0.3:  # Threshold for relevance
                relevant_interactions.append({
                    'interaction': interaction,
                    'relevance_score': relevance_score
                })
        
        # Sort by relevance score
        relevant_interactions.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return [item['interaction'] for item in relevant_interactions[:3]]  # Top 3 most relevant
    
    def _calculate_relevance_score(self, interaction: Dict[str, Any], 
                                 current_query: str, context_type: str) -> float:
        """Calculate relevance score for an interaction."""
        score = 0.0
        
        metadata = interaction.get('metadata', {})
        user_input = interaction.get('user_input', '').lower()
        ai_response = interaction.get('ai_response', '').lower()
        current_query_lower = current_query.lower()
        
        # Context type match
        if metadata.get('context_type') == context_type:
            score += 0.3
        
        # Keyword overlap
        query_words = set(current_query_lower.split())
        input_words = set(user_input.split())
        response_words = set(ai_response.split())
        
        input_overlap = len(query_words.intersection(input_words)) / max(len(query_words), 1)
        response_overlap = len(query_words.intersection(response_words)) / max(len(query_words), 1)
        
        score += (input_overlap + response_overlap) * 0.2
        
        # Recency bonus
        timestamp = metadata.get('timestamp', 0)
        recency = max(0, 1 - (time.time() - timestamp) / 3600)  # Decay over 1 hour
        score += recency * 0.1
        
        # Quality bonus
        quality = metadata.get('response_quality', 0.0)
        score += quality * 0.1
        
        return min(score, 1.0)
    
    def _generate_context_summary(self, interactions: List[Dict[str, Any]], 
                                context_type: str) -> str:
        """Generate a summary of the conversation context."""
        if not interactions:
            return "No previous context available."
        
        recent_interactions = interactions[-3:]
        
        if context_type == 'coding':
            languages = set()
            tasks = set()
            
            for interaction in recent_interactions:
                metadata = interaction.get('metadata', {})
                if metadata.get('language_detected'):
                    languages.add(metadata['language_detected'])
                if metadata.get('task_type'):
                    tasks.add(metadata['task_type'])
            
            summary = f"Recent coding discussion"
            if languages:
                summary += f" involving {', '.join(languages)}"
            if tasks:
                summary += f" focused on {', '.join(tasks)} tasks"
        else:
            summary = "Recent general conversation"
        
        return summary
    
    def _generate_recommendations(self, context_type: str, coding_score: float, 
                                topic_continuity: Dict[str, Any]) -> List[str]:
        """Generate recommendations for memory usage."""
        recommendations = []
        
        if context_type == 'coding':
            recommendations.append("Prioritize technical context and code examples")
            if coding_score > 0.8:
                recommendations.append("Deep coding session - maintain technical depth")
            if topic_continuity['score'] > 0.7:
                recommendations.append("Strong topic continuity - build on previous solutions")
        else:
            recommendations.append("Focus on conversational context and general knowledge")
            if topic_continuity['score'] < 0.3:
                recommendations.append("Topic switching detected - provide broader context")
        
        return recommendations
    
    def _create_empty_context(self) -> Dict[str, Any]:
        """Create empty context for new conversations."""
        return {
            'context_type': 'general',
            'coding_score': 0.0,
            'topic_continuity': {'score': 0.0, 'current_topic': None, 'topic_changes': 0},
            'model_usage_pattern': {'model_counts': {}, 'dominant_model': 'general'},
            'temporal_context': {'session_duration': 0, 'interaction_frequency': 0},
            'relevant_context': [],
            'context_summary': "Starting new conversation",
            'recommendations': ["Establish conversation context"]
        }


def test_memory_context_analyzer():
    """Test function for memory context analyzer."""
    analyzer = MemoryContextAnalyzer()
    
    # Sample interactions
    sample_interactions = [
        {
            'user_input': 'How do I write a Python function?',
            'ai_response': 'Here is how to write a Python function: def my_function():',
            'metadata': {
                'is_coding_query': True,
                'context_type': 'coding',
                'task_type': 'implement',
                'language_detected': 'python',
                'timestamp': time.time() - 300
            }
        },
        {
            'user_input': 'Can you debug this code?',
            'ai_response': 'I can help debug your code. Please share the code.',
            'metadata': {
                'is_coding_query': True,
                'context_type': 'coding',
                'task_type': 'debug',
                'language_detected': 'python',
                'timestamp': time.time() - 150
            }
        }
    ]
    
    current_query = "How do I handle exceptions in Python?"
    
    context = analyzer.analyze_conversation_context(sample_interactions, current_query)
    
    print("Memory Context Analysis:")
    print(f"Context Type: {context['context_type']}")
    print(f"Coding Score: {context['coding_score']:.2f}")
    print(f"Topic Continuity: {context['topic_continuity']['score']:.2f}")
    print(f"Context Summary: {context['context_summary']}")
    print(f"Recommendations: {context['recommendations']}")


if __name__ == "__main__":
    test_memory_context_analyzer()
