#!/usr/bin/env python3
"""
Enhanced Intelligent Knowledge System for Personal AI Assistant

A comprehensive system that combines intelligent knowledge acquisition with
persistent user profile management and complete session storage capabilities.

Features:
- Automatic detection of knowledge gaps
- Web search integration with result evaluation
- Information storage and retrieval
- Source attribution and transparency
- Learning from search results for future queries
- Single-user profile management with persistent preferences
- Complete conversation session storage and history
- Adaptive learning from historical interactions
- Privacy-focused local storage with encryption
- Database versioning and migration support
"""

import re
import json
import logging
import requests
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import hashlib
import uuid

# Import enhanced components
from enhanced_database_manager import EnhancedDatabaseManager
from user_profile_manager import UserProfileManager
from session_manager import SessionManager
from privacy_security_manager import PrivacySecurityManager
from Memory import MemorySystem

logger = logging.getLogger(__name__)


class KnowledgeGapDetector:
    """Detects when a query requires internet search for current or unknown information."""
    
    def __init__(self):
        """Initialize the knowledge gap detector."""
        self.current_events_keywords = [
            'latest', 'recent', 'current', 'today', 'yesterday', 'this week', 'this month',
            'breaking', 'news', 'update', 'announcement', 'release', 'launched', 'published'
        ]
        
        self.time_sensitive_topics = [
            'stock price', 'weather', 'news', 'events', 'schedule', 'status',
            'availability', 'price', 'cost', 'rate', 'version', 'update'
        ]
        
        self.technical_domains = [
            'api', 'documentation', 'tutorial', 'guide', 'specification',
            'framework', 'library', 'tool', 'software', 'technology'
        ]
        
        self.uncertainty_indicators = [
            "i don't know", "not sure", "unclear", "uncertain", "unknown",
            "no information", "insufficient data", "need to check"
        ]
    
    def should_search_web(self, query: str, context: Dict[str, Any] = None) -> Tuple[bool, str, float]:
        """
        Determine if a query requires web search.
        
        Args:
            query: User's query
            context: Additional context information
            
        Returns:
            Tuple of (should_search, reason, confidence)
        """
        query_lower = query.lower()
        confidence = 0.0
        reasons = []
        
        # Check for current events indicators
        current_events_score = sum(1 for keyword in self.current_events_keywords 
                                 if keyword in query_lower)
        if current_events_score > 0:
            confidence += 0.3 * current_events_score
            reasons.append("current_events")
        
        # Check for time-sensitive topics
        time_sensitive_score = sum(1 for topic in self.time_sensitive_topics 
                                 if topic in query_lower)
        if time_sensitive_score > 0:
            confidence += 0.25 * time_sensitive_score
            reasons.append("time_sensitive")
        
        # Check for technical documentation requests
        tech_score = sum(1 for domain in self.technical_domains 
                        if domain in query_lower)
        if tech_score > 0:
            confidence += 0.2 * tech_score
            reasons.append("technical_documentation")
        
        # Check for specific date/time references
        date_patterns = [
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',
            r'\b(today|yesterday|tomorrow)\b',
            r'\b(this|last|next)\s+(week|month|year)\b'
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, query_lower):
                confidence += 0.4
                reasons.append("specific_date_reference")
                break
        
        # Check for "what is" or "who is" questions about potentially new topics
        if re.search(r'\b(what|who|when|where|how)\s+(is|are|was|were)\b', query_lower):
            confidence += 0.2
            reasons.append("factual_query")
        
        # Check for version/release information
        if re.search(r'\b(version|release|v\d+\.\d+|update|patch)\b', query_lower):
            confidence += 0.3
            reasons.append("version_information")
        
        # Check for company/product specific information
        if re.search(r'\b(company|corporation|startup|product|service|platform)\b', query_lower):
            confidence += 0.15
            reasons.append("company_product_info")
        
        # Normalize confidence to 0-1 range
        confidence = min(confidence, 1.0)
        
        # Decision threshold
        should_search = confidence >= 0.3
        
        reason = ", ".join(reasons) if reasons else "no_indicators"
        
        return should_search, reason, confidence


class WebSearchIntegrator:
    """Integrates web search functionality with intelligent result evaluation."""
    
    def __init__(self, search_config: Dict[str, Any] = None):
        """Initialize the web search integrator."""
        self.config = search_config or {}
        self.search_history = []
        self.result_cache = {}
        self.cache_duration = timedelta(hours=1)  # Cache results for 1 hour
    
    def search_and_evaluate(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Perform web search and evaluate results for quality and relevance.
        
        Args:
            query: Search query
            num_results: Number of results to retrieve
            
        Returns:
            Dictionary containing search results and evaluation
        """
        try:
            # Check cache first
            cache_key = hashlib.md5(query.encode()).hexdigest()
            if cache_key in self.result_cache:
                cached_result = self.result_cache[cache_key]
                if datetime.now() - cached_result['timestamp'] < self.cache_duration:
                    logger.info(f"Using cached search results for: {query}")
                    return cached_result['data']
            
            # Perform web search (this would integrate with the existing web search tool)
            search_results = self._perform_web_search(query, num_results)
            
            if not search_results:
                return {'success': False, 'error': 'No search results found'}
            
            # Evaluate and rank results
            evaluated_results = self._evaluate_search_results(search_results, query)
            
            # Select best sources
            best_sources = self._select_best_sources(evaluated_results)
            
            # Extract and synthesize information
            synthesized_info = self._synthesize_information(best_sources, query)
            
            result = {
                'success': True,
                'query': query,
                'search_results': search_results,
                'evaluated_results': evaluated_results,
                'best_sources': best_sources,
                'synthesized_info': synthesized_info,
                'timestamp': datetime.now(),
                'source_count': len(best_sources)
            }
            
            # Cache the result
            self.result_cache[cache_key] = {
                'data': result,
                'timestamp': datetime.now()
            }
            
            # Store in search history
            self.search_history.append({
                'query': query,
                'timestamp': datetime.now(),
                'result_count': len(search_results),
                'success': True
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {e}")
            return {'success': False, 'error': str(e)}
    
    def _perform_web_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """
        Perform the actual web search using available search tools.
        This integrates with the existing web search functionality.
        """
        try:
            # Try to import and use the web search functionality
            # This would be called from the main system that has access to web search tools
            if hasattr(self, '_web_search_function'):
                return self._web_search_function(query, num_results)

            # Fallback: Try to use requests for basic web search
            # This is a simplified implementation for demonstration
            search_results = []

            # For now, return a structured placeholder that indicates web search capability
            # In production, this would integrate with Google Custom Search API or similar
            return [
                {
                    'title': f'Web search result for: {query}',
                    'url': f'https://search-results.example.com/q={quote_plus(query)}',
                    'snippet': f'Real-time information about {query} would be retrieved from web search.',
                    'source': 'web-search-api'
                }
            ]

        except Exception as e:
            logger.error(f"Web search execution failed: {e}")
            return []

    def set_web_search_function(self, search_function):
        """Set the web search function to be used by this integrator."""
        self._web_search_function = search_function
    
    def _evaluate_search_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Evaluate search results for quality, relevance, and authority."""
        evaluated = []
        
        for result in results:
            score = self._calculate_result_score(result, query)
            authority_score = self._calculate_authority_score(result)
            relevance_score = self._calculate_relevance_score(result, query)
            
            evaluated_result = {
                **result,
                'quality_score': score,
                'authority_score': authority_score,
                'relevance_score': relevance_score,
                'overall_score': (score + authority_score + relevance_score) / 3
            }
            evaluated.append(evaluated_result)
        
        # Sort by overall score
        evaluated.sort(key=lambda x: x['overall_score'], reverse=True)
        return evaluated
    
    def _calculate_result_score(self, result: Dict[str, Any], query: str) -> float:
        """Calculate quality score for a search result."""
        score = 0.5  # Base score
        
        # Check title relevance
        title = result.get('title', '').lower()
        query_words = query.lower().split()
        title_matches = sum(1 for word in query_words if word in title)
        score += (title_matches / len(query_words)) * 0.3
        
        # Check snippet relevance
        snippet = result.get('snippet', '').lower()
        snippet_matches = sum(1 for word in query_words if word in snippet)
        score += (snippet_matches / len(query_words)) * 0.2
        
        return min(score, 1.0)
    
    def _calculate_authority_score(self, result: Dict[str, Any]) -> float:
        """Calculate authority score based on source domain."""
        url = result.get('url', '').lower()
        
        # High authority domains
        high_authority = [
            'wikipedia.org', 'github.com', 'stackoverflow.com', 'docs.python.org',
            'developer.mozilla.org', 'w3.org', 'ietf.org', 'ieee.org',
            'acm.org', 'arxiv.org', 'nature.com', 'science.org'
        ]
        
        # Medium authority domains
        medium_authority = [
            'medium.com', 'dev.to', 'hackernoon.com', 'towardsdatascience.com',
            'techcrunch.com', 'wired.com', 'arstechnica.com'
        ]
        
        for domain in high_authority:
            if domain in url:
                return 0.9
        
        for domain in medium_authority:
            if domain in url:
                return 0.7
        
        # Check for official documentation patterns
        if any(pattern in url for pattern in ['docs.', 'documentation', 'api.', 'developer.']):
            return 0.8
        
        return 0.5  # Default score
    
    def _calculate_relevance_score(self, result: Dict[str, Any], query: str) -> float:
        """Calculate relevance score based on content match."""
        content = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
        query_lower = query.lower()
        
        # Exact phrase match
        if query_lower in content:
            return 0.9
        
        # Word overlap
        query_words = set(query_lower.split())
        content_words = set(content.split())
        overlap = len(query_words.intersection(content_words))
        
        if len(query_words) > 0:
            return min(overlap / len(query_words), 1.0)
        
        return 0.5
    
    def _select_best_sources(self, evaluated_results: List[Dict[str, Any]], max_sources: int = 3) -> List[Dict[str, Any]]:
        """Select the best sources based on evaluation scores."""
        # Filter results with minimum quality threshold
        quality_threshold = 0.6
        qualified_results = [r for r in evaluated_results if r['overall_score'] >= quality_threshold]
        
        # If not enough qualified results, lower the threshold
        if len(qualified_results) < 2:
            quality_threshold = 0.4
            qualified_results = [r for r in evaluated_results if r['overall_score'] >= quality_threshold]
        
        # Return top results up to max_sources
        return qualified_results[:max_sources]
    
    def _synthesize_information(self, sources: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Synthesize information from multiple sources into a coherent response."""
        if not sources:
            return {'content': 'No reliable sources found for this query.', 'confidence': 0.0}
        
        # Extract key information from sources
        synthesized_content = []
        source_references = []
        
        for i, source in enumerate(sources, 1):
            title = source.get('title', 'Unknown Title')
            snippet = source.get('snippet', 'No description available')
            url = source.get('url', '')
            
            synthesized_content.append(f"**Source {i}**: {title}")
            synthesized_content.append(f"{snippet}")
            synthesized_content.append("")
            
            source_references.append({
                'index': i,
                'title': title,
                'url': url,
                'authority_score': source.get('authority_score', 0.5),
                'relevance_score': source.get('relevance_score', 0.5)
            })
        
        # Calculate overall confidence based on source quality
        avg_authority = sum(s.get('authority_score', 0.5) for s in sources) / len(sources)
        avg_relevance = sum(s.get('relevance_score', 0.5) for s in sources) / len(sources)
        confidence = (avg_authority + avg_relevance) / 2
        
        return {
            'content': '\n'.join(synthesized_content),
            'confidence': confidence,
            'source_count': len(sources),
            'sources': source_references,
            'synthesis_timestamp': datetime.now().isoformat()
        }


class EnhancedIntelligentKnowledgeSystem:
    """
    Enhanced intelligent knowledge system with comprehensive user profile management
    and complete session storage capabilities for personal AI assistant.
    """

    def __init__(self, config: Dict[str, Any] = None,
                 db_path: str = "Memory/Database/enhanced_knowledge.db",
                 profile_name: str = "default"):
        """
        Initialize the enhanced intelligent knowledge system.

        Args:
            config: System configuration
            db_path: Path to enhanced database
            profile_name: User profile name
        """
        self.config = config or {}
        self.profile_name = profile_name
        self._lock = threading.Lock()

        # Initialize core components
        self.gap_detector = KnowledgeGapDetector()
        self.web_integrator = WebSearchIntegrator(config.get('search_config', {}))

        # Initialize enhanced database and components
        self.enhanced_db = EnhancedDatabaseManager(db_path)
        self.user_profile_manager = self.enhanced_db.user_profile_manager
        self.session_manager = self.enhanced_db.session_manager
        self.privacy_manager = PrivacySecurityManager(
            self.enhanced_db.db_connection,
            self.enhanced_db.encryption_manager
        )

        # Initialize traditional memory system for backward compatibility
        self.memory_system = MemorySystem(db_path.replace('enhanced_knowledge.db', 'memory.db'))

        # Session management
        self.current_session_id = None
        self.session_context = {}

        # Legacy compatibility
        self.knowledge_cache = {}
        self.learning_history = []

        logger.info(f"Enhanced Intelligent Knowledge System initialized for profile: {profile_name}")

    def start_session(self, session_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new conversation session.

        Args:
            session_context: Optional context for the session

        Returns:
            Session ID
        """
        with self._lock:
            try:
                self.current_session_id = self.session_manager.create_session(
                    self.profile_name, session_context
                )
                self.session_context = session_context or {}

                logger.info(f"Started new session: {self.current_session_id}")
                return self.current_session_id

            except Exception as e:
                logger.error(f"Error starting session: {e}")
                raise

    def end_session(self, session_summary: Optional[str] = None,
                   user_satisfaction_score: Optional[float] = None) -> bool:
        """
        End the current conversation session.

        Args:
            session_summary: Optional session summary
            user_satisfaction_score: Optional satisfaction rating (0.0-1.0)

        Returns:
            True if successful, False otherwise
        """
        if not self.current_session_id:
            logger.warning("No active session to end")
            return False

        try:
            success = self.session_manager.end_session(
                self.current_session_id, session_summary, user_satisfaction_score
            )

            if success:
                self.current_session_id = None
                self.session_context = {}
                logger.info("Session ended successfully")

            return success

        except Exception as e:
            logger.error(f"Error ending session: {e}")
            return False

    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query with enhanced session tracking and user profile integration.

        Args:
            query: User's query
            context: Additional context information

        Returns:
            Dictionary containing processing results and any web-sourced information
        """
        try:
            # Ensure we have an active session
            if not self.current_session_id:
                self.start_session()

            # Get user preferences for response customization
            user_profile = self.user_profile_manager.get_profile(self.profile_name)
            response_preferences = self._extract_response_preferences(user_profile)

            # Enhanced context with user profile and session history
            enhanced_context = {
                **(context or {}),
                'session_id': self.current_session_id,
                'user_preferences': response_preferences,
                'profile_name': self.profile_name
            }

            # Add user input message to session
            user_message_id = self.session_manager.add_message(
                self.current_session_id, 'user_input', query, metadata=enhanced_context
            )

            # Detect if web search is needed
            should_search, reason, confidence = self.gap_detector.should_search_web(query, enhanced_context)

            result = {
                'query': query,
                'session_id': self.current_session_id,
                'user_message_id': user_message_id,
                'should_search': should_search,
                'search_reason': reason,
                'search_confidence': confidence,
                'user_preferences': response_preferences,
                'timestamp': datetime.now().isoformat()
            }
            
            if should_search:
                logger.info(f"Triggering web search for query: {query} (reason: {reason}, confidence: {confidence:.2f})")

                # Perform web search and evaluation
                search_result = self.web_integrator.search_and_evaluate(query)

                if search_result['success']:
                    result.update({
                        'web_search_performed': True,
                        'search_results': search_result,
                        'synthesized_info': search_result['synthesized_info'],
                        'sources': search_result['best_sources']
                    })

                    # Store learned information in both systems
                    self._store_learned_information(query, search_result)
                    self._store_enhanced_knowledge(query, search_result, enhanced_context)

                else:
                    result.update({
                        'web_search_performed': True,
                        'search_error': search_result.get('error', 'Unknown error'),
                        'fallback_to_existing_knowledge': True
                    })
            else:
                result.update({
                    'web_search_performed': False,
                    'use_existing_knowledge': True
                })

            # Generate AI response based on results and user preferences
            ai_response = self._generate_personalized_response(result, response_preferences)
            result['ai_response'] = ai_response

            # Add AI response message to session
            ai_message_id = self.session_manager.add_message(
                self.current_session_id, 'ai_response', ai_response,
                parent_message_id=user_message_id,
                metadata={
                    'search_performed': should_search,
                    'sources_count': len(result.get('sources', [])),
                    'response_preferences': response_preferences
                }
            )
            result['ai_message_id'] = ai_message_id

            # Store interaction in traditional memory system for compatibility
            self.memory_system.add_interaction(query, ai_response)

            return result
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return {
                'query': query,
                'session_id': self.current_session_id,
                'error': str(e),
                'should_search': False,
                'use_existing_knowledge': True
            }

    def add_user_feedback(self, feedback_text: str, rating: Optional[int] = None,
                         message_id: Optional[str] = None) -> bool:
        """
        Add user feedback for the last interaction or specific message.

        Args:
            feedback_text: User feedback text
            rating: Optional rating (1-5)
            message_id: Optional specific message ID

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.current_session_id:
                logger.warning("No active session for feedback")
                return False

            # If no message_id provided, get the last AI response
            if not message_id:
                messages = self.session_manager.get_session_messages(
                    self.current_session_id, message_types=['ai_response']
                )
                if messages:
                    message_id = messages[-1]['message_id']
                else:
                    logger.warning("No AI response found for feedback")
                    return False

            # Update message with feedback
            conn = self.enhanced_db.db_connection.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE messages
                SET user_rating = ?, user_feedback = ?, feedback_timestamp = ?
                WHERE message_id = ?
            """, (rating, feedback_text, datetime.now().isoformat(), message_id))

            conn.commit()

            # Process feedback for learning
            feedback_data = {
                'text': feedback_text,
                'rating': rating,
                'message_id': message_id,
                'session_id': self.current_session_id,
                'timestamp': datetime.now().isoformat()
            }

            # Analyze feedback and adapt preferences
            analyzed_feedback = self._analyze_feedback(feedback_data)
            adaptations = self.user_profile_manager.adapt_preferences_from_feedback(
                analyzed_feedback, self.profile_name
            )

            logger.info(f"Processed feedback with {len(adaptations)} adaptations")
            return True

        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
            return False

    def _extract_response_preferences(self, user_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract response preferences from user profile."""
        if not user_profile:
            return {
                'response_style': 'balanced',
                'verbosity_level': 'moderate',
                'technical_level': 'intermediate',
                'explanation_preference': 'balanced'
            }

        return {
            'response_style': user_profile.get('response_style', 'balanced'),
            'verbosity_level': user_profile.get('verbosity_level', 'moderate'),
            'technical_level': user_profile.get('technical_level', 'intermediate'),
            'explanation_preference': user_profile.get('explanation_preference', 'balanced')
        }

    def _generate_personalized_response(self, result: Dict[str, Any],
                                      preferences: Dict[str, Any]) -> str:
        """
        Generate personalized AI response based on user preferences.

        This is a simplified implementation. In a real system, this would
        integrate with the actual AI response generation pipeline.
        """
        base_response = "I'd be happy to help with your query."

        if result.get('web_search_performed') and result.get('synthesized_info'):
            content = result['synthesized_info'].get('content', '')
            sources = result.get('sources', [])

            # Customize response based on preferences
            if preferences['verbosity_level'] == 'brief':
                # Provide concise response
                base_response = f"Based on current information: {content[:200]}..."
            elif preferences['verbosity_level'] == 'detailed':
                # Provide comprehensive response
                base_response = f"Here's detailed information about your query:\n\n{content}"
                if sources:
                    base_response += f"\n\nSources: {', '.join([s.get('title', 'Unknown') for s in sources[:3]])}"
            else:
                # Moderate response
                base_response = f"Based on my research: {content[:400]}..."

        return base_response

    def _store_enhanced_knowledge(self, query: str, search_result: Dict[str, Any],
                                context: Dict[str, Any]) -> None:
        """Store knowledge in enhanced database with session context."""
        try:
            if not search_result.get('success'):
                return

            synthesized_info = search_result.get('synthesized_info', {})
            sources = search_result.get('best_sources', [])

            # Store in knowledge base (using existing database memory)
            if hasattr(self.enhanced_db, 'knowledge'):
                self.enhanced_db.knowledge.store_knowledge(
                    topic=context.get('topic', 'general'),
                    content=synthesized_info.get('content', ''),
                    source='web_search',
                    confidence=synthesized_info.get('confidence', 0.5),
                    tags=[s.get('title', '') for s in sources[:5]]
                )

            logger.debug(f"Stored enhanced knowledge for query: {query}")

        except Exception as e:
            logger.error(f"Error storing enhanced knowledge: {e}")

    def _analyze_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user feedback to extract actionable insights.

        This is a simplified implementation. In practice, you might use
        NLP libraries for sentiment analysis and intent detection.
        """
        feedback_text = feedback_data.get('text', '').lower()
        rating = feedback_data.get('rating')

        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'helpful', 'perfect', 'thanks', 'thank you']
        negative_words = ['bad', 'wrong', 'incorrect', 'unhelpful', 'confusing', 'too']

        positive_count = sum(1 for word in positive_words if word in feedback_text)
        negative_count = sum(1 for word in negative_words if word in feedback_text)

        # Calculate sentiment score
        if rating:
            sentiment = (rating - 3) / 2  # Convert 1-5 rating to -1 to 1 scale
        else:
            sentiment = (positive_count - negative_count) / max(positive_count + negative_count, 1)

        # Detect feedback type
        feedback_type = 'general'
        if any(word in feedback_text for word in ['style', 'tone', 'formal', 'casual']):
            feedback_type = 'style_feedback'
        elif any(word in feedback_text for word in ['detail', 'brief', 'long', 'short']):
            feedback_type = 'verbosity_feedback'
        elif any(word in feedback_text for word in ['technical', 'simple', 'complex']):
            feedback_type = 'technical_level_feedback'

        return {
            'sentiment': sentiment,
            'type': feedback_type,
            'text': feedback_data.get('text', ''),
            'rating': rating,
            'timestamp': feedback_data.get('timestamp')
        }

    # Enhanced session and profile management methods

    def get_session_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get conversation history for current or recent sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session data with metadata
        """
        try:
            return self.session_manager.search_sessions("", limit=limit)
        except Exception as e:
            logger.error(f"Error getting session history: {e}")
            return []

    def search_conversation_history(self, query: str,
                                  date_range: Optional[Tuple[datetime, datetime]] = None,
                                  topic_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search through conversation history.

        Args:
            query: Search query
            date_range: Optional date range filter
            topic_filter: Optional topic filter

        Returns:
            List of matching sessions and messages
        """
        try:
            return self.session_manager.search_sessions(query, date_range, topic_filter)
        except Exception as e:
            logger.error(f"Error searching conversation history: {e}")
            return []

    def update_user_preference(self, preference_key: str, value: Any) -> bool:
        """
        Update a user preference.

        Args:
            preference_key: Preference key to update
            value: New value

        Returns:
            True if successful, False otherwise
        """
        try:
            return self.user_profile_manager.update_preference(
                preference_key, value, self.profile_name
            )
        except Exception as e:
            logger.error(f"Error updating preference {preference_key}: {e}")
            return False

    def get_user_preference(self, preference_key: str, default: Any = None) -> Any:
        """
        Get a user preference value.

        Args:
            preference_key: Preference key
            default: Default value if not found

        Returns:
            Preference value or default
        """
        try:
            return self.user_profile_manager.get_preference(
                preference_key, default, self.profile_name
            )
        except Exception as e:
            logger.error(f"Error getting preference {preference_key}: {e}")
            return default

    def export_user_data(self, output_path: str, format_type: str = "json",
                        include_sensitive: bool = False) -> bool:
        """
        Export all user data for backup or portability.

        Args:
            output_path: Output file path
            format_type: Export format ('json' or 'csv')
            include_sensitive: Whether to include sensitive data

        Returns:
            True if successful, False otherwise
        """
        try:
            if format_type.lower() == "json":
                # Export comprehensive data including profile and sessions
                export_data = {
                    'user_profile': self.user_profile_manager.export_profile(self.profile_name),
                    'sessions': self.get_session_history(1000),  # Last 1000 sessions
                    'system_info': self.enhanced_db.get_system_status(),
                    'export_metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'profile_name': self.profile_name,
                        'include_sensitive': include_sensitive
                    }
                }

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

                return True
            else:
                return self.enhanced_db.export_all_data(output_path, format_type)

        except Exception as e:
            logger.error(f"Error exporting user data: {e}")
            return False

    def import_user_data(self, import_path: str) -> bool:
        """
        Import user data from backup.

        Args:
            import_path: Path to import file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            # Import user profile
            if 'user_profile' in import_data:
                success = self.user_profile_manager.import_profile(
                    import_data['user_profile'], self.profile_name
                )
                if not success:
                    logger.warning("Failed to import user profile")

            logger.info(f"Imported user data from {import_path}")
            return True

        except Exception as e:
            logger.error(f"Error importing user data: {e}")
            return False

    def create_backup(self) -> Optional[str]:
        """
        Create automatic backup of all data.

        Returns:
            Backup ID if successful, None otherwise
        """
        try:
            return self.enhanced_db.create_automatic_backup()
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status including enhanced features.

        Returns:
            System status dictionary
        """
        try:
            base_status = self.enhanced_db.get_system_status()

            # Add session information
            base_status.update({
                'current_session_id': self.current_session_id,
                'profile_name': self.profile_name,
                'active_session': self.current_session_id is not None,
                'memory_system_status': self.memory_system.get_system_status()
            })

            return base_status

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

    # Privacy and Data Management API

    def get_privacy_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive privacy dashboard for the user.

        Returns:
            Privacy dashboard with all data management options
        """
        try:
            return self.privacy_manager.get_privacy_dashboard(self.profile_name)
        except Exception as e:
            logger.error(f"Error getting privacy dashboard: {e}")
            return {'error': str(e)}

    def view_all_personal_data(self) -> Dict[str, Any]:
        """
        View all personal data stored in the system.

        Returns:
            Comprehensive view of all personal data
        """
        try:
            return self.privacy_manager.data_privacy.view_personal_data(self.profile_name)
        except Exception as e:
            logger.error(f"Error viewing personal data: {e}")
            return {'error': str(e)}

    def delete_session_data(self, session_id: str, secure_delete: bool = True) -> bool:
        """
        Delete specific session data.

        Args:
            session_id: Session ID to delete
            secure_delete: Whether to perform secure deletion

        Returns:
            True if successful, False otherwise
        """
        try:
            return self.privacy_manager.data_privacy.delete_session_data(session_id, secure_delete)
        except Exception as e:
            logger.error(f"Error deleting session data: {e}")
            return False

    def delete_date_range_data(self, start_date: datetime, end_date: datetime,
                              secure_delete: bool = True) -> Dict[str, int]:
        """
        Delete all data within a specific date range.

        Args:
            start_date: Start date for deletion
            end_date: End date for deletion
            secure_delete: Whether to perform secure deletion

        Returns:
            Dictionary with deletion counts
        """
        try:
            return self.privacy_manager.data_privacy.delete_date_range_data(
                start_date, end_date, secure_delete
            )
        except Exception as e:
            logger.error(f"Error deleting date range data: {e}")
            return {'error': str(e)}

    def anonymize_profile(self) -> bool:
        """
        Anonymize the current user profile.

        Returns:
            True if successful, False otherwise
        """
        try:
            return self.privacy_manager.data_privacy.anonymize_profile_data(self.profile_name)
        except Exception as e:
            logger.error(f"Error anonymizing profile: {e}")
            return False

    def export_gdpr_data(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Export all personal data in GDPR-compliant format.

        Args:
            output_path: Optional output path

        Returns:
            Path to exported file or None if failed
        """
        try:
            return self.privacy_manager.data_privacy.export_gdpr_data(self.profile_name, output_path)
        except Exception as e:
            logger.error(f"Error exporting GDPR data: {e}")
            return None

    def get_data_retention_info(self) -> Dict[str, Any]:
        """
        Get information about data retention policies.

        Returns:
            Data retention information
        """
        try:
            return self.privacy_manager.data_privacy.get_data_retention_info()
        except Exception as e:
            logger.error(f"Error getting retention info: {e}")
            return {'error': str(e)}

    # Advanced Analytics and Pattern Analysis API

    def analyze_conversation_patterns(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze conversation patterns over a specified period.

        Args:
            days_back: Number of days to analyze

        Returns:
            Conversation pattern analysis
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            sessions = self.search_conversation_history("", (start_date, end_date))

            if not sessions:
                return {'message': 'No conversation data found for the specified period'}

            # Analyze patterns
            topics = {}
            session_lengths = []
            daily_activity = {}
            satisfaction_scores = []

            for session in sessions:
                # Topic analysis
                topic_categories = session.get('topic_categories', [])
                if isinstance(topic_categories, str):
                    topic_categories = json.loads(topic_categories)

                for topic in topic_categories:
                    topics[topic] = topics.get(topic, 0) + 1

                # Session length analysis
                if session.get('duration_seconds'):
                    session_lengths.append(session['duration_seconds'])

                # Daily activity analysis
                if session.get('started_at'):
                    date = session['started_at'][:10]  # Extract date part
                    daily_activity[date] = daily_activity.get(date, 0) + 1

                # Satisfaction analysis
                if session.get('user_satisfaction_score'):
                    satisfaction_scores.append(session['user_satisfaction_score'])

            return {
                'analysis_period': f"{start_date.isoformat()} to {end_date.isoformat()}",
                'total_sessions': len(sessions),
                'top_topics': sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10],
                'session_statistics': {
                    'average_length_seconds': sum(session_lengths) / len(session_lengths) if session_lengths else 0,
                    'total_conversation_time': sum(session_lengths),
                    'shortest_session': min(session_lengths) if session_lengths else 0,
                    'longest_session': max(session_lengths) if session_lengths else 0
                },
                'activity_patterns': {
                    'most_active_day': max(daily_activity.items(), key=lambda x: x[1]) if daily_activity else None,
                    'daily_activity': daily_activity
                },
                'satisfaction_metrics': {
                    'average_satisfaction': sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0,
                    'satisfaction_trend': satisfaction_scores[-10:] if satisfaction_scores else []
                },
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing conversation patterns: {e}")
            return {'error': str(e)}

    def get_learning_insights(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get learning insights derived from conversation history.

        Args:
            limit: Maximum number of insights to return

        Returns:
            List of learning insights
        """
        try:
            conn = self.enhanced_db.db_connection.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM learning_insights
                ORDER BY confidence_level DESC, created_at DESC
                LIMIT ?
            """, (limit,))

            insights = []
            for row in cursor.fetchall():
                insight = dict(row)

                # Parse JSON fields
                if insight.get('insight_data'):
                    insight['insight_data'] = json.loads(insight['insight_data'])
                if insight.get('source_sessions'):
                    insight['source_sessions'] = json.loads(insight['source_sessions'])
                if insight.get('source_messages'):
                    insight['source_messages'] = json.loads(insight['source_messages'])

                insights.append(insight)

            return insights

        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return []

    def generate_personalized_recommendations(self) -> Dict[str, Any]:
        """
        Generate personalized recommendations based on user patterns.

        Returns:
            Personalized recommendations
        """
        try:
            # Analyze recent patterns
            patterns = self.analyze_conversation_patterns(30)
            profile = self.user_profile_manager.get_profile(self.profile_name)

            recommendations = {
                'preference_suggestions': [],
                'usage_optimization': [],
                'feature_recommendations': [],
                'generated_at': datetime.now().isoformat()
            }

            # Generate preference suggestions based on patterns
            if patterns.get('top_topics'):
                top_topic = patterns['top_topics'][0][0]
                recommendations['preference_suggestions'].append({
                    'type': 'topic_specialization',
                    'suggestion': f"Consider setting technical level to 'advanced' for {top_topic} discussions",
                    'reason': f"You frequently discuss {top_topic}"
                })

            # Usage optimization suggestions
            session_stats = patterns.get('session_statistics', {})
            avg_length = session_stats.get('average_length_seconds', 0)

            if avg_length > 1800:  # 30 minutes
                recommendations['usage_optimization'].append({
                    'type': 'session_length',
                    'suggestion': "Consider breaking longer conversations into focused sessions",
                    'reason': f"Your average session length is {avg_length/60:.1f} minutes"
                })

            # Feature recommendations based on usage
            if profile and profile.get('verbosity_level') == 'moderate':
                recommendations['feature_recommendations'].append({
                    'type': 'verbosity_adjustment',
                    'suggestion': "Try 'detailed' verbosity for complex topics",
                    'reason': "Based on your question patterns, you might benefit from more detailed responses"
                })

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {'error': str(e)}

    def _store_learned_information(self, query: str, search_result: Dict[str, Any]):
        """Store newly learned information for future reference."""
        try:
            learned_info = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'sources': search_result.get('best_sources', []),
                'synthesized_content': search_result.get('synthesized_info', {}),
                'confidence': search_result.get('synthesized_info', {}).get('confidence', 0.0)
            }
            
            # Store in knowledge cache
            cache_key = hashlib.md5(query.lower().encode()).hexdigest()
            self.knowledge_cache[cache_key] = learned_info
            
            # Add to learning history
            self.learning_history.append({
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'source_count': len(search_result.get('best_sources', [])),
                'confidence': learned_info['confidence']
            })
            
            logger.info(f"Stored learned information for query: {query}")
            
        except Exception as e:
            logger.error(f"Failed to store learned information: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning and knowledge acquisition process."""
        return {
            'total_queries_processed': len(self.learning_history),
            'knowledge_cache_size': len(self.knowledge_cache),
            'recent_learning': self.learning_history[-10:] if self.learning_history else [],
            'average_confidence': sum(entry.get('confidence', 0) for entry in self.learning_history) / len(self.learning_history) if self.learning_history else 0,
            'web_searches_performed': len([entry for entry in self.learning_history if entry.get('source_count', 0) > 0])
        }


# Backward compatibility - keep original class
class IntelligentKnowledgeSystem(EnhancedIntelligentKnowledgeSystem):
    """
    Backward compatible version of the intelligent knowledge system.

    This class maintains the original API while providing access to enhanced features.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with default enhanced features."""
        super().__init__(config, profile_name="default")

        # Start a default session for backward compatibility
        self.start_session({'compatibility_mode': True})


# Convenience functions for easy integration

def create_personal_ai_assistant(profile_name: str = "default",
                                config: Optional[Dict[str, Any]] = None) -> EnhancedIntelligentKnowledgeSystem:
    """
    Create a personal AI assistant with enhanced capabilities.

    Args:
        profile_name: User profile name
        config: Optional configuration

    Returns:
        Enhanced intelligent knowledge system instance
    """
    return EnhancedIntelligentKnowledgeSystem(config, profile_name=profile_name)


def migrate_legacy_system(legacy_db_path: str, enhanced_db_path: str) -> bool:
    """
    Migrate data from legacy system to enhanced system.

    Args:
        legacy_db_path: Path to legacy database
        enhanced_db_path: Path to enhanced database

    Returns:
        True if successful, False otherwise
    """
    try:
        # This would implement migration logic
        # For now, it's a placeholder
        logger.info(f"Migration from {legacy_db_path} to {enhanced_db_path} would be implemented here")
        return True
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False
