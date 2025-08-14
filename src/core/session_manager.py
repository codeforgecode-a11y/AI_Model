#!/usr/bin/env python3
"""
Session Management System

Handles complete conversation session storage with full message history,
session metadata, conversation threading, and efficient retrieval capabilities.
"""

import json
import logging
import sqlite3
import threading
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages comprehensive session storage with full conversation history and metadata.
    """
    
    def __init__(self, db_connection, user_profile_manager=None):
        """
        Initialize session manager.
        
        Args:
            db_connection: Database connection instance
            user_profile_manager: Optional user profile manager for integration
        """
        self.db = db_connection
        self.profile_manager = user_profile_manager
        self._lock = threading.Lock()
        self._active_sessions = {}  # session_id -> session_data
        self._message_cache = defaultdict(list)  # session_id -> messages
    
    def create_session(self, profile_name: str = "default", 
                      session_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            profile_name: User profile name
            session_context: Optional context information
            
        Returns:
            Session ID
        """
        with self._lock:
            try:
                session_id = str(uuid.uuid4())
                
                # Get profile ID
                profile_id = None
                if self.profile_manager:
                    profile = self.profile_manager.get_profile(profile_name)
                    if profile:
                        profile_id = profile['id']
                
                # Prepare session data
                session_data = {
                    'session_id': session_id,
                    'profile_id': profile_id,
                    'started_at': datetime.now().isoformat(),
                    'interaction_count': 0,
                    'session_context': json.dumps(session_context or {}),
                    'device_info': json.dumps(self._get_device_info())
                }
                
                # Store in database
                conn = self.db.get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO sessions (
                        session_id, profile_id, started_at, interaction_count,
                        session_context, device_info
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id, profile_id, session_data['started_at'],
                    0, session_data['session_context'], session_data['device_info']
                ))
                
                conn.commit()
                
                # Cache active session
                self._active_sessions[session_id] = session_data
                
                logger.info(f"Created new session {session_id} for profile {profile_name}")
                return session_id
                
            except Exception as e:
                logger.error(f"Error creating session: {e}")
                raise
    
    def add_message(self, session_id: str, message_type: str, content: str,
                   parent_message_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a message to a session.
        
        Args:
            session_id: Session identifier
            message_type: Type of message (user_input, ai_response, system_message)
            content: Message content
            parent_message_id: Optional parent message for threading
            metadata: Optional message metadata
            
        Returns:
            Message ID
        """
        with self._lock:
            try:
                message_id = str(uuid.uuid4())
                timestamp = datetime.now().isoformat()
                
                # Calculate content hash for deduplication
                content_hash = hashlib.md5(content.encode()).hexdigest()
                
                # Get conversation turn number
                conversation_turn = self._get_next_turn_number(session_id)
                
                # Detect topic and intent
                topic = self._detect_topic(content, session_id)
                intent = self._detect_intent(content, message_type)
                
                # Prepare message data
                message_data = {
                    'session_id': session_id,
                    'message_id': message_id,
                    'timestamp': timestamp,
                    'message_type': message_type,
                    'content': content,
                    'content_hash': content_hash,
                    'parent_message_id': parent_message_id,
                    'conversation_turn': conversation_turn,
                    'topic': topic,
                    'intent': intent,
                    'metadata': json.dumps(metadata or {}),
                    'response_length': len(content),
                    'context_snapshot': json.dumps(self._capture_context_snapshot(session_id))
                }
                
                # Store in database
                conn = self.db.get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO messages (
                        session_id, message_id, timestamp, message_type, content,
                        content_hash, parent_message_id, conversation_turn, topic,
                        intent, metadata, response_length, context_snapshot
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, message_id, timestamp, message_type, content,
                    content_hash, parent_message_id, conversation_turn, topic,
                    intent, message_data['metadata'], message_data['response_length'],
                    message_data['context_snapshot']
                ))
                
                conn.commit()
                
                # Update session interaction count
                self._update_session_interaction_count(session_id)
                
                # Cache message
                self._message_cache[session_id].append(message_data)
                
                # Update session topic if this is a significant message
                if message_type == 'user_input' and topic:
                    self._update_session_topic(session_id, topic)
                
                logger.debug(f"Added message {message_id} to session {session_id}")
                return message_id
                
            except Exception as e:
                logger.error(f"Error adding message to session {session_id}: {e}")
                raise
    
    def end_session(self, session_id: str, session_summary: Optional[str] = None,
                   user_satisfaction_score: Optional[float] = None,
                   goal_completion_status: str = "completed") -> bool:
        """
        End a conversation session and finalize metadata.
        
        Args:
            session_id: Session identifier
            session_summary: Optional session summary
            user_satisfaction_score: Optional satisfaction rating (0.0-1.0)
            goal_completion_status: Status of goal completion
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                ended_at = datetime.now().isoformat()
                
                # Calculate session duration
                session_data = self.get_session_metadata(session_id)
                if session_data:
                    started_at = datetime.fromisoformat(session_data['started_at'])
                    ended_at_dt = datetime.fromisoformat(ended_at)
                    duration_seconds = int((ended_at_dt - started_at).total_seconds())
                else:
                    duration_seconds = 0
                
                # Analyze session for topic categories and key decisions
                topic_categories = self._analyze_session_topics(session_id)
                key_decisions = self._extract_key_decisions(session_id)
                
                # Update session in database
                conn = self.db.get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE sessions SET
                        ended_at = ?, duration_seconds = ?, session_summary = ?,
                        user_satisfaction_score = ?, goal_completion_status = ?,
                        topic_categories = ?, key_decisions = ?
                    WHERE session_id = ?
                """, (
                    ended_at, duration_seconds, session_summary,
                    user_satisfaction_score, goal_completion_status,
                    json.dumps(topic_categories), json.dumps(key_decisions),
                    session_id
                ))
                
                conn.commit()
                
                # Update user profile with session data
                if self.profile_manager and session_data:
                    profile_name = "default"  # Could be enhanced to track profile name
                    session_update_data = {
                        'duration_seconds': duration_seconds,
                        'topic_categories': topic_categories,
                        'user_satisfaction_score': user_satisfaction_score,
                        'interaction_count': session_data.get('interaction_count', 0)
                    }
                    self.profile_manager.update_interaction_history(session_update_data, profile_name)
                
                # Remove from active sessions
                if session_id in self._active_sessions:
                    del self._active_sessions[session_id]
                
                # Clear message cache for this session
                if session_id in self._message_cache:
                    del self._message_cache[session_id]
                
                logger.info(f"Ended session {session_id} with duration {duration_seconds}s")
                return True
                
            except Exception as e:
                logger.error(f"Error ending session {session_id}: {e}")
                return False
    
    def get_session_messages(self, session_id: str, 
                           include_metadata: bool = True,
                           message_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get all messages for a session.
        
        Args:
            session_id: Session identifier
            include_metadata: Whether to include message metadata
            message_types: Optional filter for message types
            
        Returns:
            List of message dictionaries
        """
        try:
            # Check cache first
            if session_id in self._message_cache and self._message_cache[session_id]:
                messages = self._message_cache[session_id]
            else:
                # Load from database
                conn = self.db.get_connection()
                cursor = conn.cursor()
                
                query = "SELECT * FROM messages WHERE session_id = ? ORDER BY conversation_turn"
                cursor.execute(query, (session_id,))
                rows = cursor.fetchall()
                
                messages = []
                for row in rows:
                    message = dict(row)
                    # Parse JSON fields
                    if message.get('metadata'):
                        message['metadata'] = json.loads(message['metadata'])
                    if message.get('context_snapshot'):
                        message['context_snapshot'] = json.loads(message['context_snapshot'])
                    messages.append(message)
                
                # Cache the messages
                self._message_cache[session_id] = messages
            
            # Apply filters
            if message_types:
                messages = [m for m in messages if m['message_type'] in message_types]
            
            if not include_metadata:
                for message in messages:
                    message.pop('metadata', None)
                    message.pop('context_snapshot', None)
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages for session {session_id}: {e}")
            return []
    
    def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session metadata.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session metadata dictionary or None if not found
        """
        try:
            # Check active sessions first
            if session_id in self._active_sessions:
                return self._active_sessions[session_id].copy()
            
            # Load from database
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            
            if row:
                session_data = dict(row)
                # Parse JSON fields
                if session_data.get('topic_categories'):
                    session_data['topic_categories'] = json.loads(session_data['topic_categories'])
                if session_data.get('key_decisions'):
                    session_data['key_decisions'] = json.loads(session_data['key_decisions'])
                if session_data.get('session_context'):
                    session_data['session_context'] = json.loads(session_data['session_context'])
                if session_data.get('device_info'):
                    session_data['device_info'] = json.loads(session_data['device_info'])
                
                return session_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting session metadata for {session_id}: {e}")
            return None
    
    def search_sessions(self, query: str, 
                       date_range: Optional[Tuple[datetime, datetime]] = None,
                       topic_filter: Optional[str] = None,
                       limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search sessions by content, date, or topic.
        
        Args:
            query: Search query
            date_range: Optional date range filter (start, end)
            topic_filter: Optional topic filter
            limit: Maximum number of results
            
        Returns:
            List of matching sessions with metadata
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Build search query
            sql_conditions = []
            params = []
            
            if query:
                # Search in session summary and messages
                sql_conditions.append("""
                    (s.session_summary LIKE ? OR 
                     EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.session_id AND m.content LIKE ?))
                """)
                params.extend([f"%{query}%", f"%{query}%"])
            
            if date_range:
                sql_conditions.append("s.started_at BETWEEN ? AND ?")
                params.extend([date_range[0].isoformat(), date_range[1].isoformat()])
            
            if topic_filter:
                sql_conditions.append("s.primary_topic = ? OR s.topic_categories LIKE ?")
                params.extend([topic_filter, f"%{topic_filter}%"])
            
            where_clause = " AND ".join(sql_conditions) if sql_conditions else "1=1"
            
            sql = f"""
                SELECT s.*, COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                WHERE {where_clause}
                GROUP BY s.session_id
                ORDER BY s.started_at DESC
                LIMIT ?
            """
            params.append(limit)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            sessions = []
            for row in rows:
                session = dict(row)
                # Parse JSON fields
                if session.get('topic_categories'):
                    session['topic_categories'] = json.loads(session['topic_categories'])
                if session.get('key_decisions'):
                    session['key_decisions'] = json.loads(session['key_decisions'])
                sessions.append(session)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error searching sessions: {e}")
            return []
    
    def get_conversation_thread(self, message_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation thread starting from a specific message.
        
        Args:
            message_id: Starting message ID
            
        Returns:
            List of messages in the thread
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Get the starting message
            cursor.execute("SELECT * FROM messages WHERE message_id = ?", (message_id,))
            start_message = cursor.fetchone()
            
            if not start_message:
                return []
            
            session_id = start_message['session_id']
            
            # Get all messages in the session and build thread
            all_messages = self.get_session_messages(session_id)
            
            # Build thread by following parent-child relationships
            thread = []
            current_id = message_id
            
            # Go backwards to find thread start
            while current_id:
                message = next((m for m in all_messages if m['message_id'] == current_id), None)
                if message:
                    thread.insert(0, message)
                    current_id = message.get('parent_message_id')
                else:
                    break
            
            # Go forwards to find thread continuation
            def find_children(parent_id):
                return [m for m in all_messages if m.get('parent_message_id') == parent_id]
            
            last_message_id = thread[-1]['message_id'] if thread else message_id
            children = find_children(last_message_id)
            
            while children:
                # Add first child (assuming linear conversation)
                if children:
                    thread.append(children[0])
                    children = find_children(children[0]['message_id'])
                else:
                    break
            
            return thread
            
        except Exception as e:
            logger.error(f"Error getting conversation thread for {message_id}: {e}")
            return []
    
    def _get_next_turn_number(self, session_id: str) -> int:
        """Get the next conversation turn number for a session."""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT MAX(conversation_turn) FROM messages WHERE session_id = ?",
                (session_id,)
            )
            result = cursor.fetchone()
            max_turn = result[0] if result and result[0] is not None else 0
            return max_turn + 1
            
        except Exception as e:
            logger.error(f"Error getting next turn number: {e}")
            return 1
    
    def _detect_topic(self, content: str, session_id: str) -> Optional[str]:
        """Simple topic detection based on content and session history."""
        # This is a simplified implementation
        # In a real system, you might use NLP libraries or ML models
        
        content_lower = content.lower()
        
        # Common topic keywords
        topic_keywords = {
            'weather': ['weather', 'temperature', 'rain', 'sunny', 'cloudy'],
            'programming': ['code', 'python', 'javascript', 'programming', 'function', 'variable'],
            'health': ['health', 'doctor', 'medicine', 'symptoms', 'treatment'],
            'travel': ['travel', 'trip', 'vacation', 'flight', 'hotel'],
            'food': ['food', 'recipe', 'cooking', 'restaurant', 'meal'],
            'technology': ['technology', 'computer', 'software', 'hardware', 'tech']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return topic
        
        return 'general'
    
    def _detect_intent(self, content: str, message_type: str) -> Optional[str]:
        """Simple intent detection based on content patterns."""
        if message_type != 'user_input':
            return None
        
        content_lower = content.lower()
        
        # Intent patterns
        if any(word in content_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return 'question'
        elif any(word in content_lower for word in ['please', 'can you', 'could you', 'help']):
            return 'request'
        elif any(word in content_lower for word in ['thank', 'thanks', 'good', 'great', 'excellent']):
            return 'appreciation'
        elif any(word in content_lower for word in ['no', 'wrong', 'incorrect', 'bad', 'not right']):
            return 'correction'
        else:
            return 'statement'
    
    def _capture_context_snapshot(self, session_id: str) -> Dict[str, Any]:
        """Capture current context snapshot for a session."""
        return {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'message_count': len(self._message_cache.get(session_id, [])),
            'active_topic': self._get_current_topic(session_id)
        }
    
    def _get_current_topic(self, session_id: str) -> Optional[str]:
        """Get the current topic for a session."""
        messages = self._message_cache.get(session_id, [])
        if messages:
            # Return the topic of the most recent user input
            for message in reversed(messages):
                if message['message_type'] == 'user_input' and message.get('topic'):
                    return message['topic']
        return None
    
    def _update_session_interaction_count(self, session_id: str) -> None:
        """Update the interaction count for a session."""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE sessions 
                SET interaction_count = interaction_count + 1 
                WHERE session_id = ?
            """, (session_id,))
            
            conn.commit()
            
            # Update cache
            if session_id in self._active_sessions:
                self._active_sessions[session_id]['interaction_count'] += 1
                
        except Exception as e:
            logger.error(f"Error updating interaction count: {e}")
    
    def _update_session_topic(self, session_id: str, topic: str) -> None:
        """Update the primary topic for a session."""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE sessions 
                SET primary_topic = ? 
                WHERE session_id = ? AND primary_topic IS NULL
            """, (topic, session_id))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error updating session topic: {e}")
    
    def _analyze_session_topics(self, session_id: str) -> List[str]:
        """Analyze all topics discussed in a session."""
        try:
            messages = self.get_session_messages(session_id, include_metadata=False)
            topics = set()
            
            for message in messages:
                if message.get('topic') and message['topic'] != 'general':
                    topics.add(message['topic'])
            
            return list(topics)
            
        except Exception as e:
            logger.error(f"Error analyzing session topics: {e}")
            return []
    
    def _extract_key_decisions(self, session_id: str) -> List[Dict[str, Any]]:
        """Extract key decisions made during the session."""
        # This is a simplified implementation
        # In practice, you might use NLP to identify decision points
        
        try:
            messages = self.get_session_messages(session_id, include_metadata=False)
            decisions = []
            
            decision_keywords = ['decide', 'choose', 'select', 'pick', 'go with', 'settle on']
            
            for message in messages:
                content_lower = message['content'].lower()
                if any(keyword in content_lower for keyword in decision_keywords):
                    decisions.append({
                        'message_id': message['message_id'],
                        'timestamp': message['timestamp'],
                        'content_snippet': message['content'][:200] + '...' if len(message['content']) > 200 else message['content']
                    })
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error extracting key decisions: {e}")
            return []
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get basic device/platform information."""
        import platform
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': platform.python_version(),
            'timestamp': datetime.now().isoformat()
        }
