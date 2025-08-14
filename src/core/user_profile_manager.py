#!/usr/bin/env python3
"""
User Profile Management System

Handles persistent user profile management for single-user personal AI assistant.
Manages AI interaction preferences, voice companion settings, learning preferences,
and behavioral pattern tracking.
"""

import json
import logging
import sqlite3
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from enhanced_database_schema import EnhancedDatabaseSchema, EncryptionManager

logger = logging.getLogger(__name__)


class UserProfileManager:
    """
    Manages comprehensive user profile with persistent preferences and adaptive learning.
    """
    
    def __init__(self, db_connection, encryption_manager: Optional[EncryptionManager] = None):
        """
        Initialize user profile manager.
        
        Args:
            db_connection: Database connection instance
            encryption_manager: Optional encryption manager for sensitive data
        """
        self.db = db_connection
        self.encryption = encryption_manager or EncryptionManager()
        self._lock = threading.Lock()
        self._profile_cache = {}
        self._ensure_default_profile()
    
    def _ensure_default_profile(self) -> None:
        """Ensure a default user profile exists."""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Check if default profile exists
            cursor.execute("SELECT id FROM user_profile WHERE profile_name = ?", ("default",))
            if not cursor.fetchone():
                self.create_profile("default")
                logger.info("Created default user profile")
        except Exception as e:
            logger.error(f"Error ensuring default profile: {e}")
    
    def create_profile(self, profile_name: str = "default", 
                      initial_preferences: Optional[Dict[str, Any]] = None) -> int:
        """
        Create a new user profile.
        
        Args:
            profile_name: Name of the profile
            initial_preferences: Initial preference settings
            
        Returns:
            Profile ID
        """
        with self._lock:
            try:
                conn = self.db.get_connection()
                cursor = conn.cursor()
                
                # Default preferences
                defaults = {
                    'response_style': 'balanced',
                    'verbosity_level': 'moderate',
                    'technical_level': 'intermediate',
                    'explanation_preference': 'balanced',
                    'preferred_tts_engine': 'elevenlabs',
                    'voice_selection': 'default',
                    'speech_rate': 1.0,
                    'speech_pitch': 1.0,
                    'speech_volume': 0.8,
                    'learning_mode': 'adaptive',
                    'feedback_frequency': 'moderate',
                    'context_retention': 'session',
                    'privacy_level': 'standard'
                }
                
                # Merge with initial preferences
                if initial_preferences:
                    defaults.update(initial_preferences)
                
                # Prepare communication patterns and interaction history
                communication_patterns = json.dumps({
                    'preferred_greeting_style': 'friendly',
                    'response_timing_preference': 'immediate',
                    'question_asking_frequency': 'moderate',
                    'clarification_seeking': 'when_needed'
                })
                
                interaction_history_summary = json.dumps({
                    'total_interactions': 0,
                    'average_session_length': 0,
                    'most_discussed_topics': [],
                    'satisfaction_trend': [],
                    'learning_progress': {}
                })
                
                # Insert profile
                cursor.execute("""
                    INSERT INTO user_profile (
                        profile_name, response_style, verbosity_level, technical_level,
                        explanation_preference, preferred_tts_engine, voice_selection,
                        speech_rate, speech_pitch, speech_volume, learning_mode,
                        feedback_frequency, context_retention, privacy_level,
                        communication_patterns, interaction_history_summary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile_name, defaults['response_style'], defaults['verbosity_level'],
                    defaults['technical_level'], defaults['explanation_preference'],
                    defaults['preferred_tts_engine'], defaults['voice_selection'],
                    defaults['speech_rate'], defaults['speech_pitch'], defaults['speech_volume'],
                    defaults['learning_mode'], defaults['feedback_frequency'],
                    defaults['context_retention'], defaults['privacy_level'],
                    communication_patterns, interaction_history_summary
                ))
                
                conn.commit()
                profile_id = cursor.lastrowid
                
                # Cache the profile
                self._profile_cache[profile_name] = self._load_profile_data(profile_id)
                
                logger.info(f"Created user profile '{profile_name}' with ID {profile_id}")
                return profile_id
                
            except Exception as e:
                logger.error(f"Error creating profile '{profile_name}': {e}")
                raise
    
    def get_profile(self, profile_name: str = "default") -> Optional[Dict[str, Any]]:
        """
        Get user profile data.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            Profile data dictionary or None if not found
        """
        try:
            # Check cache first
            if profile_name in self._profile_cache:
                return self._profile_cache[profile_name]
            
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM user_profile WHERE profile_name = ?", (profile_name,))
            row = cursor.fetchone()
            
            if row:
                profile_data = self._load_profile_data(row['id'])
                self._profile_cache[profile_name] = profile_data
                return profile_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting profile '{profile_name}': {e}")
            return None
    
    def _load_profile_data(self, profile_id: int) -> Dict[str, Any]:
        """Load complete profile data from database."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM user_profile WHERE id = ?", (profile_id,))
        row = cursor.fetchone()
        
        if not row:
            return {}
        
        profile_data = dict(row)
        
        # Parse JSON fields
        if profile_data.get('communication_patterns'):
            profile_data['communication_patterns'] = json.loads(profile_data['communication_patterns'])
        
        if profile_data.get('interaction_history_summary'):
            profile_data['interaction_history_summary'] = json.loads(profile_data['interaction_history_summary'])
        
        # Decrypt sensitive data if present
        if profile_data.get('encrypted_data'):
            try:
                decrypted = self.encryption.decrypt(profile_data['encrypted_data'])
                if decrypted:
                    profile_data['sensitive_data'] = json.loads(decrypted)
            except Exception as e:
                logger.warning(f"Could not decrypt sensitive data: {e}")
        
        return profile_data
    
    def update_preference(self, preference_key: str, value: Any, 
                         profile_name: str = "default") -> bool:
        """
        Update a specific preference.
        
        Args:
            preference_key: Key of the preference to update
            value: New value
            profile_name: Name of the profile
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                conn = self.db.get_connection()
                cursor = conn.cursor()
                
                # Check if preference column exists
                valid_columns = [
                    'response_style', 'verbosity_level', 'technical_level',
                    'explanation_preference', 'preferred_tts_engine', 'voice_selection',
                    'speech_rate', 'speech_pitch', 'speech_volume', 'learning_mode',
                    'feedback_frequency', 'context_retention', 'privacy_level'
                ]
                
                if preference_key in valid_columns:
                    # Update direct column
                    sql = f"UPDATE user_profile SET {preference_key} = ?, updated_at = ? WHERE profile_name = ?"
                    cursor.execute(sql, (value, datetime.now().isoformat(), profile_name))
                else:
                    # Update in communication_patterns JSON
                    cursor.execute(
                        "SELECT communication_patterns FROM user_profile WHERE profile_name = ?",
                        (profile_name,)
                    )
                    row = cursor.fetchone()
                    if row:
                        patterns = json.loads(row['communication_patterns'] or '{}')
                        patterns[preference_key] = value
                        
                        cursor.execute("""
                            UPDATE user_profile 
                            SET communication_patterns = ?, updated_at = ? 
                            WHERE profile_name = ?
                        """, (json.dumps(patterns), datetime.now().isoformat(), profile_name))
                
                conn.commit()
                
                # Invalidate cache
                if profile_name in self._profile_cache:
                    del self._profile_cache[profile_name]
                
                logger.debug(f"Updated preference {preference_key} = {value} for profile {profile_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error updating preference {preference_key}: {e}")
                return False
    
    def get_preference(self, preference_key: str, default: Any = None, 
                      profile_name: str = "default") -> Any:
        """
        Get a specific preference value.
        
        Args:
            preference_key: Key of the preference
            default: Default value if not found
            profile_name: Name of the profile
            
        Returns:
            Preference value or default
        """
        try:
            profile = self.get_profile(profile_name)
            if not profile:
                return default
            
            # Check direct columns first
            if preference_key in profile:
                return profile[preference_key]
            
            # Check communication patterns
            patterns = profile.get('communication_patterns', {})
            if preference_key in patterns:
                return patterns[preference_key]
            
            # Check sensitive data
            sensitive = profile.get('sensitive_data', {})
            if preference_key in sensitive:
                return sensitive[preference_key]
            
            return default
            
        except Exception as e:
            logger.error(f"Error getting preference {preference_key}: {e}")
            return default
    
    def update_interaction_history(self, session_data: Dict[str, Any], 
                                  profile_name: str = "default") -> bool:
        """
        Update interaction history summary with new session data.
        
        Args:
            session_data: Data from completed session
            profile_name: Name of the profile
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                profile = self.get_profile(profile_name)
                if not profile:
                    return False
                
                history = profile.get('interaction_history_summary', {})
                
                # Update statistics
                history['total_interactions'] = history.get('total_interactions', 0) + 1
                
                # Update average session length
                current_avg = history.get('average_session_length', 0)
                total_sessions = history['total_interactions']
                session_length = session_data.get('duration_seconds', 0)
                new_avg = ((current_avg * (total_sessions - 1)) + session_length) / total_sessions
                history['average_session_length'] = new_avg
                
                # Update most discussed topics
                topics = history.get('most_discussed_topics', [])
                session_topics = session_data.get('topic_categories', [])
                for topic in session_topics:
                    # Simple frequency counting
                    topic_found = False
                    for i, (existing_topic, count) in enumerate(topics):
                        if existing_topic == topic:
                            topics[i] = (topic, count + 1)
                            topic_found = True
                            break
                    if not topic_found:
                        topics.append((topic, 1))
                
                # Keep top 10 topics
                topics.sort(key=lambda x: x[1], reverse=True)
                history['most_discussed_topics'] = topics[:10]
                
                # Update satisfaction trend
                satisfaction_trend = history.get('satisfaction_trend', [])
                if 'user_satisfaction_score' in session_data:
                    satisfaction_trend.append({
                        'timestamp': datetime.now().isoformat(),
                        'score': session_data['user_satisfaction_score']
                    })
                    # Keep last 50 satisfaction scores
                    history['satisfaction_trend'] = satisfaction_trend[-50:]
                
                # Save updated history
                conn = self.db.get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE user_profile 
                    SET interaction_history_summary = ?, updated_at = ? 
                    WHERE profile_name = ?
                """, (json.dumps(history), datetime.now().isoformat(), profile_name))
                
                conn.commit()
                
                # Invalidate cache
                if profile_name in self._profile_cache:
                    del self._profile_cache[profile_name]
                
                logger.debug(f"Updated interaction history for profile {profile_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error updating interaction history: {e}")
                return False
    
    def adapt_preferences_from_feedback(self, feedback_data: Dict[str, Any], 
                                       profile_name: str = "default") -> List[str]:
        """
        Adapt user preferences based on feedback patterns.
        
        Args:
            feedback_data: Analyzed feedback data
            profile_name: Name of the profile
            
        Returns:
            List of adaptations made
        """
        adaptations = []
        
        try:
            # Analyze feedback for preference adjustments
            sentiment = feedback_data.get('sentiment', 0.0)
            feedback_type = feedback_data.get('type', '')
            
            # Adapt verbosity based on feedback
            if 'too detailed' in feedback_data.get('text', '').lower():
                current_verbosity = self.get_preference('verbosity_level', 'moderate', profile_name)
                if current_verbosity == 'detailed':
                    self.update_preference('verbosity_level', 'moderate', profile_name)
                    adaptations.append('Reduced verbosity level to moderate')
                elif current_verbosity == 'comprehensive':
                    self.update_preference('verbosity_level', 'detailed', profile_name)
                    adaptations.append('Reduced verbosity level to detailed')
            
            elif 'too brief' in feedback_data.get('text', '').lower():
                current_verbosity = self.get_preference('verbosity_level', 'moderate', profile_name)
                if current_verbosity == 'brief':
                    self.update_preference('verbosity_level', 'moderate', profile_name)
                    adaptations.append('Increased verbosity level to moderate')
                elif current_verbosity == 'moderate':
                    self.update_preference('verbosity_level', 'detailed', profile_name)
                    adaptations.append('Increased verbosity level to detailed')
            
            # Adapt technical level based on feedback
            if 'too technical' in feedback_data.get('text', '').lower():
                current_level = self.get_preference('technical_level', 'intermediate', profile_name)
                if current_level == 'expert':
                    self.update_preference('technical_level', 'advanced', profile_name)
                    adaptations.append('Reduced technical level to advanced')
                elif current_level == 'advanced':
                    self.update_preference('technical_level', 'intermediate', profile_name)
                    adaptations.append('Reduced technical level to intermediate')
            
            # Adapt response style based on sentiment
            if sentiment < -0.5 and feedback_type == 'style_feedback':
                current_style = self.get_preference('response_style', 'balanced', profile_name)
                if current_style == 'formal':
                    self.update_preference('response_style', 'balanced', profile_name)
                    adaptations.append('Changed response style to balanced')
                elif current_style == 'technical':
                    self.update_preference('response_style', 'casual', profile_name)
                    adaptations.append('Changed response style to casual')
            
            logger.info(f"Made {len(adaptations)} preference adaptations for profile {profile_name}")
            return adaptations
            
        except Exception as e:
            logger.error(f"Error adapting preferences: {e}")
            return []
    
    def export_profile(self, profile_name: str = "default") -> Optional[Dict[str, Any]]:
        """
        Export complete profile data for backup.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            Complete profile data or None if not found
        """
        try:
            profile = self.get_profile(profile_name)
            if profile:
                # Add export metadata
                profile['export_timestamp'] = datetime.now().isoformat()
                profile['export_version'] = '1.0'
            return profile
            
        except Exception as e:
            logger.error(f"Error exporting profile {profile_name}: {e}")
            return None
    
    def import_profile(self, profile_data: Dict[str, Any], 
                      profile_name: str = "default") -> bool:
        """
        Import profile data from backup.
        
        Args:
            profile_data: Profile data to import
            profile_name: Name for the imported profile
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create new profile with imported data
            preferences = {k: v for k, v in profile_data.items() 
                          if k not in ['id', 'created_at', 'updated_at', 'export_timestamp', 'export_version']}
            
            # Check if profile exists
            existing = self.get_profile(profile_name)
            if existing:
                # Update existing profile
                for key, value in preferences.items():
                    if key in ['communication_patterns', 'interaction_history_summary']:
                        continue  # Handle these separately
                    self.update_preference(key, value, profile_name)
            else:
                # Create new profile
                self.create_profile(profile_name, preferences)
            
            logger.info(f"Imported profile data for {profile_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing profile: {e}")
            return False
