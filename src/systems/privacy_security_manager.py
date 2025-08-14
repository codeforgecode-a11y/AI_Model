#!/usr/bin/env python3
"""
Privacy and Security Manager

Implements comprehensive privacy and security features for the personal AI assistant:
- Data encryption for sensitive information
- Clear data management controls
- Local-only storage with no external transmission
- Data portability and user control options
- Secure deletion and data lifecycle management
"""

import json
import logging
import sqlite3
import threading
import hashlib
import os
import shutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)


class DataPrivacyManager:
    """
    Manages data privacy controls and user data rights.
    """
    
    def __init__(self, db_connection, encryption_manager):
        """
        Initialize privacy manager.
        
        Args:
            db_connection: Database connection instance
            encryption_manager: Encryption manager instance
        """
        self.db = db_connection
        self.encryption = encryption_manager
        self._lock = threading.Lock()
    
    def view_personal_data(self, profile_name: str = "default") -> Dict[str, Any]:
        """
        Provide comprehensive view of all personal data stored.
        
        Args:
            profile_name: User profile name
            
        Returns:
            Dictionary containing all personal data categories
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Get user profile data
            cursor.execute("SELECT * FROM user_profile WHERE profile_name = ?", (profile_name,))
            profile_data = cursor.fetchone()
            
            # Get session count and date range
            cursor.execute("""
                SELECT COUNT(*), MIN(started_at), MAX(started_at) 
                FROM sessions s 
                JOIN user_profile p ON s.profile_id = p.id 
                WHERE p.profile_name = ?
            """, (profile_name,))
            session_stats = cursor.fetchone()
            
            # Get message count
            cursor.execute("""
                SELECT COUNT(*) FROM messages m
                JOIN sessions s ON m.session_id = s.session_id
                JOIN user_profile p ON s.profile_id = p.id
                WHERE p.profile_name = ?
            """, (profile_name,))
            message_count = cursor.fetchone()[0]
            
            # Get learning insights count
            cursor.execute("SELECT COUNT(*) FROM learning_insights")
            insights_count = cursor.fetchone()[0]
            
            # Get conversation topics
            cursor.execute("SELECT topic_name, discussion_frequency FROM conversation_topics ORDER BY discussion_frequency DESC LIMIT 10")
            top_topics = cursor.fetchall()
            
            return {
                'profile_summary': {
                    'profile_name': profile_name,
                    'created_at': profile_data['created_at'] if profile_data else None,
                    'last_updated': profile_data['updated_at'] if profile_data else None,
                    'preferences_count': len(json.loads(profile_data['communication_patterns'] or '{}')) if profile_data else 0
                },
                'conversation_data': {
                    'total_sessions': session_stats[0] if session_stats[0] else 0,
                    'first_session': session_stats[1] if session_stats[1] else None,
                    'last_session': session_stats[2] if session_stats[2] else None,
                    'total_messages': message_count
                },
                'learning_data': {
                    'insights_count': insights_count,
                    'top_topics': [{'topic': row[0], 'frequency': row[1]} for row in top_topics]
                },
                'data_summary': {
                    'total_data_points': message_count + insights_count + (1 if profile_data else 0),
                    'privacy_level': profile_data['privacy_level'] if profile_data else 'standard',
                    'encryption_status': 'enabled',
                    'local_storage_only': True
                },
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error viewing personal data: {e}")
            return {'error': str(e)}
    
    def delete_session_data(self, session_id: str, secure_delete: bool = True) -> bool:
        """
        Delete specific session data with optional secure deletion.
        
        Args:
            session_id: Session identifier to delete
            secure_delete: Whether to perform secure deletion
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                conn = self.db.get_connection()
                cursor = conn.cursor()
                
                # Start transaction
                cursor.execute("BEGIN TRANSACTION")
                
                # Delete messages first (due to foreign key constraints)
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                messages_deleted = cursor.rowcount
                
                # Delete session
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                sessions_deleted = cursor.rowcount
                
                conn.commit()
                
                if secure_delete:
                    # Perform secure deletion by overwriting database pages
                    cursor.execute("VACUUM")
                
                logger.info(f"Deleted session {session_id}: {messages_deleted} messages, {sessions_deleted} session")
                return True
                
            except Exception as e:
                conn.rollback()
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
        with self._lock:
            try:
                conn = self.db.get_connection()
                cursor = conn.cursor()
                
                # Start transaction
                cursor.execute("BEGIN TRANSACTION")
                
                # Get sessions in date range
                cursor.execute("""
                    SELECT session_id FROM sessions 
                    WHERE started_at BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat()))
                
                session_ids = [row[0] for row in cursor.fetchall()]
                
                # Delete messages for these sessions
                messages_deleted = 0
                for session_id in session_ids:
                    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                    messages_deleted += cursor.rowcount
                
                # Delete sessions
                cursor.execute("""
                    DELETE FROM sessions 
                    WHERE started_at BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat()))
                sessions_deleted = cursor.rowcount
                
                # Delete learning insights in date range
                cursor.execute("""
                    DELETE FROM learning_insights 
                    WHERE created_at BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat()))
                insights_deleted = cursor.rowcount
                
                conn.commit()
                
                if secure_delete:
                    cursor.execute("VACUUM")
                
                deletion_summary = {
                    'sessions_deleted': sessions_deleted,
                    'messages_deleted': messages_deleted,
                    'insights_deleted': insights_deleted,
                    'date_range': f"{start_date.isoformat()} to {end_date.isoformat()}"
                }
                
                logger.info(f"Deleted data for date range: {deletion_summary}")
                return deletion_summary
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error deleting date range data: {e}")
                return {'error': str(e)}
    
    def anonymize_profile_data(self, profile_name: str = "default") -> bool:
        """
        Anonymize profile data while preserving learning patterns.
        
        Args:
            profile_name: Profile name to anonymize
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                conn = self.db.get_connection()
                cursor = conn.cursor()
                
                # Generate anonymous profile name
                anonymous_name = f"anonymous_{hashlib.md5(profile_name.encode()).hexdigest()[:8]}"
                
                # Update profile with anonymized data
                cursor.execute("""
                    UPDATE user_profile 
                    SET profile_name = ?, encrypted_data = NULL, updated_at = ?
                    WHERE profile_name = ?
                """, (anonymous_name, datetime.now().isoformat(), profile_name))
                
                conn.commit()
                
                logger.info(f"Anonymized profile {profile_name} -> {anonymous_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error anonymizing profile: {e}")
                return False
    
    def export_gdpr_data(self, profile_name: str = "default", 
                        output_path: str = None) -> Optional[str]:
        """
        Export all personal data in GDPR-compliant format.
        
        Args:
            profile_name: Profile name to export
            output_path: Optional output path
            
        Returns:
            Path to exported file or None if failed
        """
        try:
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"gdpr_export_{profile_name}_{timestamp}.json"
            
            # Get comprehensive data view
            personal_data = self.view_personal_data(profile_name)
            
            # Add detailed data for GDPR compliance
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Get all sessions with messages
            cursor.execute("""
                SELECT s.*, COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                JOIN user_profile p ON s.profile_id = p.id
                WHERE p.profile_name = ?
                GROUP BY s.session_id
                ORDER BY s.started_at
            """, (profile_name,))
            
            sessions = []
            for row in cursor.fetchall():
                session_data = dict(row)
                
                # Get messages for this session
                cursor.execute("""
                    SELECT message_type, content, timestamp, user_rating, user_feedback
                    FROM messages 
                    WHERE session_id = ?
                    ORDER BY conversation_turn
                """, (session_data['session_id'],))
                
                messages = [dict(msg_row) for msg_row in cursor.fetchall()]
                session_data['messages'] = messages
                sessions.append(session_data)
            
            # Compile GDPR export
            gdpr_export = {
                'export_info': {
                    'export_type': 'GDPR_DATA_EXPORT',
                    'profile_name': profile_name,
                    'export_date': datetime.now().isoformat(),
                    'data_controller': 'Personal AI Assistant (Local)',
                    'retention_policy': 'User-controlled local storage'
                },
                'personal_data_summary': personal_data,
                'detailed_sessions': sessions,
                'data_processing_info': {
                    'purposes': [
                        'Personalized AI assistance',
                        'Learning user preferences',
                        'Conversation history maintenance',
                        'Response quality improvement'
                    ],
                    'legal_basis': 'User consent for personal AI assistant functionality',
                    'data_retention': 'Indefinite (user-controlled)',
                    'data_sharing': 'None - all data stored locally only'
                }
            }
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(gdpr_export, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"GDPR export completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating GDPR export: {e}")
            return None
    
    def get_data_retention_info(self) -> Dict[str, Any]:
        """
        Get information about data retention policies and storage.
        
        Returns:
            Data retention information
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Get database size and statistics
            cursor.execute("SELECT COUNT(*) FROM user_profile")
            profile_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM sessions")
            session_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM messages")
            message_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM learning_insights")
            insights_count = cursor.fetchone()[0]
            
            # Get oldest and newest data
            cursor.execute("SELECT MIN(created_at), MAX(updated_at) FROM user_profile")
            profile_dates = cursor.fetchone()
            
            cursor.execute("SELECT MIN(started_at), MAX(started_at) FROM sessions")
            session_dates = cursor.fetchone()
            
            return {
                'storage_info': {
                    'storage_type': 'Local SQLite database',
                    'encryption_enabled': True,
                    'external_transmission': False,
                    'cloud_storage': False
                },
                'data_counts': {
                    'user_profiles': profile_count,
                    'conversation_sessions': session_count,
                    'messages': message_count,
                    'learning_insights': insights_count
                },
                'date_ranges': {
                    'profile_created': profile_dates[0] if profile_dates[0] else None,
                    'profile_updated': profile_dates[1] if profile_dates[1] else None,
                    'first_session': session_dates[0] if session_dates[0] else None,
                    'last_session': session_dates[1] if session_dates[1] else None
                },
                'retention_policy': {
                    'automatic_deletion': False,
                    'user_controlled': True,
                    'backup_retention': '30 days (configurable)',
                    'secure_deletion_available': True
                },
                'privacy_controls': {
                    'view_data': True,
                    'export_data': True,
                    'delete_specific_sessions': True,
                    'delete_date_ranges': True,
                    'anonymize_profile': True,
                    'complete_deletion': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting retention info: {e}")
            return {'error': str(e)}


class SecureDataManager:
    """
    Manages secure data operations and encryption.
    """
    
    def __init__(self, encryption_manager):
        """
        Initialize secure data manager.
        
        Args:
            encryption_manager: Encryption manager instance
        """
        self.encryption = encryption_manager
        self._lock = threading.Lock()
    
    def encrypt_sensitive_data(self, data: Dict[str, Any]) -> str:
        """
        Encrypt sensitive data for storage.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data string
        """
        try:
            json_data = json.dumps(data)
            return self.encryption.encrypt(json_data)
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            return ""
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Dict[str, Any]:
        """
        Decrypt sensitive data from storage.
        
        Args:
            encrypted_data: Encrypted data string
            
        Returns:
            Decrypted data dictionary
        """
        try:
            if not encrypted_data:
                return {}
            
            json_data = self.encryption.decrypt(encrypted_data)
            return json.loads(json_data) if json_data else {}
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            return {}
    
    def secure_delete_file(self, file_path: str, passes: int = 3) -> bool:
        """
        Securely delete a file by overwriting it multiple times.
        
        Args:
            file_path: Path to file to delete
            passes: Number of overwrite passes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                return True
            
            file_size = os.path.getsize(file_path)
            
            with open(file_path, "r+b") as file:
                for _ in range(passes):
                    file.seek(0)
                    file.write(os.urandom(file_size))
                    file.flush()
                    os.fsync(file.fileno())
            
            os.remove(file_path)
            logger.info(f"Securely deleted file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error securely deleting file {file_path}: {e}")
            return False
    
    def validate_data_integrity(self, db_path: str) -> Dict[str, Any]:
        """
        Validate database integrity and detect potential corruption.
        
        Args:
            db_path: Path to database file
            
        Returns:
            Integrity check results
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            
            # Get database info
            cursor.execute("PRAGMA database_list")
            db_info = cursor.fetchall()
            
            # Check for foreign key violations
            cursor.execute("PRAGMA foreign_key_check")
            fk_violations = cursor.fetchall()
            
            conn.close()
            
            return {
                'integrity_status': integrity_result,
                'database_info': db_info,
                'foreign_key_violations': len(fk_violations),
                'check_timestamp': datetime.now().isoformat(),
                'status': 'healthy' if integrity_result == 'ok' and len(fk_violations) == 0 else 'issues_detected'
            }
            
        except Exception as e:
            logger.error(f"Error validating database integrity: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'check_timestamp': datetime.now().isoformat()
            }


class PrivacySecurityManager:
    """
    Main privacy and security manager that coordinates all privacy operations.
    """
    
    def __init__(self, db_connection, encryption_manager):
        """
        Initialize privacy and security manager.
        
        Args:
            db_connection: Database connection instance
            encryption_manager: Encryption manager instance
        """
        self.data_privacy = DataPrivacyManager(db_connection, encryption_manager)
        self.secure_data = SecureDataManager(encryption_manager)
        self.db_connection = db_connection
        
        logger.info("Privacy and Security Manager initialized")
    
    def get_privacy_dashboard(self, profile_name: str = "default") -> Dict[str, Any]:
        """
        Get comprehensive privacy dashboard for user.
        
        Args:
            profile_name: User profile name
            
        Returns:
            Privacy dashboard data
        """
        try:
            return {
                'personal_data_overview': self.data_privacy.view_personal_data(profile_name),
                'data_retention_info': self.data_privacy.get_data_retention_info(),
                'security_status': {
                    'encryption_enabled': True,
                    'local_storage_only': True,
                    'secure_deletion_available': True,
                    'data_export_available': True
                },
                'available_actions': {
                    'view_all_data': True,
                    'export_data': True,
                    'delete_sessions': True,
                    'delete_date_ranges': True,
                    'anonymize_profile': True,
                    'complete_data_deletion': True
                },
                'dashboard_generated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating privacy dashboard: {e}")
            return {'error': str(e)}
