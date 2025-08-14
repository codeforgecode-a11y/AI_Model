#!/usr/bin/env python3
"""
Enhanced Database Schema for Personal AI Assistant

This module defines the comprehensive database schema for single-user profile
management and complete session storage capabilities.

Schema Components:
1. User Profile Management
2. Enhanced Session Storage
3. Complete Message History
4. Learning Insights Storage
5. Database Versioning Support
"""

import sqlite3
import json
import logging
import threading
import hashlib
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)


class EnhancedDatabaseSchema:
    """
    Enhanced database schema for comprehensive user profile and session management.
    """
    
    # Database version for migration support
    CURRENT_VERSION = 1
    
    @staticmethod
    def get_schema_sql() -> Dict[str, str]:
        """
        Get SQL statements for creating all enhanced tables.
        
        Returns:
            Dictionary mapping table names to CREATE TABLE SQL statements
        """
        return {
            # Database versioning table
            'schema_version': """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            """,
            
            # Enhanced user profile table
            'user_profile': """
                CREATE TABLE IF NOT EXISTS user_profile (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_name TEXT NOT NULL DEFAULT 'default',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    
                    -- AI Interaction Preferences
                    response_style TEXT DEFAULT 'balanced',  -- formal, casual, balanced, technical
                    verbosity_level TEXT DEFAULT 'moderate',  -- brief, moderate, detailed, comprehensive
                    technical_level TEXT DEFAULT 'intermediate',  -- beginner, intermediate, advanced, expert
                    explanation_preference TEXT DEFAULT 'balanced',  -- examples, theory, practical, balanced
                    
                    -- Voice Companion Settings
                    preferred_tts_engine TEXT DEFAULT 'elevenlabs',  -- elevenlabs, mozilla, system
                    voice_selection TEXT DEFAULT 'default',
                    speech_rate REAL DEFAULT 1.0,
                    speech_pitch REAL DEFAULT 1.0,
                    speech_volume REAL DEFAULT 0.8,
                    
                    -- Learning and Behavioral Preferences
                    learning_mode TEXT DEFAULT 'adaptive',  -- adaptive, conservative, aggressive
                    feedback_frequency TEXT DEFAULT 'moderate',  -- minimal, moderate, frequent
                    context_retention TEXT DEFAULT 'session',  -- none, session, persistent, full
                    privacy_level TEXT DEFAULT 'standard',  -- minimal, standard, enhanced, maximum
                    
                    -- Communication Patterns (JSON)
                    communication_patterns TEXT,  -- Learned patterns as JSON
                    interaction_history_summary TEXT,  -- Summary statistics as JSON
                    
                    -- Encrypted sensitive data
                    encrypted_data TEXT,  -- Encrypted sensitive preferences
                    
                    UNIQUE(profile_name)
                )
            """,
            
            # Enhanced sessions table with comprehensive metadata
            'sessions': """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    profile_id INTEGER,
                    
                    -- Session metadata
                    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    ended_at TEXT,
                    duration_seconds INTEGER,
                    
                    -- Session characteristics
                    primary_topic TEXT,
                    topic_categories TEXT,  -- JSON array of topics discussed
                    interaction_count INTEGER DEFAULT 0,
                    
                    -- Session quality metrics
                    user_satisfaction_score REAL,
                    session_effectiveness REAL,
                    goal_completion_status TEXT,  -- completed, partial, abandoned
                    
                    -- Key decisions and outcomes
                    key_decisions TEXT,  -- JSON array of important decisions made
                    action_items TEXT,  -- JSON array of follow-up actions
                    session_summary TEXT,
                    
                    -- Context and environment
                    session_context TEXT,  -- JSON with environmental context
                    device_info TEXT,  -- JSON with device/platform info
                    
                    FOREIGN KEY (profile_id) REFERENCES user_profile (id)
                )
            """,
            
            # Enhanced messages table with full context relationships
            'messages': """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_id TEXT UNIQUE NOT NULL,
                    
                    -- Message content
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    message_type TEXT NOT NULL,  -- user_input, ai_response, system_message
                    content TEXT NOT NULL,
                    content_hash TEXT,  -- For deduplication
                    
                    -- Message context and relationships
                    parent_message_id TEXT,  -- For threading
                    conversation_turn INTEGER,  -- Sequential turn number
                    topic TEXT,
                    intent TEXT,  -- Detected user intent
                    
                    -- Message metadata
                    metadata TEXT,  -- JSON with additional metadata
                    processing_time_ms INTEGER,
                    confidence_score REAL,
                    
                    -- User feedback and ratings
                    user_rating INTEGER,  -- 1-5 rating
                    user_feedback TEXT,
                    feedback_timestamp TEXT,
                    
                    -- AI response characteristics
                    response_length INTEGER,
                    response_complexity REAL,
                    sources_used TEXT,  -- JSON array of sources
                    
                    -- Context preservation
                    context_snapshot TEXT,  -- JSON snapshot of context at message time
                    
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id),
                    FOREIGN KEY (parent_message_id) REFERENCES messages (message_id)
                )
            """,
            
            # Learning insights from conversation history
            'learning_insights': """
                CREATE TABLE IF NOT EXISTS learning_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight_id TEXT UNIQUE NOT NULL,
                    
                    -- Insight metadata
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    insight_type TEXT NOT NULL,  -- pattern, preference, behavior, improvement
                    confidence_level REAL DEFAULT 0.5,
                    
                    -- Insight content
                    title TEXT NOT NULL,
                    description TEXT,
                    insight_data TEXT,  -- JSON with detailed insight data
                    
                    -- Source information
                    source_sessions TEXT,  -- JSON array of session IDs that contributed
                    source_messages TEXT,  -- JSON array of message IDs that contributed
                    analysis_method TEXT,  -- How this insight was derived
                    
                    -- Application and impact
                    applied_at TEXT,
                    impact_score REAL,
                    validation_status TEXT DEFAULT 'pending',  -- pending, validated, rejected
                    
                    -- Temporal relevance
                    relevance_start TEXT,
                    relevance_end TEXT,
                    decay_rate REAL DEFAULT 0.1
                )
            """,
            
            # Enhanced conversation topics with relationships
            'conversation_topics': """
                CREATE TABLE IF NOT EXISTS conversation_topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_name TEXT UNIQUE NOT NULL,
                    
                    -- Topic metadata
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_discussed TEXT,
                    discussion_frequency INTEGER DEFAULT 1,
                    
                    -- Topic characteristics
                    category TEXT,
                    subcategory TEXT,
                    complexity_level TEXT,
                    user_expertise_level TEXT,
                    
                    -- Topic relationships
                    related_topics TEXT,  -- JSON array of related topic names
                    parent_topic TEXT,
                    
                    -- User preferences for this topic
                    preferred_detail_level TEXT,
                    preferred_explanation_style TEXT,
                    
                    -- Topic evolution
                    evolution_history TEXT  -- JSON tracking how topic discussions evolved
                )
            """,
            
            # System backups and data integrity
            'system_backups': """
                CREATE TABLE IF NOT EXISTS system_backups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backup_id TEXT UNIQUE NOT NULL,
                    
                    -- Backup metadata
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    backup_type TEXT NOT NULL,  -- full, incremental, profile_only
                    backup_size_bytes INTEGER,
                    
                    -- Backup content
                    backup_path TEXT,
                    backup_hash TEXT,  -- For integrity verification
                    compression_used TEXT,
                    encryption_used BOOLEAN DEFAULT 0,
                    
                    -- Backup status
                    status TEXT DEFAULT 'completed',  -- in_progress, completed, failed
                    verification_status TEXT,  -- verified, failed, pending
                    
                    -- Retention information
                    retention_policy TEXT,
                    expires_at TEXT
                )
            """
        }
    
    @staticmethod
    def get_indexes_sql() -> List[str]:
        """
        Get SQL statements for creating performance indexes.
        
        Returns:
            List of CREATE INDEX SQL statements
        """
        return [
            # User profile indexes
            "CREATE INDEX IF NOT EXISTS idx_user_profile_name ON user_profile(profile_name)",
            "CREATE INDEX IF NOT EXISTS idx_user_profile_updated ON user_profile(updated_at)",
            
            # Session indexes
            "CREATE INDEX IF NOT EXISTS idx_sessions_id ON sessions(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_profile ON sessions(profile_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_topic ON sessions(primary_topic)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_ended ON sessions(ended_at)",
            
            # Message indexes
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_messages_type ON messages(message_type)",
            "CREATE INDEX IF NOT EXISTS idx_messages_topic ON messages(topic)",
            "CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_message_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_turn ON messages(conversation_turn)",
            "CREATE INDEX IF NOT EXISTS idx_messages_hash ON messages(content_hash)",
            
            # Learning insights indexes
            "CREATE INDEX IF NOT EXISTS idx_insights_type ON learning_insights(insight_type)",
            "CREATE INDEX IF NOT EXISTS idx_insights_created ON learning_insights(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_insights_confidence ON learning_insights(confidence_level)",
            "CREATE INDEX IF NOT EXISTS idx_insights_validation ON learning_insights(validation_status)",
            
            # Topic indexes
            "CREATE INDEX IF NOT EXISTS idx_topics_name ON conversation_topics(topic_name)",
            "CREATE INDEX IF NOT EXISTS idx_topics_category ON conversation_topics(category)",
            "CREATE INDEX IF NOT EXISTS idx_topics_last_discussed ON conversation_topics(last_discussed)",
            
            # Backup indexes
            "CREATE INDEX IF NOT EXISTS idx_backups_created ON system_backups(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_backups_type ON system_backups(backup_type)",
            "CREATE INDEX IF NOT EXISTS idx_backups_status ON system_backups(status)"
        ]
    
    @staticmethod
    def get_migration_sql(from_version: int, to_version: int) -> List[str]:
        """
        Get SQL statements for migrating between schema versions.
        
        Args:
            from_version: Current schema version
            to_version: Target schema version
            
        Returns:
            List of SQL statements for migration
        """
        migrations = []
        
        # Migration from version 0 (legacy) to version 1
        if from_version == 0 and to_version >= 1:
            migrations.extend([
                # Add new columns to existing conversations table if it exists
                "ALTER TABLE conversations ADD COLUMN message_id TEXT",
                "ALTER TABLE conversations ADD COLUMN parent_message_id TEXT",
                "ALTER TABLE conversations ADD COLUMN conversation_turn INTEGER",
                "ALTER TABLE conversations ADD COLUMN content_hash TEXT",
                "ALTER TABLE conversations ADD COLUMN user_rating INTEGER",
                "ALTER TABLE conversations ADD COLUMN user_feedback TEXT",
                "ALTER TABLE conversations ADD COLUMN feedback_timestamp TEXT",
                "ALTER TABLE conversations ADD COLUMN context_snapshot TEXT",
                
                # Update schema version
                "INSERT OR REPLACE INTO schema_version (version, description) VALUES (1, 'Enhanced schema with user profiles and session management')"
            ])
        
        return migrations


class EncryptionManager:
    """
    Manages encryption for sensitive user data.
    """
    
    def __init__(self, key_file: str = "Memory/Database/.encryption_key"):
        """
        Initialize encryption manager.
        
        Args:
            key_file: Path to encryption key file
        """
        self.key_file = Path(key_file)
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        self._key = self._load_or_create_key()
        self._cipher = Fernet(self._key)
    
    def _load_or_create_key(self) -> bytes:
        """Load existing key or create new one."""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions
            os.chmod(self.key_file, 0o600)
            return key
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data."""
        if not data:
            return ""
        encrypted = self._cipher.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        if not encrypted_data:
            return ""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self._cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return ""
