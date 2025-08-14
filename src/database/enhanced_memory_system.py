#!/usr/bin/env python3
"""
Enhanced Memory System for AugmentCode Private Assistant

Extends the existing three-component memory system with:
- Enhanced privacy controls and encryption
- Specialized technical knowledge storage
- Advanced context analysis for technical domains
- Secure local-only operation
- Persistent conversation memory with privacy safeguards
"""

import sqlite3
import json
import logging
import hashlib
import base64
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import threading
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Import existing memory components
try:
    from Memory import MemorySystem, ContextMemory, DatabaseMemory, LearningMemory
except ImportError:
    # Fallback for standalone usage
    MemorySystem = None

logger = logging.getLogger(__name__)


class PrivacyManager:
    """Manages privacy controls and data encryption for the memory system."""
    
    def __init__(self, privacy_config: Dict[str, Any]):
        """
        Initialize privacy manager.
        
        Args:
            privacy_config: Privacy configuration dictionary
        """
        self.config = privacy_config
        self.encryption_key = None
        self._lock = threading.Lock()
        
        if self.config.get('encrypt_local_storage', False):
            self._initialize_encryption()
        
        logger.info(f"✅ Privacy Manager initialized (Level: {self.config.get('privacy_level', 'standard')})")
    
    def _initialize_encryption(self):
        """Initialize encryption for local storage."""
        try:
            # Generate or load encryption key
            key_file = Path("Memory/Database/.encryption_key")
            
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new key
                password = b"augmentcode_private_assistant"  # In production, use user-provided password
                salt = b"augmentcode_salt_2024"  # In production, use random salt
                
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password))
                self.encryption_key = key
                
                # Save key securely
                key_file.parent.mkdir(parents=True, exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(key)
                key_file.chmod(0o600)  # Restrict permissions
                
            logger.info("🔐 Encryption initialized for local storage")
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self.encryption_key = None
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        if not self.encryption_key or not self.config.get('encrypt_local_storage', False):
            return data
        
        try:
            fernet = Fernet(self.encryption_key)
            encrypted_data = fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        if not self.encryption_key or not self.config.get('encrypt_local_storage', False):
            return encrypted_data
        
        try:
            fernet = Fernet(self.encryption_key)
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_data
    
    def sanitize_sensitive_data(self, text: str) -> str:
        """Sanitize potentially sensitive information from text."""
        if not self.config.get('sanitize_sensitive_data', False):
            return text
        
        # Patterns for sensitive data
        patterns = [
            (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_ADDRESS]'),  # IP addresses
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email addresses
            (r'\b(?:password|passwd|pwd)\s*(?:is|[:=])\s*\S+', 'password [REDACTED]'),  # Passwords
            (r'\b(?:token|key|secret)\s*[:=]\s*\S+', 'token=[REDACTED]'),  # API keys/tokens
            (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD_NUMBER]'),  # Credit card numbers
        ]
        
        sanitized_text = text
        for pattern, replacement in patterns:
            sanitized_text = re.sub(pattern, replacement, sanitized_text, flags=re.IGNORECASE)
        
        return sanitized_text
    
    def should_store_interaction(self, user_input: str, ai_response: str) -> bool:
        """Determine if an interaction should be stored based on privacy settings."""
        if self.config.get('session_only_mode', False):
            return False
        
        # Check for sensitive content that shouldn't be stored
        sensitive_keywords = ['password', 'secret', 'private key', 'token', 'credential']
        combined_text = (user_input + ' ' + ai_response).lower()
        
        if any(keyword in combined_text for keyword in sensitive_keywords):
            if self.config.get('privacy_level') == 'maximum':
                return False
        
        return True


class TechnicalKnowledgeStore:
    """Specialized storage for technical knowledge and patterns."""
    
    def __init__(self, db_path: str, privacy_manager: PrivacyManager):
        """
        Initialize technical knowledge store.
        
        Args:
            db_path: Database file path
            privacy_manager: Privacy manager instance
        """
        self.db_path = db_path
        self.privacy_manager = privacy_manager
        self._lock = threading.Lock()
        self._initialize_database()
        
        logger.info("✅ Technical Knowledge Store initialized")
    
    def _initialize_database(self):
        """Initialize the technical knowledge database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Technical patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Code snippets table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS code_snippets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    language TEXT NOT NULL,
                    snippet_type TEXT NOT NULL,
                    code_content TEXT NOT NULL,
                    description TEXT,
                    usage_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Security knowledge table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    technique TEXT NOT NULL,
                    description TEXT NOT NULL,
                    tools TEXT,
                    reference_links TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Workflow templates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workflow_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_name TEXT NOT NULL,
                    workflow_type TEXT NOT NULL,
                    template_data TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def store_technical_pattern(self, pattern_type: str, pattern_data: Dict[str, Any]) -> bool:
        """Store a technical pattern for future reference."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Encrypt sensitive data if needed
                    encrypted_data = self.privacy_manager.encrypt_data(json.dumps(pattern_data))
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO technical_patterns 
                        (pattern_type, pattern_data, frequency, last_used)
                        VALUES (?, ?, 
                            COALESCE((SELECT frequency + 1 FROM technical_patterns 
                                     WHERE pattern_type = ? AND pattern_data = ?), 1),
                            CURRENT_TIMESTAMP)
                    ''', (pattern_type, encrypted_data, pattern_type, encrypted_data))
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to store technical pattern: {e}")
            return False
    
    def get_relevant_patterns(self, query_type: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant technical patterns for a query type."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT pattern_type, pattern_data, frequency, last_used
                    FROM technical_patterns
                    WHERE pattern_type LIKE ?
                    ORDER BY frequency DESC, last_used DESC
                    LIMIT ?
                ''', (f'%{query_type}%', limit))
                
                patterns = []
                for row in cursor.fetchall():
                    try:
                        decrypted_data = self.privacy_manager.decrypt_data(row[1])
                        pattern_data = json.loads(decrypted_data)
                        patterns.append({
                            'type': row[0],
                            'data': pattern_data,
                            'frequency': row[2],
                            'last_used': row[3]
                        })
                    except Exception as e:
                        logger.warning(f"Failed to decrypt pattern data: {e}")
                        continue
                
                return patterns
                
        except Exception as e:
            logger.error(f"Failed to retrieve patterns: {e}")
            return []
    
    def store_code_snippet(self, language: str, snippet_type: str, 
                          code_content: str, description: str = None) -> bool:
        """Store a code snippet for future reference."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Sanitize and encrypt code content
                    sanitized_code = self.privacy_manager.sanitize_sensitive_data(code_content)
                    encrypted_code = self.privacy_manager.encrypt_data(sanitized_code)
                    
                    # Check if snippet already exists
                    cursor.execute('''
                        SELECT id, usage_count FROM code_snippets
                        WHERE language = ? AND snippet_type = ? AND code_content = ?
                    ''', (language, snippet_type, encrypted_code))

                    existing = cursor.fetchone()
                    if existing:
                        # Update usage count
                        cursor.execute('''
                            UPDATE code_snippets SET usage_count = usage_count + 1
                            WHERE id = ?
                        ''', (existing[0],))
                    else:
                        # Insert new snippet
                        cursor.execute('''
                            INSERT INTO code_snippets
                            (language, snippet_type, code_content, description, usage_count)
                            VALUES (?, ?, ?, ?, 1)
                        ''', (language, snippet_type, encrypted_code, description))
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to store code snippet: {e}")
            return False


class EnhancedMemorySystem:
    """
    Enhanced memory system with privacy controls and technical knowledge storage.
    
    Extends the existing MemorySystem with additional capabilities for
    private assistant functionality.
    """
    
    def __init__(self, db_path: str = "Memory/Database/enhanced_memory.db",
                 privacy_config: Dict[str, Any] = None,
                 technical_config: Dict[str, Any] = None):
        """
        Initialize enhanced memory system.
        
        Args:
            db_path: Database file path
            privacy_config: Privacy configuration
            technical_config: Technical guidance configuration
        """
        self.db_path = db_path
        self.privacy_config = privacy_config or {}
        self.technical_config = technical_config or {}
        
        # Initialize privacy manager
        self.privacy_manager = PrivacyManager(self.privacy_config)
        
        # Initialize technical knowledge store
        self.technical_store = TechnicalKnowledgeStore(db_path, self.privacy_manager)
        
        # Initialize base memory system if available
        if MemorySystem:
            self.base_memory = MemorySystem(
                db_path=db_path,
                max_context_history=self.privacy_config.get('max_context_history', 100),
                context_window=self.privacy_config.get('context_window', 20)
            )
        else:
            self.base_memory = None
            logger.warning("Base memory system not available")
        
        self.session_data = {}
        self._lock = threading.Lock()
        
        logger.info("✅ Enhanced Memory System initialized")
    
    def add_interaction(self, user_input: str, ai_response: str,
                       metadata: Dict[str, Any] = None,
                       guidance_type: str = None) -> Dict[str, Any]:
        """
        Add interaction with enhanced privacy controls and technical analysis.
        
        Args:
            user_input: User's input
            ai_response: AI's response
            metadata: Additional metadata
            guidance_type: Type of technical guidance provided
            
        Returns:
            Dictionary with storage results
        """
        try:
            with self._lock:
                # Check privacy settings
                if not self.privacy_manager.should_store_interaction(user_input, ai_response):
                    return {'stored': False, 'reason': 'privacy_policy'}
                
                # Sanitize sensitive data
                sanitized_input = self.privacy_manager.sanitize_sensitive_data(user_input)
                sanitized_response = self.privacy_manager.sanitize_sensitive_data(ai_response)
                
                results = {'stored': True, 'enhanced_features': []}
                
                # Store in base memory system if available
                if self.base_memory:
                    base_result = self.base_memory.add_interaction(
                        sanitized_input, sanitized_response, metadata=metadata
                    )
                    results['base_memory'] = base_result
                
                # Extract and store technical patterns
                if guidance_type and metadata:
                    pattern_stored = self.technical_store.store_technical_pattern(
                        guidance_type, {
                            'query': sanitized_input,
                            'response_type': metadata.get('response_type'),
                            'tools_used': metadata.get('tools_recommended'),
                            'timestamp': datetime.now().isoformat()
                        }
                    )
                    if pattern_stored:
                        results['enhanced_features'].append('technical_pattern_stored')
                
                # Store code snippets if present
                if self._contains_code(ai_response):
                    code_snippets = self._extract_code_snippets(ai_response)
                    for snippet in code_snippets:
                        self.technical_store.store_code_snippet(
                            snippet['language'],
                            snippet['type'],
                            snippet['code'],
                            snippet.get('description')
                        )
                    results['enhanced_features'].append('code_snippets_stored')
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to add interaction: {e}")
            return {'stored': False, 'error': str(e)}
    
    def get_enhanced_context(self, query: str, guidance_type: str = None) -> Dict[str, Any]:
        """
        Get enhanced context including technical patterns and knowledge.
        
        Args:
            query: Current user query
            guidance_type: Type of guidance being requested
            
        Returns:
            Enhanced context dictionary
        """
        try:
            context = {}
            
            # Get base context if available
            if self.base_memory:
                base_context = self.base_memory.get_response_context()
                context.update(base_context)
            
            # Add technical patterns
            if guidance_type:
                relevant_patterns = self.technical_store.get_relevant_patterns(guidance_type)
                context['technical_patterns'] = relevant_patterns
            
            # Add privacy status
            context['privacy_status'] = {
                'level': self.privacy_config.get('privacy_level', 'standard'),
                'encryption_enabled': self.privacy_config.get('encrypt_local_storage', False),
                'offline_only': self.privacy_config.get('offline_only_mode', False)
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get enhanced context: {e}")
            return {'error': str(e)}
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code snippets."""
        code_indicators = ['```', 'def ', 'function ', 'class ', 'import ', '#include', 'SELECT ', 'INSERT ']
        return any(indicator in text for indicator in code_indicators)
    
    def _extract_code_snippets(self, text: str) -> List[Dict[str, str]]:
        """Extract code snippets from text."""
        snippets = []
        
        # Simple extraction for code blocks
        import re
        code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', text, re.DOTALL)
        
        for language, code in code_blocks:
            snippets.append({
                'language': language or 'unknown',
                'type': 'code_block',
                'code': code.strip(),
                'description': 'Extracted from AI response'
            })
        
        return snippets
    
    def export_knowledge(self) -> Dict[str, Any]:
        """Export technical knowledge for backup."""
        try:
            return {
                'technical_patterns': self.technical_store.get_relevant_patterns('', limit=1000),
                'privacy_config': self.privacy_config,
                'export_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to export knowledge: {e}")
            return {'error': str(e)}
    
    def clear_session(self, preserve_technical_knowledge: bool = True):
        """Clear session data while optionally preserving technical knowledge."""
        try:
            with self._lock:
                if self.base_memory:
                    self.base_memory.clear_session()
                
                self.session_data.clear()
                
                if not preserve_technical_knowledge:
                    # Clear technical knowledge (implement if needed)
                    pass
                
                logger.info("Session cleared successfully")
                
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
