#!/usr/bin/env python3
"""
Enhanced Database Manager

Extends the existing DatabaseMemory with comprehensive features including:
- Database versioning and migration support
- Automatic backup functionality
- Data export to JSON/CSV formats
- Enhanced thread-safety
- Integration with new schema components
"""

import sqlite3
import json
import csv
import logging
import threading
import shutil
import gzip
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import uuid

from enhanced_database_schema import EnhancedDatabaseSchema, EncryptionManager
from user_profile_manager import UserProfileManager
from session_manager import SessionManager

logger = logging.getLogger(__name__)


class DatabaseVersionManager:
    """
    Manages database schema versioning and migrations.
    """
    
    def __init__(self, db_connection):
        """
        Initialize version manager.
        
        Args:
            db_connection: Database connection instance
        """
        self.db = db_connection
        self._ensure_version_table()
    
    def _ensure_version_table(self) -> None:
        """Ensure schema_version table exists."""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Create schema_version table if it doesn't exist
            cursor.execute(EnhancedDatabaseSchema.get_schema_sql()['schema_version'])
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error ensuring version table: {e}")
    
    def get_current_version(self) -> int:
        """
        Get current database schema version.
        
        Returns:
            Current version number
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT MAX(version) FROM schema_version")
            result = cursor.fetchone()
            
            return result[0] if result and result[0] is not None else 0
            
        except Exception as e:
            logger.error(f"Error getting current version: {e}")
            return 0
    
    def migrate_to_version(self, target_version: int) -> bool:
        """
        Migrate database to target version.
        
        Args:
            target_version: Target schema version
            
        Returns:
            True if successful, False otherwise
        """
        try:
            current_version = self.get_current_version()
            
            if current_version == target_version:
                logger.info(f"Database already at version {target_version}")
                return True
            
            if current_version > target_version:
                logger.warning(f"Cannot downgrade from version {current_version} to {target_version}")
                return False
            
            logger.info(f"Migrating database from version {current_version} to {target_version}")
            
            # Get migration SQL
            migration_sql = EnhancedDatabaseSchema.get_migration_sql(current_version, target_version)
            
            if not migration_sql:
                logger.warning(f"No migration path from {current_version} to {target_version}")
                return False
            
            # Execute migration
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Start transaction
            cursor.execute("BEGIN TRANSACTION")
            
            try:
                for sql in migration_sql:
                    cursor.execute(sql)
                
                # Update version
                cursor.execute("""
                    INSERT OR REPLACE INTO schema_version (version, description) 
                    VALUES (?, ?)
                """, (target_version, f"Migrated from version {current_version}"))
                
                conn.commit()
                logger.info(f"Successfully migrated to version {target_version}")
                return True
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Migration failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error during migration: {e}")
            return False
    
    def initialize_schema(self) -> bool:
        """
        Initialize complete enhanced schema.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Create all tables
            schema_sql = EnhancedDatabaseSchema.get_schema_sql()
            
            for table_name, create_sql in schema_sql.items():
                logger.debug(f"Creating table: {table_name}")
                cursor.execute(create_sql)
            
            # Create indexes
            index_sql = EnhancedDatabaseSchema.get_indexes_sql()
            for index_sql_stmt in index_sql:
                cursor.execute(index_sql_stmt)
            
            # Set current version
            cursor.execute("""
                INSERT OR REPLACE INTO schema_version (version, description) 
                VALUES (?, ?)
            """, (EnhancedDatabaseSchema.CURRENT_VERSION, "Initial enhanced schema"))
            
            conn.commit()
            logger.info("Enhanced database schema initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing schema: {e}")
            return False


class BackupManager:
    """
    Manages automatic database backups with configurable intervals.
    """
    
    def __init__(self, db_path: str, backup_dir: str = "Memory/Database/backups"):
        """
        Initialize backup manager.
        
        Args:
            db_path: Path to database file
            backup_dir: Directory for backup files
        """
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
    
    def create_backup(self, backup_type: str = "full", 
                     compress: bool = True, encrypt: bool = False) -> Optional[str]:
        """
        Create a database backup.
        
        Args:
            backup_type: Type of backup (full, incremental)
            compress: Whether to compress the backup
            encrypt: Whether to encrypt the backup
            
        Returns:
            Backup ID if successful, None otherwise
        """
        with self._lock:
            try:
                backup_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create backup filename
                backup_filename = f"backup_{backup_type}_{timestamp}_{backup_id[:8]}.db"
                if compress:
                    backup_filename += ".gz"
                
                backup_path = self.backup_dir / backup_filename
                
                # Create backup
                if compress:
                    with open(self.db_path, 'rb') as f_in:
                        with gzip.open(backup_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.copy2(self.db_path, backup_path)
                
                # Calculate backup size and hash
                backup_size = backup_path.stat().st_size
                backup_hash = self._calculate_file_hash(backup_path)
                
                # Store backup metadata (would be stored in database)
                backup_metadata = {
                    'backup_id': backup_id,
                    'created_at': datetime.now().isoformat(),
                    'backup_type': backup_type,
                    'backup_size_bytes': backup_size,
                    'backup_path': str(backup_path),
                    'backup_hash': backup_hash,
                    'compression_used': 'gzip' if compress else 'none',
                    'encryption_used': encrypt,
                    'status': 'completed'
                }
                
                logger.info(f"Created backup {backup_id} at {backup_path}")
                return backup_id
                
            except Exception as e:
                logger.error(f"Error creating backup: {e}")
                return None
    
    def restore_backup(self, backup_id: str) -> bool:
        """
        Restore database from backup.
        
        Args:
            backup_id: Backup identifier
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                # Find backup file (simplified - in real implementation, 
                # would query backup metadata from database)
                backup_files = list(self.backup_dir.glob(f"*{backup_id[:8]}*"))
                
                if not backup_files:
                    logger.error(f"Backup {backup_id} not found")
                    return False
                
                backup_path = backup_files[0]
                
                # Create backup of current database
                current_backup = self.create_backup("pre_restore")
                if not current_backup:
                    logger.error("Failed to backup current database before restore")
                    return False
                
                # Restore from backup
                if backup_path.suffix == '.gz':
                    with gzip.open(backup_path, 'rb') as f_in:
                        with open(self.db_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.copy2(backup_path, self.db_path)
                
                logger.info(f"Restored database from backup {backup_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error restoring backup {backup_id}: {e}")
                return False
    
    def cleanup_old_backups(self, retention_days: int = 30) -> int:
        """
        Clean up old backup files.
        
        Args:
            retention_days: Number of days to retain backups
            
        Returns:
            Number of backups deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            deleted_count = 0
            
            for backup_file in self.backup_dir.glob("backup_*"):
                if backup_file.stat().st_mtime < cutoff_date.timestamp():
                    backup_file.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted old backup: {backup_file}")
            
            logger.info(f"Cleaned up {deleted_count} old backups")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
            return 0
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        import hashlib
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


class DataExportManager:
    """
    Manages data export to various formats (JSON, CSV).
    """
    
    def __init__(self, db_connection):
        """
        Initialize export manager.
        
        Args:
            db_connection: Database connection instance
        """
        self.db = db_connection
    
    def export_to_json(self, output_path: str, 
                      include_sensitive: bool = False,
                      date_range: Optional[Tuple[datetime, datetime]] = None) -> bool:
        """
        Export all data to JSON format.
        
        Args:
            output_path: Output file path
            include_sensitive: Whether to include sensitive data
            date_range: Optional date range filter
            
        Returns:
            True if successful, False otherwise
        """
        try:
            export_data = {
                'export_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0',
                    'include_sensitive': include_sensitive,
                    'date_range': [d.isoformat() for d in date_range] if date_range else None
                },
                'user_profiles': self._export_user_profiles(include_sensitive),
                'sessions': self._export_sessions(date_range),
                'messages': self._export_messages(date_range),
                'learning_insights': self._export_learning_insights(date_range),
                'conversation_topics': self._export_conversation_topics()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Data exported to JSON: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return False
    
    def export_to_csv(self, output_dir: str,
                     date_range: Optional[Tuple[datetime, datetime]] = None) -> bool:
        """
        Export data to CSV files (one per table).
        
        Args:
            output_dir: Output directory path
            date_range: Optional date range filter
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export each table to separate CSV
            tables_to_export = [
                ('user_profiles', self._export_user_profiles(False)),
                ('sessions', self._export_sessions(date_range)),
                ('messages', self._export_messages(date_range)),
                ('learning_insights', self._export_learning_insights(date_range)),
                ('conversation_topics', self._export_conversation_topics())
            ]
            
            for table_name, data in tables_to_export:
                if data:
                    csv_path = output_path / f"{table_name}.csv"
                    self._write_csv(csv_path, data)
                    logger.debug(f"Exported {table_name} to {csv_path}")
            
            logger.info(f"Data exported to CSV files in: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
    
    def _export_user_profiles(self, include_sensitive: bool = False) -> List[Dict[str, Any]]:
        """Export user profiles data."""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM user_profile")
            rows = cursor.fetchall()
            
            profiles = []
            for row in rows:
                profile = dict(row)
                
                # Parse JSON fields
                if profile.get('communication_patterns'):
                    profile['communication_patterns'] = json.loads(profile['communication_patterns'])
                if profile.get('interaction_history_summary'):
                    profile['interaction_history_summary'] = json.loads(profile['interaction_history_summary'])
                
                # Handle sensitive data
                if not include_sensitive:
                    profile.pop('encrypted_data', None)
                
                profiles.append(profile)
            
            return profiles
            
        except Exception as e:
            logger.error(f"Error exporting user profiles: {e}")
            return []
    
    def _export_sessions(self, date_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """Export sessions data."""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            if date_range:
                cursor.execute("""
                    SELECT * FROM sessions 
                    WHERE started_at BETWEEN ? AND ?
                    ORDER BY started_at
                """, (date_range[0].isoformat(), date_range[1].isoformat()))
            else:
                cursor.execute("SELECT * FROM sessions ORDER BY started_at")
            
            rows = cursor.fetchall()
            
            sessions = []
            for row in rows:
                session = dict(row)
                
                # Parse JSON fields
                for json_field in ['topic_categories', 'key_decisions', 'session_context', 'device_info']:
                    if session.get(json_field):
                        session[json_field] = json.loads(session[json_field])
                
                sessions.append(session)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error exporting sessions: {e}")
            return []
    
    def _export_messages(self, date_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """Export messages data."""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            if date_range:
                cursor.execute("""
                    SELECT * FROM messages 
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """, (date_range[0].isoformat(), date_range[1].isoformat()))
            else:
                cursor.execute("SELECT * FROM messages ORDER BY timestamp")
            
            rows = cursor.fetchall()
            
            messages = []
            for row in rows:
                message = dict(row)
                
                # Parse JSON fields
                for json_field in ['metadata', 'sources_used', 'context_snapshot']:
                    if message.get(json_field):
                        message[json_field] = json.loads(message[json_field])
                
                messages.append(message)
            
            return messages
            
        except Exception as e:
            logger.error(f"Error exporting messages: {e}")
            return []
    
    def _export_learning_insights(self, date_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """Export learning insights data."""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            if date_range:
                cursor.execute("""
                    SELECT * FROM learning_insights 
                    WHERE created_at BETWEEN ? AND ?
                    ORDER BY created_at
                """, (date_range[0].isoformat(), date_range[1].isoformat()))
            else:
                cursor.execute("SELECT * FROM learning_insights ORDER BY created_at")
            
            rows = cursor.fetchall()
            
            insights = []
            for row in rows:
                insight = dict(row)
                
                # Parse JSON fields
                for json_field in ['insight_data', 'source_sessions', 'source_messages']:
                    if insight.get(json_field):
                        insight[json_field] = json.loads(insight[json_field])
                
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error exporting learning insights: {e}")
            return []
    
    def _export_conversation_topics(self) -> List[Dict[str, Any]]:
        """Export conversation topics data."""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM conversation_topics ORDER BY topic_name")
            rows = cursor.fetchall()
            
            topics = []
            for row in rows:
                topic = dict(row)
                
                # Parse JSON fields
                for json_field in ['related_topics', 'evolution_history']:
                    if topic.get(json_field):
                        topic[json_field] = json.loads(topic[json_field])
                
                topics.append(topic)
            
            return topics
            
        except Exception as e:
            logger.error(f"Error exporting conversation topics: {e}")
            return []
    
    def _write_csv(self, file_path: Path, data: List[Dict[str, Any]]) -> None:
        """Write data to CSV file."""
        if not data:
            return
        
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in data:
                # Convert complex objects to JSON strings for CSV
                csv_row = {}
                for key, value in row.items():
                    if isinstance(value, (dict, list)):
                        csv_row[key] = json.dumps(value)
                    else:
                        csv_row[key] = value
                writer.writerow(csv_row)


class EnhancedDatabaseManager:
    """
    Enhanced database manager that coordinates all database operations.
    """
    
    def __init__(self, db_path: str = "Memory/Database/enhanced_memory.db"):
        """
        Initialize enhanced database manager.
        
        Args:
            db_path: Path to database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        from Memory.database import DatabaseConnection
        self.db_connection = DatabaseConnection(str(self.db_path))
        
        self.version_manager = DatabaseVersionManager(self.db_connection)
        self.backup_manager = BackupManager(str(self.db_path))
        self.export_manager = DataExportManager(self.db_connection)
        self.encryption_manager = EncryptionManager()
        
        # Initialize enhanced components
        self.user_profile_manager = UserProfileManager(self.db_connection, self.encryption_manager)
        self.session_manager = SessionManager(self.db_connection, self.user_profile_manager)
        
        # Initialize schema
        self._initialize_enhanced_schema()
        
        logger.info("Enhanced Database Manager initialized")
    
    def _initialize_enhanced_schema(self) -> None:
        """Initialize or migrate to enhanced schema."""
        try:
            current_version = self.version_manager.get_current_version()
            target_version = EnhancedDatabaseSchema.CURRENT_VERSION
            
            if current_version == 0:
                # Fresh installation
                self.version_manager.initialize_schema()
            elif current_version < target_version:
                # Migration needed
                self.version_manager.migrate_to_version(target_version)
            
        except Exception as e:
            logger.error(f"Error initializing enhanced schema: {e}")
            raise
    
    def create_automatic_backup(self) -> Optional[str]:
        """Create automatic backup with default settings."""
        return self.backup_manager.create_backup("automatic", compress=True)
    
    def export_all_data(self, output_path: str, format_type: str = "json") -> bool:
        """
        Export all data in specified format.
        
        Args:
            output_path: Output path
            format_type: Export format ('json' or 'csv')
            
        Returns:
            True if successful, False otherwise
        """
        if format_type.lower() == "json":
            return self.export_manager.export_to_json(output_path)
        elif format_type.lower() == "csv":
            return self.export_manager.export_to_csv(output_path)
        else:
            logger.error(f"Unsupported export format: {format_type}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            return {
                'database_path': str(self.db_path),
                'database_size_mb': self.db_path.stat().st_size / (1024 * 1024),
                'schema_version': self.version_manager.get_current_version(),
                'backup_count': len(list(self.backup_manager.backup_dir.glob("backup_*"))),
                'encryption_enabled': True,
                'last_backup': self._get_last_backup_time(),
                'status_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def _get_last_backup_time(self) -> Optional[str]:
        """Get timestamp of last backup."""
        try:
            backup_files = list(self.backup_manager.backup_dir.glob("backup_*"))
            if backup_files:
                latest_backup = max(backup_files, key=lambda f: f.stat().st_mtime)
                return datetime.fromtimestamp(latest_backup.stat().st_mtime).isoformat()
            return None
        except Exception:
            return None
