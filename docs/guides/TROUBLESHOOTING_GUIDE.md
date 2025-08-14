# Enhanced Knowledge System - Troubleshooting Guide

This guide helps resolve common issues with the Enhanced Intelligent Knowledge System.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Database Problems](#database-problems)
3. [Performance Issues](#performance-issues)
4. [Memory and Storage](#memory-and-storage)
5. [Privacy and Security](#privacy-and-security)
6. [Integration Problems](#integration-problems)
7. [Error Messages](#error-messages)
8. [Debugging Tools](#debugging-tools)

## Installation Issues

### Missing Dependencies

**Problem**: Import errors when starting the system

```
ImportError: No module named 'cryptography'
ModuleNotFoundError: No module named 'enhanced_database_manager'
```

**Solution**:
```bash
# Install required dependencies
pip install cryptography pathlib sqlite3

# Ensure all system files are in the same directory
ls -la *.py
# Should show: intelligent_knowledge_system.py, enhanced_database_manager.py, etc.
```

### Python Version Compatibility

**Problem**: Syntax errors or unexpected behavior

**Solution**:
```bash
# Check Python version (requires 3.7+)
python --version

# Use virtual environment for isolation
python -m venv enhanced_assistant_env
source enhanced_assistant_env/bin/activate  # Linux/Mac
# or
enhanced_assistant_env\Scripts\activate  # Windows

pip install -r requirements.txt
```

### File Permissions

**Problem**: Cannot create database or backup files

```
PermissionError: [Errno 13] Permission denied: 'Memory/Database/enhanced_memory.db'
```

**Solution**:
```bash
# Create directories with proper permissions
mkdir -p Memory/Database
chmod 755 Memory/Database

# Or specify a different path
assistant = create_personal_ai_assistant(
    config={'db_path': '/home/user/assistant_data/memory.db'}
)
```

## Database Problems

### Database Lock Errors

**Problem**: Database is locked errors during operations

```
sqlite3.OperationalError: database is locked
```

**Solution**:
```python
# Ensure proper connection management
try:
    # Your database operations
    result = assistant.process_query("test query")
finally:
    # Always end sessions properly
    if assistant.current_session_id:
        assistant.end_session()

# Check for zombie processes
import psutil
for proc in psutil.process_iter(['pid', 'name']):
    if 'python' in proc.info['name']:
        print(f"Python process: {proc.info}")
```

### Database Corruption

**Problem**: Database integrity check fails

```python
# Check database integrity
status = assistant.enhanced_db.version_manager.validate_data_integrity()
if status['status'] != 'healthy':
    print(f"Database issues detected: {status}")
```

**Solution**:
```python
# Restore from backup
backup_manager = assistant.enhanced_db.backup_manager
backups = list(backup_manager.backup_dir.glob("backup_*"))

if backups:
    latest_backup = max(backups, key=lambda f: f.stat().st_mtime)
    backup_id = latest_backup.name.split('_')[-1].split('.')[0]
    
    success = backup_manager.restore_backup(backup_id)
    print(f"Restore successful: {success}")
else:
    print("No backups available - manual recovery needed")
```

### Schema Migration Issues

**Problem**: Database schema version mismatch

**Solution**:
```python
# Check current version
current_version = assistant.enhanced_db.version_manager.get_current_version()
target_version = EnhancedDatabaseSchema.CURRENT_VERSION

print(f"Current: {current_version}, Target: {target_version}")

# Force migration
if current_version < target_version:
    success = assistant.enhanced_db.version_manager.migrate_to_version(target_version)
    print(f"Migration successful: {success}")

# If migration fails, backup and reinitialize
if not success:
    # Export existing data
    assistant.export_user_data("backup_before_reset.json")
    
    # Reinitialize schema
    assistant.enhanced_db.version_manager.initialize_schema()
```

## Performance Issues

### Slow Query Processing

**Problem**: Queries take too long to process

**Diagnostic**:
```python
import time

start_time = time.time()
result = assistant.process_query("test query")
processing_time = time.time() - start_time

print(f"Query processing time: {processing_time:.2f} seconds")

# Check database size
status = assistant.get_system_status()
print(f"Database size: {status['database_size_mb']:.2f} MB")
```

**Solutions**:
```python
# 1. Clean up old data
from datetime import datetime, timedelta

# Delete data older than 6 months
cutoff_date = datetime.now() - timedelta(days=180)
deletion_result = assistant.delete_date_range_data(
    datetime.min, cutoff_date, secure_delete=True
)
print(f"Cleaned up: {deletion_result}")

# 2. Optimize database
conn = assistant.enhanced_db.db_connection.get_connection()
cursor = conn.cursor()
cursor.execute("VACUUM")
cursor.execute("ANALYZE")
conn.commit()

# 3. Reduce cache sizes
assistant.session_manager._message_cache.clear()
```

### Memory Usage Issues

**Problem**: High memory consumption

**Diagnostic**:
```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

# Check cache sizes
print(f"Message cache size: {len(assistant.session_manager._message_cache)}")
print(f"Profile cache size: {len(assistant.user_profile_manager._profile_cache)}")
```

**Solutions**:
```python
# Clear caches periodically
assistant.session_manager._message_cache.clear()
assistant.user_profile_manager._profile_cache.clear()

# Reduce context history
assistant.memory_system.context_memory.conversation.max_history = 25

# End sessions promptly
if assistant.current_session_id:
    assistant.end_session()
```

### Search Performance

**Problem**: Conversation history search is slow

**Solution**:
```python
# Check database indexes
conn = assistant.enhanced_db.db_connection.get_connection()
cursor = conn.cursor()

# Verify indexes exist
cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
indexes = cursor.fetchall()
print(f"Available indexes: {[idx[0] for idx in indexes]}")

# Rebuild indexes if needed
index_sql = EnhancedDatabaseSchema.get_indexes_sql()
for sql in index_sql:
    cursor.execute(sql)
conn.commit()

# Use more specific search terms
results = assistant.search_conversation_history(
    "specific keyword",
    topic_filter="programming"  # Add topic filter
)
```

## Memory and Storage

### Disk Space Issues

**Problem**: Running out of disk space

**Diagnostic**:
```python
import shutil

# Check available space
total, used, free = shutil.disk_usage(assistant.enhanced_db.db_path.parent)
print(f"Free space: {free / 1024 / 1024 / 1024:.2f} GB")

# Check database size
db_size = assistant.enhanced_db.db_path.stat().st_size
print(f"Database size: {db_size / 1024 / 1024:.2f} MB")

# Check backup sizes
backup_dir = assistant.enhanced_db.backup_manager.backup_dir
if backup_dir.exists():
    backup_sizes = sum(f.stat().st_size for f in backup_dir.glob("*"))
    print(f"Backup total size: {backup_sizes / 1024 / 1024:.2f} MB")
```

**Solutions**:
```python
# 1. Clean up old backups
deleted_count = assistant.enhanced_db.backup_manager.cleanup_old_backups(
    retention_days=30
)
print(f"Deleted {deleted_count} old backups")

# 2. Compress database
conn = assistant.enhanced_db.db_connection.get_connection()
cursor = conn.cursor()
cursor.execute("VACUUM")
conn.commit()

# 3. Archive old sessions
old_sessions = assistant.search_conversation_history(
    "",
    date_range=(datetime.min, datetime.now() - timedelta(days=365))
)

if old_sessions:
    # Export before deletion
    assistant.export_user_data("archived_sessions.json")
    
    # Delete old sessions
    for session in old_sessions:
        assistant.delete_session_data(session['session_id'])
```

### Backup Issues

**Problem**: Backup creation fails

**Solution**:
```python
# Check backup directory permissions
backup_dir = assistant.enhanced_db.backup_manager.backup_dir
backup_dir.mkdir(parents=True, exist_ok=True)

# Manual backup
import shutil
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = backup_dir / f"manual_backup_{timestamp}.db"

try:
    shutil.copy2(assistant.enhanced_db.db_path, backup_path)
    print(f"Manual backup created: {backup_path}")
except Exception as e:
    print(f"Backup failed: {e}")

# Test backup integrity
if backup_path.exists():
    test_conn = sqlite3.connect(backup_path)
    cursor = test_conn.cursor()
    cursor.execute("PRAGMA integrity_check")
    result = cursor.fetchone()[0]
    test_conn.close()
    print(f"Backup integrity: {result}")
```

## Privacy and Security

### Encryption Issues

**Problem**: Cannot decrypt sensitive data

```python
# Test encryption functionality
from enhanced_database_schema import EncryptionManager

encryption = EncryptionManager()
test_data = "test sensitive information"

try:
    encrypted = encryption.encrypt(test_data)
    decrypted = encryption.decrypt(encrypted)
    
    if decrypted == test_data:
        print("Encryption working correctly")
    else:
        print("Encryption/decryption mismatch")
        
except Exception as e:
    print(f"Encryption error: {e}")
    
    # Regenerate encryption key (WARNING: loses existing encrypted data)
    key_file = Path("Memory/Database/.encryption_key")
    if key_file.exists():
        key_file.unlink()
    
    # Create new encryption manager
    new_encryption = EncryptionManager()
    print("New encryption key generated")
```

### Data Export Issues

**Problem**: GDPR export fails or incomplete

**Solution**:
```python
# Check export permissions
export_dir = Path("exports")
export_dir.mkdir(exist_ok=True)

# Test basic export
try:
    export_path = assistant.export_user_data("test_export.json", "json")
    if export_path:
        print(f"Export successful: {export_path}")
        
        # Verify export content
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        required_sections = ['user_profile', 'sessions', 'export_metadata']
        missing_sections = [s for s in required_sections if s not in export_data]
        
        if missing_sections:
            print(f"Missing sections: {missing_sections}")
        else:
            print("Export complete and valid")
    
except Exception as e:
    print(f"Export failed: {e}")
    
    # Manual export
    manual_export = {
        'user_profile': assistant.user_profile_manager.export_profile(),
        'sessions': assistant.get_session_history(1000),
        'privacy_info': assistant.get_data_retention_info()
    }
    
    with open("manual_export.json", 'w') as f:
        json.dump(manual_export, f, indent=2, default=str)
```

## Integration Problems

### Voice Integration Issues

**Problem**: TTS or speech recognition not working

**Solution**:
```python
# Test TTS preferences
tts_engine = assistant.get_user_preference('preferred_tts_engine', 'system')
print(f"Configured TTS engine: {tts_engine}")

# Test system TTS fallback
try:
    import pyttsx3
    engine = pyttsx3.init()
    engine.say("Test speech synthesis")
    engine.runAndWait()
    print("System TTS working")
except Exception as e:
    print(f"System TTS error: {e}")

# Test speech recognition
try:
    import speech_recognition as sr
    r = sr.Recognizer()
    print("Speech recognition available")
except ImportError:
    print("Install speech_recognition: pip install SpeechRecognition")
```

### Web Search Integration

**Problem**: Web search not working

**Solution**:
```python
# Test web search configuration
search_config = assistant.config.get('search_config', {})
print(f"Search config: {search_config}")

# Test basic web connectivity
import requests

try:
    response = requests.get("https://httpbin.org/get", timeout=5)
    print(f"Web connectivity: OK ({response.status_code})")
except Exception as e:
    print(f"Web connectivity issue: {e}")

# Mock web search for testing
def mock_web_search(query, num_results=5):
    return [{
        'title': f'Mock result for: {query}',
        'url': 'https://example.com',
        'snippet': f'Mock information about {query}'
    }]

assistant.web_integrator.set_web_search_function(mock_web_search)
```

## Error Messages

### Common Error Patterns

#### "No active session" errors

```python
# Always ensure session is active
if not assistant.current_session_id:
    assistant.start_session()

# Or check before operations
def safe_add_feedback(assistant, feedback, rating):
    if assistant.current_session_id:
        return assistant.add_user_feedback(feedback, rating)
    else:
        print("No active session for feedback")
        return False
```

#### "Profile not found" errors

```python
# Verify profile exists
profile = assistant.user_profile_manager.get_profile(assistant.profile_name)
if not profile:
    print(f"Creating profile: {assistant.profile_name}")
    assistant.user_profile_manager.create_profile(assistant.profile_name)
```

#### JSON parsing errors

```python
# Handle corrupted JSON data
import json

def safe_json_parse(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return {}

# Use in database operations
def get_session_metadata_safe(session_id):
    session_data = assistant.session_manager.get_session_metadata(session_id)
    if session_data:
        # Safely parse JSON fields
        session_data['topic_categories'] = safe_json_parse(
            session_data.get('topic_categories', '[]')
        )
    return session_data
```

## Debugging Tools

### Enable Debug Logging

```python
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Enable SQL query logging
logging.getLogger('sqlite3').setLevel(logging.DEBUG)
```

### System Diagnostics

```python
def run_system_diagnostics(assistant):
    """Run comprehensive system diagnostics."""
    
    print("=== System Diagnostics ===")
    
    # 1. Basic system status
    status = assistant.get_system_status()
    print(f"Database size: {status.get('database_size_mb', 0):.2f} MB")
    print(f"Schema version: {status.get('schema_version', 'unknown')}")
    print(f"Active session: {status.get('active_session', False)}")
    
    # 2. Database integrity
    integrity = assistant.enhanced_db.version_manager.validate_data_integrity()
    print(f"Database integrity: {integrity['status']}")
    
    # 3. Component status
    components = {
        'user_profile_manager': assistant.user_profile_manager,
        'session_manager': assistant.session_manager,
        'privacy_manager': assistant.privacy_manager,
        'memory_system': assistant.memory_system
    }
    
    for name, component in components.items():
        try:
            # Test basic functionality
            if hasattr(component, 'get_profile'):
                component.get_profile('test')
            print(f"{name}: OK")
        except Exception as e:
            print(f"{name}: ERROR - {e}")
    
    # 4. Performance test
    import time
    start_time = time.time()
    
    test_session = assistant.start_session({'diagnostic': True})
    assistant.process_query("Diagnostic test query")
    assistant.end_session()
    
    test_time = time.time() - start_time
    print(f"Basic operation time: {test_time:.2f} seconds")
    
    print("=== Diagnostics Complete ===")

# Run diagnostics
run_system_diagnostics(assistant)
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_system_performance():
    """Profile system performance."""
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run typical operations
    assistant = create_personal_ai_assistant("profile_test")
    assistant.start_session()
    
    for i in range(10):
        assistant.process_query(f"Test query {i}")
    
    assistant.end_session()
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

# Run profiling
profile_system_performance()
```

## Getting Help

### Log Analysis

```bash
# Search for specific errors
grep -i "error" debug.log | tail -20

# Find database operations
grep -i "database\|sqlite" debug.log

# Check session management
grep -i "session" debug.log
```

### Community Support

1. Check the GitHub issues for similar problems
2. Provide detailed error logs and system information
3. Include minimal reproduction code
4. Specify your Python version and operating system

### Reporting Bugs

When reporting issues, include:

```python
# System information
import sys
import platform

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"System: {platform.system()} {platform.release()}")

# Assistant status
status = assistant.get_system_status()
print(f"Database version: {status.get('schema_version')}")
print(f"Database size: {status.get('database_size_mb')} MB")

# Error details with full traceback
import traceback
try:
    # Problem operation
    pass
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
```

This troubleshooting guide covers the most common issues. For complex problems, enable debug logging and use the diagnostic tools to gather detailed information about the system state.
