# Enhanced Intelligent Knowledge System

A comprehensive personal AI assistant with persistent user profile management, complete session storage capabilities, and advanced privacy controls.

## Overview

The Enhanced Intelligent Knowledge System extends the original `intelligent_knowledge_system.py` with powerful new features designed for single-user personal AI assistants:

- **Persistent User Profile Management**: Comprehensive user preferences, behavioral patterns, and adaptive learning
- **Complete Session Storage**: Full conversation history with metadata, threading, and context preservation
- **Advanced Privacy Controls**: Local-only storage, data encryption, GDPR compliance, and user data rights
- **Database Versioning**: Schema migrations, automatic backups, and data integrity
- **Pattern Analysis**: Conversation insights, usage analytics, and personalized recommendations
- **Backward Compatibility**: Seamless integration with existing memory systems

## Quick Start

### Basic Setup

```python
from intelligent_knowledge_system import create_personal_ai_assistant

# Create your personal AI assistant
assistant = create_personal_ai_assistant(
    profile_name="your_name",
    config={
        'db_path': 'path/to/your/database.db',
        'privacy_level': 'standard',
        'auto_backup': True
    }
)

# Start a conversation session
assistant.start_session({'context': 'daily_chat'})

# Process queries with full session tracking
result = assistant.process_query("What's the weather like today?")
print(result['ai_response'])

# Add user feedback for learning
assistant.add_user_feedback("Great response, very helpful!", rating=5)

# End the session
assistant.end_session("Daily chat completed", user_satisfaction_score=0.9)
```

### Advanced Usage

```python
# Configure user preferences
assistant.update_user_preference('response_style', 'friendly')
assistant.update_user_preference('verbosity_level', 'detailed')
assistant.update_user_preference('technical_level', 'intermediate')

# Voice companion settings
assistant.update_user_preference('preferred_tts_engine', 'elevenlabs')
assistant.update_user_preference('speech_rate', 1.2)

# Search conversation history
recent_sessions = assistant.search_conversation_history(
    "machine learning", 
    date_range=(datetime.now() - timedelta(days=30), datetime.now())
)

# Analyze conversation patterns
patterns = assistant.analyze_conversation_patterns(days_back=30)
print(f"Total sessions: {patterns['total_sessions']}")
print(f"Top topics: {patterns['top_topics']}")

# Get personalized recommendations
recommendations = assistant.generate_personalized_recommendations()
```

## Core Components

### 1. Enhanced Database Manager (`enhanced_database_manager.py`)

Manages the comprehensive database with versioning, backups, and exports:

```python
from enhanced_database_manager import EnhancedDatabaseManager

db_manager = EnhancedDatabaseManager("path/to/database.db")

# Create automatic backup
backup_id = db_manager.create_automatic_backup()

# Export data in various formats
db_manager.export_all_data("backup.json", format_type="json")
db_manager.export_all_data("backup_csv/", format_type="csv")

# Get system status
status = db_manager.get_system_status()
```

### 2. User Profile Manager (`user_profile_manager.py`)

Handles persistent user profiles with adaptive learning:

```python
from user_profile_manager import UserProfileManager

profile_manager = UserProfileManager(db_connection, encryption_manager)

# Create and manage profiles
profile_manager.create_profile("user_name", {
    'response_style': 'casual',
    'technical_level': 'advanced'
})

# Adaptive preference learning
adaptations = profile_manager.adapt_preferences_from_feedback({
    'text': 'too technical',
    'sentiment': -0.3,
    'type': 'technical_level_feedback'
}, "user_name")
```

### 3. Session Manager (`session_manager.py`)

Comprehensive session and message management:

```python
from session_manager import SessionManager

session_manager = SessionManager(db_connection, profile_manager)

# Create session with context
session_id = session_manager.create_session("user_name", {
    'device': 'mobile',
    'location': 'home'
})

# Add messages with threading
user_msg_id = session_manager.add_message(
    session_id, 'user_input', "How do I learn Python?"
)

ai_msg_id = session_manager.add_message(
    session_id, 'ai_response', "Here's a comprehensive guide...",
    parent_message_id=user_msg_id
)

# Search and retrieve conversations
sessions = session_manager.search_sessions(
    "Python programming",
    topic_filter="programming"
)
```

### 4. Privacy & Security Manager (`privacy_security_manager.py`)

Comprehensive privacy controls and data management:

```python
from privacy_security_manager import PrivacySecurityManager

privacy_manager = PrivacySecurityManager(db_connection, encryption_manager)

# View all personal data
personal_data = privacy_manager.data_privacy.view_personal_data("user_name")

# Delete specific sessions
privacy_manager.data_privacy.delete_session_data("session_id", secure_delete=True)

# Export GDPR-compliant data
gdpr_export_path = privacy_manager.data_privacy.export_gdpr_data("user_name")

# Get privacy dashboard
dashboard = privacy_manager.get_privacy_dashboard("user_name")
```

## Database Schema

The enhanced system uses a comprehensive SQLite schema with the following key tables:

### Core Tables

- **`user_profile`**: User preferences, settings, and behavioral patterns
- **`sessions`**: Session metadata with duration, topics, and satisfaction scores
- **`messages`**: Individual messages with full content and context relationships
- **`learning_insights`**: Extracted patterns from conversation history
- **`conversation_topics`**: Topic tracking with frequency and relationships

### Supporting Tables

- **`schema_version`**: Database versioning for migrations
- **`system_backups`**: Backup metadata and integrity tracking

### Key Features

- **Foreign Key Constraints**: Maintain data integrity
- **JSON Fields**: Flexible metadata storage
- **Encryption Support**: Sensitive data protection
- **Full-Text Search**: Efficient content searching
- **Indexing**: Optimized query performance

## Privacy and Security

### Local-Only Storage

- All data stored locally in SQLite database
- No external transmission or cloud storage
- Complete user control over data

### Encryption

```python
from enhanced_database_schema import EncryptionManager

encryption = EncryptionManager()
encrypted_data = encryption.encrypt("sensitive information")
decrypted_data = encryption.decrypt(encrypted_data)
```

### Data Rights and Controls

- **View All Data**: Complete transparency of stored information
- **Export Data**: GDPR-compliant data portability
- **Delete Specific Data**: Granular deletion controls
- **Anonymize Profile**: Remove personal identifiers while preserving patterns
- **Secure Deletion**: Overwrite data for complete removal

### Privacy Dashboard

```python
# Get comprehensive privacy overview
dashboard = assistant.get_privacy_dashboard()

# Includes:
# - Personal data summary
# - Data retention information
# - Available privacy controls
# - Security status
```

## Configuration

### Basic Configuration

```python
config = {
    'db_path': 'Memory/Database/personal_assistant.db',
    'profile_name': 'default',
    'privacy_level': 'standard',  # minimal, standard, enhanced, maximum
    'auto_backup': True,
    'backup_interval_hours': 24,
    'search_config': {
        'enable_web_search': True,
        'cache_duration_hours': 1
    }
}

assistant = EnhancedIntelligentKnowledgeSystem(config)
```

### Privacy Levels

- **Minimal**: Basic functionality, minimal data retention
- **Standard**: Balanced privacy and functionality (default)
- **Enhanced**: Additional privacy protections
- **Maximum**: Maximum privacy, limited learning capabilities

### Voice Companion Integration

```python
# Configure TTS preferences
assistant.update_user_preference('preferred_tts_engine', 'elevenlabs')
assistant.update_user_preference('voice_selection', 'professional_female')
assistant.update_user_preference('speech_rate', 1.1)
assistant.update_user_preference('speech_pitch', 1.0)
assistant.update_user_preference('speech_volume', 0.8)

# Fallback configuration
assistant.update_user_preference('tts_fallback_engine', 'mozilla')
assistant.update_user_preference('cost_management', 'free_tier_only')
```

## Migration from Legacy System

### Automatic Migration

The enhanced system automatically detects and migrates from legacy databases:

```python
# Legacy system data is automatically migrated
assistant = EnhancedIntelligentKnowledgeSystem(
    db_path="path/to/legacy/memory.db"
)
```

### Manual Migration

```python
from intelligent_knowledge_system import migrate_legacy_system

success = migrate_legacy_system(
    legacy_db_path="old_system/memory.db",
    enhanced_db_path="new_system/enhanced_memory.db"
)
```

## Performance Optimization

### Database Optimization

- Automatic indexing on frequently queried fields
- Connection pooling for efficient database access
- Configurable cache sizes and retention policies

### Memory Management

```python
# Configure memory limits
config = {
    'max_context_history': 100,
    'context_window': 15,
    'message_cache_size': 1000,
    'session_cache_duration_hours': 24
}
```

### Backup Management

```python
# Configure automatic backups
assistant.enhanced_db.backup_manager.cleanup_old_backups(retention_days=30)

# Manual backup with compression
backup_id = assistant.enhanced_db.backup_manager.create_backup(
    backup_type="full",
    compress=True,
    encrypt=True
)
```

## Testing

### Run Test Suite

```bash
# Run all tests
python test_enhanced_knowledge_system.py

# Run performance benchmark
python test_enhanced_knowledge_system.py benchmark
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component functionality
- **Performance Tests**: Large-scale data handling
- **Privacy Tests**: Data protection and user rights
- **Backward Compatibility**: Legacy system integration

## Troubleshooting

### Common Issues

#### Database Lock Errors

```python
# Ensure proper connection management
with assistant.enhanced_db.db_connection.get_connection() as conn:
    # Perform database operations
    pass
```

#### Memory Growth

```python
# Monitor and manage cache sizes
status = assistant.get_system_status()
if status['database_size_mb'] > 100:
    assistant.create_backup()
    # Consider data cleanup or archiving
```

#### Performance Issues

```python
# Check database integrity
integrity = assistant.enhanced_db.version_manager.validate_data_integrity()
if integrity['status'] != 'healthy':
    # Run database maintenance
    pass
```

### Logging Configuration

```python
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_assistant.log'),
        logging.StreamHandler()
    ]
)
```

### Data Recovery

```python
# Restore from backup
success = assistant.enhanced_db.backup_manager.restore_backup(backup_id)

# Validate data integrity after recovery
integrity = assistant.enhanced_db.version_manager.validate_data_integrity()
```

## API Reference

### Main System Methods

- `start_session(context)`: Start new conversation session
- `end_session(summary, satisfaction)`: End current session
- `process_query(query, context)`: Process user query with full tracking
- `add_user_feedback(text, rating)`: Add feedback for learning

### Profile Management

- `update_user_preference(key, value)`: Update user preference
- `get_user_preference(key, default)`: Get preference value
- `get_user_profile()`: Get complete profile data

### Session Management

- `get_session_history(limit)`: Get recent sessions
- `search_conversation_history(query, filters)`: Search conversations
- `get_conversation_thread(message_id)`: Get message thread

### Privacy Controls

- `get_privacy_dashboard()`: Get privacy overview
- `view_all_personal_data()`: View all stored data
- `delete_session_data(session_id)`: Delete specific session
- `export_gdpr_data(path)`: Export GDPR-compliant data
- `anonymize_profile()`: Anonymize user profile

### Analytics

- `analyze_conversation_patterns(days)`: Analyze usage patterns
- `generate_personalized_recommendations()`: Get recommendations
- `get_learning_insights(limit)`: Get learning insights

### System Management

- `create_backup()`: Create system backup
- `export_user_data(path, format)`: Export all data
- `get_system_status()`: Get system status and metrics

## License and Privacy

This enhanced system is designed with privacy-first principles:

- **Local Storage Only**: No data leaves your device
- **User Control**: Complete control over your data
- **Transparency**: Full visibility into stored information
- **Data Rights**: Easy export, deletion, and anonymization
- **Security**: Encryption for sensitive information

The system respects user privacy and provides tools for complete data management and control.

## Installation and Dependencies

### Required Dependencies

```bash
pip install sqlite3 cryptography pathlib json logging threading
```

### Optional Dependencies

```bash
# For voice integration
pip install speech_recognition pyttsx3

# For analytics visualization
pip install matplotlib pandas

# For advanced NLP features
pip install nltk spacy

# For web search integration
pip install requests beautifulsoup4
```

### Installation

1. Clone or download the enhanced knowledge system files
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python test_enhanced_knowledge_system.py`
4. Start using: `python your_assistant_script.py`
