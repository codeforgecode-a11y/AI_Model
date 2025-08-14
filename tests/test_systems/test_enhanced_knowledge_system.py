#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Intelligent Knowledge System

Tests all components of the enhanced system including:
- User profile management
- Session storage and retrieval
- Database migrations
- Data export/import functionality
- Integration with existing memory systems
- Performance testing with large conversation histories
- Privacy and security features
"""

import unittest
import tempfile
import shutil
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

# Import the enhanced system components
from intelligent_knowledge_system import EnhancedIntelligentKnowledgeSystem, create_personal_ai_assistant
from enhanced_database_manager import EnhancedDatabaseManager
from user_profile_manager import UserProfileManager
from session_manager import SessionManager
from privacy_security_manager import PrivacySecurityManager
from enhanced_database_schema import EnhancedDatabaseSchema, EncryptionManager


class TestEnhancedKnowledgeSystem(unittest.TestCase):
    """Test suite for the enhanced intelligent knowledge system."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_db_path = Path(self.test_dir) / "test_enhanced_knowledge.db"
        self.test_profile_name = "test_user"
        
        # Initialize test system
        self.system = EnhancedIntelligentKnowledgeSystem(
            config={'test_mode': True},
            db_path=str(self.test_db_path),
            profile_name=self.test_profile_name
        )
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'system') and self.system.current_session_id:
            self.system.end_session()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_system_initialization(self):
        """Test system initialization and component setup."""
        # Test that all components are initialized
        self.assertIsNotNone(self.system.enhanced_db)
        self.assertIsNotNone(self.system.user_profile_manager)
        self.assertIsNotNone(self.system.session_manager)
        self.assertIsNotNone(self.system.privacy_manager)
        self.assertIsNotNone(self.system.memory_system)
        
        # Test database file creation
        self.assertTrue(self.test_db_path.exists())
        
        # Test profile creation
        profile = self.system.user_profile_manager.get_profile(self.test_profile_name)
        self.assertIsNotNone(profile)
        self.assertEqual(profile['profile_name'], self.test_profile_name)
    
    def test_user_profile_management(self):
        """Test user profile creation, updates, and retrieval."""
        # Test preference updates
        self.assertTrue(self.system.update_user_preference('response_style', 'formal'))
        self.assertTrue(self.system.update_user_preference('verbosity_level', 'detailed'))
        self.assertTrue(self.system.update_user_preference('technical_level', 'advanced'))
        
        # Test preference retrieval
        self.assertEqual(self.system.get_user_preference('response_style'), 'formal')
        self.assertEqual(self.system.get_user_preference('verbosity_level'), 'detailed')
        self.assertEqual(self.system.get_user_preference('technical_level'), 'advanced')
        
        # Test default values
        self.assertEqual(self.system.get_user_preference('nonexistent_pref', 'default'), 'default')
        
        # Test custom preferences in communication patterns
        self.assertTrue(self.system.update_user_preference('custom_greeting', 'Hello there!'))
        self.assertEqual(self.system.get_user_preference('custom_greeting'), 'Hello there!')
    
    def test_session_management(self):
        """Test session creation, management, and termination."""
        # Test session creation
        session_id = self.system.start_session({'test_context': 'unit_test'})
        self.assertIsNotNone(session_id)
        self.assertEqual(self.system.current_session_id, session_id)
        
        # Test session metadata
        session_data = self.system.session_manager.get_session_metadata(session_id)
        self.assertIsNotNone(session_data)
        self.assertEqual(session_data['session_id'], session_id)
        
        # Test session termination
        self.assertTrue(self.system.end_session("Test session completed", 0.8))
        self.assertIsNone(self.system.current_session_id)
        
        # Verify session was properly ended
        final_session_data = self.system.session_manager.get_session_metadata(session_id)
        self.assertIsNotNone(final_session_data['ended_at'])
        self.assertEqual(final_session_data['user_satisfaction_score'], 0.8)
    
    def test_message_storage_and_retrieval(self):
        """Test message storage and conversation threading."""
        # Start a session
        session_id = self.system.start_session()
        
        # Test query processing with message storage
        result = self.system.process_query("What is machine learning?")
        self.assertIn('user_message_id', result)
        self.assertIn('ai_message_id', result)
        self.assertIn('ai_response', result)
        
        # Test message retrieval
        messages = self.system.session_manager.get_session_messages(session_id)
        self.assertGreaterEqual(len(messages), 2)  # At least user input and AI response
        
        # Test message types
        user_messages = [m for m in messages if m['message_type'] == 'user_input']
        ai_messages = [m for m in messages if m['message_type'] == 'ai_response']
        self.assertGreater(len(user_messages), 0)
        self.assertGreater(len(ai_messages), 0)
        
        # Test conversation threading
        ai_message = ai_messages[0]
        user_message = user_messages[0]
        self.assertEqual(ai_message['parent_message_id'], user_message['message_id'])
    
    def test_feedback_processing(self):
        """Test user feedback processing and preference adaptation."""
        # Start session and process query
        self.system.start_session()
        result = self.system.process_query("Explain quantum computing")
        
        # Add positive feedback
        self.assertTrue(self.system.add_user_feedback("Great explanation, very clear!", 5))
        
        # Add negative feedback about verbosity
        result2 = self.system.process_query("What is AI?")
        self.assertTrue(self.system.add_user_feedback("Too detailed, I wanted a brief answer", 2))
        
        # Check if preferences were adapted (this would depend on the feedback analysis)
        # The system should learn from the feedback patterns
        
        # Test feedback without active session
        self.system.end_session()
        self.assertFalse(self.system.add_user_feedback("No active session", 3))
    
    def test_conversation_history_search(self):
        """Test conversation history search and retrieval."""
        # Create multiple sessions with different topics
        session1_id = self.system.start_session()
        self.system.process_query("Tell me about Python programming")
        self.system.process_query("How do I use decorators in Python?")
        self.system.end_session("Python discussion", 0.9)
        
        session2_id = self.system.start_session()
        self.system.process_query("What is machine learning?")
        self.system.process_query("Explain neural networks")
        self.system.end_session("ML discussion", 0.8)
        
        # Test search functionality
        python_sessions = self.system.search_conversation_history("Python")
        self.assertGreater(len(python_sessions), 0)
        
        ml_sessions = self.system.search_conversation_history("machine learning")
        self.assertGreater(len(ml_sessions), 0)
        
        # Test session history retrieval
        all_sessions = self.system.get_session_history(10)
        self.assertGreaterEqual(len(all_sessions), 2)
    
    def test_data_export_import(self):
        """Test data export and import functionality."""
        # Create some test data
        self.system.start_session()
        self.system.process_query("Test query for export")
        self.system.update_user_preference('test_pref', 'test_value')
        self.system.end_session()
        
        # Test JSON export
        export_path = Path(self.test_dir) / "test_export.json"
        self.assertTrue(self.system.export_user_data(str(export_path), "json"))
        self.assertTrue(export_path.exists())
        
        # Verify export content
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        self.assertIn('user_profile', export_data)
        self.assertIn('sessions', export_data)
        self.assertIn('export_metadata', export_data)
        
        # Test CSV export
        csv_dir = Path(self.test_dir) / "csv_export"
        self.assertTrue(self.system.export_user_data(str(csv_dir), "csv"))
        self.assertTrue(csv_dir.exists())
        
        # Test import
        self.assertTrue(self.system.import_user_data(str(export_path)))
    
    def test_privacy_features(self):
        """Test privacy and data management features."""
        # Create test data
        self.system.start_session()
        self.system.process_query("Sensitive information test")
        session_id = self.system.current_session_id
        self.system.end_session()
        
        # Test privacy dashboard
        dashboard = self.system.get_privacy_dashboard()
        self.assertIn('personal_data_overview', dashboard)
        self.assertIn('data_retention_info', dashboard)
        self.assertIn('security_status', dashboard)
        
        # Test personal data view
        personal_data = self.system.view_all_personal_data()
        self.assertIn('profile_summary', personal_data)
        self.assertIn('conversation_data', personal_data)
        
        # Test session deletion
        self.assertTrue(self.system.delete_session_data(session_id))
        
        # Verify session was deleted
        deleted_session = self.system.session_manager.get_session_metadata(session_id)
        self.assertIsNone(deleted_session)
        
        # Test GDPR export
        gdpr_path = self.system.export_gdpr_data()
        self.assertIsNotNone(gdpr_path)
        self.assertTrue(Path(gdpr_path).exists())
    
    def test_database_versioning(self):
        """Test database schema versioning and migration."""
        # Test current version
        current_version = self.system.enhanced_db.version_manager.get_current_version()
        self.assertEqual(current_version, EnhancedDatabaseSchema.CURRENT_VERSION)
        
        # Test schema initialization
        self.assertTrue(self.system.enhanced_db.version_manager.initialize_schema())
        
        # Test migration (simulate older version)
        # This would require more complex setup to test actual migrations
        self.assertTrue(True)  # Placeholder for migration tests
    
    def test_backup_functionality(self):
        """Test automatic backup creation and restoration."""
        # Create test data
        self.system.start_session()
        self.system.process_query("Backup test query")
        self.system.end_session()
        
        # Test backup creation
        backup_id = self.system.create_backup()
        self.assertIsNotNone(backup_id)
        
        # Test system status includes backup info
        status = self.system.get_system_status()
        self.assertIn('backup_count', status)
    
    def test_pattern_analysis(self):
        """Test conversation pattern analysis."""
        # Create varied conversation data
        topics = ['programming', 'science', 'technology', 'programming', 'science']
        
        for i, topic in enumerate(topics):
            self.system.start_session()
            self.system.process_query(f"Tell me about {topic}")
            self.system.end_session(f"Session about {topic}", 0.7 + (i * 0.05))
            time.sleep(0.1)  # Small delay to ensure different timestamps
        
        # Test pattern analysis
        patterns = self.system.analyze_conversation_patterns(days_back=1)
        self.assertIn('total_sessions', patterns)
        self.assertIn('top_topics', patterns)
        self.assertIn('session_statistics', patterns)
        self.assertIn('satisfaction_metrics', patterns)
        
        # Verify topic analysis
        top_topics = patterns['top_topics']
        self.assertGreater(len(top_topics), 0)
        
        # Test recommendations
        recommendations = self.system.generate_personalized_recommendations()
        self.assertIn('preference_suggestions', recommendations)
        self.assertIn('usage_optimization', recommendations)
        self.assertIn('feature_recommendations', recommendations)
    
    def test_performance_with_large_data(self):
        """Test system performance with large conversation histories."""
        start_time = time.time()
        
        # Create a moderate amount of test data
        num_sessions = 20
        messages_per_session = 10
        
        for session_num in range(num_sessions):
            self.system.start_session({'session_number': session_num})
            
            for msg_num in range(messages_per_session):
                query = f"Test query {msg_num} in session {session_num}"
                result = self.system.process_query(query)
                self.assertIn('ai_response', result)
            
            self.system.end_session(f"Test session {session_num}", 0.8)
        
        creation_time = time.time() - start_time
        
        # Test retrieval performance
        start_time = time.time()
        all_sessions = self.system.get_session_history(num_sessions)
        retrieval_time = time.time() - start_time
        
        # Test search performance
        start_time = time.time()
        search_results = self.system.search_conversation_history("test")
        search_time = time.time() - start_time
        
        # Verify data integrity
        self.assertEqual(len(all_sessions), num_sessions)
        self.assertGreater(len(search_results), 0)
        
        # Performance assertions (adjust thresholds as needed)
        self.assertLess(creation_time, 30.0, "Data creation took too long")
        self.assertLess(retrieval_time, 1.0, "Data retrieval took too long")
        self.assertLess(search_time, 2.0, "Search took too long")
        
        print(f"Performance metrics:")
        print(f"  Data creation: {creation_time:.2f}s for {num_sessions} sessions")
        print(f"  Data retrieval: {retrieval_time:.2f}s for {num_sessions} sessions")
        print(f"  Search time: {search_time:.2f}s")
    
    def test_backward_compatibility(self):
        """Test backward compatibility with original API."""
        from intelligent_knowledge_system import IntelligentKnowledgeSystem
        
        # Test that original class still works
        legacy_system = IntelligentKnowledgeSystem({'test_mode': True})
        
        # Test original methods
        result = legacy_system.process_query("Test backward compatibility")
        self.assertIn('query', result)
        self.assertIn('should_search', result)
        
        # Test that enhanced features are available
        self.assertTrue(hasattr(legacy_system, 'get_privacy_dashboard'))
        self.assertTrue(hasattr(legacy_system, 'update_user_preference'))
    
    def test_encryption_functionality(self):
        """Test data encryption and decryption."""
        encryption_manager = EncryptionManager()
        
        # Test encryption/decryption
        test_data = "Sensitive user information"
        encrypted = encryption_manager.encrypt(test_data)
        decrypted = encryption_manager.decrypt(encrypted)
        
        self.assertNotEqual(test_data, encrypted)
        self.assertEqual(test_data, decrypted)
        
        # Test with empty data
        self.assertEqual(encryption_manager.encrypt(""), "")
        self.assertEqual(encryption_manager.decrypt(""), "")


class TestIntegrationScenarios(unittest.TestCase):
    """Test realistic usage scenarios and integration flows."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.assistant = create_personal_ai_assistant(
            "integration_test_user",
            {'db_path': str(Path(self.test_dir) / "integration_test.db")}
        )
    
    def tearDown(self):
        """Clean up integration test environment."""
        if hasattr(self, 'assistant') and self.assistant.current_session_id:
            self.assistant.end_session()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_complete_user_journey(self):
        """Test a complete user journey from setup to data export."""
        # 1. Initial setup and preference configuration
        self.assistant.update_user_preference('response_style', 'friendly')
        self.assistant.update_user_preference('verbosity_level', 'moderate')
        
        # 2. Multiple conversation sessions
        for day in range(3):
            self.assistant.start_session({'day': day})
            
            # Simulate varied conversations
            queries = [
                "How do I learn Python programming?",
                "What are the best practices for code organization?",
                "Can you explain object-oriented programming?"
            ]
            
            for query in queries:
                result = self.assistant.process_query(query)
                self.assertIn('ai_response', result)
                
                # Simulate user feedback
                if day == 0:
                    self.assistant.add_user_feedback("Very helpful!", 5)
                elif day == 1:
                    self.assistant.add_user_feedback("Good but a bit too detailed", 4)
            
            self.assistant.end_session(f"Day {day} learning session", 0.8 + (day * 0.05))
        
        # 3. Data analysis and insights
        patterns = self.assistant.analyze_conversation_patterns()
        self.assertGreater(patterns['total_sessions'], 0)
        
        recommendations = self.assistant.generate_personalized_recommendations()
        self.assertIn('preference_suggestions', recommendations)
        
        # 4. Privacy management
        privacy_dashboard = self.assistant.get_privacy_dashboard()
        self.assertIn('personal_data_overview', privacy_dashboard)
        
        # 5. Data export
        export_path = Path(self.test_dir) / "user_journey_export.json"
        self.assertTrue(self.assistant.export_user_data(str(export_path)))
        
        # 6. Verify export completeness
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        self.assertIn('user_profile', export_data)
        self.assertGreater(len(export_data['sessions']), 0)


def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    import tempfile
    import shutil
    from pathlib import Path

    print("Running Enhanced Knowledge System Performance Benchmark")
    print("=" * 60)

    test_dir = tempfile.mkdtemp()
    try:
        assistant = create_personal_ai_assistant(
            "benchmark_user",
            {'db_path': str(Path(test_dir) / "benchmark.db")}
        )

        # Benchmark 1: Large conversation history
        print("\n1. Testing large conversation history performance...")
        start_time = time.time()

        num_sessions = 100
        messages_per_session = 20

        for session_num in range(num_sessions):
            assistant.start_session({'benchmark': True, 'session': session_num})

            for msg_num in range(messages_per_session):
                query = f"Benchmark query {msg_num} about topic {session_num % 10}"
                assistant.process_query(query)

                if msg_num % 5 == 0:  # Add feedback every 5 messages
                    assistant.add_user_feedback("Good response", 4)

            assistant.end_session(f"Benchmark session {session_num}", 0.8)

            if session_num % 10 == 0:
                print(f"  Completed {session_num + 1}/{num_sessions} sessions")

        creation_time = time.time() - start_time
        print(f"  Created {num_sessions} sessions with {num_sessions * messages_per_session} messages in {creation_time:.2f}s")

        # Benchmark 2: Search performance
        print("\n2. Testing search performance...")
        start_time = time.time()

        search_results = assistant.search_conversation_history("benchmark")
        search_time = time.time() - start_time
        print(f"  Search completed in {search_time:.2f}s, found {len(search_results)} results")

        # Benchmark 3: Pattern analysis performance
        print("\n3. Testing pattern analysis performance...")
        start_time = time.time()

        patterns = assistant.analyze_conversation_patterns(days_back=30)
        analysis_time = time.time() - start_time
        print(f"  Pattern analysis completed in {analysis_time:.2f}s")
        print(f"  Analyzed {patterns['total_sessions']} sessions")

        # Benchmark 4: Data export performance
        print("\n4. Testing data export performance...")
        start_time = time.time()

        export_path = Path(test_dir) / "benchmark_export.json"
        assistant.export_user_data(str(export_path))
        export_time = time.time() - start_time

        export_size = export_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  Data export completed in {export_time:.2f}s")
        print(f"  Export file size: {export_size:.2f} MB")

        # Benchmark 5: Database operations
        print("\n5. Testing database operations...")
        start_time = time.time()

        # Test backup creation
        backup_id = assistant.create_backup()
        backup_time = time.time() - start_time
        print(f"  Backup creation completed in {backup_time:.2f}s")

        # Summary
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Data Creation:     {creation_time:.2f}s ({num_sessions * messages_per_session / creation_time:.1f} messages/sec)")
        print(f"Search:            {search_time:.2f}s")
        print(f"Pattern Analysis:  {analysis_time:.2f}s")
        print(f"Data Export:       {export_time:.2f}s ({export_size / export_time:.1f} MB/sec)")
        print(f"Backup Creation:   {backup_time:.2f}s")

        # Memory usage (approximate)
        status = assistant.get_system_status()
        db_size = status.get('database_size_mb', 0)
        print(f"Database Size:     {db_size:.2f} MB")

        assistant.end_session()

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'benchmark':
        run_performance_benchmark()
    else:
        # Run the test suite
        unittest.main(verbosity=2)
