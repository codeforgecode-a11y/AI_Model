#!/usr/bin/env python3
"""
Test script for Qwen2.5-Coder-Tools integration

This script tests the integration of the Qwen model with the existing
voice companion system, verifying model selection, coding assistance,
and memory integration.
"""

import sys
import time
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from qwen_coder_interface import QwenCoderInterface, CodeAnalyzer
        print("✅ Qwen Coder Interface imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Qwen Coder Interface: {e}")
        return False
    
    try:
        from model_selector import ModelSelector, ModelType, QueryClassifier
        print("✅ Model Selector imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Model Selector: {e}")
        return False
    
    try:
        from Memory import MemorySystem
        print("✅ Memory System imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Memory System: {e}")
        return False
    
    return True

def test_code_analyzer():
    """Test the code analyzer functionality."""
    print("\n🔍 Testing Code Analyzer...")
    
    try:
        from qwen_coder_interface import CodeAnalyzer
        analyzer = CodeAnalyzer()
        
        # Test coding query detection
        test_cases = [
            ("How do I write a Python function?", True),
            ("def hello(): print('world')", True),
            ("What's the weather like?", False),
            ("Debug this JavaScript code", True),
            ("Tell me a joke", False),
            ("class MyClass:", True),
            ("import numpy as np", True),
        ]
        
        for query, expected in test_cases:
            result = analyzer.is_coding_query(query)
            status = "✅" if result == expected else "❌"
            print(f"  {status} '{query}' -> {result} (expected {expected})")
        
        # Test language detection
        print("\n  Language Detection:")
        lang_tests = [
            ("def hello(): print('world')", "python"),
            ("function test() { console.log('hello'); }", "javascript"),
            ("public class Test {}", "java"),
            ("#include <iostream>", "cpp"),
        ]
        
        for code, expected in lang_tests:
            result = analyzer.detect_language(code)
            status = "✅" if result == expected else "❌"
            print(f"  {status} '{code}' -> {result} (expected {expected})")
        
        return True
        
    except Exception as e:
        print(f"❌ Code Analyzer test failed: {e}")
        return False

def test_model_selector():
    """Test the model selector functionality."""
    print("\n🔍 Testing Model Selector...")
    
    try:
        from model_selector import ModelSelector, ModelType
        selector = ModelSelector()
        
        # Test query classification
        test_queries = [
            ("How do I write a Python function?", ModelType.CODING),
            ("What's the weather like today?", ModelType.GENERAL),
            ("Debug this code: def test():", ModelType.CODING),
            ("Tell me a story", ModelType.GENERAL),
            ("Explain recursion in programming", ModelType.CODING),
            ("How are you doing?", ModelType.GENERAL),
        ]
        
        for query, expected in test_queries:
            model_type, confidence = selector.select_model(query)
            status = "✅" if model_type == expected else "❌"
            print(f"  {status} '{query}' -> {model_type.value} (conf: {confidence:.2f})")
        
        # Test performance recording
        selector.record_performance(ModelType.CODING, 1.5, True)
        selector.record_performance(ModelType.GENERAL, 0.8, True)
        
        stats = selector.get_model_stats()
        print(f"\n  Model Statistics: {len(stats)} models tracked")
        
        return True
        
    except Exception as e:
        print(f"❌ Model Selector test failed: {e}")
        return False

def test_qwen_interface():
    """Test the Qwen interface (if model is available)."""
    print("\n🔍 Testing Qwen Interface...")
    
    try:
        from qwen_coder_interface import QwenCoderInterface
        
        # Initialize interface (this will check model availability)
        qwen = QwenCoderInterface()
        print("✅ Qwen interface initialized")
        
        # Test prompt creation
        messages = qwen.create_coding_prompt(
            "How do I write a Python function?",
            task_type="implement"
        )
        
        if messages and len(messages) >= 2:
            print("✅ Coding prompt creation successful")
            print(f"  System prompt length: {len(messages[0]['content'])}")
            print(f"  User prompt: {messages[1]['content']}")
        else:
            print("❌ Coding prompt creation failed")
            return False
        
        # Test task type determination
        task_types = [
            ("debug this code", "debug"),
            ("implement a function", "implement"),
            ("explain how this works", "explain"),
            ("review my code", "review"),
        ]
        
        for query, expected in task_types:
            result = qwen.determine_task_type(query)
            status = "✅" if result == expected else "❌"
            print(f"  {status} '{query}' -> {result} (expected {expected})")
        
        return True
        
    except Exception as e:
        print(f"❌ Qwen Interface test failed: {e}")
        print("  Note: This is expected if the Qwen model is not yet downloaded")
        return False

def test_memory_integration():
    """Test memory system integration."""
    print("\n🔍 Testing Memory Integration...")
    
    try:
        from Memory import MemorySystem
        
        # Initialize memory system
        memory = MemorySystem(
            db_path="Memory/Database/test_memory.db",
            max_context_history=10,
            context_window=5
        )
        print("✅ Memory system initialized")
        
        # Test adding interactions with model metadata
        memory.add_interaction(
            user_input="How do I write a Python function?",
            ai_response="Here's how to write a Python function: def my_function():",
            metadata={
                'model_used': 'qwen_coder',
                'model_confidence': 0.85,
                'response_time': 1.2,
                'task_type': 'implement'
            }
        )
        print("✅ Interaction with model metadata stored")
        
        # Test context retrieval
        context = memory.get_response_context()
        if context and 'context_memory' in context:
            print("✅ Context retrieval successful")
        else:
            print("❌ Context retrieval failed")
            return False
        
        # Test system status
        status = memory.get_system_status()
        print(f"✅ Memory status: {status['session_id'][:8]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory Integration test failed: {e}")
        return False

def test_full_integration():
    """Test the full integration with text chat companion."""
    print("\n🔍 Testing Full Integration...")

    try:
        # Import the updated text chat companion
        from text_chat_companion import TextChatCompanion

        print("✅ TextChatCompanion imported successfully")

        # Note: We won't actually initialize it here to avoid conflicts
        # with the running Ollama models, but we can verify the class exists
        # and has the expected methods

        required_methods = [
            'process_user_input',
            'show_model_stats',
            'show_memory_status',
            '__init__'
        ]

        for method in required_methods:
            if hasattr(TextChatCompanion, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} missing")
                return False

        return True

    except Exception as e:
        print(f"❌ Full Integration test failed: {e}")
        return False

def test_model_switching():
    """Test model switching logic."""
    print("\n🔍 Testing Model Switching Logic...")

    try:
        from model_selector import ModelSelector, ModelType
        from qwen_coder_interface import QwenCoderInterface

        selector = ModelSelector()

        # Test switching scenarios
        test_scenarios = [
            {
                'query': 'How do I write a Python function?',
                'expected': ModelType.CODING,
                'description': 'Coding query should select coding model'
            },
            {
                'query': 'What is the weather like today?',
                'expected': ModelType.GENERAL,
                'description': 'General query should select general model'
            },
            {
                'query': 'def hello(): print("world")',
                'expected': ModelType.CODING,
                'description': 'Code snippet should select coding model'
            },
            {
                'query': 'Tell me a joke',
                'expected': ModelType.GENERAL,
                'description': 'Casual conversation should select general model'
            }
        ]

        for scenario in test_scenarios:
            model_type, confidence = selector.select_model(scenario['query'])
            success = model_type == scenario['expected']
            status = "✅" if success else "❌"
            print(f"  {status} {scenario['description']}")
            print(f"    Query: '{scenario['query']}'")
            print(f"    Expected: {scenario['expected'].value}, Got: {model_type.value} (conf: {confidence:.2f})")

            if not success:
                return False

        print("✅ All model switching tests passed")
        return True

    except Exception as e:
        print(f"❌ Model switching test failed: {e}")
        return False

def test_memory_context_analyzer():
    """Test the memory context analyzer."""
    print("\n🔍 Testing Memory Context Analyzer...")

    try:
        from memory_context_analyzer import MemoryContextAnalyzer

        analyzer = MemoryContextAnalyzer()

        # Create sample interactions
        sample_interactions = [
            {
                'user_input': 'How do I write a Python function?',
                'ai_response': 'Here is how to write a Python function: def my_function():',
                'metadata': {
                    'is_coding_query': True,
                    'context_type': 'coding',
                    'task_type': 'implement',
                    'language_detected': 'python',
                    'timestamp': time.time() - 300
                }
            },
            {
                'user_input': 'Can you debug this code?',
                'ai_response': 'I can help debug your code. Please share the code.',
                'metadata': {
                    'is_coding_query': True,
                    'context_type': 'coding',
                    'task_type': 'debug',
                    'language_detected': 'python',
                    'timestamp': time.time() - 150
                }
            }
        ]

        # Test context analysis
        current_query = "How do I handle exceptions in Python?"
        context = analyzer.analyze_conversation_context(sample_interactions, current_query)

        # Verify context analysis results
        expected_checks = [
            ('context_type', 'coding'),
            ('coding_score', lambda x: x > 0.5),
            ('topic_continuity', lambda x: isinstance(x, dict) and 'score' in x),
            ('relevant_context', lambda x: isinstance(x, list)),
            ('context_summary', lambda x: isinstance(x, str) and len(x) > 0),
            ('recommendations', lambda x: isinstance(x, list))
        ]

        for key, expected in expected_checks:
            if key not in context:
                print(f"❌ Missing key: {key}")
                return False

            value = context[key]
            if callable(expected):
                if not expected(value):
                    print(f"❌ Invalid value for {key}: {value}")
                    return False
            else:
                if value != expected:
                    print(f"❌ Expected {key}={expected}, got {value}")
                    return False

            print(f"✅ {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ Memory context analyzer test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Qwen Integration Tests\n")
    
    tests = [
        ("Imports", test_imports),
        ("Code Analyzer", test_code_analyzer),
        ("Model Selector", test_model_selector),
        ("Qwen Interface", test_qwen_interface),
        ("Memory Integration", test_memory_integration),
        ("Memory Context Analyzer", test_memory_context_analyzer),
        ("Model Switching", test_model_switching),
        ("Full Integration", test_full_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} Test")
        print('='*50)
        
        start_time = time.time()
        try:
            result = test_func()
            duration = time.time() - start_time
            results.append((test_name, result, duration))
            
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status} - {test_name} ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            results.append((test_name, False, duration))
            print(f"\n❌ FAILED - {test_name} ({duration:.2f}s): {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, duration in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name:<20} ({duration:.2f}s)")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Qwen integration is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
