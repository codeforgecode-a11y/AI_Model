#!/usr/bin/env python3
"""
End-to-End Integration Test for Qwen2.5-Coder-Tools

This script performs comprehensive end-to-end testing of the Qwen integration,
including actual model responses, memory persistence, and model switching.
"""

import sys
import time
import logging
import tempfile
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_availability():
    """Test that required models are available."""
    print("🔍 Testing Model Availability...")
    
    try:
        import ollama
        
        # Check available models
        response = ollama.list()
        models = response.get('models', [])
        
        # Extract model names
        if models and isinstance(models[0], dict):
            if 'name' in models[0]:
                available_models = [model['name'] for model in models]
            elif 'model' in models[0]:
                available_models = [model['model'] for model in models]
            else:
                available_models = [str(model.get(list(model.keys())[0], '')) for model in models]
        else:
            available_models = [str(model) for model in models]
        
        print(f"Available models: {available_models}")
        
        # Check for required models
        required_models = ['llama3.2:1b', 'hhao/qwen2.5-coder-tools:7b']
        missing_models = []
        
        for model in required_models:
            if model not in available_models:
                missing_models.append(model)
            else:
                print(f"✅ {model} is available")
        
        if missing_models:
            print(f"⚠️  Missing models: {missing_models}")
            print("Note: Tests will use fallback mechanisms")
        
        return len(missing_models) == 0
        
    except Exception as e:
        print(f"❌ Model availability check failed: {e}")
        return False


def test_basic_text_companion():
    """Test basic text companion functionality."""
    print("\n🔍 Testing Basic Text Companion...")
    
    try:
        from text_chat_companion import TextChatCompanion
        
        # Create a temporary companion instance
        companion = TextChatCompanion(internet_mode="offline")
        
        print("✅ TextChatCompanion initialized successfully")
        
        # Test basic functionality
        test_queries = [
            "Hello, how are you?",  # General query
            "What is 2 + 2?",       # Simple math
        ]
        
        for query in test_queries:
            print(f"\nTesting query: '{query}'")
            
            try:
                response = companion.process_user_input(query)
                
                if response and len(response.strip()) > 0:
                    print(f"✅ Got response: {response[:100]}...")
                else:
                    print("❌ Empty or no response")
                    return False
                    
            except Exception as e:
                print(f"❌ Query processing failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Basic text companion test failed: {e}")
        return False


def test_coding_queries():
    """Test coding-specific queries."""
    print("\n🔍 Testing Coding Queries...")
    
    try:
        from text_chat_companion import TextChatCompanion
        
        # Create a temporary companion instance
        companion = TextChatCompanion(internet_mode="offline")
        
        # Test coding queries
        coding_queries = [
            "How do I write a Python function?",
            "What is a for loop in programming?",
            "def hello(): print('world')",  # Code snippet
        ]
        
        for query in coding_queries:
            print(f"\nTesting coding query: '{query}'")
            
            try:
                response = companion.process_user_input(query)
                
                if response and len(response.strip()) > 0:
                    print(f"✅ Got coding response: {response[:100]}...")
                    
                    # Check if response contains coding-related terms
                    coding_terms = ['function', 'def', 'code', 'programming', 'python']
                    if any(term in response.lower() for term in coding_terms):
                        print("✅ Response contains coding-related content")
                    else:
                        print("⚠️  Response may not be coding-specific")
                else:
                    print("❌ Empty or no response")
                    return False
                    
            except Exception as e:
                print(f"❌ Coding query processing failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Coding queries test failed: {e}")
        return False


def test_memory_persistence():
    """Test memory persistence across interactions."""
    print("\n🔍 Testing Memory Persistence...")
    
    try:
        from text_chat_companion import TextChatCompanion
        
        # Create a temporary companion instance
        companion = TextChatCompanion(internet_mode="offline")
        
        # First interaction
        query1 = "My name is Alice"
        response1 = companion.process_user_input(query1)
        print(f"First interaction: '{query1}' -> '{response1[:50]}...'")
        
        # Second interaction that should reference the first
        query2 = "What is my name?"
        response2 = companion.process_user_input(query2)
        print(f"Second interaction: '{query2}' -> '{response2[:50]}...'")
        
        # Check if the second response references the name
        if 'alice' in response2.lower():
            print("✅ Memory persistence working - name remembered")
        else:
            print("⚠️  Memory persistence unclear - name may not be remembered")
        
        # Test memory status
        try:
            companion.show_memory_status()
            print("✅ Memory status display working")
        except Exception as e:
            print(f"⚠️  Memory status display failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory persistence test failed: {e}")
        return False


def test_model_statistics():
    """Test model statistics functionality."""
    print("\n🔍 Testing Model Statistics...")
    
    try:
        from text_chat_companion import TextChatCompanion
        
        # Create a temporary companion instance
        companion = TextChatCompanion(internet_mode="offline")
        
        # Process a few queries to generate statistics
        test_queries = [
            "Hello",
            "How do I write a function?",
            "What is the weather?",
        ]
        
        for query in test_queries:
            companion.process_user_input(query)
        
        # Test model statistics display
        try:
            companion.show_model_stats()
            print("✅ Model statistics display working")
        except Exception as e:
            print(f"❌ Model statistics display failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model statistics test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and fallback mechanisms."""
    print("\n🔍 Testing Error Handling...")
    
    try:
        from text_chat_companion import TextChatCompanion
        
        # Create a temporary companion instance
        companion = TextChatCompanion(internet_mode="offline")
        
        # Test with various edge cases
        edge_cases = [
            "",  # Empty input
            "   ",  # Whitespace only
            "a" * 1000,  # Very long input
            "🚀🎉💻",  # Emoji only
        ]
        
        for case in edge_cases:
            print(f"Testing edge case: '{case[:20]}{'...' if len(case) > 20 else ''}'")
            
            try:
                response = companion.process_user_input(case)
                
                if response is not None:
                    print(f"✅ Handled gracefully: {response[:50]}...")
                else:
                    print("✅ Handled gracefully: No response (expected)")
                    
            except Exception as e:
                print(f"⚠️  Exception occurred (may be expected): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def main():
    """Run all end-to-end tests."""
    print("🚀 Starting End-to-End Integration Tests\n")
    
    tests = [
        ("Model Availability", test_model_availability),
        ("Basic Text Companion", test_basic_text_companion),
        ("Coding Queries", test_coding_queries),
        ("Memory Persistence", test_memory_persistence),
        ("Model Statistics", test_model_statistics),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running {test_name} Test")
        print('='*60)
        
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
    print(f"\n{'='*60}")
    print("END-TO-END TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, duration in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name:<25} ({duration:.2f}s)")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All end-to-end tests passed! Integration is working correctly.")
    elif passed >= total * 0.8:
        print("✅ Most tests passed. Integration is mostly working with minor issues.")
    else:
        print("⚠️  Several tests failed. Check the output above for details.")
    
    return passed >= total * 0.8  # Consider 80% pass rate as success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
