#!/usr/bin/env python3
"""
Test script for the text-based chat companion
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import ollama
        print("✅ Ollama import successful")
    except ImportError as e:
        print(f"❌ Ollama import failed: {e}")
        return False
    
    try:
        import requests
        print("✅ Requests import successful")
    except ImportError as e:
        print(f"❌ Requests import failed: {e}")
        return False
    
    try:
        import colorama
        print("✅ Colorama import successful")
    except ImportError as e:
        print(f"❌ Colorama import failed: {e}")
        return False
    
    try:
        from Memory import MemorySystem
        print("✅ Memory system import successful")
    except ImportError as e:
        print(f"❌ Memory system import failed: {e}")
        return False
    
    try:
        from enhanced_llm import EnhancedLLMInterface
        print("✅ Enhanced LLM import successful")
    except ImportError as e:
        print(f"❌ Enhanced LLM import failed: {e}")
        return False
    
    try:
        from text_chat_companion import TextChatCompanion, Config
        print("✅ Text chat companion import successful")
    except ImportError as e:
        print(f"❌ Text chat companion import failed: {e}")
        return False
    
    return True

def test_ollama_connection():
    """Test Ollama connection and model availability."""
    print("\n🔗 Testing Ollama connection...")
    
    try:
        import ollama
        response = ollama.list()
        models = response.get('models', [])
        
        if not models:
            print("❌ No Ollama models found. Please install a model:")
            print("   ollama pull llama3.2:1b")
            return False

        print(f"✅ Ollama connection successful. Found {len(models)} models:")
        for model in models[:3]:  # Show first 3 models
            # Handle different response formats
            if isinstance(model, dict):
                model_name = model.get('name', model.get('model', str(model)))
            else:
                model_name = str(model)
            print(f"   - {model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return False

def test_memory_system():
    """Test memory system initialization."""
    print("\n🧠 Testing memory system...")
    
    try:
        from Memory import MemorySystem
        
        # Initialize memory system with test database
        memory = MemorySystem(
            db_path="Memory/Database/test_memory.db",
            max_context_history=10,
            context_window=5
        )
        
        # Test adding an interaction
        result = memory.add_interaction(
            user_input="Hello, this is a test",
            ai_response="Hello! This is a test response.",
            metadata={"test": True}
        )
        
        if result.get('context_updated') and result.get('database_stored'):
            print("✅ Memory system test successful")
            
            # Test getting context
            context = memory.get_response_context()
            if context and 'context_memory' in context:
                print("✅ Context retrieval successful")
                return True
            else:
                print("❌ Context retrieval failed")
                return False
        else:
            print("❌ Memory system test failed")
            return False
            
    except Exception as e:
        print(f"❌ Memory system test failed: {e}")
        return False

def test_enhanced_llm():
    """Test enhanced LLM interface."""
    print("\n🤖 Testing enhanced LLM interface...")
    
    try:
        from enhanced_llm import EnhancedLLMInterface
        
        # Initialize with a lightweight model
        llm = EnhancedLLMInterface(
            model_name="llama3.2:1b",
            config={"temperature": 0.7, "max_tokens": 50}
        )
        
        # Test response generation
        response, metadata = llm.generate_response(
            "Hello, this is a test message. Please respond briefly.",
            additional_context=""
        )
        
        if response and len(response.strip()) > 0:
            print("✅ Enhanced LLM test successful")
            print(f"   Response: {response[:100]}...")
            print(f"   Quality score: {metadata.get('validation_score', 'N/A')}")
            return True
        else:
            print("❌ Enhanced LLM test failed - empty response")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced LLM test failed: {e}")
        return False

def test_text_chat_companion():
    """Test text chat companion initialization."""
    print("\n💬 Testing text chat companion...")
    
    try:
        from text_chat_companion import TextChatCompanion
        
        # Initialize companion
        companion = TextChatCompanion(
            ollama_model="llama3.2:1b",
            internet_mode="offline"
        )
        
        print("✅ Text chat companion initialization successful")
        
        # Test input processing
        test_input = "Hello, this is a test message"
        response = companion.process_user_input(test_input)
        
        if response and len(response.strip()) > 0:
            print("✅ Input processing test successful")
            print(f"   Response: {response[:100]}...")
            return True
        else:
            print("❌ Input processing test failed")
            return False
            
    except Exception as e:
        print(f"❌ Text chat companion test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Text Chat Companion Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Ollama Connection", test_ollama_connection),
        ("Memory System", test_memory_system),
        ("Enhanced LLM", test_enhanced_llm),
        ("Text Chat Companion", test_text_chat_companion)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The text chat companion is ready to use.")
        print("\nTo start the application, run:")
        print("   python text_chat_companion.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        
        if passed == 0:
            print("\n💡 Quick troubleshooting:")
            print("   1. Make sure Ollama is running: ollama serve")
            print("   2. Install a model: ollama pull llama3.2:1b")
            print("   3. Check dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
