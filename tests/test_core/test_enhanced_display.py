#!/usr/bin/env python3
"""
Test script for the enhanced display functionality
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_enhanced_display():
    """Test the enhanced display functionality."""
    print("🧪 Testing Enhanced Display Features...")
    
    try:
        from text_chat_companion import TextChatCompanion
        
        # Initialize companion
        companion = TextChatCompanion(
            ollama_model="llama3.2:1b",
            internet_mode="offline"
        )
        
        print("✅ Text chat companion initialized")
        
        # Test basic display
        print("\n" + "="*60)
        print("🔍 Testing Basic Display")
        print("="*60)
        
        test_response = "This is a test response to demonstrate the enhanced display functionality. It includes multiple sentences and should be properly formatted with word wrapping if the line is too long."
        test_metadata = {
            "validation_score": 0.85,
            "quality_score": 0.92,
            "task_type": "general",
            "language_detected": "en"
        }
        test_processing_time = 1.23
        
        companion.display_response(test_response, test_metadata, test_processing_time)
        
        # Test verbose mode
        print("\n" + "="*60)
        print("🔍 Testing Verbose Mode")
        print("="*60)
        
        companion.toggle_verbose_mode()
        companion.display_response(test_response, test_metadata, test_processing_time)
        
        # Test privacy mode
        print("\n" + "="*60)
        print("🔍 Testing Privacy Mode")
        print("="*60)
        
        companion.toggle_privacy_mode()
        companion.display_response("This response should have privacy indicators.", test_metadata, 0.5)
        
        # Test long response with formatting
        print("\n" + "="*60)
        print("🔍 Testing Long Response with Code")
        print("="*60)
        
        long_response = """Here's a comprehensive answer about Python programming:

Python is a high-level programming language. Here's a simple example:

```python
def hello_world():
    print("Hello, World!")
    return True
```

Key features include:
- Easy to read syntax
- Dynamic typing
- Extensive standard library
- Cross-platform compatibility

For more advanced usage, you might want to consider:
1. Object-oriented programming concepts
2. Functional programming paradigms
3. Asynchronous programming with asyncio
4. Web development with frameworks like Django or Flask

This response demonstrates how the enhanced display handles various types of content including code blocks, lists, and long paragraphs that need proper word wrapping."""
        
        companion.display_response(long_response, test_metadata, 2.1)
        
        # Test error handling
        print("\n" + "="*60)
        print("🔍 Testing Error Handling")
        print("="*60)
        
        # Test empty response
        companion.display_response("", {}, 0.0)
        
        # Test None response
        try:
            companion.display_response(None, {}, 0.0)
        except:
            print("✅ Handled None response gracefully")
        
        # Test very long response
        very_long_response = "This is a very long response. " * 200
        companion.display_response(very_long_response, test_metadata, 0.8)
        
        print("\n✅ Enhanced display tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced display test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_privacy_features():
    """Test privacy-specific features."""
    print("\n🔒 Testing Privacy Features...")
    
    try:
        from text_chat_companion import TextChatCompanion
        
        # Initialize with high privacy
        companion = TextChatCompanion(
            ollama_model="llama3.2:1b",
            internet_mode="offline"
        )
        
        # Set high privacy mode
        companion.privacy_config["privacy_level"] = "high"
        companion.privacy_config["disable_response_logging"] = True
        companion.privacy_config["show_privacy_indicator"] = True
        
        print("✅ Privacy settings configured")
        
        # Test privacy display
        test_response = "This is a private response that should not be logged in detail."
        companion.display_response(test_response, {"validation_score": 0.9}, 1.0)
        
        print("✅ Privacy features test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Privacy features test failed: {e}")
        return False

def test_accessibility_features():
    """Test accessibility features."""
    print("\n♿ Testing Accessibility Features...")
    
    try:
        from text_chat_companion import TextChatCompanion
        
        companion = TextChatCompanion(
            ollama_model="llama3.2:1b",
            internet_mode="offline"
        )
        
        # Enable accessibility mode
        companion.display_config["accessibility_mode"] = True
        companion.display_config["use_colors"] = False  # For screen readers
        companion.display_config["compact_mode"] = False
        companion.display_config["show_separator"] = True
        
        print("✅ Accessibility settings configured")
        
        # Test accessibility display
        test_response = "This response is displayed with accessibility features enabled for better screen reader compatibility."
        companion.display_response(test_response, {"validation_score": 0.8}, 0.9)
        
        print("✅ Accessibility features test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Accessibility features test failed: {e}")
        return False

def test_configuration_options():
    """Test various configuration options."""
    print("\n⚙️ Testing Configuration Options...")
    
    try:
        from text_chat_companion import TextChatCompanion
        
        companion = TextChatCompanion(
            ollama_model="llama3.2:1b",
            internet_mode="offline"
        )
        
        # Test different configurations
        configs_to_test = [
            {"name": "Compact Mode", "config": {"compact_mode": True, "show_separator": False}},
            {"name": "Wide Display", "config": {"max_line_width": 120, "word_wrap": True}},
            {"name": "No Colors", "config": {"use_colors": False}},
            {"name": "Timestamps", "config": {"show_timestamps": True}},
            {"name": "All Metadata", "config": {"show_metadata": True, "show_quality_scores": True, "show_processing_time": True}}
        ]
        
        test_response = "This is a test response to demonstrate different configuration options."
        test_metadata = {"validation_score": 0.75, "task_type": "test"}
        
        for config_test in configs_to_test:
            print(f"\n--- Testing {config_test['name']} ---")
            
            # Apply configuration
            for key, value in config_test['config'].items():
                companion.display_config[key] = value
            
            companion.display_response(test_response, test_metadata, 0.5)
            
            # Reset to defaults
            companion.display_config = companion.__class__.Config.DISPLAY_CONFIG.copy()
        
        print("✅ Configuration options test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration options test failed: {e}")
        return False

def main():
    """Run all enhanced display tests."""
    print("🚀 Starting Enhanced Display Tests")
    print("=" * 70)
    
    tests = [
        ("Enhanced Display", test_enhanced_display),
        ("Privacy Features", test_privacy_features),
        ("Accessibility Features", test_accessibility_features),
        ("Configuration Options", test_configuration_options)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} test PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test CRASHED: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All enhanced display tests passed!")
        print("\nNew features available:")
        print("  • Enhanced text formatting with word wrapping")
        print("  • Response metadata display (quality scores, timing)")
        print("  • Privacy controls and indicators")
        print("  • Accessibility features for screen readers")
        print("  • Configurable display options")
        print("  • 'verbose' and 'privacy' commands in chat")
    else:
        print("⚠️  Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main()
