#!/usr/bin/env python3
"""
Test script to verify the hacking question issue is fixed
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_hacking_question():
    """Test the specific hacking question that was causing issues."""
    print("🧪 Testing the hacking question issue...")
    
    try:
        from text_chat_companion import TextChatCompanion
        
        # Initialize companion in offline mode to avoid internet confusion
        companion = TextChatCompanion(
            ollama_model="llama3.2:1b",
            internet_mode="offline"
        )
        
        print("✅ Text chat companion initialized")
        
        # Test the problematic question
        test_input = "Do you know about Hacking?"
        print(f"\n📝 Testing input: '{test_input}'")
        
        response = companion.process_user_input(test_input)
        
        print(f"\n🤖 AI Response:")
        print(f"'{response}'")
        
        # Check if response is relevant
        response_lower = response.lower()
        relevant_keywords = [
            'hack', 'security', 'cyber', 'computer', 'network', 'ethical',
            'penetration', 'vulnerability', 'programming', 'system'
        ]
        
        irrelevant_keywords = [
            'llama', 'model', 'download', 'transformers', 'httpx', 'github',
            'repository', 'library', 'pip install', 'modified_at'
        ]
        
        relevant_count = sum(1 for keyword in relevant_keywords if keyword in response_lower)
        irrelevant_count = sum(1 for keyword in irrelevant_keywords if keyword in response_lower)
        
        print(f"\n📊 Response Analysis:")
        print(f"Relevant keywords found: {relevant_count}")
        print(f"Irrelevant keywords found: {irrelevant_count}")
        
        if irrelevant_count > 0:
            print("❌ Response contains irrelevant technical content!")
            print("The response still seems to be confused by system context.")
            return False
        elif relevant_count > 0:
            print("✅ Response appears relevant to the hacking question!")
            return True
        else:
            print("⚠️  Response doesn't contain obvious irrelevant content, but also lacks clear relevant content.")
            print("This might be a generic response - check manually.")
            return True  # Not clearly broken
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_multiple_questions():
    """Test multiple questions to ensure consistency."""
    print("\n🧪 Testing multiple questions for consistency...")
    
    test_questions = [
        "What is Python programming?",
        "Tell me about cybersecurity",
        "How does machine learning work?",
        "What are the basics of networking?",
        "Explain ethical hacking"
    ]
    
    try:
        from text_chat_companion import TextChatCompanion
        
        companion = TextChatCompanion(
            ollama_model="llama3.2:1b",
            internet_mode="offline"
        )
        
        results = []
        
        for question in test_questions:
            print(f"\n📝 Testing: '{question}'")
            response = companion.process_user_input(question)
            
            # Check for irrelevant technical content
            response_lower = response.lower()
            irrelevant_keywords = [
                'llama', 'model', 'download', 'transformers', 'httpx', 'github',
                'repository', 'library', 'pip install', 'modified_at', 'ollama'
            ]
            
            irrelevant_count = sum(1 for keyword in irrelevant_keywords if keyword in response_lower)
            
            if irrelevant_count > 0:
                print(f"❌ Response contains {irrelevant_count} irrelevant technical terms")
                results.append(False)
            else:
                print("✅ Response appears clean and relevant")
                results.append(True)
        
        success_rate = sum(results) / len(results) * 100
        print(f"\n📊 Overall Results: {success_rate:.1f}% of responses were clean")
        
        return success_rate >= 80  # 80% success rate threshold
        
    except Exception as e:
        print(f"❌ Multiple question test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Hacking Question Fix")
    print("=" * 50)
    
    tests = [
        ("Hacking Question", test_hacking_question),
        ("Multiple Questions", test_multiple_questions)
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
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The hacking question issue appears to be fixed.")
    else:
        print("⚠️  Some tests failed. The issue may not be fully resolved.")
        print("\n💡 If tests are still failing, the issue might be:")
        print("   1. Enhanced LLM context management needs further fixes")
        print("   2. Prompt engineering needs adjustment")
        print("   3. Model confusion from previous conversations")

if __name__ == "__main__":
    main()
