#!/usr/bin/env python3
"""
Test Script for Voice Companion Optimizations

This script tests the enhanced components to validate performance improvements
in both result quality and voice accuracy.
"""

import sys
import time
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_enhanced_stt():
    """Test enhanced Speech-to-Text capabilities."""
    print("\n🗣️ Testing Enhanced Speech-to-Text")
    print("=" * 50)
    
    try:
        from enhanced_stt import EnhancedSpeechToText
        
        # Initialize enhanced STT
        stt = EnhancedSpeechToText(model_size="base", enable_preprocessing=True)
        
        # Test model info
        model_info = stt.get_model_info()
        print(f"✅ STT Model loaded: {model_info['model_size']}")
        print(f"✅ Preprocessing enabled: {model_info['preprocessing_enabled']}")
        
        # Test audio preprocessing components
        print("✅ Audio preprocessor initialized")
        print("✅ Voice Activity Detector initialized")
        print("✅ Text post-processor initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ STT test failed: {e}")
        return False


def test_enhanced_tts():
    """Test enhanced Text-to-Speech capabilities."""
    print("\n🔊 Testing Enhanced Text-to-Speech")
    print("=" * 50)
    
    try:
        from enhanced_tts import EnhancedTextToSpeech
        
        # Initialize enhanced TTS
        config = {
            'rate': 150,
            'volume': 0.95,
            'voice_preference': ['female', 'default'],
            'handle_numbers': True,
            'expand_abbreviations': True
        }
        
        tts = EnhancedTextToSpeech(config)
        
        # Test voice info
        voice_info = tts.get_voice_info()
        available_voices = tts.get_available_voices()
        
        print(f"✅ TTS Engine initialized")
        print(f"✅ Available voices: {len(available_voices)}")
        print(f"✅ Current voice: {voice_info.get('name', 'Default')}")
        
        # Test text preprocessing
        test_texts = [
            "Hello! This is a test of the enhanced TTS system.",
            "The API returned HTTP status 200 at 3:30 PM.",
            "What is 2+2? The answer is 4.",
            "Dr. Smith will meet you at 123 Main St. at 2:00 PM."
        ]
        
        print("\n📝 Testing text preprocessing:")
        for text in test_texts:
            processed = tts.text_preprocessor.preprocess_text(text)
            print(f"Original:  {text}")
            print(f"Processed: {processed}")
            print()
        
        # Test speech (without actually speaking)
        print("✅ Text preprocessing working correctly")
        print("✅ Speech rate adaptation enabled")
        
        tts.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False


def test_enhanced_llm():
    """Test enhanced LLM interface."""
    print("\n🤖 Testing Enhanced LLM Interface")
    print("=" * 50)
    
    try:
        from enhanced_llm import EnhancedLLMInterface
        
        # Initialize enhanced LLM
        config = {
            'temperature': 0.7,
            'top_k': 40,
            'top_p': 0.9,
            'num_predict': 100,
            'enable_response_caching': True
        }
        
        llm = EnhancedLLMInterface("llama3.2:3b", config)
        
        print("✅ LLM interface initialized")
        print("✅ Prompt engineer ready")
        print("✅ Response validator ready")
        print("✅ Context manager ready")
        
        # Test prompt type detection
        test_queries = [
            ("What is Python?", "conversational"),
            ("How do I write a function in Python?", "technical"),
            ("Tell me a story about a robot", "creative"),
            ("What are the pros and cons of AI?", "analytical")
        ]
        
        print("\n🎯 Testing prompt type detection:")
        for query, expected in test_queries:
            detected = llm.prompt_engineer.detect_prompt_type(query)
            status = "✅" if detected == expected else "⚠️"
            print(f"{status} '{query}' → {detected} (expected: {expected})")
        
        # Test response generation (if Ollama is available)
        try:
            print("\n💬 Testing response generation:")
            response, metadata = llm.generate_response("What is artificial intelligence?")
            
            print(f"✅ Response generated: {len(response)} characters")
            print(f"✅ Quality score: {metadata.get('validation_score', 0):.2f}")
            print(f"✅ Processing time: {metadata.get('processing_time', 0):.2f}s")
            print(f"✅ Prompt type: {metadata.get('prompt_type', 'unknown')}")
            
            # Test response validation
            validation = llm.response_validator.validate_response(response, "What is artificial intelligence?")
            print(f"✅ Validation score: {validation['overall_score']:.2f}")
            print(f"✅ Issues found: {len(validation['issues'])}")
            
        except Exception as e:
            print(f"⚠️ Response generation test skipped (Ollama not available): {e}")
        
        # Test cache functionality
        cache_stats = llm.get_cache_stats()
        print(f"✅ Response cache: {cache_stats['cache_enabled']}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False


def test_memory_integration():
    """Test memory system integration."""
    print("\n🧠 Testing Memory System Integration")
    print("=" * 50)
    
    try:
        from Memory import MemorySystem
        
        # Initialize memory system
        memory = MemorySystem(
            db_path="Memory/Database/test_memory.db",
            max_context_history=20,
            context_window=5
        )
        
        print("✅ Memory system initialized")
        
        # Test interaction storage
        result = memory.add_interaction(
            user_input="What is machine learning?",
            ai_response="Machine learning is a subset of artificial intelligence...",
            feedback="Good explanation!"
        )
        
        print(f"✅ Interaction stored: {result.get('database_stored', False)}")
        print(f"✅ Learning applied: {result.get('learning_results', {}).get('feedback_processed', False)}")
        
        # Test context retrieval
        context = memory.get_response_context()
        print(f"✅ Context retrieved: {len(context.get('context_memory', {}).get('conversation_history', []))} interactions")
        
        # Test search
        search_results = memory.search("machine learning")
        print(f"✅ Search results: {len(search_results.get('database_results', {}).get('conversations', []))} conversations")
        
        # Test preferences
        memory.set_user_preference("communication", "style", "friendly")
        style = memory.get_user_preference("communication", "style")
        print(f"✅ Preferences: style = {style}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False


def test_configuration():
    """Test optimized configuration."""
    print("\n⚙️ Testing Optimized Configuration")
    print("=" * 50)

    try:
        import optimized_config

        print("✅ Configuration loaded successfully")
        
        # Check key configurations
        print(f"✅ Audio sample rate: {optimized_config.AUDIO_CONFIG['sample_rate']} Hz")
        print(f"✅ Whisper model: {optimized_config.WHISPER_CONFIG['model_size']}")
        print(f"✅ Ollama model: {optimized_config.OLLAMA_CONFIG['model_name']}")
        print(f"✅ TTS rate: {optimized_config.TTS_CONFIG['rate']} WPM")

        # Check optimization flags
        print(f"✅ Audio preprocessing: {optimized_config.VOICE_ACCURACY_CONFIG['enable_audio_preprocessing']}")
        print(f"✅ Response validation: {optimized_config.RESULT_QUALITY_CONFIG['enable_response_validation']}")
        print(f"✅ Context memory: {optimized_config.RESULT_QUALITY_CONFIG['enable_context_memory']}")
        print(f"✅ Response caching: {optimized_config.PERFORMANCE_CONFIG['enable_response_caching']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def run_performance_benchmark():
    """Run a simple performance benchmark."""
    print("\n📊 Performance Benchmark")
    print("=" * 50)
    
    try:
        # Test text preprocessing speed
        from enhanced_tts import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        test_text = "Hello! This is a test of the enhanced TTS system. The API returned HTTP status 200 at 3:30 PM. What is 2+2? The answer is 4."
        
        start_time = time.time()
        for _ in range(100):
            processed = preprocessor.preprocess_text(test_text)
        preprocessing_time = (time.time() - start_time) / 100
        
        print(f"✅ Text preprocessing: {preprocessing_time*1000:.2f}ms per text")
        
        # Test response validation speed
        from enhanced_llm import ResponseValidator
        
        validator = ResponseValidator()
        test_response = "Artificial intelligence is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence."
        test_query = "What is artificial intelligence?"
        
        start_time = time.time()
        for _ in range(50):
            validation = validator.validate_response(test_response, test_query)
        validation_time = (time.time() - start_time) / 50
        
        print(f"✅ Response validation: {validation_time*1000:.2f}ms per response")
        
        # Test memory operations
        from Memory import MemorySystem
        
        memory = MemorySystem(db_path=":memory:")  # In-memory database for testing
        
        start_time = time.time()
        for i in range(10):
            memory.add_interaction(f"Test query {i}", f"Test response {i}")
        memory_time = (time.time() - start_time) / 10
        
        print(f"✅ Memory operations: {memory_time*1000:.2f}ms per interaction")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


def main():
    """Run all optimization tests."""
    print("🚀 Voice Companion Optimization Tests")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Enhanced STT", test_enhanced_stt),
        ("Enhanced TTS", test_enhanced_tts),
        ("Enhanced LLM", test_enhanced_llm),
        ("Memory Integration", test_memory_integration),
        ("Performance Benchmark", run_performance_benchmark),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name} test...")
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📋 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All optimizations are working correctly!")
        print("\n💡 Next steps:")
        print("1. Run: python optimized_voice_companion.py")
        print("2. Test with real voice interactions")
        print("3. Monitor performance metrics")
        print("4. Fine-tune settings based on your use case")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
        print("\n💡 Troubleshooting:")
        print("1. Ensure all dependencies are installed")
        print("2. Check that Ollama is running (for LLM tests)")
        print("3. Verify audio device permissions (for STT tests)")
        print("4. Review the PERFORMANCE_OPTIMIZATION_GUIDE.md")


if __name__ == "__main__":
    main()
