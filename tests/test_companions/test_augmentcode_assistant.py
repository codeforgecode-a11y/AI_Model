#!/usr/bin/env python3
"""
Test Suite for AugmentCode Private Assistant

Comprehensive testing of all enhanced features:
- Technical guidance system
- Enhanced memory with privacy controls
- Model selection and response quality
- Privacy and security features
- Tool-assisted workflows
"""

import sys
import time
import logging
from pathlib import Path

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing module imports...")
    
    try:
        from technical_guidance_system import TechnicalGuidanceSystem, GuidanceType, SecurityContext
        print("✅ Technical guidance system imported successfully")
        
        from enhanced_memory_system import EnhancedMemorySystem, PrivacyManager
        print("✅ Enhanced memory system imported successfully")
        
        # Test existing modules
        from Memory import MemorySystem
        print("✅ Base memory system imported successfully")
        
        from enhanced_llm import EnhancedLLMInterface
        print("✅ Enhanced LLM interface imported successfully")
        
        from model_selector import ModelSelector, ModelType
        print("✅ Model selector imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_technical_guidance_system():
    """Test the technical guidance system."""
    print("\n🛡️ Testing Technical Guidance System...")
    
    try:
        from technical_guidance_system import TechnicalGuidanceSystem, GuidanceType
        
        # Initialize guidance system
        guidance = TechnicalGuidanceSystem()
        
        # Test query analysis
        test_queries = [
            "How do I perform a penetration test on a web application?",
            "Write a Python function to calculate fibonacci numbers",
            "Debug this memory leak in my C++ application",
            "What tools should I use for network reconnaissance?"
        ]
        
        for query in test_queries:
            guidance_type, confidence, security_context = guidance.analyze_query_type(query)
            print(f"Query: '{query[:50]}...'")
            print(f"  Type: {guidance_type.value}, Confidence: {confidence:.2f}")
            print(f"  Security Context: {security_context.value}")
        
        # Test guidance generation
        response, metadata = guidance.generate_technical_guidance(
            "How do I use Nmap for network scanning?",
            "Authorized penetration testing context"
        )
        
        if response and len(response) > 100:
            print("✅ Technical guidance generation successful")
            print(f"  Response length: {len(response)} characters")
            print(f"  Metadata: {metadata}")
            return True
        else:
            print("❌ Technical guidance generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Technical guidance test failed: {e}")
        return False

def test_enhanced_memory_system():
    """Test the enhanced memory system with privacy controls."""
    print("\n🧠 Testing Enhanced Memory System...")
    
    try:
        from enhanced_memory_system import EnhancedMemorySystem
        
        # Test configuration
        privacy_config = {
            'privacy_level': 'high',
            'encrypt_local_storage': True,
            'sanitize_sensitive_data': True,
            'offline_only_mode': True
        }
        
        technical_config = {
            'enabled': True,
            'cybersecurity_mode': True,
            'step_by_step_explanations': True
        }
        
        # Initialize enhanced memory
        enhanced_memory = EnhancedMemorySystem(
            db_path="Memory/Database/test_enhanced_memory.db",
            privacy_config=privacy_config,
            technical_config=technical_config
        )
        
        # Test interaction storage
        test_interaction = enhanced_memory.add_interaction(
            user_input="How do I use Burp Suite for web application testing?",
            ai_response="Burp Suite is a comprehensive web application security testing tool...",
            metadata={
                'guidance_type': 'cybersecurity',
                'tools_recommended': True,
                'response_quality': 0.85
            },
            guidance_type='cybersecurity'
        )
        
        if test_interaction.get('stored'):
            print("✅ Enhanced memory interaction storage successful")
            print(f"  Enhanced features: {test_interaction.get('enhanced_features', [])}")
        else:
            print("❌ Enhanced memory interaction storage failed")
            return False
        
        # Test context retrieval
        context = enhanced_memory.get_enhanced_context(
            "Follow-up question about web security",
            "cybersecurity"
        )
        
        if context and 'privacy_status' in context:
            print("✅ Enhanced context retrieval successful")
            print(f"  Privacy level: {context['privacy_status']['level']}")
            return True
        else:
            print("❌ Enhanced context retrieval failed")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced memory test failed: {e}")
        return False

def test_privacy_manager():
    """Test privacy controls and data sanitization."""
    print("\n🔒 Testing Privacy Manager...")
    
    try:
        from enhanced_memory_system import PrivacyManager
        
        privacy_config = {
            'privacy_level': 'high',
            'encrypt_local_storage': True,
            'sanitize_sensitive_data': True
        }
        
        privacy_manager = PrivacyManager(privacy_config)
        
        # Test data sanitization
        sensitive_text = "My password is secret123 and my email is user@example.com"
        sanitized = privacy_manager.sanitize_sensitive_data(sensitive_text)

        print(f"  Original: {sensitive_text}")
        print(f"  Sanitized: {sanitized}")

        # Check if sensitive data was removed
        password_removed = 'secret123' not in sanitized
        email_removed = 'user@example.com' not in sanitized

        if password_removed and email_removed:
            print("✅ Data sanitization successful")
        else:
            print("❌ Data sanitization failed")
            if not password_removed:
                print("  Password not sanitized")
            if not email_removed:
                print("  Email not sanitized")
            return False
        
        # Test encryption (if available)
        if privacy_manager.encryption_key:
            test_data = "This is sensitive technical information"
            encrypted = privacy_manager.encrypt_data(test_data)
            decrypted = privacy_manager.decrypt_data(encrypted)
            
            if decrypted == test_data and encrypted != test_data:
                print("✅ Encryption/decryption successful")
                return True
            else:
                print("❌ Encryption/decryption failed")
                return False
        else:
            print("⚠️ Encryption not available (cryptography package may be missing)")
            return True
            
    except Exception as e:
        print(f"❌ Privacy manager test failed: {e}")
        return False

def test_model_integration():
    """Test integration with existing model systems."""
    print("\n🤖 Testing Model Integration...")
    
    try:
        # Test if Ollama is available
        import ollama
        
        try:
            models = ollama.list()
            print(f"✅ Ollama connection successful")
            print(f"  Available models: {len(models.get('models', []))}")
        except Exception as e:
            print(f"⚠️ Ollama not available: {e}")
            print("  This is expected if Ollama is not running")
            return True  # Not a failure for testing
        
        # Test model selector
        from model_selector import ModelSelector, ModelType
        
        selector = ModelSelector()
        
        test_queries = [
            "Write a Python function",
            "How do I perform SQL injection testing?",
            "What's the weather like today?"
        ]
        
        for query in test_queries:
            model_type, confidence = selector.select_model(query, {})
            print(f"Query: '{query}'")
            print(f"  Selected model: {model_type.value}, Confidence: {confidence:.2f}")
        
        print("✅ Model integration test successful")
        return True
        
    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print("\n⚙️ Testing Configuration System...")
    
    try:
        # Import the main module to test configuration
        sys.path.insert(0, str(Path(__file__).parent))
        from text_chat_companion import Config
        
        # Test privacy configuration
        privacy_config = Config.PRIVACY_CONFIG
        required_privacy_keys = [
            'privacy_level', 'encrypt_local_storage', 'offline_only_mode',
            'sanitize_sensitive_data', 'local_processing_only'
        ]
        
        for key in required_privacy_keys:
            if key not in privacy_config:
                print(f"❌ Missing privacy config key: {key}")
                return False
        
        # Test technical guidance configuration
        tech_config = Config.TECHNICAL_GUIDANCE_CONFIG
        required_tech_keys = [
            'enabled', 'cybersecurity_mode', 'penetration_testing_mode',
            'step_by_step_explanations', 'minimal_disclaimers'
        ]
        
        for key in required_tech_keys:
            if key not in tech_config:
                print(f"❌ Missing technical config key: {key}")
                return False
        
        print("✅ Configuration system test successful")
        print(f"  Privacy level: {privacy_config.get('privacy_level')}")
        print(f"  Technical guidance: {'Enabled' if tech_config.get('enabled') else 'Disabled'}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("🚀 Starting AugmentCode Assistant Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Technical Guidance System", test_technical_guidance_system),
        ("Enhanced Memory System", test_enhanced_memory_system),
        ("Privacy Manager", test_privacy_manager),
        ("Model Integration", test_model_integration),
        ("Configuration System", test_configuration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        start_time = time.time()
        
        try:
            result = test_func()
            duration = time.time() - start_time
            results.append((test_name, result, duration))
            
            if result:
                print(f"✅ {test_name} PASSED ({duration:.2f}s)")
            else:
                print(f"❌ {test_name} FAILED ({duration:.2f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            results.append((test_name, False, duration))
            print(f"💥 {test_name} CRASHED: {e} ({duration:.2f}s)")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    total_time = sum(duration for _, _, duration in results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    
    if passed == total:
        print("\n🎉 All tests passed! AugmentCode Assistant is ready to use.")
        return True
    else:
        print(f"\n⚠️ {total-passed} test(s) failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
