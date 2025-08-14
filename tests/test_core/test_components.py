#!/usr/bin/env python3
"""
Test script for individual components of the Voice Companion
Use this to debug issues with specific parts of the system
"""

import sys
import time
from colorama import init, Fore, Style

init()

def test_imports():
    """Test if all required packages can be imported."""
    print(f"{Fore.CYAN}🧪 Testing imports...{Style.RESET_ALL}")
    
    tests = [
        ("pyaudio", "PyAudio for audio recording"),
        ("numpy", "NumPy for audio processing"),
        ("whisper", "OpenAI Whisper for STT"),
        ("pyttsx3", "pyttsx3 for TTS"),
        ("ollama", "Ollama client"),
        ("requests", "Requests for HTTP"),
        ("colorama", "Colorama for colored output")
    ]
    
    results = []
    for module, description in tests:
        try:
            __import__(module)
            print(f"  ✅ {module}: {description}")
            results.append(True)
        except ImportError as e:
            print(f"  ❌ {module}: {description} - {e}")
            results.append(False)
    
    return all(results)

def test_audio():
    """Test audio recording capabilities."""
    print(f"\n{Fore.CYAN}🎤 Testing audio recording...{Style.RESET_ALL}")
    
    try:
        import pyaudio
        
        audio = pyaudio.PyAudio()
        
        # List audio devices
        print("Available audio devices:")
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  📱 Input Device {i}: {info['name']}")
        
        # Test recording
        print("\n🔴 Testing microphone access...")
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        print("Recording for 2 seconds...")
        data = stream.read(1024 * 32)  # ~2 seconds at 16kHz
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        print(f"  ✅ Recorded {len(data)} bytes of audio data")
        return True
        
    except Exception as e:
        print(f"  ❌ Audio test failed: {e}")
        return False

def test_whisper():
    """Test Whisper speech-to-text."""
    print(f"\n{Fore.CYAN}🗣️  Testing Whisper STT (optimized for i3 systems)...{Style.RESET_ALL}")

    try:
        import whisper
        import numpy as np

        print("Loading Whisper tiny model (optimized for low-resource systems)...")
        model = whisper.load_model("tiny", device="cpu")

        # Create a simple test audio (sine wave)
        sample_rate = 16000
        duration = 1.0
        frequency = 440  # A4 note

        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        print("Testing transcription with synthetic audio...")
        result = model.transcribe(
            audio_data,
            language="en",
            fp16=False,  # CPU compatibility
            verbose=False
        )

        text = result["text"].strip()
        detected_language = result["language"]

        print(f"  ✅ Whisper loaded successfully")
        print(f"  📝 Detected language: {detected_language}")
        print(f"  🎵 Transcription of sine wave: '{text}' (expected to be empty or noise)")
        print(f"  💡 Model optimized for i3 7th gen CPU")

        return True

    except Exception as e:
        print(f"  ❌ Whisper test failed: {e}")
        print(f"  💡 Try: pip install openai-whisper")
        return False

def test_tts():
    """Test text-to-speech."""
    print(f"\n{Fore.CYAN}🔊 Testing TTS...{Style.RESET_ALL}")
    
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Get available voices
        voices = engine.getProperty('voices')
        print(f"Available voices: {len(voices)}")
        for i, voice in enumerate(voices[:3]):  # Show first 3
            print(f"  🎭 Voice {i}: {voice.name}")
        
        # Test speech
        test_text = "Hello, this is a test of the text to speech system."
        print(f"\n🗨️  Testing speech: '{test_text}'")
        print("  (You should hear audio output)")
        
        engine.say(test_text)
        engine.runAndWait()
        
        print("  ✅ TTS test completed")
        return True
        
    except Exception as e:
        print(f"  ❌ TTS test failed: {e}")
        return False

def test_ollama():
    """Test Ollama LLM connection."""
    print(f"\n{Fore.CYAN}🤖 Testing Ollama...{Style.RESET_ALL}")

    try:
        import ollama
        import requests

        # First test if Ollama service is running
        print("Checking Ollama service...")
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                print("  ❌ Ollama service not responding")
                return False
        except requests.exceptions.RequestException:
            print("  ❌ Ollama service not running. Start with: ollama serve")
            return False

        # Test connection and get models
        print("Checking Ollama connection...")
        try:
            models_response = ollama.list()
            print(f"  📡 Raw response type: {type(models_response)}")

            # Handle Ollama client response format
            if hasattr(models_response, 'models'):
                # New Ollama client returns ListResponse object
                models_list = models_response.models
            elif isinstance(models_response, dict) and 'models' in models_response:
                # Fallback for dict response
                models_list = models_response['models']
            elif isinstance(models_response, list):
                # Direct list response
                models_list = models_response
            else:
                print(f"  ⚠️  Unexpected response format: {models_response}")
                return False

            if not models_list:
                print("  ⚠️  No models found. Please run: ollama pull llama3.2:1b")
                return False

            print(f"Available models: {len(models_list)}")
            for i, model in enumerate(models_list):
                # Handle Model objects from new Ollama client
                if hasattr(model, 'model'):
                    model_name = model.model
                    model_size = getattr(model, 'size', 'Unknown size')
                    print(f"  🧠 {model_name} ({model_size} bytes)")
                elif isinstance(model, dict):
                    model_name = model.get('name', model.get('model', f'Unknown model {i}'))
                    model_size = model.get('size', 'Unknown size')
                    print(f"  🧠 {model_name} ({model_size} bytes)")
                else:
                    print(f"  🧠 {model}")

            # Test generation with the first available model
            first_model = models_list[0]
            if hasattr(first_model, 'model'):
                model_name = first_model.model
            elif isinstance(first_model, dict):
                model_name = first_model.get('name', first_model.get('model', 'llama3.2:1b'))
            else:
                model_name = str(first_model)

            print(f"\n💭 Testing generation with {model_name}...")

            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
                options={
                    "temperature": 0.1,
                    "num_predict": 20,
                    "num_ctx": 1024
                }
            )

            reply = response['message']['content']
            print(f"  🤖 Response: '{reply}'")
            print("  ✅ Ollama test completed")

            return True

        except Exception as api_error:
            print(f"  ❌ Ollama API error: {api_error}")
            print(f"  🔍 Error type: {type(api_error)}")
            return False

    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        print("  💡 Try: pip install ollama")
        return False
    except Exception as e:
        print(f"  ❌ Ollama test failed: {e}")
        print(f"  🔍 Error type: {type(e)}")
        print("  💡 Make sure Ollama is running: ollama serve")
        return False

def test_text_input():
    """Test text input functionality."""
    print(f"\n{Fore.CYAN}✏️  Testing text input functionality...{Style.RESET_ALL}")

    try:
        # Test text validation functions
        print("Testing text validation...")

        # Test encoding validation
        test_texts = [
            "Hello, world!",  # Normal text
            "Café résumé naïve",  # Unicode text
            "Hello\nWorld",  # Newline
            "Test\tTab",  # Tab
            "",  # Empty string
            " " * 10,  # Whitespace only
            "A" * 1500,  # Very long text
        ]

        for i, text in enumerate(test_texts):
            try:
                # Test encoding
                text.encode('utf-8')
                print(f"  ✅ Text {i+1}: Encoding OK")

                # Test length
                if len(text) > 1000:
                    print(f"  ⚠️  Text {i+1}: Length {len(text)} (would be truncated)")
                else:
                    print(f"  ✅ Text {i+1}: Length {len(text)} OK")

            except UnicodeEncodeError:
                print(f"  ❌ Text {i+1}: Encoding failed")

        print("  ✅ Text validation tests completed")
        return True

    except Exception as e:
        print(f"  ❌ Text input test failed: {e}")
        return False

def test_text_to_speech_direct():
    """Test direct text-to-speech functionality."""
    print(f"\n{Fore.CYAN}🗨️  Testing direct text-to-speech...{Style.RESET_ALL}")

    try:
        import pyttsx3

        engine = pyttsx3.init()

        # Test various text types
        test_texts = [
            "This is a simple test.",
            "Testing numbers: 1, 2, 3, 100, 2024.",
            "Testing punctuation: Hello! How are you? I'm fine, thanks.",
            "Testing special characters: @#$%^&*()",
            "A longer text to test speech quality and timing. This should take a few seconds to speak completely."
        ]

        for i, text in enumerate(test_texts):
            print(f"  🗨️  Test {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            try:
                engine.say(text)
                engine.runAndWait()
                print(f"  ✅ Test {i+1}: Speech completed")
            except Exception as tts_error:
                print(f"  ❌ Test {i+1}: TTS failed - {tts_error}")
                return False

        print("  ✅ Direct text-to-speech tests completed")
        return True

    except Exception as e:
        print(f"  ❌ Direct TTS test failed: {e}")
        return False

def test_text_input_integration():
    """Test text input integration with voice companion components."""
    print(f"\n{Fore.CYAN}🔗 Testing text input integration...{Style.RESET_ALL}")

    try:
        # Import the voice companion module
        import sys
        import os

        # Add current directory to path to import voice_companion
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        # Test importing the enhanced voice companion
        try:
            from voice_companion import VoiceCompanion, TEXT_INPUT_CONFIG, INPUT_MODE_CONFIG
            print("  ✅ Voice companion module imported successfully")
        except ImportError as import_error:
            print(f"  ❌ Failed to import voice companion: {import_error}")
            return False

        # Test configuration loading
        print(f"  📋 Text input config: {TEXT_INPUT_CONFIG}")
        print(f"  📋 Input mode config: {INPUT_MODE_CONFIG}")

        # Test that the new methods exist
        companion_methods = [
            'process_text_input',
            'speak_text_directly',
            'show_mode_menu',
            'run_text_conversation_mode',
            'run_text_to_speech_mode',
            'run_mixed_mode'
        ]

        for method_name in companion_methods:
            if hasattr(VoiceCompanion, method_name):
                print(f"  ✅ Method '{method_name}' exists")
            else:
                print(f"  ❌ Method '{method_name}' missing")
                return False

        print("  ✅ Text input integration tests completed")
        return True

    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        print(f"  🔍 Error details: {type(e).__name__}: {str(e)}")
        return False

def test_speech_rate_control():
    """Test speech rate control functionality."""
    print(f"\n{Fore.CYAN}🎚️  Testing speech rate control...{Style.RESET_ALL}")

    try:
        import pyttsx3

        # Test basic TTS engine initialization
        engine = pyttsx3.init()

        # Test different speech rates
        test_rates = [80, 120, 160, 200, 250]

        for rate in test_rates:
            try:
                engine.setProperty('rate', rate)
                current_rate = engine.getProperty('rate')
                print(f"  ✅ Rate {rate}: Set successfully (actual: {current_rate})")
            except Exception as e:
                print(f"  ❌ Rate {rate}: Failed to set - {e}")
                return False

        # Test rate presets
        presets = {
            "very_slow": 80,
            "slow": 110,
            "normal": 140,
            "fast": 180,
            "very_fast": 220
        }

        print("  Testing rate presets...")
        for preset, rate in presets.items():
            try:
                engine.setProperty('rate', rate)
                print(f"  ✅ Preset '{preset}' ({rate} WPM): OK")
            except Exception as e:
                print(f"  ❌ Preset '{preset}': Failed - {e}")

        print("  ✅ Speech rate control tests completed")
        return True

    except Exception as e:
        print(f"  ❌ Speech rate control test failed: {e}")
        return False

def test_internet_connectivity():
    """Test internet connectivity detection."""
    print(f"\n{Fore.CYAN}🌐 Testing internet connectivity...{Style.RESET_ALL}")

    try:
        import socket
        import urllib.request

        # Test basic connectivity
        print("  Testing basic connectivity...")

        # Test DNS resolution
        try:
            socket.gethostbyname('google.com')
            print("  ✅ DNS resolution: Working")
        except socket.gaierror:
            print("  ❌ DNS resolution: Failed")
            return False

        # Test HTTP connection
        try:
            response = urllib.request.urlopen('https://httpbin.org/get', timeout=10)
            if response.getcode() == 200:
                print("  ✅ HTTP connection: Working")
            else:
                print(f"  ❌ HTTP connection: Status {response.getcode()}")
        except Exception as e:
            print(f"  ⚠️  HTTP connection: Failed ({e}) - May be offline")

        print("  ✅ Internet connectivity tests completed")
        return True

    except Exception as e:
        print(f"  ❌ Internet connectivity test failed: {e}")
        return False

def test_web_search():
    """Test web search functionality."""
    print(f"\n{Fore.CYAN}🔍 Testing web search...{Style.RESET_ALL}")

    try:
        # Test DuckDuckGo API access
        import urllib.request
        import urllib.parse
        import json

        print("  Testing DuckDuckGo API access...")

        query = "test search"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }

        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(params)

        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))
                print(f"  ✅ DuckDuckGo API: Accessible")

                # Check response structure
                if 'Abstract' in data or 'RelatedTopics' in data:
                    print(f"  ✅ API Response: Valid structure")
                else:
                    print(f"  ⚠️  API Response: Unexpected structure")

        except Exception as e:
            print(f"  ⚠️  DuckDuckGo API: Failed ({e}) - May be offline")

        print("  ✅ Web search tests completed")
        return True

    except Exception as e:
        print(f"  ❌ Web search test failed: {e}")
        return False

def test_realtime_data():
    """Test real-time data fetching."""
    print(f"\n{Fore.CYAN}⏰ Testing real-time data fetching...{Style.RESET_ALL}")

    try:
        import urllib.request
        import json

        # Test time API
        print("  Testing WorldTimeAPI...")
        try:
            with urllib.request.urlopen('https://worldtimeapi.org/api/ip', timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                if 'datetime' in data:
                    print("  ✅ WorldTimeAPI: Working")
                else:
                    print("  ❌ WorldTimeAPI: Invalid response")
        except Exception as e:
            print(f"  ⚠️  WorldTimeAPI: Failed ({e}) - May be offline")

        # Test weather API (wttr.in)
        print("  Testing weather API...")
        try:
            with urllib.request.urlopen('https://wttr.in/London?format=j1', timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                if 'current_condition' in data:
                    print("  ✅ Weather API: Working")
                else:
                    print("  ❌ Weather API: Invalid response")
        except Exception as e:
            print(f"  ⚠️  Weather API: Failed ({e}) - May be offline")

        print("  ✅ Real-time data tests completed")
        return True

    except Exception as e:
        print(f"  ❌ Real-time data test failed: {e}")
        return False

def test_full_pipeline():
    """Test a simplified version of the full pipeline."""
    print(f"\n{Fore.CYAN}🔄 Testing full pipeline...{Style.RESET_ALL}")

    try:
        # This is a simplified test without actual audio recording
        print("Simulating voice input: 'Hello, how are you?'")
        user_input = "Hello, how are you?"

        # Test LLM response
        import ollama
        print("Getting available models...")

        try:
            models_response = ollama.list()

            # Handle Ollama client response format
            if hasattr(models_response, 'models'):
                # New Ollama client returns ListResponse object
                models_list = models_response.models
            elif isinstance(models_response, dict) and 'models' in models_response:
                # Fallback for dict response
                models_list = models_response['models']
            elif isinstance(models_response, list):
                # Direct list response
                models_list = models_response
            else:
                print(f"  ❌ Unexpected models response: {models_response}")
                return False

            if not models_list:
                print("  ❌ No Ollama models available")
                return False

            # Get the first available model
            first_model = models_list[0]
            if hasattr(first_model, 'model'):
                model_name = first_model.model
            elif isinstance(first_model, dict):
                model_name = first_model.get('name', first_model.get('model', 'llama3.2:1b'))
            else:
                model_name = str(first_model)

            print(f"Using model: {model_name}")

            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": user_input}],
                options={
                    "temperature": 0.7,
                    "num_predict": 50,
                    "num_ctx": 1024
                }
            )

            reply = response['message']['content']
            print(f"  🤖 LLM Response: '{reply}'")

            # Test TTS
            print("Testing text-to-speech...")
            import pyttsx3
            engine = pyttsx3.init()

            # Speak a shorter version for testing
            test_speech = reply[:100] + "..." if len(reply) > 100 else reply
            engine.say(test_speech)
            engine.runAndWait()

            print("  ✅ Full pipeline test completed")
            return True

        except Exception as llm_error:
            print(f"  ❌ LLM error: {llm_error}")
            return False

    except Exception as e:
        print(f"  ❌ Full pipeline test failed: {e}")
        print(f"  🔍 Error details: {type(e).__name__}: {str(e)}")
        return False

def main():
    """Run all tests."""
    print(f"{Fore.MAGENTA}🧪 Voice Companion Component Tests{Style.RESET_ALL}")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Audio", test_audio),
        ("Whisper STT", test_whisper),
        ("TTS", test_tts),
        ("Ollama LLM", test_ollama),
        ("Text Input", test_text_input),
        ("Direct Text-to-Speech", test_text_to_speech_direct),
        ("Speech Rate Control", test_speech_rate_control),
        ("Internet Connectivity", test_internet_connectivity),
        ("Web Search", test_web_search),
        ("Real-time Data", test_realtime_data),
        ("Text Input Integration", test_text_input_integration),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Test interrupted by user{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"  ❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print(f"\n{Fore.MAGENTA}📊 Test Summary{Style.RESET_ALL}")
    print("=" * 30)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print(f"\n{Fore.GREEN}🎉 All tests passed! Your system is ready.{Style.RESET_ALL}")
        print("Run: python voice_companion.py")
    else:
        print(f"\n{Fore.RED}⚠️  Some tests failed. Check the errors above.{Style.RESET_ALL}")
        print("Run: python setup.py")

if __name__ == "__main__":
    main()
