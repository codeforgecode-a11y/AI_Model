#!/usr/bin/env python3
"""
Offline AI Voice Jarvis
A complete offline voice Jarvis using:
- Whisper for Speech-to-Text
- Ollama for LLM responses
- pyttsx3 for Text-to-Speech
"""

import os
import sys
import time
import threading
import queue
import logging
import signal
import json
import urllib.request
import urllib.parse
import urllib.error
import socket
from typing import Optional, Dict, Any, List
from pathlib import Path

# Suppress ALSA warnings by redirecting stderr
os.environ['ALSA_PCM_CARD'] = '0'
os.environ['ALSA_PCM_DEVICE'] = '0'

# Suppress PyAudio and ALSA warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Redirect ALSA error messages to null
import contextlib
import sys

class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Suppress unused parameter warnings
        _ = exc_type, exc_val, exc_tb
        sys.stderr.close()
        sys.stderr = self._original_stderr

# Third-party imports
import pyaudio
import numpy as np
import whisper
import pyttsx3
import ollama
from colorama import init, Fore, Style

# Local imports
try:
    from config.config import (TEXT_INPUT_CONFIG, INPUT_MODE_CONFIG, TTS_CONFIG,
                       INTERNET_CONFIG, WEB_SEARCH_CONFIG, REALTIME_DATA_CONFIG)
except ImportError:
    # Fallback configuration if config.py is not available
    TEXT_INPUT_CONFIG = {
        "enabled": True,
        "max_length": 1000,
        "min_length": 1,
        "strip_whitespace": True,
        "allow_empty": False,
        "encoding": "utf-8",
        "prompt": "Enter text: ",
        "multiline": False,
        "timeout": 300
    }
    INPUT_MODE_CONFIG = {
        "default_mode": "voice",
        "allow_mode_switching": True,
        "text_only_mode": False,
        "voice_only_mode": False,
        "show_mode_menu": True
    }
    TTS_CONFIG = {
        "engine": "pyttsx3",
        "rate": 140,
        "volume": 0.9,
        "voice_preference": ["female", "zira", "default"],
        "rate_min": 80,
        "rate_max": 300,
        "rate_step": 10,
        "allow_runtime_adjustment": True,
        "rate_presets": {
            "very_slow": 80,
            "slow": 110,
            "normal": 140,
            "fast": 180,
            "very_fast": 220
        }
    }
    INTERNET_CONFIG = {
        "enabled": True,
        "default_mode": "hybrid",
        "fallback_to_offline": True,
        "connection_timeout": 10,
        "request_timeout": 30,
        "max_retries": 3,
        "user_agent": "VoiceCompanion/1.0 (Educational AI Assistant)",
        "rate_limit_delay": 1.0,
    }
    WEB_SEARCH_CONFIG = {
        "enabled": True,
        "engine": "duckduckgo",
        "max_results": 5,
        "snippet_length": 200,
        "search_timeout": 15,
        "safe_search": True,
        "region": "us-en",
    }
    REALTIME_DATA_CONFIG = {
        "weather": {"enabled": True, "api_provider": "openweathermap", "api_key": "", "default_location": "auto", "units": "metric", "timeout": 10},
        "news": {"enabled": True, "api_provider": "newsapi", "api_key": "", "country": "us", "category": "general", "max_articles": 5, "timeout": 15},
        "time": {"enabled": True, "timezone_api": "worldtimeapi", "timeout": 5}
    }

# Initialize colorama for cross-platform colored output
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_companion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AudioRecorder:
    """Handles microphone input and audio recording."""

    def __init__(self, sample_rate: int = 44100, chunk_size: int = 1024):
        # Use 44100 Hz which is supported by most hardware
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # Suppress PyAudio and ALSA initialization messages
        with SuppressStderr():
            self.audio = pyaudio.PyAudio()

        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.input_device_index, self.supported_rate = self._find_best_input_device()

    def _find_best_input_device(self) -> tuple[Optional[int], int]:
        """Find the best available input device and supported sample rate."""
        # Common sample rates to try (in order of preference for Whisper)
        preferred_rates = [44100, 48000, 22050, 16000, 8000]

        try:
            default_device = self.audio.get_default_input_device_info()
            device_index = default_device['index']

            # Test which sample rates work with this device
            for rate in preferred_rates:
                try:
                    # Test if this rate works
                    test_stream = self.audio.open(
                        format=pyaudio.paInt16,
                        channels=1,  # Try mono first
                        rate=rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=1024
                    )
                    test_stream.close()
                    logger.info(f"✅ Found working audio: Device {device_index}, Rate {rate}Hz")
                    return device_index, rate
                except Exception:
                    continue

            # If mono doesn't work, try stereo
            for rate in preferred_rates:
                try:
                    test_stream = self.audio.open(
                        format=pyaudio.paInt16,
                        channels=2,  # Try stereo
                        rate=rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=1024
                    )
                    test_stream.close()
                    logger.info(f"✅ Found working audio: Device {device_index}, Rate {rate}Hz (stereo)")
                    return device_index, rate
                except Exception:
                    continue

        except Exception:
            pass

        # Fallback: find any working input device
        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    for rate in preferred_rates:
                        try:
                            test_stream = self.audio.open(
                                format=pyaudio.paInt16,
                                channels=1,
                                rate=rate,
                                input=True,
                                input_device_index=i,
                                frames_per_buffer=1024
                            )
                            test_stream.close()
                            logger.info(f"✅ Fallback audio: Device {i}, Rate {rate}Hz")
                            return i, rate
                        except Exception:
                            continue
            except Exception:
                continue

        logger.error("❌ No working audio device found")
        return None, 44100

    def start_recording(self) -> None:
        """Start recording audio from microphone."""
        try:
            # Use the detected working sample rate
            self.sample_rate = self.supported_rate

            # Suppress ALSA errors during stream creation
            with SuppressStderr():
                # Try mono first, then stereo if needed
                try:
                    self.stream = self.audio.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=self.supported_rate,
                        input=True,
                        input_device_index=self.input_device_index,
                        frames_per_buffer=self.chunk_size,
                        stream_callback=self._audio_callback
                    )
                    self.channels = 1
                except Exception:
                    # Fallback to stereo
                    self.stream = self.audio.open(
                        format=pyaudio.paInt16,
                        channels=2,
                        rate=self.supported_rate,
                        input=True,
                        input_device_index=self.input_device_index,
                        frames_per_buffer=self.chunk_size,
                        stream_callback=self._audio_callback
                    )
                    self.channels = 2

            self.is_recording = True
            self.stream.start_stream()
            logger.info(f"🎤 Recording started: {self.supported_rate}Hz, {self.channels} channel(s)")
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            raise
    
    def stop_recording(self) -> bytes:
        """Stop recording and return audio data."""
        if self.is_recording:
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()
            
            # Collect all audio data
            audio_data = b''
            while not self.audio_queue.empty():
                audio_data += self.audio_queue.get()
            
            logger.info("🛑 Recording stopped")
            return audio_data
        return b''
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream."""
        # Suppress unused parameter warnings
        _ = frame_count, time_info, status

        if self.is_recording:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def cleanup(self):
        """Clean up audio resources."""
        if hasattr(self, 'stream') and self.stream.is_active():
            self.stream.close()
        self.audio.terminate()


class SpeechToText:
    """Handles speech-to-text conversion using Whisper."""

    def __init__(self, model_size: str = "tiny"):
        """
        Initialize Whisper model.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
                       tiny is recommended for i3 7th gen systems
        """
        self.model_size = model_size
        logger.info(f"🧠 Loading Whisper model: {model_size} (optimized for low-resource systems)")

        try:
            # Load model with CPU-optimized settings
            self.model = whisper.load_model(
                model_size,
                device="cpu",  # Force CPU for i3 systems
                download_root=None,
                in_memory=True  # Keep model in memory for faster inference
            )
            logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_data: bytes, sample_rate: int = 44100, channels: int = 1) -> str:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate

        Returns:
            Transcribed text
        """
        try:
            # Convert bytes to numpy array and normalize
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Handle stereo audio by taking only the left channel
            if channels == 2:
                audio_np = audio_np[::2]  # Take every other sample (left channel)

            # Apply automatic gain control to fix saturated audio
            max_amplitude = np.max(np.abs(audio_np))
            if max_amplitude > 0.95:  # Audio is clipped/saturated
                # Reduce gain significantly
                gain_factor = 0.3 / max_amplitude
                audio_np = audio_np * gain_factor
                logger.info(f"Applied AGC: reduced gain by {gain_factor:.3f} (max was {max_amplitude:.3f})")
            elif max_amplitude > 0.7:  # Audio is too loud
                # Moderate gain reduction
                gain_factor = 0.5 / max_amplitude
                audio_np = audio_np * gain_factor
                logger.info(f"Applied AGC: reduced gain by {gain_factor:.3f} (max was {max_amplitude:.3f})")

            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                # Simple downsampling by taking every nth sample
                downsample_factor = sample_rate // 16000
                if downsample_factor > 1:
                    audio_np = audio_np[::downsample_factor]
                    effective_sample_rate = 16000
                else:
                    effective_sample_rate = sample_rate
            else:
                effective_sample_rate = sample_rate

            # Check if audio is long enough to process
            if len(audio_np) < effective_sample_rate * 0.5:  # Less than 0.5 seconds
                logger.info("Audio too short, skipping transcription")
                return ""

            # Check audio volume/energy to detect if there's actual speech
            audio_energy = np.sqrt(np.mean(audio_np ** 2))

            # Adaptive energy threshold based on audio characteristics
            if max_amplitude > 0.95:
                # For previously saturated audio, use lower threshold
                energy_threshold = 0.005
            else:
                # Normal threshold
                energy_threshold = 0.01

            if audio_energy < energy_threshold:
                logger.info(f"Audio too quiet (energy: {audio_energy:.4f}, threshold: {energy_threshold:.4f}), skipping transcription")
                return ""

            logger.info(f"Processing audio: {len(audio_np)} samples, energy: {audio_energy:.4f}, max: {max_amplitude:.3f}")

            # Transcribe with optimized settings for low-resource systems
            result = self.model.transcribe(
                audio_np,
                language="en",           # Fixed language to avoid detection overhead
                task="transcribe",       # Explicit task
                fp16=False,             # Disable FP16 for CPU compatibility
                verbose=False,          # Reduce logging overhead
                temperature=0.0,        # Deterministic output
                best_of=1,              # Single beam search
                beam_size=1,            # Minimal beam search
                patience=1.0,           # Faster decoding
                length_penalty=1.0,     # No length penalty
                suppress_tokens="-1",   # Default suppression
                initial_prompt=None,    # No initial prompt
                condition_on_previous_text=False,  # Don't use previous context
                compression_ratio_threshold=2.4,   # Default threshold
                logprob_threshold=-1.0, # Default threshold
                no_speech_threshold=0.6 # Higher threshold to avoid false positives
            )

            # Extract text from result
            text = result["text"].strip()

            logger.info(f"🗣️  Transcribed: {text}")
            return text

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""


class TextToSpeech:
    """Handles text-to-speech conversion with configurable speech rate."""

    def __init__(self):
        """Initialize TTS engine with configurable settings."""
        try:
            self.engine = pyttsx3.init()
            self.config = TTS_CONFIG

            # Configure voice settings
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to use a preferred voice
                voice_preferences = self.config.get("voice_preference", ["female", "zira", "default"])
                for preference in voice_preferences:
                    for voice in voices:
                        if preference.lower() in voice.name.lower():
                            self.engine.setProperty('voice', voice.id)
                            logger.info(f"🎭 Selected voice: {voice.name}")
                            break
                    else:
                        continue
                    break
                else:
                    # Use first available voice as fallback
                    self.engine.setProperty('voice', voices[0].id)
                    logger.info(f"🎭 Using fallback voice: {voices[0].name}")

            # Set initial speech rate and volume from config
            self.current_rate = self.config.get("rate", 140)
            self.engine.setProperty('rate', self.current_rate)
            self.engine.setProperty('volume', self.config.get("volume", 0.9))

            logger.info(f"🔊 TTS engine initialized with rate: {self.current_rate} WPM")

        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            raise
    
    def speak(self, text: str) -> None:
        """
        Convert text to speech and play it.

        Args:
            text: Text to speak
        """
        try:
            logger.info(f"🗨️  Speaking: {text} (Rate: {self.current_rate} WPM)")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS failed: {e}")

    def set_rate(self, rate: int) -> bool:
        """
        Set the speech rate.

        Args:
            rate: Speech rate in words per minute

        Returns:
            True if successful, False otherwise
        """
        try:
            min_rate = self.config.get("rate_min", 80)
            max_rate = self.config.get("rate_max", 300)

            # Clamp rate to valid range
            rate = max(min_rate, min(max_rate, rate))

            self.engine.setProperty('rate', rate)
            self.current_rate = rate
            logger.info(f"🎚️  Speech rate set to: {rate} WPM")
            return True

        except Exception as e:
            logger.error(f"Failed to set speech rate: {e}")
            return False

    def get_rate(self) -> int:
        """Get the current speech rate."""
        return self.current_rate

    def adjust_rate(self, delta: int) -> bool:
        """
        Adjust the speech rate by a delta value.

        Args:
            delta: Change in words per minute (positive to increase, negative to decrease)

        Returns:
            True if successful, False otherwise
        """
        new_rate = self.current_rate + delta
        return self.set_rate(new_rate)

    def set_rate_preset(self, preset: str) -> bool:
        """
        Set speech rate using a preset.

        Args:
            preset: Preset name (very_slow, slow, normal, fast, very_fast)

        Returns:
            True if successful, False otherwise
        """
        try:
            presets = self.config.get("rate_presets", {})
            if preset in presets:
                return self.set_rate(presets[preset])
            else:
                logger.warning(f"Unknown rate preset: {preset}")
                return False
        except Exception as e:
            logger.error(f"Failed to set rate preset: {e}")
            return False

    def get_available_presets(self) -> dict:
        """Get available rate presets."""
        return self.config.get("rate_presets", {})

    def show_rate_info(self) -> str:
        """Get current rate information as a formatted string."""
        presets = self.get_available_presets()
        current_preset = None

        # Find which preset matches current rate
        for preset_name, preset_rate in presets.items():
            if preset_rate == self.current_rate:
                current_preset = preset_name
                break

        info = f"Current rate: {self.current_rate} WPM"
        if current_preset:
            info += f" ({current_preset})"

        return info


class InternetConnectivity:
    """Manages internet connectivity detection and error handling."""

    def __init__(self):
        """Initialize internet connectivity manager."""
        self.config = INTERNET_CONFIG
        self.is_connected = False
        self.last_check_time = 0
        self.check_interval = 30  # Check connectivity every 30 seconds

    def check_connectivity(self, force_check: bool = False) -> bool:
        """
        Check internet connectivity.

        Args:
            force_check: Force a new connectivity check regardless of cache

        Returns:
            True if internet is available, False otherwise
        """
        current_time = time.time()

        # Use cached result if recent and not forced
        if not force_check and (current_time - self.last_check_time) < self.check_interval:
            return self.is_connected

        try:
            timeout = self.config.get("connection_timeout", 10)

            # Try to connect to multiple reliable hosts
            test_hosts = [
                ("8.8.8.8", 53),      # Google DNS
                ("1.1.1.1", 53),      # Cloudflare DNS
                ("208.67.222.222", 53) # OpenDNS
            ]

            for host, port in test_hosts:
                try:
                    socket.setdefaulttimeout(timeout)
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    result = sock.connect_ex((host, port))
                    sock.close()

                    if result == 0:
                        self.is_connected = True
                        self.last_check_time = current_time
                        logger.info("🌐 Internet connectivity confirmed")
                        return True

                except Exception:
                    continue

            # All hosts failed
            self.is_connected = False
            self.last_check_time = current_time
            logger.warning("🚫 No internet connectivity detected")
            return False

        except Exception as e:
            logger.error(f"Connectivity check failed: {e}")
            self.is_connected = False
            self.last_check_time = current_time
            return False

    def make_request(self, url: str, data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make an HTTP request with error handling and retries.

        Args:
            url: URL to request
            data: Optional POST data
            headers: Optional headers

        Returns:
            Response data as dict or None if failed
        """
        if not self.check_connectivity():
            logger.warning("No internet connectivity for request")
            return None

        max_retries = self.config.get("max_retries", 3)
        timeout = self.config.get("request_timeout", 30)
        user_agent = self.config.get("user_agent", "VoiceCompanion/1.0")

        # Default headers
        request_headers = {
            "User-Agent": user_agent,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9"
        }

        if headers:
            request_headers.update(headers)

        for attempt in range(max_retries):
            try:
                # Prepare request
                if data:
                    # POST request
                    post_data = urllib.parse.urlencode(data).encode('utf-8')
                    request = urllib.request.Request(url, data=post_data, headers=request_headers)
                else:
                    # GET request
                    request = urllib.request.Request(url, headers=request_headers)

                # Make request
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    response_data = response.read().decode('utf-8')

                    # Try to parse as JSON
                    try:
                        return json.loads(response_data)
                    except json.JSONDecodeError:
                        # Return as text if not JSON
                        return {"text": response_data}

            except urllib.error.HTTPError as e:
                logger.warning(f"HTTP error {e.code} on attempt {attempt + 1}: {e.reason}")
                if attempt == max_retries - 1:
                    return None

            except urllib.error.URLError as e:
                logger.warning(f"URL error on attempt {attempt + 1}: {e.reason}")
                if attempt == max_retries - 1:
                    return None

            except socket.timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    return None

            except Exception as e:
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return None

            # Wait before retry
            if attempt < max_retries - 1:
                delay = self.config.get("rate_limit_delay", 1.0) * (attempt + 1)
                time.sleep(delay)

        return None


class WebSearchEngine:
    """Handles web search functionality using DuckDuckGo."""

    def __init__(self, connectivity_manager: InternetConnectivity):
        """Initialize web search engine."""
        self.connectivity = connectivity_manager
        self.config = WEB_SEARCH_CONFIG

    def search(self, query: str) -> List[Dict[str, str]]:
        """
        Search the web for information.

        Args:
            query: Search query

        Returns:
            List of search results with title, snippet, and URL
        """
        if not self.config.get("enabled", True):
            logger.info("Web search is disabled")
            return []

        if not self.connectivity.check_connectivity():
            logger.warning("No internet connectivity for web search")
            return []

        try:
            # Use DuckDuckGo instant answer API
            return self._search_duckduckgo(query)

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    def _search_duckduckgo(self, query: str) -> List[Dict[str, str]]:
        """Search using DuckDuckGo instant answer API."""
        try:
            # DuckDuckGo instant answer API
            base_url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
                "no_redirect": "1"
            }

            url = base_url + "?" + urllib.parse.urlencode(params)

            response = self.connectivity.make_request(url)
            if not response:
                return []

            results = []

            # Extract instant answer
            if response.get("Abstract"):
                results.append({
                    "title": response.get("Heading", "Instant Answer"),
                    "snippet": response.get("Abstract", "")[:self.config.get("snippet_length", 200)],
                    "url": response.get("AbstractURL", ""),
                    "source": "DuckDuckGo Instant Answer"
                })

            # Extract related topics
            related_topics = response.get("RelatedTopics", [])
            max_results = self.config.get("max_results", 5)

            for topic in related_topics[:max_results-len(results)]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "").split(" - ")[0] if " - " in topic.get("Text", "") else "Related Topic",
                        "snippet": topic.get("Text", "")[:self.config.get("snippet_length", 200)],
                        "url": topic.get("FirstURL", ""),
                        "source": "DuckDuckGo Related"
                    })

            # If no results, try a simple web search approach
            if not results:
                results = self._simple_web_search(query)

            logger.info(f"🔍 Found {len(results)} search results for: {query}")
            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    def _simple_web_search(self, query: str) -> List[Dict[str, str]]:
        """Fallback simple web search."""
        try:
            # This is a simplified approach - in a real implementation,
            # you might want to use a proper search API
            search_url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}"

            # For now, return a placeholder result indicating search was attempted
            return [{
                "title": f"Search Results for: {query}",
                "snippet": f"Web search was performed for '{query}'. For detailed results, please use a web browser.",
                "url": search_url,
                "source": "Web Search"
            }]

        except Exception as e:
            logger.error(f"Simple web search failed: {e}")
            return []

    def format_search_results(self, results: List[Dict[str, str]]) -> str:
        """Format search results for LLM consumption."""
        if not results:
            return "No search results found."

        formatted = "Web Search Results:\n\n"

        for i, result in enumerate(results, 1):
            title = result.get("title", "No Title")
            snippet = result.get("snippet", "No description available")
            source = result.get("source", "Unknown")

            formatted += f"{i}. {title}\n"
            formatted += f"   {snippet}\n"
            formatted += f"   Source: {source}\n\n"

        return formatted


class RealTimeDataFetcher:
    """Handles real-time data fetching (weather, news, time, etc.)."""

    def __init__(self, connectivity_manager: InternetConnectivity):
        """Initialize real-time data fetcher."""
        self.connectivity = connectivity_manager
        self.config = REALTIME_DATA_CONFIG

    def get_weather(self, location: str = None) -> Optional[Dict]:
        """
        Get current weather information.

        Args:
            location: Location for weather (optional, uses default if not provided)

        Returns:
            Weather data dict or None if failed
        """
        weather_config = self.config.get("weather", {})
        if not weather_config.get("enabled", True):
            return None

        if not self.connectivity.check_connectivity():
            return None

        try:
            # Use a free weather API (OpenWeatherMap alternative)
            # For demo purposes, using a simple weather service
            if not location:
                location = weather_config.get("default_location", "auto")

            # Use wttr.in service which doesn't require API key
            url = f"https://wttr.in/{urllib.parse.quote(location)}?format=j1"

            response = self.connectivity.make_request(url)
            if response and "current_condition" in response:
                current = response["current_condition"][0]

                return {
                    "location": location,
                    "temperature": current.get("temp_C", "N/A"),
                    "condition": current.get("weatherDesc", [{}])[0].get("value", "N/A"),
                    "humidity": current.get("humidity", "N/A"),
                    "wind_speed": current.get("windspeedKmph", "N/A"),
                    "feels_like": current.get("FeelsLikeC", "N/A")
                }

        except Exception as e:
            logger.error(f"Weather fetch failed: {e}")

        return None

    def get_current_time(self, timezone: str = None) -> Optional[Dict]:
        """
        Get current time information.

        Args:
            timezone: Timezone (optional)

        Returns:
            Time data dict or None if failed
        """
        time_config = self.config.get("time", {})
        if not time_config.get("enabled", True):
            return None

        if not self.connectivity.check_connectivity():
            return None

        try:
            # Use WorldTimeAPI
            if timezone:
                url = f"https://worldtimeapi.org/api/timezone/{timezone}"
            else:
                url = "https://worldtimeapi.org/api/ip"

            response = self.connectivity.make_request(url)
            if response:
                return {
                    "datetime": response.get("datetime", ""),
                    "timezone": response.get("timezone", ""),
                    "utc_offset": response.get("utc_offset", ""),
                    "day_of_week": response.get("day_of_week", ""),
                    "day_of_year": response.get("day_of_year", "")
                }

        except Exception as e:
            logger.error(f"Time fetch failed: {e}")

        return None

    def get_news_headlines(self, category: str = None, country: str = None) -> List[Dict]:
        """
        Get current news headlines.

        Args:
            category: News category (optional)
            country: Country code (optional)

        Returns:
            List of news articles
        """
        news_config = self.config.get("news", {})
        if not news_config.get("enabled", True):
            return []

        if not self.connectivity.check_connectivity():
            return []

        try:
            # Use a free news API alternative
            # For demo purposes, return a placeholder
            return [{
                "title": "Latest News",
                "description": "Real-time news fetching is available. Configure API keys in config.py for full functionality.",
                "source": "News Service",
                "published_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }]

        except Exception as e:
            logger.error(f"News fetch failed: {e}")
            return []

    def format_weather_data(self, weather_data: Dict) -> str:
        """Format weather data for speech."""
        if not weather_data:
            return "Weather information is not available."

        location = weather_data.get("location", "your location")
        temp = weather_data.get("temperature", "unknown")
        condition = weather_data.get("condition", "unknown")
        humidity = weather_data.get("humidity", "unknown")
        feels_like = weather_data.get("feels_like", "unknown")

        return (f"The weather in {location} is {condition} with a temperature of {temp} degrees Celsius. "
                f"It feels like {feels_like} degrees with {humidity}% humidity.")

    def format_time_data(self, time_data: Dict) -> str:
        """Format time data for speech."""
        if not time_data:
            return "Time information is not available."

        datetime_str = time_data.get("datetime", "")
        timezone = time_data.get("timezone", "")

        if datetime_str:
            # Parse and format the datetime
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                formatted_time = dt.strftime("%I:%M %p on %A, %B %d, %Y")
                return f"The current time is {formatted_time} in {timezone}."
            except:
                return f"The current time information is available for {timezone}."

        return "Time information is not available."

    def format_news_data(self, news_data: List[Dict]) -> str:
        """Format news data for speech."""
        if not news_data:
            return "No news headlines are available."

        formatted = "Here are the latest news headlines:\n\n"
        for i, article in enumerate(news_data[:3], 1):  # Limit to top 3
            title = article.get("title", "No title")
            description = article.get("description", "")
            formatted += f"{i}. {title}\n"
            if description:
                formatted += f"   {description[:100]}...\n"

        return formatted


class LLMInterface:
    """Enhanced interface for Ollama LLM with internet learning capabilities."""

    def __init__(self, model_name: str = "llama3.2:1b", internet_mode: str = "hybrid"):
        """
        Initialize Ollama interface with internet capabilities.

        Args:
            model_name: Name of the Ollama model to use
            internet_mode: Internet mode - "offline", "online", or "hybrid"
        """
        self.model_name = model_name
        self.conversation_history = []
        self.internet_mode = internet_mode

        # Initialize internet learning components
        self.connectivity = InternetConnectivity()
        self.web_search = WebSearchEngine(self.connectivity)
        self.realtime_data = RealTimeDataFetcher(self.connectivity)

        # Test Ollama connection
        try:
            logger.info("🔍 Testing Ollama connection...")
            response = ollama.list()

            # Handle Ollama client response format
            if hasattr(response, 'models'):
                # New Ollama client returns ListResponse object
                models_list = response.models
            elif isinstance(response, dict) and 'models' in response:
                # Fallback for dict response
                models_list = response['models']
            elif isinstance(response, list):
                # Direct list response
                models_list = response
            else:
                logger.error(f"Unexpected Ollama response format: {response}")
                raise Exception("Invalid Ollama response format")

            # Extract model names safely
            available_models = []
            for model in models_list:
                if hasattr(model, 'model'):
                    # New Ollama client Model object
                    available_models.append(model.model)
                elif isinstance(model, dict):
                    # Dict format
                    model_name_field = model.get('name', model.get('model', None))
                    if model_name_field:
                        available_models.append(model_name_field)
                else:
                    available_models.append(str(model))

            logger.info(f"Available models: {available_models}")

            if not available_models:
                raise Exception("No Ollama models available. Please run: ollama pull llama3.2:1b")

            # Check if requested model is available
            model_found = False
            for available_model in available_models:
                if model_name in available_model or available_model in model_name:
                    self.model_name = available_model
                    model_found = True
                    break

            if not model_found:
                logger.warning(f"Model {model_name} not found. Available models: {available_models}")
                # Use the first available model
                self.model_name = available_models[0]
                logger.info(f"Using fallback model: {self.model_name}")

            logger.info(f"🤖 LLM ready with model: {self.model_name}")
            logger.info(f"🌐 Internet mode: {self.internet_mode}")

        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.error("Make sure Ollama is running: ollama serve")
            logger.error("And that you have models installed: ollama pull llama3.2:1b")
            raise
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate response from LLM with internet learning capabilities.

        Args:
            user_input: User's input text

        Returns:
            LLM response
        """
        try:
            # Check if this requires internet information
            enhanced_input = user_input
            context_info = ""

            if self.internet_mode in ["online", "hybrid"]:
                context_info = self._gather_internet_context(user_input)
                if context_info:
                    enhanced_input = f"{user_input}\n\nAdditional Context:\n{context_info}"

            # Add user input to conversation history
            self.conversation_history.append({"role": "user", "content": enhanced_input})

            # Keep conversation history manageable (last 10 exchanges)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            # Generate response with optimized settings for i3 systems
            response = ollama.chat(
                model=self.model_name,
                messages=self.conversation_history,
                options={
                    "temperature": 0.7,
                    "num_predict": 150,     # Slightly increased for internet-enhanced responses
                    "num_ctx": 2048,        # Reduced context window to save memory
                    "top_k": 20,            # Reduced top-k for faster sampling
                    "top_p": 0.9,           # Nucleus sampling
                    "repeat_last_n": 64,    # Prevent repetition
                    "repeat_penalty": 1.1,  # Slight penalty for repetition
                    "num_thread": 2         # Limit threads for dual-core i3
                }
            )

            assistant_response = response['message']['content'].strip()

            # Add assistant response to history (without the enhanced context)
            self.conversation_history.append({"role": "assistant", "content": assistant_response})

            # Add internet mode indicator if context was used
            if context_info and self.internet_mode != "offline":
                mode_indicator = " 🌐" if self.connectivity.is_connected else " 📴"
                logger.info(f"🤖 LLM Response{mode_indicator}: {assistant_response}")
            else:
                logger.info(f"🤖 LLM Response: {assistant_response}")

            return assistant_response

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            fallback_msg = "I'm sorry, I'm having trouble processing that right now."

            # If internet mode and offline fallback is enabled
            if self.internet_mode == "online" and INTERNET_CONFIG.get("fallback_to_offline", True):
                fallback_msg += " I'm switching to offline mode."
                self.internet_mode = "offline"

            return fallback_msg

    def _gather_internet_context(self, user_input: str) -> str:
        """Gather relevant internet context for the user input."""
        context_parts = []
        user_lower = user_input.lower()

        try:
            # Check for weather-related queries
            weather_keywords = ["weather", "temperature", "rain", "sunny", "cloudy", "forecast"]
            if any(keyword in user_lower for keyword in weather_keywords):
                weather_data = self.realtime_data.get_weather()
                if weather_data:
                    context_parts.append(self.realtime_data.format_weather_data(weather_data))

            # Check for time-related queries
            time_keywords = ["time", "date", "today", "now", "current"]
            if any(keyword in user_lower for keyword in time_keywords):
                time_data = self.realtime_data.get_current_time()
                if time_data:
                    context_parts.append(self.realtime_data.format_time_data(time_data))

            # Check for news-related queries
            news_keywords = ["news", "headlines", "latest", "current events", "happening"]
            if any(keyword in user_lower for keyword in news_keywords):
                news_data = self.realtime_data.get_news_headlines()
                if news_data:
                    context_parts.append(self.realtime_data.format_news_data(news_data))

            # Check if this might benefit from web search
            search_indicators = ["what is", "who is", "how to", "when did", "where is", "why", "search", "find"]
            if any(indicator in user_lower for indicator in search_indicators):
                search_results = self.web_search.search(user_input)
                if search_results:
                    context_parts.append(self.web_search.format_search_results(search_results))

        except Exception as e:
            logger.error(f"Error gathering internet context: {e}")

        return "\n\n".join(context_parts) if context_parts else ""

    def set_internet_mode(self, mode: str) -> bool:
        """
        Set the internet mode.

        Args:
            mode: "offline", "online", or "hybrid"

        Returns:
            True if successful, False otherwise
        """
        valid_modes = ["offline", "online", "hybrid"]
        if mode in valid_modes:
            self.internet_mode = mode
            logger.info(f"🌐 Internet mode set to: {mode}")
            return True
        else:
            logger.warning(f"Invalid internet mode: {mode}")
            return False

    def get_internet_status(self) -> Dict[str, Any]:
        """Get current internet status and capabilities."""
        return {
            "mode": self.internet_mode,
            "connected": self.connectivity.is_connected,
            "last_check": self.connectivity.last_check_time,
            "web_search_enabled": WEB_SEARCH_CONFIG.get("enabled", True),
            "weather_enabled": REALTIME_DATA_CONFIG.get("weather", {}).get("enabled", True),
            "news_enabled": REALTIME_DATA_CONFIG.get("news", {}).get("enabled", True),
            "time_enabled": REALTIME_DATA_CONFIG.get("time", {}).get("enabled", True)
        }


class VoiceCompanion:
    """Main voice companion orchestrator with internet learning capabilities."""

    def __init__(self, whisper_model: str = "tiny", ollama_model: str = "llama3.2:1b", internet_mode: str = "hybrid"):
        """Initialize all components optimized for i3 7th gen systems."""
        logger.info("🚀 Initializing Voice Companion (optimized for low-resource systems)...")

        try:
            # Initialize components with resource-conscious settings
            self.audio_recorder = AudioRecorder(sample_rate=16000, chunk_size=1024)
            self.stt = SpeechToText(whisper_model)
            self.tts = TextToSpeech()
            self.llm = LLMInterface(ollama_model, internet_mode)
            self.internet_mode = internet_mode

            self.is_running = False
            logger.info("✅ Voice Companion initialized successfully!")
            logger.info(f"💡 Using Whisper '{whisper_model}' and Ollama '{ollama_model}' for optimal performance")
            logger.info(f"🌐 Internet mode: {internet_mode}")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    def listen_for_wake_word(self, wake_word: str = "hey jarvis") -> bool:
        """
        Listen for wake word using continuous audio monitoring.
        """
        print(f"\n{Fore.CYAN}🎧 Listening for wake word: '{wake_word}'{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 Say '{wake_word}' to activate, or press Ctrl+C to exit{Style.RESET_ALL}")

        try:
            while True:
                # Record a short audio snippet for wake word detection
                self.audio_recorder.start_recording()
                time.sleep(3)  # Listen for 3 seconds
                audio_data = self.audio_recorder.stop_recording()

                if audio_data:
                    # Transcribe the audio
                    text = self.stt.transcribe_audio(
                        audio_data,
                        sample_rate=self.audio_recorder.supported_rate,
                        channels=getattr(self.audio_recorder, 'channels', 1)
                    )

                    if text:
                        print(f"{Fore.BLUE}🔍 Heard: '{text}'{Style.RESET_ALL}")

                        # Check if wake word is detected (flexible matching)
                        text_lower = text.lower().strip()
                        wake_word_lower = wake_word.lower().strip()

                        # Multiple ways to match the wake word
                        wake_word_detected = (
                            wake_word_lower in text_lower or
                            "hey jarvis" in text_lower or
                            "jarvis" in text_lower or
                            ("hey" in text_lower and "jarvis" in text_lower) or
                            text_lower.replace(" ", "") == wake_word_lower.replace(" ", "")
                        )

                        if wake_word_detected:
                            print(f"{Fore.GREEN}✅ Wake word detected!{Style.RESET_ALL}")
                            return True
                        else:
                            print(f"{Fore.GRAY}⏳ Still listening...{Style.RESET_ALL}")

                # Small delay before next listening cycle
                time.sleep(0.5)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}👋 Exiting wake word detection{Style.RESET_ALL}")
            return False
        except Exception as e:
            logger.error(f"Wake word detection failed: {e}")
            print(f"{Fore.RED}❌ Wake word detection error. Falling back to Enter key.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")
            input()
            return True
    
    def process_voice_input(self) -> Optional[str]:
        """Record and transcribe voice input."""
        try:
            print(f"\n{Fore.GREEN}🎤 Listening... (Press Enter when done speaking){Style.RESET_ALL}")

            self.audio_recorder.start_recording()
            input()  # Wait for user to press Enter
            audio_data = self.audio_recorder.stop_recording()

            if not audio_data:
                return None

            # Transcribe audio with correct parameters
            text = self.stt.transcribe_audio(
                audio_data,
                sample_rate=self.audio_recorder.supported_rate,
                channels=getattr(self.audio_recorder, 'channels', 1)
            )
            return text if text else None

        except Exception as e:
            logger.error(f"Voice input processing failed: {e}")
            return None

    def process_text_input(self) -> Optional[str]:
        """Process manual text input from user."""
        try:
            config = TEXT_INPUT_CONFIG
            prompt = config.get("prompt", "Enter text: ")
            max_length = config.get("max_length", 1000)
            min_length = config.get("min_length", 1)
            strip_whitespace = config.get("strip_whitespace", True)
            allow_empty = config.get("allow_empty", False)

            print(f"\n{Fore.CYAN}✏️  Text Input Mode{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}💡 Type your message (max {max_length} characters){Style.RESET_ALL}")
            print(f"{Fore.YELLOW}💡 Type 'voice' to switch to voice mode, 'quit' to exit{Style.RESET_ALL}")

            # Handle timeout for input
            timeout = config.get("timeout", 300)

            def timeout_handler(signum, frame):
                raise TimeoutError("Input timeout")

            # Set up timeout (Unix-like systems only)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)

            try:
                user_input = input(f"{Fore.GREEN}{prompt}{Style.RESET_ALL}")

                # Cancel timeout
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)

                # Validate encoding first
                try:
                    user_input.encode(config.get("encoding", "utf-8"))
                except UnicodeEncodeError as e:
                    logger.error(f"Text encoding error: {e}")
                    print(f"{Fore.RED}❌ Text contains invalid characters. Please use standard text.{Style.RESET_ALL}")
                    return None

                # Process input according to configuration
                if strip_whitespace:
                    user_input = user_input.strip()

                # Check for empty input
                if not user_input:
                    if allow_empty:
                        return ""
                    else:
                        print(f"{Fore.RED}❌ Empty input not allowed. Please enter some text.{Style.RESET_ALL}")
                        return None

                # Check length constraints
                if len(user_input) < min_length:
                    print(f"{Fore.RED}❌ Input too short. Minimum {min_length} characters required.{Style.RESET_ALL}")
                    return None

                if len(user_input) > max_length:
                    print(f"{Fore.YELLOW}⚠️  Input truncated to {max_length} characters.{Style.RESET_ALL}")
                    user_input = user_input[:max_length]

                # Check for potentially problematic content
                if '\x00' in user_input:
                    print(f"{Fore.RED}❌ Input contains null characters. Please enter valid text.{Style.RESET_ALL}")
                    return None

                # Check for excessive whitespace or control characters
                control_chars = sum(1 for c in user_input if ord(c) < 32 and c not in ['\t', '\n', '\r'])
                if control_chars > 0:
                    print(f"{Fore.YELLOW}⚠️  Input contains {control_chars} control characters. This might affect processing.{Style.RESET_ALL}")

                # Check for special commands
                user_input_lower = user_input.lower().strip()
                if user_input_lower in ['voice', 'switch to voice', 'voice mode']:
                    return "SWITCH_TO_VOICE"
                elif user_input_lower in ['quit', 'exit', 'goodbye', 'stop']:
                    return "QUIT"
                elif user_input_lower in ['help', '?']:
                    self._show_text_input_help()
                    return None
                elif user_input_lower.startswith('rate '):
                    # Handle speech rate commands
                    self._handle_rate_command(user_input_lower)
                    return None
                elif user_input_lower in ['rate', 'speech rate', 'speed']:
                    self._show_rate_info()
                    return None
                elif user_input_lower in ['internet', 'connectivity', 'online', 'status']:
                    self._show_internet_status()
                    return None
                elif user_input_lower.startswith('internet '):
                    self._handle_internet_command(user_input_lower)
                    return None

                logger.info(f"📝 Text input received: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
                return user_input

            except TimeoutError:
                print(f"\n{Fore.YELLOW}⏰ Input timeout. Returning to main menu.{Style.RESET_ALL}")
                return None
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}⏸️  Input cancelled by user.{Style.RESET_ALL}")
                return None

        except Exception as e:
            logger.error(f"Text input processing failed: {e}")
            print(f"{Fore.RED}❌ Text input error: {e}{Style.RESET_ALL}")
            return None

    def _show_text_input_help(self):
        """Show help information for text input mode."""
        print(f"\n{Fore.CYAN}📖 Text Input Help{Style.RESET_ALL}")
        print("=" * 40)
        print(f"{Fore.GREEN}Commands:{Style.RESET_ALL}")
        print(f"  • {Fore.YELLOW}'voice'{Style.RESET_ALL} - Switch to voice input mode")
        print(f"  • {Fore.YELLOW}'quit' or 'exit'{Style.RESET_ALL} - Exit the application")
        print(f"  • {Fore.YELLOW}'help' or '?'{Style.RESET_ALL} - Show this help")
        print(f"  • {Fore.YELLOW}'rate'{Style.RESET_ALL} - Show current speech rate")
        print(f"  • {Fore.YELLOW}'rate <number>'{Style.RESET_ALL} - Set speech rate (80-300 WPM)")
        print(f"  • {Fore.YELLOW}'rate <preset>'{Style.RESET_ALL} - Use preset (slow, normal, fast)")
        print(f"  • {Fore.YELLOW}'internet'{Style.RESET_ALL} - Show internet status and capabilities")
        print(f"  • {Fore.YELLOW}'internet <mode>'{Style.RESET_ALL} - Set internet mode (offline/online/hybrid)")
        print(f"\n{Fore.GREEN}Speech Rate Presets:{Style.RESET_ALL}")
        presets = self.tts.get_available_presets()
        for preset, rate in presets.items():
            print(f"  • {Fore.CYAN}{preset}{Style.RESET_ALL}: {rate} WPM")

        # Show internet features if available
        if hasattr(self, 'internet_mode') and self.internet_mode != 'offline':
            print(f"\n{Fore.GREEN}Internet Learning Features:{Style.RESET_ALL}")
            print(f"  • Ask about current weather, time, or news")
            print(f"  • Search for recent information online")
            print(f"  • Get real-time data and updates")

        print(f"\n{Fore.GREEN}Tips:{Style.RESET_ALL}")
        print(f"  • Maximum text length: {TEXT_INPUT_CONFIG.get('max_length', 1000)} characters")
        print(f"  • Use clear, simple text for best speech quality")
        print(f"  • Avoid excessive special characters")
        print(f"  • Press Enter to submit your text")
        print("=" * 40)

    def _handle_rate_command(self, command: str):
        """Handle speech rate adjustment commands."""
        try:
            parts = command.split()
            if len(parts) < 2:
                self._show_rate_info()
                return

            rate_value = parts[1]

            # Try to parse as number first
            try:
                rate_num = int(rate_value)
                if self.tts.set_rate(rate_num):
                    print(f"{Fore.GREEN}✅ Speech rate set to {rate_num} WPM{Style.RESET_ALL}")
                    # Test the new rate
                    self.tts.speak("Speech rate has been adjusted.")
                else:
                    print(f"{Fore.RED}❌ Failed to set speech rate{Style.RESET_ALL}")
            except ValueError:
                # Try as preset
                if self.tts.set_rate_preset(rate_value):
                    new_rate = self.tts.get_rate()
                    print(f"{Fore.GREEN}✅ Speech rate set to {rate_value} preset ({new_rate} WPM){Style.RESET_ALL}")
                    # Test the new rate
                    self.tts.speak(f"Speech rate set to {rate_value}.")
                else:
                    print(f"{Fore.RED}❌ Unknown rate preset: {rate_value}{Style.RESET_ALL}")
                    self._show_rate_info()

        except Exception as e:
            logger.error(f"Rate command error: {e}")
            print(f"{Fore.RED}❌ Error processing rate command: {e}{Style.RESET_ALL}")

    def _show_rate_info(self):
        """Show current speech rate information."""
        try:
            rate_info = self.tts.show_rate_info()
            print(f"\n{Fore.CYAN}🎚️  Speech Rate Information{Style.RESET_ALL}")
            print("=" * 35)
            print(f"{Fore.GREEN}{rate_info}{Style.RESET_ALL}")

            print(f"\n{Fore.GREEN}Available presets:{Style.RESET_ALL}")
            presets = self.tts.get_available_presets()
            for preset, rate in presets.items():
                current = " (current)" if rate == self.tts.get_rate() else ""
                print(f"  • {Fore.CYAN}{preset}{Style.RESET_ALL}: {rate} WPM{current}")

            print(f"\n{Fore.GREEN}Usage:{Style.RESET_ALL}")
            print(f"  • Type 'rate <number>' to set specific rate (80-300)")
            print(f"  • Type 'rate <preset>' to use a preset")
            print("=" * 35)

        except Exception as e:
            logger.error(f"Rate info error: {e}")
            print(f"{Fore.RED}❌ Error showing rate info: {e}{Style.RESET_ALL}")

    def _show_internet_status(self):
        """Show current internet status and capabilities."""
        try:
            status = self.llm.get_internet_status()

            print(f"\n{Fore.CYAN}🌐 Internet Status{Style.RESET_ALL}")
            print("=" * 35)

            # Connection status
            connectivity_icon = "🟢" if status['connected'] else "🔴"
            print(f"{Fore.GREEN}Connection:{Style.RESET_ALL} {connectivity_icon} {'Connected' if status['connected'] else 'Disconnected'}")
            print(f"{Fore.GREEN}Mode:{Style.RESET_ALL} {status['mode'].upper()}")

            if status['last_check']:
                last_check = time.strftime("%H:%M:%S", time.localtime(status['last_check']))
                print(f"{Fore.GREEN}Last Check:{Style.RESET_ALL} {last_check}")

            print(f"\n{Fore.GREEN}Available Features:{Style.RESET_ALL}")
            features = [
                ("Web Search", status['web_search_enabled']),
                ("Weather Data", status['weather_enabled']),
                ("News Headlines", status['news_enabled']),
                ("Time Information", status['time_enabled'])
            ]

            for feature, enabled in features:
                icon = "✅" if enabled else "❌"
                print(f"  {icon} {feature}")

            print(f"\n{Fore.GREEN}Commands:{Style.RESET_ALL}")
            print(f"  • 'internet offline' - Switch to offline mode")
            print(f"  • 'internet online' - Switch to online mode")
            print(f"  • 'internet hybrid' - Switch to hybrid mode")
            print(f"  • 'internet test' - Test connectivity")
            print("=" * 35)

        except Exception as e:
            logger.error(f"Internet status error: {e}")
            print(f"{Fore.RED}❌ Error showing internet status: {e}{Style.RESET_ALL}")

    def _handle_internet_command(self, command: str):
        """Handle internet-related commands."""
        try:
            parts = command.split()
            if len(parts) < 2:
                self._show_internet_status()
                return

            action = parts[1].lower()

            if action in ['offline', 'online', 'hybrid']:
                if self.llm.set_internet_mode(action):
                    self.internet_mode = action
                    print(f"{Fore.GREEN}✅ Internet mode set to {action}{Style.RESET_ALL}")

                    # Show new status
                    if action != 'offline':
                        connected = self.llm.connectivity.check_connectivity(force_check=True)
                        icon = "🟢" if connected else "🔴"
                        print(f"{Fore.CYAN}📡 Connectivity: {icon} {'Connected' if connected else 'Disconnected'}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ Failed to set internet mode{Style.RESET_ALL}")

            elif action == 'test':
                print(f"{Fore.YELLOW}🔍 Testing internet connectivity...{Style.RESET_ALL}")
                connected = self.llm.connectivity.check_connectivity(force_check=True)
                icon = "🟢" if connected else "🔴"
                print(f"{Fore.CYAN}📡 Result: {icon} {'Connected' if connected else 'Disconnected'}{Style.RESET_ALL}")

                if connected and self.internet_mode != 'offline':
                    print(f"{Fore.GREEN}🌐 Internet learning features are available{Style.RESET_ALL}")

            else:
                print(f"{Fore.RED}❌ Unknown internet command: {action}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Available commands: offline, online, hybrid, test{Style.RESET_ALL}")

        except Exception as e:
            logger.error(f"Internet command error: {e}")
            print(f"{Fore.RED}❌ Error processing internet command: {e}{Style.RESET_ALL}")

    def speak_text_directly(self, text: str) -> bool:
        """
        Speak text directly without LLM processing.
        Used for text-to-speech only mode.

        Args:
            text: Text to speak

        Returns:
            True if successful, False otherwise
        """
        try:
            if not text or not text.strip():
                print(f"{Fore.RED}❌ No text to speak.{Style.RESET_ALL}")
                return False

            # Validate text encoding
            try:
                text.encode('utf-8')
            except UnicodeEncodeError as e:
                logger.error(f"Text encoding error: {e}")
                print(f"{Fore.RED}❌ Text contains invalid characters: {e}{Style.RESET_ALL}")
                return False

            # Check for extremely long text that might cause TTS issues
            max_tts_length = 5000  # Reasonable limit for TTS
            if len(text) > max_tts_length:
                print(f"{Fore.YELLOW}⚠️  Text is very long ({len(text)} chars). Truncating to {max_tts_length} characters.{Style.RESET_ALL}")
                text = text[:max_tts_length] + "..."

            # Check for potentially problematic characters
            problematic_chars = ['<', '>', '&', '"', "'"]
            if any(char in text for char in problematic_chars):
                print(f"{Fore.YELLOW}⚠️  Text contains special characters that might affect speech quality.{Style.RESET_ALL}")

            print(f"\n{Fore.GREEN}🗨️  Speaking: {text[:100]}{'...' if len(text) > 100 else ''}{Style.RESET_ALL}")

            # Attempt TTS with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.tts.speak(text)
                    return True
                except Exception as tts_error:
                    logger.warning(f"TTS attempt {attempt + 1} failed: {tts_error}")
                    if attempt < max_retries - 1:
                        print(f"{Fore.YELLOW}⚠️  TTS attempt {attempt + 1} failed, retrying...{Style.RESET_ALL}")
                        time.sleep(1)  # Brief pause before retry
                    else:
                        raise tts_error

            return False

        except Exception as e:
            logger.error(f"Direct text-to-speech failed: {e}")
            print(f"{Fore.RED}❌ Text-to-speech error: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}💡 Try shorter text or check your audio system.{Style.RESET_ALL}")
            return False

    def show_mode_menu(self) -> tuple[str, str]:
        """
        Show input and internet mode selection menu.

        Returns:
            Tuple of (input_mode, internet_mode)
        """
        try:
            # First, show input mode selection
            print(f"\n{Fore.MAGENTA}🎛️  Input Mode Selection{Style.RESET_ALL}")
            print("=" * 50)
            print(f"{Fore.CYAN}1. Voice Mode{Style.RESET_ALL} - Talk to Jarvis using your microphone")
            print(f"{Fore.CYAN}2. Text Mode{Style.RESET_ALL} - Type messages to Jarvis")
            print(f"{Fore.CYAN}3. Text-to-Speech Only{Style.RESET_ALL} - Type text to hear it spoken (no AI)")
            print(f"{Fore.CYAN}4. Mixed Mode{Style.RESET_ALL} - Switch between voice and text during conversation")
            print(f"{Fore.RED}5. Quit{Style.RESET_ALL}")
            print("=" * 50)

            input_mode = None
            while not input_mode:
                try:
                    choice = input(f"{Fore.GREEN}Select input mode (1-5): {Style.RESET_ALL}").strip()

                    if choice == '1':
                        input_mode = 'voice'
                    elif choice == '2':
                        input_mode = 'text'
                    elif choice == '3':
                        input_mode = 'text_only'
                    elif choice == '4':
                        input_mode = 'mixed'
                    elif choice == '5':
                        return 'quit', 'offline'
                    else:
                        print(f"{Fore.RED}❌ Invalid choice. Please select 1-5.{Style.RESET_ALL}")
                        continue

                except KeyboardInterrupt:
                    print(f"\n{Fore.YELLOW}👋 Exiting...{Style.RESET_ALL}")
                    return 'quit', 'offline'

            # Skip internet mode selection for text-only mode
            if input_mode == 'text_only':
                return input_mode, 'offline'

            # Show internet mode selection
            print(f"\n{Fore.MAGENTA}🌐 Internet Mode Selection{Style.RESET_ALL}")
            print("=" * 50)

            # Check connectivity status
            connectivity_status = "🟢 Connected" if self.llm.connectivity.check_connectivity() else "🔴 Disconnected"
            print(f"Internet Status: {connectivity_status}")
            print()

            print(f"{Fore.CYAN}1. Offline Mode{Style.RESET_ALL} - Use only local AI (faster, private)")
            print(f"{Fore.CYAN}2. Online Mode{Style.RESET_ALL} - Use internet for current information")
            print(f"{Fore.CYAN}3. Hybrid Mode{Style.RESET_ALL} - Combine local AI with internet when needed (recommended)")
            print("=" * 50)

            internet_mode = None
            while not internet_mode:
                try:
                    choice = input(f"{Fore.GREEN}Select internet mode (1-3): {Style.RESET_ALL}").strip()

                    if choice == '1':
                        internet_mode = 'offline'
                    elif choice == '2':
                        internet_mode = 'online'
                    elif choice == '3':
                        internet_mode = 'hybrid'
                    else:
                        print(f"{Fore.RED}❌ Invalid choice. Please select 1-3.{Style.RESET_ALL}")
                        continue

                except KeyboardInterrupt:
                    print(f"\n{Fore.YELLOW}👋 Exiting...{Style.RESET_ALL}")
                    return 'quit', 'offline'

            return input_mode, internet_mode

        except Exception as e:
            logger.error(f"Mode menu error: {e}")
            print(f"{Fore.RED}❌ Menu error: {e}{Style.RESET_ALL}")
            return 'voice', 'hybrid'  # Default fallback
    
    def run(self):
        """Main conversation loop with support for multiple input modes and internet learning."""
        self.is_running = True

        print(f"\n{Fore.MAGENTA}🎉 Voice Jarvis is ready!{Style.RESET_ALL}")

        # Show mode selection menu if enabled
        if INPUT_MODE_CONFIG.get("show_mode_menu", True):
            input_mode, internet_mode = self.show_mode_menu()
            if input_mode == 'quit':
                return
        else:
            input_mode = INPUT_MODE_CONFIG.get("default_mode", "voice")
            internet_mode = INTERNET_CONFIG.get("default_mode", "hybrid")

        # Set internet mode
        self.llm.set_internet_mode(internet_mode)
        self.internet_mode = internet_mode

        print(f"\n{Fore.CYAN}🎛️  Input Mode: {input_mode.upper()}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🌐 Internet Mode: {internet_mode.upper()}{Style.RESET_ALL}")

        # Show internet status
        if internet_mode != 'offline':
            status = self.llm.get_internet_status()
            connectivity_icon = "🟢" if status['connected'] else "🔴"
            print(f"{Fore.CYAN}📡 Connectivity: {connectivity_icon} {'Connected' if status['connected'] else 'Disconnected'}{Style.RESET_ALL}")

        # Mode-specific instructions
        if input_mode == 'voice':
            print(f"{Fore.CYAN}Say 'hey jarvis' to start a conversation.{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Say 'quit', 'exit', or 'goodbye' to stop.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}💡 If microphone issues occur, run: python test_microphone.py{Style.RESET_ALL}")
        elif input_mode == 'text':
            print(f"{Fore.CYAN}Type your messages to chat with Jarvis.{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Type 'quit', 'exit', or 'goodbye' to stop.{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Type 'rate' to adjust speech speed, 'internet' for connectivity info.{Style.RESET_ALL}")
        elif input_mode == 'text_only':
            print(f"{Fore.CYAN}Text-to-Speech Mode: Type text to hear it spoken.{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Type 'quit', 'exit', or 'goodbye' to stop.{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Type 'rate' to adjust speech speed.{Style.RESET_ALL}")
        elif input_mode == 'mixed':
            print(f"{Fore.CYAN}Mixed Mode: Use voice or text input as needed.{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Say 'hey jarvis' for voice or type messages directly.{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Type 'voice' to switch to voice mode, 'text' for text mode.{Style.RESET_ALL}")

        # Show internet capabilities if enabled
        if internet_mode != 'offline':
            print(f"\n{Fore.GREEN}🌐 Internet Learning Features Available:{Style.RESET_ALL}")
            print(f"  • Web search for current information")
            print(f"  • Real-time weather data")
            print(f"  • Current time and date")
            print(f"  • Latest news headlines")

        try:
            if input_mode == 'text_only':
                self.run_text_to_speech_mode()
            elif input_mode == 'text':
                self.run_text_conversation_mode()
            elif input_mode == 'mixed':
                self.run_mixed_mode()
            else:  # voice mode
                self.run_voice_conversation_mode()

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}👋 Shutting down gracefully...{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.cleanup()
    
    def run_voice_conversation_mode(self):
        """Run the original voice conversation mode."""
        while self.is_running:
            # Wait for wake word
            if not self.listen_for_wake_word():
                continue

            # Process voice input
            user_input = self.process_voice_input()

            if not user_input:
                print(f"{Fore.RED}❌ No speech detected. Try again.{Style.RESET_ALL}")
                continue

            print(f"\n{Fore.BLUE}👤 You said: {user_input}{Style.RESET_ALL}")

            # Check for exit commands
            if any(word in user_input.lower() for word in ['quit', 'exit', 'goodbye', 'stop']):
                self.tts.speak("Goodbye! It was nice talking with you.")
                break

            # Generate and speak response
            response = self.llm.generate_response(user_input)
            print(f"{Fore.GREEN}🤖 Jarvis: {response}{Style.RESET_ALL}")
            self.tts.speak(response)

    def run_text_conversation_mode(self):
        """Run text-based conversation mode with LLM."""
        while self.is_running:
            # Process text input
            user_input = self.process_text_input()

            if not user_input:
                continue

            # Handle special commands
            if user_input == "QUIT":
                self.tts.speak("Goodbye! It was nice talking with you.")
                break
            elif user_input == "SWITCH_TO_VOICE":
                print(f"{Fore.CYAN}🔄 Switching to voice mode...{Style.RESET_ALL}")
                self.run_voice_conversation_mode()
                break

            print(f"\n{Fore.BLUE}👤 You typed: {user_input}{Style.RESET_ALL}")

            # Generate and speak response
            response = self.llm.generate_response(user_input)
            print(f"{Fore.GREEN}🤖 Jarvis: {response}{Style.RESET_ALL}")
            self.tts.speak(response)

    def run_text_to_speech_mode(self):
        """Run text-to-speech only mode (no LLM processing)."""
        print(f"\n{Fore.MAGENTA}🔊 Text-to-Speech Mode Active{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 Enter text and hear it spoken. No AI processing.{Style.RESET_ALL}")

        while self.is_running:
            user_input = self.process_text_input()

            if not user_input:
                continue

            # Handle special commands
            if user_input == "QUIT":
                print(f"{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
                break
            elif user_input == "SWITCH_TO_VOICE":
                print(f"{Fore.CYAN}🔄 Switching to voice mode...{Style.RESET_ALL}")
                self.run_voice_conversation_mode()
                break

            # Speak the text directly
            self.speak_text_directly(user_input)

    def run_mixed_mode(self):
        """Run mixed mode allowing both voice and text input."""
        print(f"\n{Fore.MAGENTA}🎛️  Mixed Mode Active{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 Choose your input method for each interaction.{Style.RESET_ALL}")

        while self.is_running:
            # Show input options
            print(f"\n{Fore.CYAN}Choose input method:{Style.RESET_ALL}")
            print(f"  {Fore.GREEN}1. Voice input (say 'hey jarvis'){Style.RESET_ALL}")
            print(f"  {Fore.GREEN}2. Text input{Style.RESET_ALL}")
            print(f"  {Fore.GREEN}3. Text-to-speech only{Style.RESET_ALL}")
            print(f"  {Fore.RED}4. Quit{Style.RESET_ALL}")

            try:
                choice = input(f"{Fore.CYAN}Select (1-4): {Style.RESET_ALL}").strip()

                if choice == '1':
                    # Voice input
                    if not self.listen_for_wake_word():
                        continue

                    user_input = self.process_voice_input()
                    if not user_input:
                        print(f"{Fore.RED}❌ No speech detected. Try again.{Style.RESET_ALL}")
                        continue

                    print(f"\n{Fore.BLUE}👤 You said: {user_input}{Style.RESET_ALL}")

                    # Check for exit commands
                    if any(word in user_input.lower() for word in ['quit', 'exit', 'goodbye', 'stop']):
                        self.tts.speak("Goodbye! It was nice talking with you.")
                        break

                    # Generate and speak response
                    response = self.llm.generate_response(user_input)
                    print(f"{Fore.GREEN}🤖 Jarvis: {response}{Style.RESET_ALL}")
                    self.tts.speak(response)

                elif choice == '2':
                    # Text input with LLM
                    user_input = self.process_text_input()
                    if not user_input:
                        continue

                    if user_input == "QUIT":
                        self.tts.speak("Goodbye! It was nice talking with you.")
                        break

                    print(f"\n{Fore.BLUE}👤 You typed: {user_input}{Style.RESET_ALL}")

                    # Generate and speak response
                    response = self.llm.generate_response(user_input)
                    print(f"{Fore.GREEN}🤖 Jarvis: {response}{Style.RESET_ALL}")
                    self.tts.speak(response)

                elif choice == '3':
                    # Text-to-speech only
                    user_input = self.process_text_input()
                    if not user_input:
                        continue

                    if user_input == "QUIT":
                        break

                    self.speak_text_directly(user_input)

                elif choice == '4':
                    print(f"{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
                    break

                else:
                    print(f"{Fore.RED}❌ Invalid choice. Please select 1-4.{Style.RESET_ALL}")

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}👋 Exiting mixed mode...{Style.RESET_ALL}")
                break

    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        if hasattr(self, 'audio_recorder'):
            self.audio_recorder.cleanup()
        logger.info("🧹 Cleanup completed")


def main():
    """Main entry point."""
    try:
        print(f"{Fore.CYAN}🤖 Starting Voice Jarvis with Text Input Support (optimized for i3 7th gen)...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}⏳ Initializing audio and AI components...{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✨ New Features:{Style.RESET_ALL}")
        print(f"   • Text input mode for typing messages")
        print(f"   • Text-to-speech only mode")
        print(f"   • Mixed mode (voice + text)")
        print(f"   • Enhanced error handling")

        # Create voice companion with optimized settings
        companion = VoiceCompanion(
            whisper_model="tiny",      # Optimized for i3 systems
            ollama_model="llama3.2:1b", # Lightweight model for 8GB RAM
            internet_mode="hybrid"     # Default to hybrid mode
        )

        # Start the conversation loop
        companion.run()

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        print(f"{Fore.RED}❌ Failed to start Voice Jarvis: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 Troubleshooting tips:{Style.RESET_ALL}")
        print("   1. Make sure Ollama is running: ollama serve")
        print("   2. Check if models are installed: ollama list")
        print("   3. Install required model: ollama pull llama3.2:1b")
        print("   4. Test components: python test_components.py")
        print("   5. Fix audio warnings: ./fix_audio_warnings.sh")
        print("   6. Check text input config in config.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
