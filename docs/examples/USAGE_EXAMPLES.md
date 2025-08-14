# Enhanced Knowledge System - Usage Examples

This document provides practical examples for using the Enhanced Intelligent Knowledge System in various scenarios.

## Table of Contents

1. [Basic Personal AI Assistant Setup](#basic-personal-ai-assistant-setup)
2. [Voice Companion Integration](#voice-companion-integration)
3. [Learning and Adaptation Examples](#learning-and-adaptation-examples)
4. [Privacy and Data Management](#privacy-and-data-management)
5. [Advanced Analytics and Insights](#advanced-analytics-and-insights)
6. [Integration with Existing Systems](#integration-with-existing-systems)

## Basic Personal AI Assistant Setup

### Simple Daily Assistant

```python
#!/usr/bin/env python3
"""
Simple daily AI assistant with persistent memory and learning.
"""

from intelligent_knowledge_system import create_personal_ai_assistant
from datetime import datetime

def main():
    # Create your personal assistant
    assistant = create_personal_ai_assistant(
        profile_name="daily_user",
        config={
            'db_path': 'personal_assistant.db',
            'privacy_level': 'standard',
            'auto_backup': True
        }
    )
    
    # Configure initial preferences
    assistant.update_user_preference('response_style', 'friendly')
    assistant.update_user_preference('verbosity_level', 'moderate')
    assistant.update_user_preference('technical_level', 'intermediate')
    
    print("Personal AI Assistant Ready!")
    print("Type 'quit' to exit, 'help' for commands")
    
    # Start a daily session
    session_context = {
        'session_type': 'daily_chat',
        'time_of_day': datetime.now().strftime('%H:%M'),
        'date': datetime.now().strftime('%Y-%m-%d')
    }
    assistant.start_session(session_context)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'help':
            print_help()
            continue
        elif user_input.lower().startswith('feedback:'):
            feedback = user_input[9:].strip()
            assistant.add_user_feedback(feedback, rating=4)
            print("Thank you for the feedback!")
            continue
        
        # Process the query
        result = assistant.process_query(user_input)
        print(f"\nAssistant: {result['ai_response']}")
        
        # Show if web search was performed
        if result.get('web_search_performed'):
            print("(Used web search for current information)")
    
    # End session with summary
    assistant.end_session("Daily chat session completed")
    print("\nSession ended. Have a great day!")

def print_help():
    print("\nAvailable commands:")
    print("- Type any question or request")
    print("- 'feedback: <your feedback>' - Provide feedback on responses")
    print("- 'quit' - Exit the assistant")
    print("- 'help' - Show this help message")

if __name__ == "__main__":
    main()
```

### Specialized Learning Assistant

```python
#!/usr/bin/env python3
"""
Specialized learning assistant for programming education.
"""

from intelligent_knowledge_system import create_personal_ai_assistant
import json

class ProgrammingLearningAssistant:
    def __init__(self, student_name):
        self.assistant = create_personal_ai_assistant(
            profile_name=f"student_{student_name}",
            config={
                'db_path': f'learning_assistant_{student_name}.db',
                'privacy_level': 'enhanced'
            }
        )
        
        # Configure for learning context
        self.assistant.update_user_preference('response_style', 'educational')
        self.assistant.update_user_preference('verbosity_level', 'detailed')
        self.assistant.update_user_preference('explanation_preference', 'examples')
        
        self.current_topic = None
        self.learning_session_active = False
    
    def start_learning_session(self, topic):
        """Start a focused learning session on a specific topic."""
        self.current_topic = topic
        session_context = {
            'session_type': 'learning',
            'topic': topic,
            'learning_goal': f'Master {topic} concepts'
        }
        
        session_id = self.assistant.start_session(session_context)
        self.learning_session_active = True
        
        print(f"Starting learning session on: {topic}")
        print(f"Session ID: {session_id}")
        
        # Get personalized recommendations for this topic
        recommendations = self.assistant.generate_personalized_recommendations()
        if recommendations.get('feature_recommendations'):
            print("\nPersonalized suggestions:")
            for rec in recommendations['feature_recommendations'][:2]:
                print(f"- {rec['suggestion']}")
    
    def ask_question(self, question):
        """Ask a learning question with context."""
        if not self.learning_session_active:
            print("Please start a learning session first!")
            return
        
        # Add topic context to the question
        contextualized_query = f"In the context of {self.current_topic}: {question}"
        
        result = self.assistant.process_query(contextualized_query)
        
        print(f"\nAnswer: {result['ai_response']}")
        
        # Prompt for understanding feedback
        understanding = input("\nDid this help? (yes/no/partially): ").lower()
        
        if understanding == 'yes':
            self.assistant.add_user_feedback("Clear and helpful explanation", rating=5)
        elif understanding == 'no':
            feedback = input("What was unclear? ")
            self.assistant.add_user_feedback(f"Unclear: {feedback}", rating=2)
        elif understanding == 'partially':
            feedback = input("What could be improved? ")
            self.assistant.add_user_feedback(f"Partially helpful: {feedback}", rating=3)
    
    def end_learning_session(self):
        """End the current learning session with reflection."""
        if not self.learning_session_active:
            return
        
        # Get session insights
        patterns = self.assistant.analyze_conversation_patterns(days_back=1)
        
        session_summary = f"Completed learning session on {self.current_topic}"
        satisfaction = float(input("Rate this session (0.0-1.0): "))
        
        self.assistant.end_session(session_summary, satisfaction)
        self.learning_session_active = False
        
        print(f"\nSession completed! You asked {patterns.get('total_sessions', 0)} questions.")
        print("Keep up the great learning!")
    
    def get_learning_progress(self):
        """Show learning progress and insights."""
        patterns = self.assistant.analyze_conversation_patterns(days_back=30)
        insights = self.assistant.get_learning_insights(limit=10)
        
        print("\n=== Learning Progress ===")
        print(f"Total learning sessions: {patterns['total_sessions']}")
        print(f"Average session satisfaction: {patterns['satisfaction_metrics']['average_satisfaction']:.2f}")
        
        if patterns['top_topics']:
            print("\nMost studied topics:")
            for topic, count in patterns['top_topics'][:5]:
                print(f"  {topic}: {count} sessions")
        
        if insights:
            print(f"\nRecent insights: {len(insights)} learning patterns identified")

# Example usage
if __name__ == "__main__":
    assistant = ProgrammingLearningAssistant("alice")
    
    assistant.start_learning_session("Python Object-Oriented Programming")
    assistant.ask_question("What is the difference between a class and an object?")
    assistant.ask_question("How do I implement inheritance in Python?")
    assistant.end_learning_session()
    
    assistant.get_learning_progress()
```

## Voice Companion Integration

### Multi-TTS Voice Assistant

```python
#!/usr/bin/env python3
"""
Voice-enabled personal assistant with TTS integration.
"""

from intelligent_knowledge_system import create_personal_ai_assistant
import speech_recognition as sr
import pyttsx3
from datetime import datetime

class VoicePersonalAssistant:
    def __init__(self, user_name):
        # Initialize the enhanced knowledge system
        self.assistant = create_personal_ai_assistant(
            profile_name=user_name,
            config={
                'db_path': f'voice_assistant_{user_name}.db',
                'privacy_level': 'standard'
            }
        )
        
        # Configure voice preferences
        self.setup_voice_preferences()
        
        # Initialize speech recognition and TTS
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        
        # Configure TTS based on user preferences
        self.configure_tts()
    
    def setup_voice_preferences(self):
        """Configure voice-specific preferences."""
        # TTS Engine preferences
        self.assistant.update_user_preference('preferred_tts_engine', 'elevenlabs')
        self.assistant.update_user_preference('tts_fallback_engine', 'mozilla')
        
        # Voice characteristics
        self.assistant.update_user_preference('voice_selection', 'professional_female')
        self.assistant.update_user_preference('speech_rate', 1.1)
        self.assistant.update_user_preference('speech_pitch', 1.0)
        self.assistant.update_user_preference('speech_volume', 0.8)
        
        # Cost management for premium TTS
        self.assistant.update_user_preference('cost_management', 'free_tier_priority')
        self.assistant.update_user_preference('max_monthly_cost', 10.0)
    
    def configure_tts(self):
        """Configure TTS engine based on user preferences."""
        rate = self.assistant.get_user_preference('speech_rate', 1.0)
        volume = self.assistant.get_user_preference('speech_volume', 0.8)
        
        # Configure pyttsx3 as fallback
        voices = self.tts_engine.getProperty('voices')
        self.tts_engine.setProperty('rate', int(200 * rate))
        self.tts_engine.setProperty('volume', volume)
        
        # Set voice based on preference
        voice_pref = self.assistant.get_user_preference('voice_selection', 'default')
        if 'female' in voice_pref and len(voices) > 1:
            self.tts_engine.setProperty('voice', voices[1].id)
    
    def listen_for_speech(self, timeout=5):
        """Listen for user speech input."""
        try:
            with self.microphone as source:
                print("Listening...")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=timeout)
            
            print("Processing speech...")
            text = self.recognizer.recognize_google(audio)
            return text
        
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            self.speak("Sorry, I didn't understand that.")
            return None
        except sr.RequestError as e:
            self.speak("Speech recognition error.")
            return None
    
    def speak(self, text):
        """Convert text to speech using preferred TTS engine."""
        # In a real implementation, this would check TTS preferences
        # and use ElevenLabs API or Mozilla TTS accordingly
        
        tts_engine = self.assistant.get_user_preference('preferred_tts_engine', 'system')
        
        if tts_engine == 'elevenlabs':
            # Use ElevenLabs API (mock implementation)
            print(f"[ElevenLabs TTS]: {text}")
            # self.elevenlabs_speak(text)
        elif tts_engine == 'mozilla':
            # Use Mozilla TTS (mock implementation)
            print(f"[Mozilla TTS]: {text}")
            # self.mozilla_speak(text)
        else:
            # Fallback to system TTS
            print(f"[System TTS]: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
    
    def start_voice_session(self):
        """Start an interactive voice session."""
        session_context = {
            'session_type': 'voice_interaction',
            'input_method': 'speech',
            'tts_engine': self.assistant.get_user_preference('preferred_tts_engine')
        }
        
        self.assistant.start_session(session_context)
        
        self.speak("Hello! I'm your personal voice assistant. How can I help you today?")
        
        while True:
            # Listen for wake word or continuous conversation
            user_speech = self.listen_for_speech()
            
            if user_speech is None:
                continue
            
            print(f"You said: {user_speech}")
            
            # Check for exit commands
            if any(phrase in user_speech.lower() for phrase in ['goodbye', 'exit', 'stop']):
                self.speak("Goodbye! Have a wonderful day!")
                break
            
            # Process the speech query
            result = self.assistant.process_query(user_speech)
            
            # Speak the response
            self.speak(result['ai_response'])
            
            # Adaptive speech rate based on response length
            response_length = len(result['ai_response'])
            if response_length > 200:
                # Slow down for longer responses
                current_rate = self.assistant.get_user_preference('speech_rate', 1.0)
                self.assistant.update_user_preference('speech_rate', max(0.8, current_rate - 0.1))
                self.configure_tts()
        
        # End the session
        self.assistant.end_session("Voice interaction session completed")

# Example usage
if __name__ == "__main__":
    voice_assistant = VoicePersonalAssistant("user")
    voice_assistant.start_voice_session()
```

## Learning and Adaptation Examples

### Adaptive Preference Learning

```python
#!/usr/bin/env python3
"""
Example of how the system learns and adapts to user preferences over time.
"""

from intelligent_knowledge_system import create_personal_ai_assistant
import time

def demonstrate_adaptive_learning():
    assistant = create_personal_ai_assistant("adaptive_user")
    
    print("=== Adaptive Learning Demonstration ===\n")
    
    # Initial preferences
    print("1. Setting initial preferences...")
    assistant.update_user_preference('verbosity_level', 'moderate')
    assistant.update_user_preference('technical_level', 'intermediate')
    
    # Simulate user interactions with feedback
    print("2. Simulating user interactions with feedback...\n")
    
    # Session 1: User finds responses too brief
    assistant.start_session({'demo': 'adaptive_learning_1'})
    
    result1 = assistant.process_query("Explain machine learning")
    print(f"Response 1: {result1['ai_response'][:100]}...")
    
    # User feedback indicates response was too brief
    assistant.add_user_feedback("Too brief, I need more detail", rating=2)
    print("Feedback: Too brief, I need more detail (rating: 2)")
    
    assistant.end_session()
    
    # Session 2: System adapts to provide more detail
    time.sleep(1)  # Small delay for timestamp difference
    assistant.start_session({'demo': 'adaptive_learning_2'})
    
    result2 = assistant.process_query("Explain neural networks")
    print(f"\nResponse 2: {result2['ai_response'][:100]}...")
    
    # User feedback indicates better response
    assistant.add_user_feedback("Much better, good level of detail", rating=4)
    print("Feedback: Much better, good level of detail (rating: 4)")
    
    assistant.end_session()
    
    # Session 3: User finds technical level too high
    time.sleep(1)
    assistant.start_session({'demo': 'adaptive_learning_3'})
    
    result3 = assistant.process_query("How does backpropagation work?")
    print(f"\nResponse 3: {result3['ai_response'][:100]}...")
    
    # User feedback about technical complexity
    assistant.add_user_feedback("Too technical, please simplify", rating=2)
    print("Feedback: Too technical, please simplify (rating: 2)")
    
    assistant.end_session()
    
    # Show how preferences have adapted
    print("\n3. Checking adapted preferences...")
    current_verbosity = assistant.get_user_preference('verbosity_level')
    current_technical = assistant.get_user_preference('technical_level')
    
    print(f"Adapted verbosity level: {current_verbosity}")
    print(f"Adapted technical level: {current_technical}")
    
    # Show learning insights
    insights = assistant.get_learning_insights(limit=5)
    print(f"\n4. Learning insights generated: {len(insights)}")
    
    for insight in insights:
        print(f"   - {insight['title']}: {insight['description'][:50]}...")
    
    # Generate recommendations based on learning
    recommendations = assistant.generate_personalized_recommendations()
    print("\n5. Personalized recommendations:")
    
    for category, recs in recommendations.items():
        if recs and category != 'generated_at':
            print(f"   {category}:")
            for rec in recs[:2]:
                print(f"     - {rec['suggestion']}")

if __name__ == "__main__":
    demonstrate_adaptive_learning()
```

## Privacy and Data Management

### Complete Privacy Control Example

```python
#!/usr/bin/env python3
"""
Comprehensive example of privacy controls and data management.
"""

from intelligent_knowledge_system import create_personal_ai_assistant
from datetime import datetime, timedelta
import json

def privacy_management_demo():
    assistant = create_personal_ai_assistant(
        "privacy_conscious_user",
        config={'privacy_level': 'enhanced'}
    )
    
    print("=== Privacy and Data Management Demo ===\n")
    
    # Create some sample data
    print("1. Creating sample conversation data...")
    for i in range(3):
        assistant.start_session({'demo_session': i})
        assistant.process_query(f"Sample query {i} about privacy")
        assistant.add_user_feedback("Good response", rating=4)
        assistant.end_session(f"Demo session {i}")
    
    # Privacy Dashboard
    print("\n2. Viewing Privacy Dashboard...")
    dashboard = assistant.get_privacy_dashboard()
    
    print(f"   Total sessions: {dashboard['personal_data_overview']['conversation_data']['total_sessions']}")
    print(f"   Total messages: {dashboard['personal_data_overview']['conversation_data']['total_messages']}")
    print(f"   Privacy level: {dashboard['personal_data_overview']['data_summary']['privacy_level']}")
    print(f"   Encryption status: {dashboard['personal_data_overview']['data_summary']['encryption_status']}")
    
    # View all personal data
    print("\n3. Viewing all personal data...")
    personal_data = assistant.view_all_personal_data()
    
    print("   Profile summary:")
    profile_summary = personal_data['profile_summary']
    for key, value in profile_summary.items():
        print(f"     {key}: {value}")
    
    # Data retention information
    print("\n4. Data retention information...")
    retention_info = assistant.get_data_retention_info()
    
    print("   Storage info:")
    for key, value in retention_info['storage_info'].items():
        print(f"     {key}: {value}")
    
    print("   Privacy controls available:")
    for control, available in retention_info['privacy_controls'].items():
        print(f"     {control}: {'✓' if available else '✗'}")
    
    # Export GDPR data
    print("\n5. Exporting GDPR-compliant data...")
    gdpr_export_path = assistant.export_gdpr_data()
    
    if gdpr_export_path:
        print(f"   GDPR export created: {gdpr_export_path}")
        
        # Show export structure
        with open(gdpr_export_path, 'r') as f:
            export_data = json.load(f)
        
        print("   Export includes:")
        for section in export_data.keys():
            print(f"     - {section}")
    
    # Demonstrate selective data deletion
    print("\n6. Demonstrating selective data deletion...")
    
    # Get session list
    sessions = assistant.get_session_history(10)
    if sessions:
        # Delete the oldest session
        oldest_session = sessions[-1]
        session_id = oldest_session['session_id']
        
        print(f"   Deleting session: {session_id}")
        success = assistant.delete_session_data(session_id, secure_delete=True)
        print(f"   Deletion successful: {success}")
        
        # Verify deletion
        updated_sessions = assistant.get_session_history(10)
        print(f"   Sessions before deletion: {len(sessions)}")
        print(f"   Sessions after deletion: {len(updated_sessions)}")
    
    # Demonstrate date range deletion
    print("\n7. Demonstrating date range deletion...")
    
    # Delete data older than 1 day (for demo purposes)
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=30)
    
    deletion_result = assistant.delete_date_range_data(start_date, end_date)
    print(f"   Deletion result: {deletion_result}")
    
    # Profile anonymization (optional - commented out to preserve demo data)
    print("\n8. Profile anonymization available...")
    print("   assistant.anonymize_profile() - would anonymize the profile")
    
    print("\n=== Privacy Demo Complete ===")
    print("All user data remains under complete user control!")

if __name__ == "__main__":
    privacy_management_demo()
```

## Advanced Analytics and Insights

### Conversation Pattern Analysis

```python
#!/usr/bin/env python3
"""
Advanced analytics and pattern recognition examples.
"""

from intelligent_knowledge_system import create_personal_ai_assistant
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

def analytics_demo():
    assistant = create_personal_ai_assistant("analytics_user")
    
    print("=== Advanced Analytics Demo ===\n")
    
    # Create diverse conversation data
    print("1. Creating diverse conversation data...")
    
    topics = ['programming', 'science', 'health', 'travel', 'cooking']
    satisfaction_scores = [0.9, 0.7, 0.8, 0.95, 0.85]
    
    for i, (topic, satisfaction) in enumerate(zip(topics, satisfaction_scores)):
        assistant.start_session({'topic_focus': topic})
        
        # Multiple queries per session
        for j in range(3):
            query = f"Tell me about {topic} - question {j+1}"
            result = assistant.process_query(query)
            
            # Vary feedback
            if j == 0:
                assistant.add_user_feedback("Great start!", rating=5)
            elif j == 1:
                assistant.add_user_feedback("Good information", rating=4)
            else:
                assistant.add_user_feedback("Helpful", rating=4)
        
        assistant.end_session(f"Session about {topic}", satisfaction)
    
    # Analyze conversation patterns
    print("\n2. Analyzing conversation patterns...")
    patterns = assistant.analyze_conversation_patterns(days_back=30)
    
    print(f"   Total sessions analyzed: {patterns['total_sessions']}")
    print(f"   Average session length: {patterns['session_statistics']['average_length_seconds']:.1f} seconds")
    print(f"   Average satisfaction: {patterns['satisfaction_metrics']['average_satisfaction']:.2f}")
    
    print("\n   Top discussion topics:")
    for topic, count in patterns['top_topics'][:5]:
        print(f"     {topic}: {count} sessions")
    
    # Generate personalized recommendations
    print("\n3. Generating personalized recommendations...")
    recommendations = assistant.generate_personalized_recommendations()
    
    for category, recs in recommendations.items():
        if recs and category != 'generated_at':
            print(f"\n   {category.replace('_', ' ').title()}:")
            for rec in recs:
                print(f"     - {rec['suggestion']}")
                print(f"       Reason: {rec['reason']}")
    
    # Learning insights
    print("\n4. Learning insights...")
    insights = assistant.get_learning_insights(limit=10)
    
    print(f"   Generated {len(insights)} learning insights:")
    for insight in insights:
        print(f"     - {insight['insight_type']}: {insight['title']}")
        print(f"       Confidence: {insight['confidence_level']:.2f}")
    
    # Export analytics data
    print("\n5. Exporting analytics data...")
    analytics_export = {
        'conversation_patterns': patterns,
        'recommendations': recommendations,
        'learning_insights': insights,
        'export_timestamp': datetime.now().isoformat()
    }
    
    with open('analytics_export.json', 'w') as f:
        json.dump(analytics_export, f, indent=2, default=str)
    
    print("   Analytics data exported to 'analytics_export.json'")
    
    # Visualize patterns (if matplotlib is available)
    try:
        visualize_patterns(patterns)
    except ImportError:
        print("   Install matplotlib for pattern visualization")

def visualize_patterns(patterns):
    """Create visualizations of conversation patterns."""
    print("\n6. Creating visualizations...")
    
    # Topic distribution
    if patterns['top_topics']:
        topics, counts = zip(*patterns['top_topics'][:5])
        
        plt.figure(figsize=(12, 4))
        
        # Topic frequency chart
        plt.subplot(1, 2, 1)
        plt.bar(topics, counts)
        plt.title('Most Discussed Topics')
        plt.xlabel('Topics')
        plt.ylabel('Session Count')
        plt.xticks(rotation=45)
        
        # Satisfaction trend
        plt.subplot(1, 2, 2)
        satisfaction_trend = patterns['satisfaction_metrics']['satisfaction_trend']
        if satisfaction_trend:
            plt.plot(satisfaction_trend, marker='o')
            plt.title('Satisfaction Trend')
            plt.xlabel('Recent Sessions')
            plt.ylabel('Satisfaction Score')
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('conversation_patterns.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("   Visualizations saved as 'conversation_patterns.png'")

if __name__ == "__main__":
    analytics_demo()
```

## Integration with Existing Systems

### Legacy System Migration

```python
#!/usr/bin/env python3
"""
Example of migrating from legacy intelligent_knowledge_system to enhanced version.
"""

from intelligent_knowledge_system import IntelligentKnowledgeSystem, EnhancedIntelligentKnowledgeSystem
import shutil
from pathlib import Path

def migration_example():
    print("=== Legacy System Migration Example ===\n")
    
    # 1. Create legacy system with some data
    print("1. Setting up legacy system...")
    legacy_config = {'test_mode': True}
    legacy_system = IntelligentKnowledgeSystem(legacy_config)
    
    # Add some data to legacy system
    legacy_result = legacy_system.process_query("What is artificial intelligence?")
    print(f"   Legacy system response: {legacy_result['query']}")
    
    # 2. Initialize enhanced system (automatically migrates)
    print("\n2. Initializing enhanced system...")
    enhanced_system = EnhancedIntelligentKnowledgeSystem(
        config={'migrate_from_legacy': True},
        profile_name="migrated_user"
    )
    
    print("   Enhanced system initialized with migration support")
    
    # 3. Verify enhanced features work
    print("\n3. Testing enhanced features...")
    
    # Start session (new feature)
    session_id = enhanced_system.start_session({'migration_test': True})
    print(f"   Started session: {session_id}")
    
    # Process query with enhanced tracking
    result = enhanced_system.process_query("Tell me about machine learning")
    print(f"   Enhanced query processed with session tracking")
    
    # Use new privacy features
    dashboard = enhanced_system.get_privacy_dashboard()
    print(f"   Privacy dashboard accessible: {len(dashboard)} sections")
    
    # End session
    enhanced_system.end_session("Migration test completed")
    print("   Session ended successfully")
    
    # 4. Show backward compatibility
    print("\n4. Testing backward compatibility...")
    
    # Legacy API still works
    legacy_compatible_result = enhanced_system.process_query("Backward compatibility test")
    print(f"   Legacy API compatibility: ✓")
    
    # But enhanced features are available
    patterns = enhanced_system.analyze_conversation_patterns()
    print(f"   Enhanced analytics available: ✓")
    
    print("\n=== Migration Complete ===")
    print("Legacy functionality preserved, enhanced features available!")

if __name__ == "__main__":
    migration_example()
```

### Custom Integration Example

```python
#!/usr/bin/env python3
"""
Example of integrating the enhanced system with custom applications.
"""

from intelligent_knowledge_system import create_personal_ai_assistant
import asyncio
import json
from datetime import datetime

class CustomApplicationIntegration:
    """Example integration with a custom application."""
    
    def __init__(self, app_name, user_id):
        self.app_name = app_name
        self.user_id = user_id
        
        # Initialize personal assistant for this user
        self.assistant = create_personal_ai_assistant(
            profile_name=f"{app_name}_{user_id}",
            config={
                'db_path': f'app_data/{app_name}_{user_id}.db',
                'privacy_level': 'enhanced',
                'auto_backup': True
            }
        )
        
        # Configure for application context
        self.assistant.update_user_preference('response_style', 'professional')
        self.assistant.update_user_preference('context_retention', 'persistent')
    
    async def handle_user_request(self, request_data):
        """Handle incoming user request with full context tracking."""
        
        # Extract request information
        user_query = request_data.get('query', '')
        request_context = request_data.get('context', {})
        session_metadata = request_data.get('session_metadata', {})
        
        # Start or continue session
        if not hasattr(self, 'current_session') or not self.current_session:
            session_context = {
                'app_name': self.app_name,
                'user_id': self.user_id,
                'request_timestamp': datetime.now().isoformat(),
                **session_metadata
            }
            self.current_session = self.assistant.start_session(session_context)
        
        # Process the query
        result = self.assistant.process_query(user_query, request_context)
        
        # Prepare response
        response = {
            'response_id': result.get('ai_message_id'),
            'content': result['ai_response'],
            'session_id': self.current_session,
            'web_search_used': result.get('web_search_performed', False),
            'confidence': result.get('search_confidence', 1.0),
            'sources': result.get('sources', []),
            'timestamp': datetime.now().isoformat()
        }
        
        return response
    
    async def process_user_feedback(self, feedback_data):
        """Process user feedback for learning."""
        
        feedback_text = feedback_data.get('text', '')
        rating = feedback_data.get('rating')
        message_id = feedback_data.get('message_id')
        
        success = self.assistant.add_user_feedback(feedback_text, rating, message_id)
        
        return {
            'feedback_processed': success,
            'learning_updated': success,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_user_analytics(self):
        """Get user analytics for the application."""
        
        patterns = self.assistant.analyze_conversation_patterns(days_back=30)
        recommendations = self.assistant.generate_personalized_recommendations()
        
        return {
            'usage_patterns': patterns,
            'personalized_recommendations': recommendations,
            'user_preferences': {
                'response_style': self.assistant.get_user_preference('response_style'),
                'verbosity_level': self.assistant.get_user_preference('verbosity_level'),
                'technical_level': self.assistant.get_user_preference('technical_level')
            }
        }
    
    def export_user_data(self, export_format='json'):
        """Export user data for compliance or backup."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = f'exports/{self.app_name}_{self.user_id}_{timestamp}.{export_format}'
        
        success = self.assistant.export_user_data(export_path, export_format)
        
        return {
            'export_successful': success,
            'export_path': export_path if success else None,
            'export_timestamp': datetime.now().isoformat()
        }
    
    def cleanup_session(self, session_summary=None):
        """Clean up current session."""
        
        if hasattr(self, 'current_session') and self.current_session:
            self.assistant.end_session(session_summary or "Application session ended")
            self.current_session = None

# Example usage
async def main():
    # Initialize integration for a custom app
    integration = CustomApplicationIntegration("MyApp", "user123")
    
    # Simulate user requests
    request1 = {
        'query': 'How do I optimize my workflow?',
        'context': {'feature': 'productivity', 'user_goal': 'efficiency'},
        'session_metadata': {'app_version': '2.1.0'}
    }
    
    response1 = await integration.handle_user_request(request1)
    print(f"Response 1: {response1['content'][:100]}...")
    
    # Simulate user feedback
    feedback = {
        'text': 'Very helpful suggestions!',
        'rating': 5,
        'message_id': response1['response_id']
    }
    
    feedback_result = await integration.process_user_feedback(feedback)
    print(f"Feedback processed: {feedback_result['feedback_processed']}")
    
    # Get analytics
    analytics = integration.get_user_analytics()
    print(f"User has {analytics['usage_patterns']['total_sessions']} sessions")
    
    # Export data
    export_result = integration.export_user_data()
    print(f"Data exported: {export_result['export_successful']}")
    
    # Cleanup
    integration.cleanup_session("Demo session completed")

if __name__ == "__main__":
    asyncio.run(main())
```

These examples demonstrate the full capabilities of the Enhanced Intelligent Knowledge System, from basic personal assistant setup to advanced analytics and custom application integration. Each example can be adapted and extended based on specific use cases and requirements.
