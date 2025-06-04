
import sys
import time
import os
import numpy as np
import soundfile as sf
from pathlib import Path
from fastrtc import (
    ReplyOnPause, 
    Stream, 
    get_stt_model, 
    get_tts_model, 
    AlgoOptions, 
    SileroVadOptions, 
    ReplyOnStopWords, 
    KokoroTTSOptions
)
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional, Union
from mem0 import Memory
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("🎙️ FastRTC Voice Assistant with Mem0 - Debug & Fixed Version")
print("=" * 60)

def print_status(message):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# Initialize models
print_status("🧠 Loading STT model (Moonshine)...")
try:
    stt_model = get_stt_model("moonshine/base")
    print_status("✅ STT model loaded!")
except Exception as e:
    print_status(f"❌ STT model failed: {e}")
    sys.exit(1)

print_status("🗣️ Loading TTS model (Kokoro)...")
try:
    tts_model = get_tts_model("kokoro")
    print_status("✅ TTS model loaded!")
except Exception as e:
    print_status(f"❌ TTS model failed: {e}")
    sys.exit(1)

# Fixed configuration
LM_STUDIO_URL = "http://192.168.1.5:1234/v1"  # Fixed: Added /v1
LM_STUDIO_MODEL = "mistral-7b-instruct"
OLLAMA_URL = "http://localhost:11434"

class VoiceAssistantWithMem0:
    """Voice assistant using Mem0 with proper debugging"""
    
    def __init__(self):
        print_status("🧠 Initializing Mem0 with proper configuration...")
        
        # Set required environment variable
        os.environ["OPENAI_API_KEY"] = "dummy-key-for-local-use"
        
        # Fixed configuration - corrected LM Studio URL
        self.config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "api_key": "dummy-key-for-local-use",
                    "model": LM_STUDIO_MODEL,
                    "openai_base_url": LM_STUDIO_URL,  # Now correctly points to /v1
                    "temperature": 0.7,
                    "max_tokens": 150
                }
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text",
                    "ollama_base_url": OLLAMA_URL
                }
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "voice_assistant_memories",
                    "path": "./mem0_voice_db"
                }
            },
            "version": "v1.1"
        }
        
        try:
            # Initialize Mem0 with proper config
            self.memory = Memory.from_config(self.config)
            print_status("✅ Mem0 initialized successfully!")
            
            # Test the memory system
            print_status("🧪 Testing memory system...")
            test_result = self.memory.add(
                messages="Test memory initialization", 
                user_id="test_user",
                infer=False,  # Disable inference for testing
            )
            print_status(f"🧪 Test result: {type(test_result)} - {test_result}")
            
        except Exception as e:
            print_status(f"❌ Mem0 initialization failed: {e}")
            print_status(f"📊 Error details: {type(e).__name__}: {str(e)}")
            sys.exit(1)
        
        # Session management
        self.user_id = "voice_user"
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_context = []
        
        print_status(f"👤 User ID: {self.user_id}")
        print_status(f"🎯 Session: {self.session_id}")
    
    def process_audio_array(self, audio_data):
        """Process audio data properly"""
        if isinstance(audio_data, tuple):
            sample_rate, audio_array = audio_data
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            return sample_rate, audio_array
        return audio_data
    
    def add_conversation_to_memory(self, user_text: str, assistant_text: str):
        """Add conversation to Mem0 with detailed debugging"""
        try:
            print_status(f"🔍 Adding to memory: User='{user_text[:50]}...', Assistant='{assistant_text[:50]}...'")
            
            # Create messages array as shown in documentation
            messages = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text}
            ]
            
            # Add metadata
            metadata = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "category": "conversation"
            }
            
            print_status(f"🔍 Calling memory.add() with messages: {len(messages)} items")
            
            # Use Mem0 add method with detailed error catching
            result = self.memory.add(
                messages=messages, 
                user_id=self.user_id, 
                metadata=metadata,
                infer=False  # Disable inference for debugging
            )
            
            print_status(f"🔍 Memory.add() returned: {type(result)} - {result}")
            
            # Update local context
            self.conversation_context.append({
                "user": user_text,
                "assistant": assistant_text,
                "timestamp": datetime.now()
            })
            
            # Keep only recent context
            if len(self.conversation_context) > 5:
                self.conversation_context.pop(0)
            
            if result is not None:
                print_status(f"💾 Memory added successfully! Result: {result}")
            else:
                print_status(f"⚠️ Memory.add() returned None - this indicates a problem")
                
            return result
            
        except Exception as e:
            print_status(f"❌ Memory addition failed: {type(e).__name__}: {e}")
            import traceback
            print_status(f"📊 Full traceback: {traceback.format_exc()}")
            return None
    
    def search_memories(self, query: str, limit: int = 3):
        """Search memories with debugging"""
        try:
            print_status(f"🔍 Searching memories for: '{query}'")
            memories = self.memory.search(
                query=query, 
                user_id=self.user_id,
                limit=limit
            )
            print_status(f"🔍 Search returned: {type(memories)} with {len(memories) if memories else 0} items")
            return memories
        except Exception as e:
            print_status(f"❌ Memory search failed: {type(e).__name__}: {e}")
            return []
    
    def get_all_memories(self):
        """Get all memories with debugging"""
        try:
            print_status(f"🔍 Getting all memories for user: {self.user_id}")
            all_memories = self.memory.get_all(user_id=self.user_id)
            print_status(f"🔍 Retrieved {len(all_memories) if all_memories else 0} total memories")
            return all_memories
        except Exception as e:
            print_status(f"❌ Memory retrieval failed: {type(e).__name__}: {e}")
            return []
    
    def handle_memory_commands(self, text: str) -> Optional[str]:
        """Handle special memory commands"""
        lower_text = text.lower()
        
        if "what do you remember" in lower_text or "what do you know about me" in lower_text:
            print_status("🧠 Processing memory recall command...")
            memories = self.get_all_memories()
            
            if not memories:
                return "I don't have any memories about you yet. Start a conversation and I'll remember!"
            
            # Format memories for display
            memory_texts = []
            print_status(f"🔍 Processing {len(memories)} memories for display...")
            
            for i, memory in enumerate(memories[-5:]):  # Show last 5
                print_status(f"🔍 Memory {i}: {type(memory)} - {memory}")
                
                if isinstance(memory, dict):
                    if 'memory' in memory:
                        memory_texts.append(memory['memory'][:100])
                    elif 'text' in memory:
                        memory_texts.append(memory['text'][:100])
                    else:
                        memory_texts.append(str(memory)[:100])
                elif isinstance(memory, str):
                    memory_texts.append(memory[:100])
                else:
                    memory_texts.append(str(memory)[:100])
            
            if memory_texts:
                result = f"Here's what I remember: {'; '.join(memory_texts)}"
                print_status(f"🧠 Returning memory summary: {result[:100]}...")
                return result
            else:
                return "I have memories stored but they're in an unexpected format."
        
        elif "forget everything" in lower_text or "clear memory" in lower_text:
            try:
                print_status("🗑️ Clearing all memories...")
                self.memory.delete_all(user_id=self.user_id)
                self.conversation_context = []
                return "I've cleared all my memories. Starting fresh!"
            except Exception as e:
                print_status(f"❌ Memory clearing failed: {e}")
                return f"I had trouble clearing memories: {e}"
        
        return None
    
    def get_context_for_response(self, user_input: str) -> str:
        """Get relevant context for the response"""
        print_status(f"🔍 Getting context for: '{user_input}'")
        context_parts = []
        
        # Search for relevant memories
        relevant_memories = self.search_memories(user_input)
        if relevant_memories:
            context_parts.append("Relevant memories:")
            for memory in relevant_memories:
                if isinstance(memory, dict) and 'memory' in memory:
                    context_parts.append(f"- {memory['memory']}")
                elif isinstance(memory, str):
                    context_parts.append(f"- {memory}")
        
        # Add recent conversation context
        if self.conversation_context:
            context_parts.append("Recent conversation:")
            for turn in self.conversation_context[-2:]:
                context_parts.append(f"User: {turn['user']}")
                context_parts.append(f"Assistant: {turn['assistant']}")
        
        context = "\
".join(context_parts) if context_parts else ""
        print_status(f"🔍 Context prepared ({len(context)} chars): {context[:100]}...")
        return context

# Initialize the voice assistant
print_status("🚀 Initializing Voice Assistant with Mem0...")
assistant = VoiceAssistantWithMem0()

def get_llm_response_with_memory(text):
    """Get LLM response with memory context"""
    
    # Check for memory commands first
    memory_response = assistant.handle_memory_commands(text)
    if memory_response:
        return memory_response
    
    # Get context from memories
    context = assistant.get_context_for_response(text)
    
    # Build the prompt with context
    system_prompt = f"""You are Mistral, a friendly voice assistant with memory capabilities.

{context}

Respond naturally and conversationally. If you have relevant memories about the user, reference them appropriately."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    
    try:
        print_status(f"🤖 Calling LM Studio at: {LM_STUDIO_URL}/chat/completions")
        
        # Call LM Studio directly for the response
        response = requests.post(f"{LM_STUDIO_URL}/chat/completions", 
            json={
                "model": LM_STUDIO_MODEL,
                "messages": messages,
                "max_tokens": 256,
                "temperature": 0.7
            },
            timeout=15
        )
        
        print_status(f"🤖 LM Studio response: {response.status_code}")
        
        if response.status_code == 200:
            assistant_text = response.json()["choices"][0]["message"]["content"]
            
            # Add this conversation to memory
            assistant.add_conversation_to_memory(text, assistant_text)
            
            return assistant_text
        else:
            print_status(f"⚠️ LM Studio request failed: {response.status_code} - {response.text}")
            return "Sorry, I couldn't process that request."
            
    except Exception as e:
        print_status(f"❌ LLM request failed: {type(e).__name__}: {e}")
        return f"I heard you say: {text}"

def voice_assistant(audio):
    """Main voice assistant pipeline"""
    try:
        # Process audio
        sample_rate, audio_array = assistant.process_audio_array(audio)
        
        print_status("🎧 Processing audio...")
        
        # Speech to text
        user_text = stt_model.stt(audio)
        
        if not user_text or not user_text.strip():
            print_status("🤫 No speech detected")
            return
        
        print_status(f"👤 User: {user_text}")
        
        # Get response with memory
        assistant_text = get_llm_response_with_memory(user_text)
        print_status(f"🤖 Assistant: {assistant_text}")
        
        # Show detailed memory stats
        all_memories = assistant.get_all_memories()
        memory_count = len(all_memories) if all_memories else 0
        session_turns = len(assistant.conversation_context)
        print_status(f"💾 Memory Stats: {memory_count} total memories, {session_turns} session turns")
        
        # Text to speech
        options = KokoroTTSOptions(
            voice="af_heart",
            speed=1.0, 
            lang="en-us"
        )
        
        for audio_chunk in tts_model.stream_tts_sync(assistant_text, options):
            yield audio_chunk
            
    except Exception as e:
        print_status(f"❌ Error in voice_assistant: {e}")
        logger.error(f"Voice assistant error: {e}")

# Create the FastRTC stream
print_status("🌐 Creating FastRTC stream...")

stream = Stream(
    ReplyOnPause(
        voice_assistant,
        can_interrupt=True,
        algo_options=AlgoOptions(
            audio_chunk_duration=3.0,
            started_talking_threshold=0.2,
            speech_threshold=0.1
        ),
        model_options=SileroVadOptions(
            threshold=0.5,
            min_speech_duration_ms=500,
            min_silence_duration_ms=2500
        )
    ), 
    modality="audio", 
    mode="send-receive",
    track_constraints={
        "echoCancellation": True,
        "noiseSuppression": {"exact": True},
        "autoGainControl": {"exact": True},
        "sampleRate": {"ideal": 24000},
        "sampleSize": {"ideal": 16},
        "channelCount": {"exact": 1},
    }
)

print("=" * 60)
print("🎉 SUCCESS! Mem0 Voice Assistant Ready (Debug Version)!")
print("=" * 60)
print("📋 Instructions:")
print("   1. Click the URL that opens in your browser")
print("   2. Allow microphone access")
print("   3. Start talking naturally")
print("   4. Watch the detailed logs for debugging")
print()
print("💡 Debug Features:")
print("   ✅ Detailed memory operation logging")
print("   ✅ Fixed LM Studio URL (/v1 endpoint)")
print("   ✅ Memory addition error tracking")
print("   ✅ Result type and content inspection")
print()
print("🧠 Try saying:")
print("   • 'My name is [your name]'")
print("   • 'I love [something]'")
print("   • 'What do you remember about me?'")
print()
print("🛑 To stop: Press Ctrl+C")
print("=" * 60)

# Launch the application
try:
    stream.ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
except KeyboardInterrupt:
    print_status("🛑 Voice assistant stopped")
except Exception as e:
    print_status(f"❌ Launch error: {e}")
    sys.exit(1)


# #!/usr/bin/env python3
# """
# FastRTC Voice Assistant with Mem0 Intelligent Memory
# Enhanced with Mem0 for superior memory management and AI-powered inference
# """

# import sys
# import time
# import os
# import numpy as np
# import soundfile as sf
# from pathlib import Path
# from fastrtc import (
#     ReplyOnPause, 
#     Stream, 
#     get_stt_model, 
#     get_tts_model, 
#     AlgoOptions, 
#     SileroVadOptions, 
#     ReplyOnStopWords, 
#     KokoroTTSOptions
# )
# import requests
# import json
# from datetime import datetime
# from typing import Dict, List, Optional, Union
# from mem0 import Memory
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# print("🎙️ FastRTC Voice Assistant with Mem0 - Proper Implementation")
# print("=" * 60)

# def print_status(message):
#     timestamp = time.strftime("%H:%M:%S")
#     print(f"[{timestamp}] {message}")

# # Initialize models
# print_status("🧠 Loading STT model (Moonshine)...")
# try:
#     stt_model = get_stt_model("moonshine/base")
#     print_status("✅ STT model loaded!")
# except Exception as e:
#     print_status(f"❌ STT model failed: {e}")
#     sys.exit(1)

# print_status("🗣️ Loading TTS model (Kokoro)...")
# try:
#     tts_model = get_tts_model("kokoro")
#     print_status("✅ TTS model loaded!")
# except Exception as e:
#     print_status(f"❌ TTS model failed: {e}")
#     sys.exit(1)

# # Configuration
# LM_STUDIO_URL = "http://192.168.1.5:1234"
# LM_STUDIO_MODEL = "mistral-7b-instruct"
# OLLAMA_URL = "http://localhost:11434"

# class VoiceAssistantWithMem0:
#     """Voice assistant using Mem0 exactly as documented"""
    
#     def __init__(self):
#         print_status("🧠 Initializing Mem0 with proper configuration...")
        
#         # Set required environment variable
#         os.environ["OPENAI_API_KEY"] = "dummy-key-for-local-use"
        
#         # Configure Mem0 properly according to documentation
#         self.config = {
#             "llm": {
#                 "provider": "openai",  # Use OpenAI-compatible API
#                 "config": {
#                     "api_key": "dummy-key-for-local-use",
#                     "model": LM_STUDIO_MODEL,
#                     "openai_base_url": LM_STUDIO_URL,  # Point to LM Studio
#                     "temperature": 0.7,
#                     "max_tokens": 150
#                 }
#             },
#             "embedder": {
#                 "provider": "ollama",  # Use Ollama for embeddings
#                 "config": {
#                     "model": "nomic-embed-text",
#                     "ollama_base_url": OLLAMA_URL
#                 }
#             },
#             "vector_store": {
#                 "provider": "chroma",  # Use ChromaDB
#                 "config": {
#                     "collection_name": "voice_assistant_memories",
#                     "path": "./mem0_voice_db"
#                 }
#             },
#             "version": "v1.1"
#         }
        
#         try:
#             # Initialize Mem0 with proper config
#             self.memory = Memory.from_config(self.config)
#             print_status("✅ Mem0 initialized successfully!")
#         except Exception as e:
#             print_status(f"❌ Mem0 initialization failed: {e}")
#             sys.exit(1)
        
#         # Session management
#         self.user_id = "voice_user"
#         self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         self.conversation_context = []
        
#         print_status(f"👤 User ID: {self.user_id}")
#         print_status(f"🎯 Session: {self.session_id}")
    
#     def process_audio_array(self, audio_data):
#         """Process audio data properly"""
#         if isinstance(audio_data, tuple):
#             sample_rate, audio_array = audio_data
#             if audio_array.dtype != np.float32:
#                 audio_array = audio_array.astype(np.float32)
#             if np.max(np.abs(audio_array)) > 1.0:
#                 audio_array = audio_array / np.max(np.abs(audio_array))
#             return sample_rate, audio_array
#         return audio_data
    
#     def add_conversation_to_memory(self, user_text: str, assistant_text: str):
#         """Add conversation to Mem0 using the documented API"""
#         try:
#             # Create messages array as shown in documentation
#             messages = [
#                 {"role": "user", "content": user_text},
#                 {"role": "assistant", "content": assistant_text}
#             ]
            
#             # Add metadata as shown in documentation
#             metadata = {
#                 "session_id": self.session_id,
#                 "timestamp": datetime.now().isoformat(),
#                 "category": "conversation"
#             }
            
#             # Use Mem0 add method exactly as documented
#             result = self.memory.add(
#                 messages=messages, 
#                 user_id=self.user_id, 
#                 metadata=metadata
#             )
            
#             # Update local context
#             self.conversation_context.append({
#                 "user": user_text,
#                 "assistant": assistant_text,
#                 "timestamp": datetime.now()
#             })
            
#             # Keep only recent context
#             if len(self.conversation_context) > 5:
#                 self.conversation_context.pop(0)
            
#             print_status(f"💾 Added to memory successfully")
#             return result
            
#         except Exception as e:
#             print_status(f"⚠️ Memory addition failed: {e}")
#             return None
    
#     def search_memories(self, query: str, limit: int = 3):
#         """Search memories using documented API"""
#         try:
#             # Use search method exactly as documented
#             memories = self.memory.search(
#                 query=query, 
#                 user_id=self.user_id,
#                 limit=limit
#             )
#             return memories
#         except Exception as e:
#             print_status(f"⚠️ Memory search failed: {e}")
#             return []
    
#     def get_all_memories(self):
#         """Get all memories using documented API"""
#         try:
#             # Use get_all method exactly as documented
#             all_memories = self.memory.get_all(user_id=self.user_id)
#             return all_memories
#         except Exception as e:
#             print_status(f"⚠️ Memory retrieval failed: {e}")
#             return []
    
#     def handle_memory_commands(self, text: str) -> Optional[str]:
#         """Handle special memory commands"""
#         lower_text = text.lower()
        
#         if "what do you remember" in lower_text or "what do you know about me" in lower_text:
#             memories = self.get_all_memories()
#             if not memories:
#                 return "I don't have any memories about you yet. Start a conversation and I'll remember!"
            
#             # Format memories for display
#             memory_texts = []
#             for memory in memories[-5:]:  # Show last 5
#                 if isinstance(memory, dict) and 'memory' in memory:
#                     memory_texts.append(memory['memory'][:100])
#                 elif isinstance(memory, str):
#                     memory_texts.append(memory[:100])
            
#             if memory_texts:
#                 return f"Here's what I remember: {'; '.join(memory_texts)}"
#             else:
#                 return "I have memories stored but they're in an unexpected format."
        
#         elif "forget everything" in lower_text or "clear memory" in lower_text:
#             try:
#                 self.memory.delete_all(user_id=self.user_id)
#                 self.conversation_context = []
#                 return "I've cleared all my memories. Starting fresh!"
#             except Exception as e:
#                 return f"I had trouble clearing memories: {e}"
        
#         return None
    
#     def get_context_for_response(self, user_input: str) -> str:
#         """Get relevant context for the response"""
#         context_parts = []
        
#         # Search for relevant memories
#         relevant_memories = self.search_memories(user_input)
#         if relevant_memories:
#             context_parts.append("Relevant memories:")
#             for memory in relevant_memories:
#                 if isinstance(memory, dict) and 'memory' in memory:
#                     context_parts.append(f"- {memory['memory']}")
#                 elif isinstance(memory, str):
#                     context_parts.append(f"- {memory}")
        
#         # Add recent conversation context
#         if self.conversation_context:
#             context_parts.append("Recent conversation:")
#             for turn in self.conversation_context[-2:]:
#                 context_parts.append(f"User: {turn['user']}")
#                 context_parts.append(f"Assistant: {turn['assistant']}")
        
#         return "\
# ".join(context_parts) if context_parts else ""

# # Initialize the voice assistant
# print_status("🚀 Initializing Voice Assistant with Mem0...")
# assistant = VoiceAssistantWithMem0()

# def get_llm_response_with_memory(text):
#     """Get LLM response with memory context"""
    
#     # Check for memory commands first
#     memory_response = assistant.handle_memory_commands(text)
#     if memory_response:
#         return memory_response
    
#     # Get context from memories
#     context = assistant.get_context_for_response(text)
    
#     # Build the prompt with context
#     system_prompt = f"""You are Mistral, a friendly voice assistant with memory capabilities.

# {context}

# Respond naturally and conversationally. If you have relevant memories about the user, reference them appropriately."""

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": text}
#     ]
    
#     try:
#         # Call LM Studio directly for the response
#         response = requests.post(f"{LM_STUDIO_URL}/v1/chat/completions", 
#             json={
#                 "model": LM_STUDIO_MODEL,
#                 "messages": messages,
#                 "max_tokens": 256,
#                 "temperature": 0.7
#             },
#             timeout=15
#         )
        
#         if response.status_code == 200:
#             assistant_text = response.json()["choices"][0]["message"]["content"]
            
#             # Add this conversation to memory
#             assistant.add_conversation_to_memory(text, assistant_text)
            
#             return assistant_text
#         else:
#             print_status(f"⚠️ LLM request failed: {response.status_code}")
#             return "Sorry, I couldn't process that request."
            
#     except Exception as e:
#         print_status(f"🔄 LLM request failed: {e}")
#         return f"I heard you say: {text}"

# def voice_assistant(audio):
#     """Main voice assistant pipeline"""
#     try:
#         # Process audio
#         sample_rate, audio_array = assistant.process_audio_array(audio)
        
#         print_status("🎧 Processing audio...")
        
#         # Speech to text
#         user_text = stt_model.stt(audio)
        
#         if not user_text or not user_text.strip():
#             print_status("🤫 No speech detected")
#             return
        
#         print_status(f"👤 User: {user_text}")
        
#         # Get response with memory
#         assistant_text = get_llm_response_with_memory(user_text)
#         print_status(f"🤖 Assistant: {assistant_text}")
        
#         # Show memory stats
#         all_memories = assistant.get_all_memories()
#         memory_count = len(all_memories) if all_memories else 0
#         print_status(f"💾 Total memories: {memory_count}")
        
#         # Text to speech
#         options = KokoroTTSOptions(
#             voice="af_heart",
#             speed=1.0, 
#             lang="en-us"
#         )
        
#         for audio_chunk in tts_model.stream_tts_sync(assistant_text, options):
#             yield audio_chunk
            
#     except Exception as e:
#         print_status(f"❌ Error in voice_assistant: {e}")
#         logger.error(f"Voice assistant error: {e}")

# # Create the FastRTC stream
# print_status("🌐 Creating FastRTC stream...")


# stream = Stream(
#     ReplyOnPause(
#         voice_assistant,
#         can_interrupt=True,
#          algo_options=AlgoOptions(
#             audio_chunk_duration=1.0,
#             started_talking_threshold=0.2,
#             speech_threshold=0.1
#         ),
#         model_options=SileroVadOptions(
#             threshold=0.5,
#             min_speech_duration_ms=200,
#             min_silence_duration_ms=2500
#         )
#     ), 
#     modality="audio", 
#     mode="send-receive",
#     track_constraints={
#             "echoCancellation": True,
#             "noiseSuppression": {"exact": True},
#             "autoGainControl": {"exact": True},
#             "sampleRate": {"ideal": 24000},
#             "sampleSize": {"ideal": 16},
#             "channelCount": {"exact": 1},
#         }
# )

# print("=" * 50)
# print("🎉 SUCCESS! Mem0 Voice Assistant is ready!")
# print("=" * 50)
# print("📋 What to do next:")
# print("   1. Click the URL that opens in your browser")
# print("   2. Click 'Allow' when asked for microphone permission")
# print("   3. Start talking - no buttons needed!")
# print("   4. The assistant will respond automatically")
# print()
# print("💡 Enhanced Features:")
# print("   ✅ AI-powered memory inference (Mem0)")
# print("   ✅ Fixed memory format handling")
# print("   ✅ LM Studio compatibility improvements")
# print("   ✅ Robust error handling")
# print("   ✅ Memory persistence across sessions")
# print("   ✅ Real-time audio streaming")
# print("   ✅ Conversation interruption")
# print()
# print("🧠 Memory Commands:")
# print("   • 'What do you remember about me?'")
# print("   • 'Forget everything' or 'Clear memory'")
# print()
# print("🛑 To stop: Press Ctrl+C")
# print("=" * 50)

# # Launch the UI
# try:
#     stream.ui.launch(
#         server_name="0.0.0.0",
#         server_port=7860,
#         share=False
#     )
# except KeyboardInterrupt:
#     print_status("🛑 Voice assistant stopped by user")
# except Exception as e:
#     print_status(f"❌ Error: {e}")
#     logger.error(f"Launch error: {e}")
#     sys.exit(1)