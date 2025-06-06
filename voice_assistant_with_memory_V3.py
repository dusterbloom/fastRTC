#!/usr/bin/env python3
"""
FastRTC Voice Assistant with A-MEM (Agentic Memory) + Bluetooth Optimization
Replaces mem0 with A-MEM for intelligent memory management with dynamic organization
"""

import sys
import time
import os
import numpy as np
import asyncio
import aiohttp
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastrtc import (
    ReplyOnPause, 
    Stream, 
    get_stt_model, 
    get_tts_model, 
    AlgoOptions, 
    SileroVadOptions, 
    KokoroTTSOptions
)
from fastrtc.utils import AdditionalOutputs 
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import logging
from collections import deque
import hashlib

# Import A-MEM components
from a_mem.memory_system import AgenticMemorySystem
from qdrant_client.models import Distance, VectorParams
from qdrant_client import QdrantClient
from a_mem.memory_system import AgenticMemorySystem, MemoryNote



# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aioice.ice").setLevel(logging.WARNING) 
logging.getLogger("aiortc").setLevel(logging.WARNING) 
logging.getLogger("phonemizer").setLevel(logging.WARNING) 

logger = logging.getLogger(__name__) 

print("🧠 FastRTC Voice Assistant - A-MEM + Bluetooth Optimization")
print("=" * 75)

def print_status(message):
    timestamp = time.strftime("%H:%M:%S")
    logger.info(f"{message}") 

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
    if hasattr(tts_model, 'model') and hasattr(tts_model.model, 'voices'):
        available_voices = getattr(tts_model.model, 'voices', [])
        if available_voices:
            print_status(f"Kokoro TTS: Available voice names (first few): {list(available_voices)[:5]}")
        else:
            print_status("Kokoro TTS: Could not list specific voice names from model.")
    else:
        print_status("Kokoro TTS: Voice listing not directly available via tts_model.model.voices.")
    print_status("✅ TTS model loaded!")
except Exception as e:
    print_status(f"❌ TTS model failed: {e}")
    sys.exit(1)

# --- Configuration ---
USE_OLLAMA_FOR_CONVERSATION = True # Set to True to use Ollama, False for LM Studio

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") 
OLLAMA_CONVERSATIONAL_MODEL = os.getenv("OLLAMA_CONVERSATIONAL_MODEL", "llama3:8b-instruct-q4_K_M")

# LM Studio settings (only used if USE_OLLAMA_FOR_CONVERSATION is False)
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://192.168.1.5:1234/v1") 
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "mistral-nemo-instruct-2407") 

# A-MEM settings
AMEM_LLM_MODEL = os.getenv("AMEM_LLM_MODEL", "llama3.2:3b") 
AMEM_EMBEDDER_MODEL = os.getenv("AMEM_EMBEDDER_MODEL", "nomic-embed-text")

KOKORO_PREFERRED_VOICE = "af_heart" 
KOKORO_FALLBACK_VOICE_1 = "af_alloy"  
KOKORO_FALLBACK_VOICE_2 = "af_bella" 

AUDIO_SAMPLE_RATE = 16000 
MINIMAL_SILENT_FRAME_DURATION_MS = 20 
MINIMAL_SILENT_SAMPLES = int(AUDIO_SAMPLE_RATE * (MINIMAL_SILENT_FRAME_DURATION_MS / 1000.0))
SILENT_AUDIO_CHUNK_ARRAY = np.zeros(MINIMAL_SILENT_SAMPLES, dtype=np.float32)
SILENT_AUDIO_FRAME_TUPLE = (AUDIO_SAMPLE_RATE, SILENT_AUDIO_CHUNK_ARRAY)
EMPTY_AUDIO_YIELD_OUTPUT = (SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs())


class BluetoothAudioProcessor:
    def __init__(self):
        self.audio_buffer = deque(maxlen=10)
        self.noise_floor = None
        self.calibration_frames = 0
        self.min_calibration_frames = 15
        self.voice_detection_stats = deque(maxlen=30)
        
    def calibrate_noise_floor(self, audio_data):
        if self.calibration_frames < self.min_calibration_frames:
            rms = np.sqrt(np.mean(audio_data**2)) if audio_data.size > 0 else 0.0
            if self.noise_floor is None: self.noise_floor = rms
            else: self.noise_floor = 0.9 * self.noise_floor + 0.1 * rms
            self.calibration_frames += 1
            if self.calibration_frames == self.min_calibration_frames:
                print_status(f"🎙️ Bluetooth calibration complete: noise_floor={self.noise_floor:.6f}")
        
    def preprocess_bluetooth_audio(self, audio_data):
        if isinstance(audio_data, tuple) and len(audio_data) == 2 : 
            sample_rate, audio_array = audio_data
            if not isinstance(sample_rate, int) or not isinstance(audio_array, np.ndarray):
                 return AUDIO_SAMPLE_RATE, np.array([], dtype=np.float32) 
        elif isinstance(audio_data, np.ndarray): 
            sample_rate, audio_array = AUDIO_SAMPLE_RATE, audio_data
        else:
            return AUDIO_SAMPLE_RATE, np.array([], dtype=np.float32) 

        if audio_array.size == 0:
            return sample_rate, np.array([], dtype=np.float32)

        if audio_array.dtype != np.float32: audio_array = audio_array.astype(np.float32)
        
        max_abs = np.max(np.abs(audio_array))
        if max_abs > 1.0 and max_abs > 1e-6 : audio_array = audio_array / max_abs
        
        self.calibrate_noise_floor(audio_array)
        
        if len(audio_array) > 1: 
            diff = np.diff(audio_array, prepend=audio_array[0])
            audio_array = audio_array + 0.15 * diff
        
        if self.noise_floor is not None: 
            current_rms = np.sqrt(np.mean(audio_array**2)) if audio_array.size > 0 else 0.0
            if current_rms > self.noise_floor * 1.5 and current_rms > 1e-6:
                target_rms = 0.08
                if current_rms < target_rms * 0.4:
                    gain = min(2.5, (target_rms * 0.6 / current_rms) if current_rms > 1e-6 else 1.0)
                    audio_array = audio_array * gain
        
        max_abs_processed = np.max(np.abs(audio_array)) 
        if max_abs_processed > 1.0 and max_abs_processed > 1e-6:
             audio_array = audio_array / max_abs_processed
        
        self.voice_detection_stats.append({'rms': np.sqrt(np.mean(audio_array**2)) if audio_array.size > 0 else 0.0, 'timestamp': time.time()})
        return sample_rate, audio_array
    
    def get_detection_stats(self):
        if not self.voice_detection_stats: return None
        recent_rms = [s['rms'] for s in list(self.voice_detection_stats)[-10:] if s['rms'] > 1e-9]
        return {
            'avg_rms': np.mean(recent_rms) if recent_rms else 0.0,
            'noise_floor': self.noise_floor,
            'calibrated': self.calibration_frames >= self.min_calibration_frames
        }


class AMemMemoryManager:
    """A-MEM based memory manager for voice assistant"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="amem_exec")
        self.memory_cache = {'user_name': None, 'preferences': {}, 'facts': {}, 'last_updated': None}
        self.memory_queue = asyncio.Queue()
        self.background_task: Optional[asyncio.Task] = None
        self.memory_operations = 0
        self.cache_hits = 0
        
        # Initialize A-MEM system
        try:
            self.amem_system = AgenticMemorySystem(
                model_name='all-MiniLM-L6-v2',
                llm_backend="ollama",
                llm_model=AMEM_LLM_MODEL,
                evo_threshold=50  # Trigger evolution more frequently for voice interactions
            )
            print_status("🧠 A-MEM system initialized successfully")
        except Exception as e:
            print_status(f"❌ A-MEM initialization failed: {e}")
            raise
        
        self._load_existing_memories() 
        print_status("🧠 A-MEM memory manager initialized")
    
    async def start_background_processor(self):
        if self.memory_queue and (self.background_task is None or self.background_task.done()):
            self.background_task = asyncio.create_task(self._process_memory_queue())
            print_status("🚀 Background A-MEM processor started.")
    
    
    async def _process_memory_queue(self):
        print_status("Background A-MEM processor listening...")
        while True:
            try:
                # Attempt to get an item. This is a primary await point where cancellation can occur.
                op, user_text, assistant_text, category = await self.memory_queue.get()
                
                # If memory_queue.get() completed, an item was successfully dequeued.
                # We are now responsible for calling task_done() for this item.
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self.executor, self._store_memory_background,
                        user_text, assistant_text, category
                    )
                except Exception as processing_error:
                    # Log errors that occur during the execution of _store_memory_background
                    print_status(f"❌ Error during background A-MEM storage execution: {processing_error}")
                    import traceback
                    traceback.print_exc()
                    # The task for the item will still be marked as done in the finally block.
                finally:
                    # This block ensures task_done() is called for the item retrieved by get(),
                    # regardless of whether its processing was successful or raised an exception.
                    # It also runs if the outer try block is cancelled after get() but during run_in_executor.
                    self.memory_queue.task_done()
                    # You can add a log here for debugging queue size if needed:
                    # print_status(f"A-MEM queue task_done called. Current qsize: {self.memory_queue.qsize()}")

            except asyncio.CancelledError:
                print_status("Background A-MEM processor task was cancelled.")
                # If cancellation occurred during `await self.memory_queue.get()`, no item was retrieved,
                # so no corresponding task_done() is needed from this iteration.
                # If an item was retrieved and cancellation occurred during its processing,
                # the inner `finally` block (above) is responsible for calling task_done().
                break # Exit the while True loop
            except Exception as e: # Catch other unexpected errors, e.g., from memory_queue.get() if not CancelledError
                print_status(f"❌ Unexpected error in A-MEM queue processing loop: {e}")
                import traceback
                traceback.print_exc()
                # Avoid busy-looping if there's a persistent issue with get()
                await asyncio.sleep(1)


    def _parse_timestamp(self, timestamp_str: Optional[str]) -> datetime:
 

        if not timestamp_str: 
            # logger.info(f"DEBUG _parse_timestamp: Empty timestamp_str, returning datetime.min") # Optional
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            # Attempt ISO format first (common for external sources, though not our primary format)
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
            # logger.info(f"DEBUG _parse_timestamp: Parsed '{timestamp_str}' as ISO, returning {dt}") # Optional
            return dt
        except ValueError:
            # Try our specific formats
            for fmt in ('%Y%m%d%H%M', # Primary format used by MemoryNote
                        '%Y-%m-%d %H:%M:%S.%f%z', '%Y-%m-%d %H:%M:%S%z', 
                        '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                try:
                    dt = datetime.strptime(timestamp_str, fmt)
                    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
                    # logger.info(f"DEBUG _parse_timestamp: Parsed '{timestamp_str}' with format '{fmt}', returning {dt}") # Optional
                    return dt
                except ValueError: 
                    continue
            logger.warning(f"Could not parse timestamp string: '{timestamp_str}' with any known format. Returning datetime.min.")
            return datetime.min.replace(tzinfo=timezone.utc)

    def _extract_name_from_memory_text(self, memory_text: str) -> Optional[str]:
        # Only process if this is clearly from a user message
        if "Assistant:" in memory_text:
            # Split and only look at user portions
            user_parts = []
            for part in memory_text.split('\n'):
                if part.startswith("User:"):
                    user_parts.append(part[5:].strip())  # Remove "User:" prefix
            text_to_analyze = " ".join(user_parts)
        else:
            text_to_analyze = memory_text
        
        # Keep original text for name extraction (don't lowercase yet)
        text_lower = text_to_analyze.lower()
        
        patterns = [
            r"my name is (\w+)(?:\s+\w+)?",
            r"i'?m (\w+)(?:\s+\w+)?",
            r"call me (\w+)(?:\s+\w+)?",
        ]
        
        invalid_names = {
            'not', 'no', 'actually', 'sorry', 'really', 'very',
            'hungry', 'tired', 'happy', 'sad', 'excited', 'bored',
            'going', 'doing', 'feeling', 'thinking', 'wondering',
            'shocked', 'surprised', 'glad', 'sure', 'okay', 'fine',
            'great', 'good', 'bad', 'terrible', 'wonderful'
        }
        
        for pattern in patterns:
            # Search in lowercase but extract from original
            match = re.search(r'\b' + pattern + r'\b', text_lower)
            if match:
                # Get the actual position in original text
                start_pos = match.start(1)
                end_pos = match.end(1)
                
                # Extract from original text to preserve case
                name = text_to_analyze[start_pos:end_pos].strip()
                
                if name and name.lower() not in invalid_names and len(name) > 1:
                    # Don't force capitalization - preserve original case
                    return name
        
        return None
    # def _load_existing_memories(self): 
    #     """Load existing memories from A-MEM system"""
    #     try:
    #         # Get all memories from A-MEM
    #         all_memories = list(self.amem_system.memories.values())
            
        #     if all_memories:
        #         print_status(f"📚 Loading {len(all_memories)} existing A-MEM memories for user '{self.user_id}'...")
        #         name_entries = []
        #         pref_entries = {} 

        #         for memory in all_memories:
        #             memory_text = memory.content
        #             timestamp_str = memory.timestamp
                    
        #             item_timestamp = self._parse_timestamp(timestamp_str)

        #             potential_name = self._extract_name_from_memory_text(memory_text)
        #             if potential_name:
        #                 name_entries.append({'name': potential_name, 'timestamp': item_timestamp})
                    
        #             # Extract preferences from memory tags
        #             if "preference" in memory.tags:
        #                 pref_match = re.search(r"i like ([\w\s.,'-]+)|i love ([\w\s.,'-]+)|my favorite [\w\s]+ is ([\w\s.,'-]+)", memory_text, re.IGNORECASE)
        #                 if pref_match:
        #                     preference_text = (pref_match.group(1) or pref_match.group(2) or pref_match.group(3) or "").strip()
        #                     if preference_text and len(preference_text) > 2 and len(preference_text) < 150:
        #                         pref_key = hashlib.md5(preference_text.lower().encode()).hexdigest()[:10]
        #                         if pref_key not in pref_entries or item_timestamp > pref_entries[pref_key]['timestamp']:
        #                             pref_entries[pref_key] = {'text': preference_text, 'timestamp': item_timestamp}
                
        #         if name_entries:
        #             name_entries.sort(key=lambda x: x['timestamp'], reverse=True)
        #             self.memory_cache['user_name'] = name_entries[0]['name']
        #             print_status(f"👤 Restored user name from A-MEM: {self.memory_cache['user_name']}")
        #         else: 
        #             print_status("👤 No user name found in A-MEM memories.")

        #         if pref_entries:
        #             self.memory_cache['preferences'] = {k: v['text'] for k, v in pref_entries.items()}
        #             print_status(f"👍 Restored {len(self.memory_cache['preferences'])} preferences from A-MEM")
        #         else: 
        #             print_status("👍 No preferences found in A-MEM memories.")
                
        #         self.memory_cache['last_updated'] = datetime.now(timezone.utc)
        #         print_status(f"✅ A-MEM cache populated. Name: '{self.memory_cache['user_name']}', Prefs: {len(self.memory_cache['preferences'])}")
        #     else: 
        #         print_status(f"📚 No existing A-MEM memories found for user '{self.user_id}'.")
        # except Exception as e:
        #     print_status(f"⚠️ Failed to load existing A-MEM memories: {e}")
        #     import traceback; traceback.print_exc()


    def _load_existing_memories(self): 
        """Load existing memories from A-MEM system"""
        try:
                # First check if memories are already loaded in A-MEM system
                all_memories = list(self.amem_system.memories.values())

                if not all_memories:
                    print_status(f"📚 No memories in A-MEM system, checking ChromaDB...")
                    try:
                        collection = self.amem_system.retriever.collection
                        results = collection.get()  # Get all documents

                        if results and 'ids' in results and results['ids']:
                            print_status(f"📚 Loading {len(results['ids'])} memories from ChromaDB...")
                            self.amem_system._load_memories_from_chromadb()
                            all_memories = list(self.amem_system.memories.values())
                    except Exception as e:
                        print_status(f"⚠️ Error accessing ChromaDB: {e}")
                        return

                if all_memories:
                    print_status(f"📚 Processing {len(all_memories)} A-MEM memories for user '{self.user_id}'...")
                    personal_entries = []
                    fallback_entries = []
                    pref_entries = {} 
    
                    for memory in all_memories:
                        memory_text = memory.content
                        timestamp_str = memory.timestamp

                        # ---- ADD THIS ----
                        self.memory_cache['_currently_processing_content'] = memory_text 
                        # ---- END ADD ----

                        item_timestamp = self._parse_timestamp(timestamp_str)

                        
                        
                        del self.memory_cache['_currently_processing_content']
                    
                        # First collect personal_info tagged memories
                        if "personal_info" in memory.tags:
                            potential_name = self._extract_name_from_memory_text(memory_text)
                            if potential_name:
                                personal_entries.append({'name': potential_name, 'timestamp': item_timestamp})
                                logger.info(f"DEBUG _load_existing_memories: Added to personal_entries. Name: {potential_name}, Tags: {memory.tags}")

                        else:
                            potential_name = self._extract_name_from_memory_text(memory_text)
                            if potential_name:
                                fallback_entries.append({'name': potential_name, 'timestamp': item_timestamp})
    
                        if "preference" in memory.tags or "preferences" in memory_text.lower():
                            pref_match = re.search(r"i like ([\w\s.,'-]+)|i love ([\w\s.,'-]+)|my favorite [\w\s]+ is ([\w\s.,'-]+)", memory_text, re.IGNORECASE)
                            if pref_match:
                                preference_text = (pref_match.group(1) or pref_match.group(2) or pref_match.group(3) or "").strip()
                                if preference_text and 2 < len(preference_text) < 150:
                                    pref_key = hashlib.md5(preference_text.lower().encode()).hexdigest()[:10]
                                    if pref_key not in pref_entries or item_timestamp > pref_entries[pref_key]['timestamp']:
                                        pref_entries[pref_key] = {'text': preference_text, 'timestamp': item_timestamp}
    
                    # Use personal_info memories if available; otherwise fallback entries.
                    if personal_entries:
                        personal_entries.sort(key=lambda x: x['timestamp'], reverse=True)
                        top_personal_entry_name = personal_entries[0]['name']
                        self.memory_cache['user_name'] = top_personal_entry_name
                        # ---- ADD THIS LOG ----
                        logger.info(f"DEBUG _load_existing_memories: Sorted personal_entries (Top 3): {personal_entries[:3]}")
                        # ---- END ADD LOG ----

                        print_status(f"👤 Restored user name from A-MEM (personal_info): {self.memory_cache['user_name']}")
                    elif fallback_entries:
                        fallback_entries.sort(key=lambda x: x['timestamp'], reverse=True)

                        # ---- ADD THIS LOG (optional, for completeness) ----
                        logger.info(f"DEBUG _load_existing_memories: Sorted fallback_entries (Top 3): {fallback_entries[:3]}")
                        # ---- END ADD LOG ----
                        self.memory_cache['user_name'] = fallback_entries[0]['name']
                        print_status(f"👤 Restored user name from A-MEM (fallback): {self.memory_cache['user_name']}")
                    else:
                        print_status("👤 No user name found in A-MEM memories.")
    
                    if pref_entries:
                        self.memory_cache['preferences'] = {k: v['text'] for k, v in pref_entries.items()}
                        print_status(f"👍 Restored {len(self.memory_cache['preferences'])} preferences from A-MEM")
                    else:
                        print_status("👍 No preferences found in A-MEM memories.")

                    self.memory_cache['last_updated'] = datetime.now(timezone.utc)
                    print_status(f"✅ A-MEM cache populated. Name: '{self.memory_cache['user_name']}', Prefs: {len(self.memory_cache['preferences'])}")
                else: 
                    print_status(f"📚 No existing A-MEM memories found for user '{self.user_id}'.")
        except Exception as e:
            print_status(f"⚠️ Failed to load existing A-MEM memories: {e}")
            import traceback; traceback.print_exc()

    def extract_user_name(self, text: str) -> Optional[str]:
        return self._extract_name_from_memory_text(text)
    
    def is_name_correction(self, text: str) -> bool:
        patterns = [
            r"no,?\s+my name is", r"actually,?\s+my name is", r"it's\s+([\w\s]+)",
            r"not\s+[\w\s]+,?\s+it's\s+([\w\s]+)", r"no,?\s+it's\s+([\w\s]+)", r"no,?\s+i'm\s+([\w\s]+)"
        ]
        return any(re.search(p, text.lower()) for p in patterns)
    
    def should_store_memory(self, user_text: str, assistant_text: str) -> tuple[bool, str]:
        user_lower = user_text.lower().strip()
        if len(user_lower) == 0: return False, "empty_input"
        if len(user_lower) <= 3 and not user_lower.startswith("my name is"): return False, "too_short"
        
        common_transient = ['yes', 'no', 'ok', 'okay', 'thanks', 'thank you', 'um', 'uh', 'got it', 'good', 'fine', 'alright']
        if user_lower in common_transient: return False, "acknowledgment"
        if user_lower.startswith("and ") and len(user_lower.split()) < 4: return False, "minor_continuation"

        if any(p in user_lower for p in ['my name is', 'i am ', 'call me ']): return True, "personal_info"
        if any(p in user_lower for p in ['i live in', 'i work at', 'i was born in']): return True, "personal_info"
        if any(p in user_lower for p in ['i like', 'i love', 'i hate', 'my favorite', 'i prefer', 'i dislike']): return True, "preference"
        if any(p in user_lower for p in ['remember this', 'don\'t forget', 'important to know', 'make a note']): return True, "important"
        if any(p in user_lower for p in ['what do you remember', 'what do you know about me', 'tell me about yourself']): return False, "recall_request"
        if len(user_text.split()) > 7: return True, "conversation_turn"
        return False, "filtered_by_default"
    
    def update_local_cache(self, user_text: str, category: str, is_current_turn_extraction: bool = False):
        updated_cache = False
        if category == "personal_info":
            name = self.extract_user_name(user_text)
            if name:
                current_cached_name = self.memory_cache.get('user_name')
                if name != current_cached_name or self.is_name_correction(user_text) or not current_cached_name or is_current_turn_extraction:
                    self.memory_cache['user_name'] = name
                    updated_cache = True
        
        elif category == "preference":
            text_lower = user_text.lower()
            preference_text = None
            pref_patterns = [r"i like ([\w\s.,'-]+)", r"i love ([\w\s.,'-]+)", r"my favorite [\w\s]+ is ([\w\s.,'-]+)", r"i prefer ([\w\s.,'-]+)", r"i dislike ([\w\s.,'-]+)"]
            for pat_str in pref_patterns:
                match = re.search(pat_str, text_lower)
                if match:
                    captured_groups = [g for g in match.groups() if g is not None] 
                    if captured_groups:
                        preference_text = captured_groups[-1].strip().split('.')[0].split(',')[0] 

                    if preference_text and len(preference_text) > 2 and len(preference_text) < 150:
                        pref_key = hashlib.md5(preference_text.lower().encode()).hexdigest()[:10]
                        if self.memory_cache['preferences'].get(pref_key) != preference_text:
                            self.memory_cache['preferences'][pref_key] = preference_text
                            updated_cache = True
                        break 
        
        if updated_cache:
            self.memory_cache['last_updated'] = datetime.now(timezone.utc)

    async def add_to_memory_smart(self, user_text: str, assistant_text: str):
        should_store, category = self.should_store_memory(user_text, assistant_text)
        if not should_store:
            return None
        
        # Only update cache from user text, not assistant text
        self.update_local_cache(user_text, category, is_current_turn_extraction=False) 
        
        if self.memory_queue:
            # Store the conversation but mark that name extraction should only happen from user text
            await self.memory_queue.put(('add', user_text, assistant_text, category))
        else:
            print_status("⚠️ Memory queue not available, attempting direct threaded storage (fallback).")
            threading.Thread(target=self._store_memory_background, args=(user_text, assistant_text, category), daemon=True).start()
        
        self.memory_operations += 1
        return category



    def _store_memory_background(self, user_text: str, assistant_text: str, category: str):
        try:
            # Create conversation content for A-MEM
            conversation_content = f"User: {user_text}\nAssistant: {assistant_text}"
            
            # Add memory to A-MEM system with metadata
            memory_id = self.amem_system.add_note(
                content=conversation_content,
                tags=[category, "conversation"],
                category=category,
                timestamp=datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
            )
            
            print_status(f"✅ Stored memory in A-MEM: {memory_id}")
            
        except Exception as e:
            print_status(f"❌ Background A-MEM storage failed: {e}")
            import traceback; traceback.print_exc()
    
    def get_user_context(self) -> str: 
        parts = []
        if self.memory_cache.get('user_name'): 
            parts.append(f"The user's name is {self.memory_cache['user_name']}")
        if self.memory_cache.get('preferences'):
            prefs = list(self.memory_cache['preferences'].values())
            if prefs: 
                parts.append(f"You know that the user likes: {', '.join(prefs[:3])}")
        if not parts: 
            return "You don't have specific prior context about the user yet."
        return "Key things you remember about the user: " + ". ".join(parts) + "."
    
    def forget_all_memories(self):
        """Delete all memories from A-MEM and clear local cache."""
        try:
            # Clear A-MEM memories
            self.amem_system.memories.clear()
            # Reset ChromaDB
            self.amem_system.consolidate_memories()
            # Clear local cache
            self.memory_cache = {'user_name': None, 'preferences': {}, 'facts': {}, 'last_updated': None}
            print_status("✅ All A-MEM memories deleted for this user.")
            return True
        except Exception as e:
            print_status(f"❌ Failed to delete all A-MEM memories: {e}")
            return False
    
    async def search_memories_smart(self, query: str) -> str:
        query_lower = query.lower()
        if any(p in query_lower for p in ['what is my name', 'who am i']):
            self.cache_hits += 1
            if self.memory_cache.get('user_name'): 
                return f"The user's name is {self.memory_cache['user_name']}."

        try:
            loop = asyncio.get_event_loop()
            if self.memory_queue.qsize() > 0: 
                await asyncio.sleep(0.2) 

            memories = await loop.run_in_executor(self.executor, self._search_memories_sync, query)
            if memories:
                texts = [m.get('content', '') for m in memories[:3]]
                texts = [t for t in texts if t]
                if texts: 
                    return f"Regarding '{query}', I found these related memories: {'; '.join(t[:120] for t in texts)}..."
                else: 
                    return f"I found some entries for '{query}' but couldn't extract clear details from them."
            else:
                if any(p in query_lower for p in ['what is my name', 'who am i']):
                    return "I don't seem to have your name stored yet. What would you like me to call you?"
                return f"I don't have specific memories directly related to '{query}' at the moment."
        except Exception as e:
            print_status(f"⚠️ A-MEM search operation failed: {e}")
            return "I'm having some trouble accessing my long-term memories right now."
    
    def _search_memories_sync(self, query: str):
        try:
            logger.info(f"AMEM_SEARCH_DEBUG: Searching with query: '{query}' for user_id: '{self.user_id}'")
            
            # Use A-MEM search functionality
            results = self.amem_system.search_agentic(query, k=5)
            
            logger.info(f"AMEM_SEARCH_DEBUG: Found {len(results)} results")
            return results
            
        except Exception as e:
            print_status(f"❌ A-MEM search (sync call) failed: {e}")
            logger.error(f"❌ A-MEM search (sync call) failed for query '{query}': {e}", exc_info=True)
            return []
    
    def get_stats(self) -> dict:
        ops, hits = self.memory_operations, self.cache_hits
        last_upd_ts = self.memory_cache.get('last_updated')
        amem_memory_count = len(self.amem_system.memories)
        
        return {
            'mem_ops': ops, 
            'cache_hits': hits,
            'cache_eff': f"{(hits / max(ops, 1) * 100):.1f}%",
            'user_name_cache': self.memory_cache.get('user_name'),
            'prefs_cache_#': len(self.memory_cache.get('preferences', {})),
            'last_cache_upd': last_upd_ts.strftime("%H:%M:%S %Z") if last_upd_ts else "N/A",
            'mem_q_size': self.memory_queue.qsize() if self.memory_queue else -1,
            'amem_memories': amem_memory_count,
            'amem_evolution_ops': self.amem_system.evo_cnt
        }
    
    async def shutdown(self):
        print_status("Initiating A-MEM MemoryManager shutdown...")
        if self.background_task and not self.background_task.done():
            print_status("Cancelling background A-MEM processor task...")
            self.background_task.cancel()
            try: 
                await self.background_task
            except asyncio.CancelledError: 
                print_status("Background A-MEM processor task successfully cancelled.")
            except Exception as e: 
                print_status(f"Error during background task cancellation: {e}")
        
        if self.memory_queue and self.memory_queue.qsize() > 0:
            print_status(f"Waiting for {self.memory_queue.qsize()} items in A-MEM queue to be processed...")
            try:
                await asyncio.wait_for(self.memory_queue.join(), timeout=10.0) 
                print_status("A-MEM queue processing complete.")
            except asyncio.TimeoutError:
                print_status("⚠️ Timeout waiting for A-MEM queue to join. Some items might not be processed.")
        elif self.memory_queue: 
             await self.memory_queue.join() 

        print_status("Shutting down A-MEM executor...")
        self.executor.shutdown(wait=True)
        print_status("A-MEM MemoryManager shutdown complete.")


class SmartVoiceAssistant:
    def __init__(self):
        print_status("🧠 Initializing SmartVoiceAssistant with A-MEM...")
        os.environ["OPENAI_API_KEY"] = "dummy-key-for-local-use" 

        # Initialize Qdrant for A-MEM
        qclient = QdrantClient(host="localhost", port=6333)
        collections_response = qclient.get_collections()
        collection_names = [c.name for c in collections_response.collections]

        if "amem_voice_collection" in collection_names:
            print_status("✅ Qdrant collection 'amem_voice_collection' already exists.")
        else:
            print_status("Creating Qdrant collection 'amem_voice_collection'...")
            qclient.create_collection(
                collection_name="amem_voice_collection",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),  # all-MiniLM-L6-v2 size
            )
        
        self.user_id = "amem_voice_user_01" 
        self.session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        self.amem_memory = AMemMemoryManager(self.user_id)
        self.audio_processor = BluetoothAudioProcessor()
        self.conversation_buffer = deque(maxlen=100) 
        self.response_cache = {} 
        self.cache_ttl_seconds = 180 
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.turn_count = 0
        self.total_response_time = deque(maxlen=20)
        self.voice_detection_successes = 0
        print_status(f"👤 User ID: {self.user_id}, Session: {self.session_id}")
        if USE_OLLAMA_FOR_CONVERSATION:
            print_status(f"🗣️ Conversational LLM: Ollama ({OLLAMA_CONVERSATIONAL_MODEL} via {OLLAMA_URL})")
        else:
            print_status(f"🗣️ Conversational LLM: LM Studio ({LM_STUDIO_MODEL} via {LM_STUDIO_URL})")
        print_status(f"🧠 A-MEM System: {AMEM_LLM_MODEL} with {AMEM_EMBEDDER_MODEL} embeddings")
        
    async def initialize_async(self):
        print_status("Initializing async components for SmartVoiceAssistant...")
        connector = aiohttp.TCPConnector(limit_per_host=5, ssl=False) 
        timeout = aiohttp.ClientTimeout(total=20, connect=5, sock_read=15) 
        self.http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        print_status("✅ aiohttp ClientSession created.")
        await self.amem_memory.start_background_processor()
        print_status("✅ Async components initialized for SmartVoiceAssistant.")
    
    async def cleanup_async(self):
        print_status("🧹 Starting async cleanup for SmartVoiceAssistant...")
        if self.http_session: 
            await self.http_session.close()
            print_status("aiohttp ClientSession closed.")
        await self.amem_memory.shutdown() 
        print_status("🧹 Async cleanup completed for SmartVoiceAssistant.")
    
    def get_cached_response(self, text: str) -> Optional[str]:
        text_hash = hashlib.md5(text.lower().encode()).hexdigest()
        item = self.response_cache.get(text_hash)
        if item and (datetime.now(timezone.utc) - item['timestamp'] < timedelta(seconds=self.cache_ttl_seconds)):
            return item['response']
        return None
    
    def cache_response(self, text: str, response: str):
        text_hash = hashlib.md5(text.lower().encode()).hexdigest()
        self.response_cache[text_hash] = {'response': response, 'timestamp': datetime.now(timezone.utc)}
    
    def _get_llm_context_prompt(self) -> str:
        memory_context = self.amem_memory.get_user_context()
        recent_conv = ""
        if self.conversation_buffer:
            turns = []
            for turn in list(self.conversation_buffer)[-3:]: 
                turns.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")
            recent_conv = "\n---\nRecent Conversation:\n" + "\n".join(turns) if turns else ""
        
        system_prompt = f"""You are Echo, a helpful and friendly voice assistant with advanced memory capabilities.
        Keep responses concise and natural for voice interaction.

        {memory_context}

        Remember:
        - Your name is Echo (the assistant)
        - You have an advanced agentic memory system that learns and evolves
        - You can form connections between different memories and concepts
        - Ask users their name and save it so you can address them personally
        - Be warm and conversational
        {recent_conv}"""
        return system_prompt.strip()

    async def get_llm_response_smart(self, user_text: str) -> str:
        cached_llm_response = self.get_cached_response(user_text)
        if cached_llm_response: 
            return cached_llm_response
        
        # Handle memory recall requests
        recall_phrases = ['what do you remember', 'what do you know about me', 'tell me about myself', 'what is my name', 'who am i']
        if any(phrase in user_text.lower() for phrase in recall_phrases):
            memory_search_result = await self.amem_memory.search_memories_smart(user_text)
            self.cache_response(user_text, memory_search_result) 
            return memory_search_result

        # Handle memory reset requests
        if any(phrase in user_text.lower() for phrase in ["delete all your memory", "reset", "forget everything"]):
            result = self.amem_memory.forget_all_memories()
            if result:
                response = "I've erased all my memories and reset my knowledge network."
                self.cache_response(user_text, response)
                return response
            else:
                response = "Sorry, I couldn't erase my memories due to an internal error."
                self.cache_response(user_text, response)
                return response
            
        # Handle name extraction and immediate response
        potential_name_in_turn = self.amem_memory.extract_user_name(user_text)
        if potential_name_in_turn:
            self.amem_memory.update_local_cache(user_text, "personal_info", is_current_turn_extraction=True)
            if re.fullmatch(r"(my name is|i'?m|call me|i am)\s+" + re.escape(potential_name_in_turn) + r"\s*\.?", user_text.lower().strip(), re.IGNORECASE):
                ack = f"Got it, {potential_name_in_turn}! I'll remember that and my memory system will create connections with this information."
                await self.amem_memory.add_to_memory_smart(user_text, ack) 
                self.cache_response(user_text, ack)
                return ack
        
        system_prompt = self._get_llm_context_prompt()
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
        
        assistant_response_text = f"I heard you say: '{user_text}'. (Default if LLM fails)" 
        
        if not self.http_session:
            print_status("❌ aiohttp session not available for LLM call.")
            return "I'm having trouble connecting to my language abilities right now."

        try:
            if USE_OLLAMA_FOR_CONVERSATION:
                payload = {
                    "model": OLLAMA_CONVERSATIONAL_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 250
                    }
                }
                async with self.http_session.post(f"{OLLAMA_URL}/api/chat", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        assistant_response_text = data.get("message", {}).get("content", "").strip()
                        if not assistant_response_text: 
                            assistant_response_text = "I'm not sure how to respond to that."
                    else:
                        error_body = await response.text()
                        print_status(f"⚠️ Ollama LLM request failed: Status {response.status}, Body: {error_body[:200]}")
                        assistant_response_text = "I'm sorry, I encountered an issue trying to respond via Ollama."
            else: # Use LM Studio
                payload = {
                    "model": LM_STUDIO_MODEL, 
                    "messages": messages, 
                    "max_tokens": 250, 
                    "temperature": 0.7, 
                    "stream": False
                }
                async with self.http_session.post(f"{LM_STUDIO_URL}/chat/completions", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        assistant_response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                        if not assistant_response_text: 
                            assistant_response_text = "I'm not sure how to respond to that."
                    else:
                        error_body = await response.text()
                        print_status(f"⚠️ LM Studio LLM request failed: Status {response.status}, Body: {error_body[:200]}")
                        assistant_response_text = "I'm sorry, I encountered an issue trying to respond via LM Studio."
        
        except aiohttp.ClientConnectorError as e:
            url_used = OLLAMA_URL if USE_OLLAMA_FOR_CONVERSATION else LM_STUDIO_URL
            print_status(f"❌ LLM Connection Error: {e}. Is the server running at {url_used}?")
            assistant_response_text = "I'm unable to connect to my language processing unit. Please check the connection."
        except asyncio.TimeoutError:
            url_used = OLLAMA_URL if USE_OLLAMA_FOR_CONVERSATION else LM_STUDIO_URL
            print_status(f"❌ LLM Request Timed Out to {url_used}.")
            assistant_response_text = "It's taking me a bit longer than usual to think. Could you try that again in a moment?"
        except Exception as e:
            print_status(f"❌ Unexpected LLM request error: {e}")
            import traceback; traceback.print_exc()
            assistant_response_text = "I've run into a little hiccup. Let's try that again."

        # Store in A-MEM system
        await self.amem_memory.add_to_memory_smart(user_text, assistant_response_text)
        self.cache_response(user_text, assistant_response_text)
        return assistant_response_text

    def process_audio_array(self, audio_data):
        return self.audio_processor.preprocess_bluetooth_audio(audio_data)

main_event_loop: Optional[asyncio.AbstractEventLoop] = None
assistant_instance: Optional[SmartVoiceAssistant] = None
async_worker_thread: Optional[threading.Thread] = None

def setup_async_environment():
    global main_event_loop, assistant_instance, async_worker_thread
    assistant_instance = SmartVoiceAssistant() 

    def run_async_loop_in_thread():
        global main_event_loop, assistant_instance
        main_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(main_event_loop)
        if assistant_instance: 
            main_event_loop.run_until_complete(assistant_instance.initialize_async())
        else: 
            print_status("🚨 Assistant instance is None in async thread. Cannot initialize.")
            return
        
        try: 
            main_event_loop.run_forever()
        except KeyboardInterrupt: 
            print_status("Async loop interrupted in thread.")
        finally:
            if assistant_instance and main_event_loop and not main_event_loop.is_closed(): 
                print_status("Cleaning up assistant resources in async thread...")
                main_event_loop.run_until_complete(assistant_instance.cleanup_async())
            if main_event_loop and not main_event_loop.is_closed(): 
                 main_event_loop.close()
            print_status("Async event loop closed.")

    async_worker_thread = threading.Thread(target=run_async_loop_in_thread, daemon=True, name="AsyncWorkerThread")
    async_worker_thread.start()
    
    for _ in range(100): 
        if main_event_loop and main_event_loop.is_running() and \
           assistant_instance and assistant_instance.http_session and \
           assistant_instance.amem_memory and assistant_instance.amem_memory.background_task and \
           not assistant_instance.amem_memory.background_task.done():
            print_status("✅ Async environment and A-MEM components are ready.")
            return
        time.sleep(0.1)
    print_status("⚠️ Async environment or A-MEM components did not confirm readiness in time.")

def run_coro_from_sync_thread(coro) -> any:
    global main_event_loop
    if main_event_loop and main_event_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, main_event_loop)
        try: 
            return future.result(timeout=25) 
        except TimeoutError:
            print_status(f"❌ Async task timed out after 25s: {coro}")
            return "I'm sorry, that took a bit too long to process."
        except Exception as e:
            print_status(f"❌ Error running async task '{coro}': {e}")
            import traceback; traceback.print_exc()
            return "There was an internal error processing your request."
    else:
        print_status("❌ Async event loop is not available. Cannot run async task.")
        return "My async processing unit is not ready. Please try again."

def smart_voice_assistant_callback_rt(audio_data_tuple: tuple): 
    global assistant_instance, stt_model, tts_model 

    if not assistant_instance:
        print_status("🚨 Assistant instance is NOT available in callback!")
        yield EMPTY_AUDIO_YIELD_OUTPUT 
        return

    try:
        sample_rate, audio_array = assistant_instance.process_audio_array(audio_data_tuple)
        
        if audio_array.size == 0 and audio_data_tuple[1].size > 0 : 
             yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs() 
             return

        if assistant_instance.turn_count < 5 and \
           assistant_instance.audio_processor.calibration_frames < assistant_instance.audio_processor.min_calibration_frames:
            audio_stats_calib = assistant_instance.audio_processor.get_detection_stats()
            if audio_stats_calib and not audio_stats_calib['calibrated']:
                pass 

        threshold = assistant_instance.audio_processor.noise_floor * 15

        user_text = ""
        if audio_array.size > 0: 
            user_text = stt_model.stt((sample_rate, audio_array))
        
        if not user_text or not user_text.strip():
            yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs() 
            return

        assistant_instance.voice_detection_successes += 1
        print_status(f"👤 User: {user_text}")
        
        start_turn_time = time.monotonic()
        assistant_response_text = run_coro_from_sync_thread(
            assistant_instance.get_llm_response_smart(user_text)
        )
        turn_processing_time = time.monotonic() - start_turn_time
        
        if not assistant_response_text: 
            assistant_response_text = "I'm having trouble formulating a response right now."
        
        print_status(f"🤖 Assistant: {assistant_response_text}")
        print_status(f"⏱️ Turn Processing Time: {turn_processing_time:.2f}s")
        
        assistant_instance.conversation_buffer.append({
            'user': user_text, 'assistant': assistant_response_text, 
            'timestamp': datetime.now(timezone.utc)
        })
        assistant_instance.turn_count += 1
        assistant_instance.total_response_time.append(turn_processing_time)
        
        if assistant_instance.turn_count % 3 == 0: 
            avg_resp = np.mean(list(assistant_instance.total_response_time)) if assistant_instance.total_response_time else 0
            mem_s = assistant_instance.amem_memory.get_stats()
            audio_s = assistant_instance.audio_processor.get_detection_stats()
            audio_q_str = f" | AudioRMS: {audio_s['avg_rms']:.3f}" if audio_s and audio_s['calibrated'] else "(Audio uncalibrated)"
            print_status(f"📊 A-MEM Stats (Turn {assistant_instance.turn_count}): AvgResp={avg_resp:.2f}s | MemOps={mem_s['mem_ops']}(Q:{mem_s['mem_q_size']}) | A-MEM_Memories={mem_s['amem_memories']} | Evolutions={mem_s['amem_evolution_ops']} | CacheEff={mem_s['cache_eff']} | User='{mem_s['user_name_cache']}'{audio_q_str}")
        
        additional_outputs = AdditionalOutputs()

        tts_voices_to_try = [KOKORO_PREFERRED_VOICE, KOKORO_FALLBACK_VOICE_1, KOKORO_FALLBACK_VOICE_2, None] 
        tts_success = False
        for voice_id in tts_voices_to_try:
            try:
                options_params = {"speed": 1.05, "lang": "en-us"}
                if voice_id: options_params["voice"] = voice_id

                tts_options = KokoroTTSOptions(**options_params)

                chunk_count = 0
                total_samples = 0

                for tts_output_item in tts_model.stream_tts_sync(assistant_response_text, tts_options):
                    if isinstance(tts_output_item, tuple) and len(tts_output_item) == 2 and isinstance(tts_output_item[1], np.ndarray):
                        current_sr, current_chunk_array = tts_output_item
                        if current_chunk_array.size > 0:
                            chunk_count += 1
                            total_samples += current_chunk_array.size
                            logger.debug(f"TTS yielding chunk {chunk_count}: SR={current_sr}, Samples={current_chunk_array.size}, Total Samples={total_samples}")
                            yield (current_sr, current_chunk_array), additional_outputs
                    elif isinstance(tts_output_item, np.ndarray):
                        if tts_output_item.size > 0:
                            chunk_count += 1
                            total_samples += tts_output_item.size
                            logger.debug(f"TTS yielding chunk {chunk_count} (ndarray): Samples={tts_output_item.size}, Total Samples={total_samples}")
                            yield (sample_rate, tts_output_item), additional_outputs
                    else:
                        logger.debug(f"TTS yielded unexpected item type: {type(tts_output_item)}")
                        continue

                tts_success = True
                logger.info(f"TTS stream for voice '{voice_id}' completed. Total chunks: {chunk_count}, Total samples: {total_samples}")
                if voice_id != KOKORO_PREFERRED_VOICE and voice_id is not None:
                    print_status(f"TTS successful with fallback voice: {voice_id}")
                break
            except AssertionError as e:
                print_status(f"⚠️ Kokoro TTS voice '{voice_id}' not found: {e}. Trying next fallback.")
            except Exception as e:
                print_status(f"❌ Kokoro TTS error with voice '{voice_id}': {e}. Traceback below:")
                import traceback
                traceback.print_exc()
                break
        
        if not tts_success:
            print_status(f"❌ All TTS attempts failed for: '{assistant_response_text[:50]}...'")
            yield SILENT_AUDIO_FRAME_TUPLE, additional_outputs 
            
    except Exception as e:
        print_status(f"❌ UNHANDLED CRITICAL Error in smart_voice_assistant_callback: {e}")
        import traceback; traceback.print_exc()
        try: 
            error_msg = "I've encountered a system error. Please try again later."
            tts_options = KokoroTTSOptions(speed=1.0, lang="en-us") 
            for tts_err_chunk in tts_model.stream_tts_sync(error_msg, tts_options):
                if isinstance(tts_err_chunk, tuple) and len(tts_err_chunk) == 2 and isinstance(tts_err_chunk[1], np.ndarray):
                     sr_err, arr_err = tts_err_chunk
                     if arr_err.size > 0: yield (sr_err, arr_err), AdditionalOutputs()
                elif isinstance(tts_err_chunk, np.ndarray) and tts_err_chunk.size > 0:
                    yield (AUDIO_SAMPLE_RATE, tts_err_chunk), AdditionalOutputs()
        except Exception as tts_err_final:
            print_status(f"❌ Failed to TTS critical error message: {tts_err_final}")
            yield EMPTY_AUDIO_YIELD_OUTPUT

if __name__ == "__main__":
    setup_async_environment() 

    # Calculate threshold before creating Stream
    if not assistant_instance or not assistant_instance.audio_processor.noise_floor:
        threshold = 0.15  # Default fallback threshold
    else:
        threshold = assistant_instance.audio_processor.noise_floor * 15

    print_status("🌐 Creating FastRTC stream with A-MEM + Bluetooth optimization...")
    try:
        stream = Stream(
            ReplyOnPause(
                smart_voice_assistant_callback_rt, 
                can_interrupt=True,
                algo_options=AlgoOptions(
                    audio_chunk_duration=2.0,  # Reduced from 3.5
                    started_talking_threshold=0.15,  # Reduced from 0.2
                    speech_threshold=threshold
                ),
                model_options=SileroVadOptions(
                    threshold=0.3,  # Reduced from 0.5
                    min_speech_duration_ms=250,  # Reduced from 300
                    min_silence_duration_ms=2000,  # Reduced from 3000
                    speech_pad_ms=200,  # Reduced from 300
                    window_size_samples=512  # Reduced from 1024
                )
            ), 
            modality="audio", mode="send-receive",
            track_constraints={ 
                "echoCancellation": True, 
                "noiseSuppression": True, 
                "autoGainControl": True,
                "sampleRate": {"ideal": 16000}, 
                "sampleSize": {"ideal": 16},
                "channelCount": {"exact": 1}, 
                "latency": {"ideal": 0.01},  # Reduced from 0.05
            }
        )
        print("=" * 70)
        print("🧠 A-MEM SMART MEMORY + BLUETOOTH Voice Assistant Ready!")
        print("="*70)
        print("💡 Test Commands:")
        print("   • 'My name is [Your Name]'")
        print("   • 'No, my name is [Corrected Name]'")
        print("   • 'What is my name?' / 'Who am I?'")
        print("   • 'I like [something interesting]'")
        print("   • 'What do you remember about me?'")
        print("   • Ask a general question.")
        print("   • Notice how memories evolve and connect!")
        print("\n🛑 To stop: Press Ctrl+C in the terminal")
        print("=" * 70)

        stream.ui.launch(server_name="0.0.0.0", server_port=7860, quiet=False, share=False)

    except KeyboardInterrupt:
        print_status("🛑 KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        print_status(f"❌ Launch error or unhandled exception in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print_status("🏁 Main thread initiating shutdown sequence...")
        if main_event_loop and not main_event_loop.is_closed():
            print_status("Requesting async event loop to stop...")
            main_event_loop.call_soon_threadsafe(main_event_loop.stop) 
        
        if async_worker_thread and async_worker_thread.is_alive():
            print_status("Waiting for async worker thread to join...")
            async_worker_thread.join(timeout=15) 
            if async_worker_thread.is_alive():
                print_status("⚠️ Async worker thread did not join in time.")
        
        print_status("👋 A-MEM Voice assistant shutdown process complete.")
        sys.exit(0)