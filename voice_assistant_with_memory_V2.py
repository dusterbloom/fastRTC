#!/usr/bin/env python3
"""
FastRTC Voice Assistant with Smart Memory + Bluetooth Optimization
Combines intelligent memory management with Nothing Ear Buds optimization
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
from mem0 import Memory 
import logging
from collections import deque
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mem0.llms.ollama").setLevel(logging.WARNING) 
logging.getLogger("mem0.memory.main").setLevel(logging.INFO) 
logging.getLogger("mem0.embeddings.ollama").setLevel(logging.WARNING)
logging.getLogger("mem0.vector_stores.chroma").setLevel(logging.WARNING)
logging.getLogger("aioice.ice").setLevel(logging.WARNING) 
logging.getLogger("aiortc").setLevel(logging.WARNING) 
logging.getLogger("phonemizer").setLevel(logging.WARNING) 

logger = logging.getLogger(__name__) 

print("🧠 FastRTC Voice Assistant - Smart Memory + Bluetooth Optimization")
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

# Configuration
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://192.168.1.5:1234/v1") 
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "mistral-nemo-instruct-2407")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") 
OLLAMA_MEM0_LLM_MODEL = os.getenv("OLLAMA_MEM0_LLM_MODEL", "mistral:7b") 
OLLAMA_EMBEDDER_MODEL = os.getenv("OLLAMA_EMBEDDER_MODEL", "nomic-embed-text")

KOKORO_PREFERRED_VOICE = "af_heart" 
KOKORO_FALLBACK_VOICE_1 = "af_alloy"  
KOKORO_FALLBACK_VOICE_2 = "af_bella" 

AUDIO_SAMPLE_RATE = 16000 
MINIMAL_SILENT_FRAME_DURATION_MS = 20 
MINIMAL_SILENT_SAMPLES = int(AUDIO_SAMPLE_RATE * (MINIMAL_SILENT_FRAME_DURATION_MS / 1000.0))
SILENT_AUDIO_CHUNK_ARRAY = np.zeros(MINIMAL_SILENT_SAMPLES, dtype=np.float32)
SILENT_AUDIO_FRAME_TUPLE = (AUDIO_SAMPLE_RATE, SILENT_AUDIO_CHUNK_ARRAY)
EMPTY_AUDIO_YIELD_OUTPUT = (SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs())


class BluetoothAudioProcessor: # No changes
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

class SmartMemoryManager: # No changes
    def __init__(self, memory_instance, user_id):
        self.memory = memory_instance
        self.user_id = user_id
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="smart_mem_exec")
        self.memory_cache = {'user_name': None, 'preferences': {}, 'facts': {}, 'last_updated': None}
        self.memory_queue = asyncio.Queue()
        self.background_task: Optional[asyncio.Task] = None
        self.memory_operations = 0
        self.cache_hits = 0
        self._load_existing_memories() 
        print_status("🧠 Smart memory manager initialized")
    
    async def start_background_processor(self):
        if self.memory_queue and (self.background_task is None or self.background_task.done()):
            self.background_task = asyncio.create_task(self._process_memory_queue())
            print_status("🚀 Background memory processor task started.")
    
    async def _process_memory_queue(self):
        print_status("Background memory processor listening to queue...")
        while True:
            try:
                op, user_text, assistant_text, category = await self.memory_queue.get()
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor, self._store_memory_background,
                    user_text, assistant_text, category
                )
                self.memory_queue.task_done()
            except asyncio.CancelledError:
                print_status("Background memory processor task was cancelled.")
                break
            except Exception as e:
                print_status(f"❌ Unhandled error in memory queue processing: {e}")
                import traceback; traceback.print_exc()
                await asyncio.sleep(5) 
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> datetime:
        if not timestamp_str: return datetime.min.replace(tzinfo=timezone.utc)
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            for fmt in ('%Y-%m-%d %H:%M:%S.%f%z', '%Y-%m-%d %H:%M:%S%z', 
                        '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                try:
                    dt = datetime.strptime(timestamp_str, fmt)
                    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc) 
                    return dt
                except ValueError: continue
            return datetime.min.replace(tzinfo=timezone.utc)

    def _extract_name_from_memory_text(self, memory_text: str) -> Optional[str]:
        text_lower = memory_text.lower()
        patterns = [
            r"my name is ([\w\s]+)", r"i'm ([\w\s]+)", r"call me ([\w\s]+)",
            r"i am ([\w\s]+)", r"name is ([\w\s]+)"
        ]
        for pattern_str in patterns:
            match = re.search(pattern_str, text_lower)
            if match:
                name = match.group(1).strip().split('.')[0].split(',')[0]
                name = re.sub(r'[^\w\s-]', '', name) 
                name = ' '.join(name.split())
                invalid_names = ['not', 'no', 'actually', 'really', 'from', 'and', 'the', 'a', 'an', 'you', 'me', 'is', 'assistant']
                if len(name) > 1 and len(name) < 50 and name.lower() not in invalid_names:
                    return name.title()
        return None

    def _load_existing_memories(self):
        try:
            result = self.memory.get_all(user_id=self.user_id)
            existing_memories = result.get("results", []) if isinstance(result, dict) else (result if isinstance(result, list) else [])
            
            if existing_memories:
                print_status(f"📚 Loading {len(existing_memories)} existing memories for user '{self.user_id}'...")
                name_entries = []
                pref_entries = {} 

                for item in existing_memories:
                    memory_text = ""
                    timestamp_str = None
                    metadata = {}

                    if isinstance(item, dict):
                        memory_text = item.get('memory') or item.get('content') or item.get('text', "")
                        metadata = item.get('metadata', {})
                        if isinstance(metadata, dict): timestamp_str = metadata.get('timestamp')
                        if not timestamp_str: timestamp_str = item.get('created_at') or item.get('timestamp')
                    elif isinstance(item, str): memory_text = item
                    
                    item_timestamp = self._parse_timestamp(timestamp_str)

                    potential_name = self._extract_name_from_memory_text(memory_text)
                    if potential_name:
                        name_entries.append({'name': potential_name, 'timestamp': item_timestamp})
                    
                    if isinstance(metadata, dict) and metadata.get("category") == "preference":
                        pref_match = re.search(r"i like ([\w\s.,'-]+)|i love ([\w\s.,'-]+)|my favorite [\w\s]+ is ([\w\s.,'-]+)", memory_text, re.IGNORECASE)
                        if pref_match:
                            preference_text = (pref_match.group(1) or pref_match.group(2) or pref_match.group(3) or "").strip()
                            if preference_text and len(preference_text) > 2 and len(preference_text) < 150:
                                pref_key = hashlib.md5(preference_text.lower().encode()).hexdigest()[:10]
                                if pref_key not in pref_entries or item_timestamp > pref_entries[pref_key]['timestamp']:
                                    pref_entries[pref_key] = {'text': preference_text, 'timestamp': item_timestamp}
                
                if name_entries:
                    name_entries.sort(key=lambda x: x['timestamp'], reverse=True)
                    self.memory_cache['user_name'] = name_entries[0]['name']
                    print_status(f"👤 Restored user name from DB (most recent): {self.memory_cache['user_name']}")
                else: print_status("👤 No user name found in DB memories.")

                if pref_entries:
                    self.memory_cache['preferences'] = {k: v['text'] for k, v in pref_entries.items()}
                    print_status(f"👍 Restored {len(self.memory_cache['preferences'])} preferences from DB: {', '.join(list(self.memory_cache['preferences'].values())[:3])}...")
                else: print_status("👍 No preferences found in DB memories.")
                
                self.memory_cache['last_updated'] = datetime.now(timezone.utc)
                print_status(f"✅ Memory cache populated. Name: '{self.memory_cache['user_name']}', Prefs: {len(self.memory_cache['preferences'])}")
            else: print_status(f"📚 No existing memories found in DB for user '{self.user_id}'.")
        except Exception as e:
            print_status(f"⚠️ Failed to load existing memories from DB: {e}")
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
        
        self.update_local_cache(user_text, category, is_current_turn_extraction=False) 
        
        if self.memory_queue:
            await self.memory_queue.put(('add', user_text, assistant_text, category))
        else:
            print_status("⚠️ Memory queue not available, attempting direct threaded storage (fallback).")
            threading.Thread(target=self._store_memory_background, args=(user_text, assistant_text, category), daemon=True).start()
        
        self.memory_operations += 1
        return category
    
    def _store_memory_background(self, user_text: str, assistant_text: str, category: str):
        try:
            messages = [{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_text}]
            metadata = {"category": category, "timestamp": datetime.now(timezone.utc).isoformat()}
            if category == "personal_info":
                extracted_name = self.extract_user_name(user_text)
                if extracted_name: metadata["name_stated"] = extracted_name
            elif category == "preference":
                 pref_match = re.search(r"i like ([\w\s.,'-]+)|i love ([\w\s.,'-]+)|my favorite [\w\s]+ is ([\w\s.,'-]+)", user_text, re.IGNORECASE)
                 if pref_match: metadata["preference_item"] = (pref_match.group(1) or pref_match.group(2) or pref_match.group(3) or "").strip()
            
            self.memory.add(messages=messages, user_id=self.user_id, metadata=metadata)
        except Exception as e:
            print_status(f"❌ Background memory storage to Mem0 failed: {e}")
            import traceback; traceback.print_exc()
    
    def get_user_context(self) -> str: 
        parts = []
        if self.memory_cache.get('user_name'): parts.append(f"Your name is {self.memory_cache['user_name']}") 
        if self.memory_cache.get('preferences'):
            prefs = list(self.memory_cache['preferences'].values())
            if prefs: parts.append(f"You know that the user likes: {', '.join(prefs[:3])}")
        if not parts: return "You don't have specific prior context about the user yet."
        return "Key things you remember about the user: " + ". ".join(parts) + "."
    
    async def search_memories_smart(self, query: str) -> str:
        query_lower = query.lower()
        if any(p in query_lower for p in ['what is my name', 'who am i']):
            self.cache_hits += 1
            if self.memory_cache.get('user_name'): return f"Your name is {self.memory_cache['user_name']}."

        try:
            loop = asyncio.get_event_loop()
            if self.memory_queue.qsize() > 0: 
                await asyncio.sleep(0.2) 

            memories = await loop.run_in_executor(self.executor, self._search_memories_sync, query)
            if memories:
                texts = [m.get('memory') or m.get('content') or m.get('text', '') for m in memories[:3]]
                texts = [t for t in texts if t]
                if texts: return f"Regarding '{query}', I found these related memories: {'; '.join(t[:120] for t in texts)}..."
                else: return f"I found some entries for '{query}' but couldn't extract clear details from them."
            else:
                if any(p in query_lower for p in ['what is my name', 'who am i']):
                    return "I don't seem to have your name stored yet. What would you like me to call you?"
                return f"I don't have specific memories directly related to '{query}' at the moment."
        except Exception as e:
            print_status(f"⚠️ Memory search operation failed: {e}"); return "I'm having some trouble accessing my long-term memories right now."
    
    def _search_memories_sync(self, query: str):
        try:
            result = self.memory.search(query=query, user_id=self.user_id, limit=3)
            return result.get("results", []) if isinstance(result, dict) else []
        except Exception as e: print_status(f"❌ Mem0 search (sync call) failed: {e}"); return []
    
    def get_stats(self) -> dict:
        ops, hits = self.memory_operations, self.cache_hits
        last_upd_ts = self.memory_cache.get('last_updated')
        return {
            'mem_ops': ops, 'cache_hits': hits,
            'cache_eff': f"{(hits / max(ops, 1) * 100):.1f}%",
            'user_name_cache': self.memory_cache.get('user_name'),
            'prefs_cache_#': len(self.memory_cache.get('preferences', {})),
            'last_cache_upd': last_upd_ts.strftime("%H:%M:%S %Z") if last_upd_ts else "N/A",
            'mem_q_size': self.memory_queue.qsize() if self.memory_queue else -1
        }
    async def shutdown(self):
        print_status("Initiating SmartMemoryManager shutdown...")
        if self.background_task and not self.background_task.done():
            print_status("Cancelling background memory processor task...")
            self.background_task.cancel()
            try: await self.background_task
            except asyncio.CancelledError: print_status("Background memory processor task successfully cancelled.")
            except Exception as e: print_status(f"Error during background task cancellation: {e}")
        
        if self.memory_queue and self.memory_queue.qsize() > 0:
            print_status(f"Waiting for {self.memory_queue.qsize()} items in memory queue to be processed...")
            try:
                await asyncio.wait_for(self.memory_queue.join(), timeout=10.0) 
                print_status("Memory queue processing complete.")
            except asyncio.TimeoutError:
                print_status("⚠️ Timeout waiting for memory queue to join. Some items might not be processed.")
        elif self.memory_queue: 
             await self.memory_queue.join() 

        print_status("Shutting down executor...")
        self.executor.shutdown(wait=True)
        print_status("SmartMemoryManager shutdown complete.")

class SmartVoiceAssistant: # No changes
    def __init__(self):
        print_status("🧠 Initializing SmartVoiceAssistant...")
        os.environ["OPENAI_API_KEY"] = "dummy-key-for-local-use" 
        
        self.mem0_config = {
            "llm": { 
                "provider": "ollama", "config": {
                    "model": OLLAMA_MEM0_LLM_MODEL, "ollama_base_url": OLLAMA_URL,
                    "temperature": 0.5, "max_tokens": 2048, "top_p": 0.9,
                }
            },
            "embedder": {"provider": "ollama", "config": {"model": OLLAMA_EMBEDDER_MODEL, "ollama_base_url": OLLAMA_URL}},
            "vector_store": {"provider": "chroma", "config": {"collection_name": "voice_assistant_memories_v4", "path": "./mem0_voice_db_v4"}},
            "history_db_path": "./history_v4.db", "version": "v1.3",
            "custom_fact_extraction_prompt": None, 
            "custom_update_memory_prompt": None,
        }
        try:
            self.memory = Memory.from_config(self.mem0_config)
            print_status(f"✅ Mem0 initialized (LLM: {OLLAMA_MEM0_LLM_MODEL}, Embedder: {OLLAMA_EMBEDDER_MODEL})")
        except Exception as e: 
            print_status(f"❌ Mem0 initialization failed: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)
        
        self.user_id = "smart_voice_user_01" 
        self.session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        self.smart_memory = SmartMemoryManager(self.memory, self.user_id)
        self.audio_processor = BluetoothAudioProcessor()
        self.conversation_buffer = deque(maxlen=6) 
        self.response_cache = {} 
        self.cache_ttl_seconds = 180 
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.turn_count = 0
        self.total_response_time = deque(maxlen=20)
        self.voice_detection_successes = 0
        print_status(f"👤 User ID: {self.user_id}, Session: {self.session_id}")
        print_status(f"🗣️ Conversational LLM: LM Studio ({LM_STUDIO_MODEL} via {LM_STUDIO_URL})")
        
    async def initialize_async(self):
        print_status("Initializing async components for SmartVoiceAssistant...")
        connector = aiohttp.TCPConnector(limit_per_host=5, ssl=False) 
        timeout = aiohttp.ClientTimeout(total=20, connect=5, sock_read=15) 
        self.http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        print_status("✅ aiohttp ClientSession created.")
        await self.smart_memory.start_background_processor()
        print_status("✅ Async components initialized for SmartVoiceAssistant.")
    
    async def cleanup_async(self):
        print_status("🧹 Starting async cleanup for SmartVoiceAssistant...")
        if self.http_session: 
            await self.http_session.close()
            print_status("aiohttp ClientSession closed.")
        await self.smart_memory.shutdown() 
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
        memory_context = self.smart_memory.get_user_context()
        recent_conv = ""
        if self.conversation_buffer:
            turns = []
            for turn in list(self.conversation_buffer)[-3:]: 
                turns.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")
            recent_conv = "\n---\nRecent Conversation:\n" + "\n".join(turns) if turns else ""
        full_context = f"{memory_context}{recent_conv}"
        system_prompt = f"""You are a helpful, friendly, and conversational voice assistant. 
Your goal is to assist the user naturally and efficiently. 
Use a warm and engaging tone. Keep responses concise for voice interaction.
{full_context}"""
        return system_prompt.strip()

    async def get_llm_response_smart(self, user_text: str) -> str:
        cached_llm_response = self.get_cached_response(user_text)
        if cached_llm_response: return cached_llm_response
        
        recall_phrases = ['what do you remember', 'what do you know about me', 'tell me about myself', 'what is my name', 'who am i']
        if any(phrase in user_text.lower() for phrase in recall_phrases):
            memory_search_result = await self.smart_memory.search_memories_smart(user_text)
            self.cache_response(user_text, memory_search_result) 
            return memory_search_result

        potential_name_in_turn = self.smart_memory.extract_user_name(user_text)
        if potential_name_in_turn:
            self.smart_memory.update_local_cache(user_text, "personal_info", is_current_turn_extraction=True)
            if re.fullmatch(r"(my name is|i'?m|call me|i am)\s+" + re.escape(potential_name_in_turn) + r"\s*\.?", user_text.lower().strip(), re.IGNORECASE):
                ack = f"Got it, {potential_name_in_turn}! I'll remember that."
                await self.smart_memory.add_to_memory_smart(user_text, ack) 
                self.cache_response(user_text, ack)
                return ack
        
        system_prompt = self._get_llm_context_prompt()
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
        
        assistant_response_text = f"I heard you say: '{user_text}'. (Default if LLM fails)" 
        
        if not self.http_session:
            print_status("❌ aiohttp session not available for LLM call.")
            return "I'm having trouble connecting to my language abilities right now."

        try:
            async with self.http_session.post(
                f"{LM_STUDIO_URL}/chat/completions",
                json={"model": LM_STUDIO_MODEL, "messages": messages, "max_tokens": 150, "temperature": 0.7, "stream": False}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    assistant_response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if not assistant_response_text: assistant_response_text = "I'm not sure how to respond to that."
                else:
                    error_body = await response.text()
                    print_status(f"⚠️ LLM request failed: Status {response.status}, Body: {error_body[:200]}")
                    assistant_response_text = "I'm sorry, I encountered an issue trying to respond."
        except aiohttp.ClientConnectorError as e:
            print_status(f"❌ LLM Connection Error: {e}. Is LM Studio running at {LM_STUDIO_URL}?")
            assistant_response_text = "I'm unable to connect to my language processing unit. Please check the connection."
        except asyncio.TimeoutError:
            print_status(f"❌ LLM Request Timed Out to {LM_STUDIO_URL}.")
            assistant_response_text = "It's taking me a bit longer than usual to think. Could you try again in a moment?"
        except Exception as e:
            print_status(f"❌ Unexpected LLM request error: {e}")
            import traceback; traceback.print_exc()
            assistant_response_text = "I've run into a little hiccup. Let's try that again."

        await self.smart_memory.add_to_memory_smart(user_text, assistant_response_text)
        self.cache_response(user_text, assistant_response_text)
        return assistant_response_text

    def process_audio_array(self, audio_data):
        return self.audio_processor.preprocess_bluetooth_audio(audio_data)

main_event_loop: Optional[asyncio.AbstractEventLoop] = None
assistant_instance: Optional[SmartVoiceAssistant] = None
async_worker_thread: Optional[threading.Thread] = None

def setup_async_environment(): # No changes
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
        
        try: main_event_loop.run_forever()
        except KeyboardInterrupt: print_status("Async loop interrupted in thread.")
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
           assistant_instance.smart_memory and assistant_instance.smart_memory.background_task and \
           not assistant_instance.smart_memory.background_task.done():
            print_status("✅ Async environment and SVA components are ready.")
            return
        time.sleep(0.1)
    print_status("⚠️ Async environment or SVA components did not confirm readiness in time.")

def run_coro_from_sync_thread(coro) -> any: # No changes
    global main_event_loop
    if main_event_loop and main_event_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, main_event_loop)
        try: return future.result(timeout=25) 
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
            mem_s = assistant_instance.smart_memory.get_stats()
            audio_s = assistant_instance.audio_processor.get_detection_stats()
            audio_q_str = f" | AudioRMS: {audio_s['avg_rms']:.3f}" if audio_s and audio_s['calibrated'] else "(Audio uncalibrated)"
            print_status(f"📊 Stats (Turn {assistant_instance.turn_count}): AvgResp={avg_resp:.2f}s | MemOps={mem_s['mem_ops']}(Q:{mem_s['mem_q_size']}) CacheEff={mem_s['cache_eff']} User='{mem_s['user_name_cache']}'{audio_q_str}")
        
        additional_outputs = AdditionalOutputs()
        # Example: Populate additional_outputs if needed for UI
        # additional_outputs.transcript = f"User: {user_text}\nAssistant: {assistant_response_text}"
        # additional_outputs.llm_response = assistant_response_text

        tts_voices_to_try = [KOKORO_PREFERRED_VOICE, KOKORO_FALLBACK_VOICE_1, KOKORO_FALLBACK_VOICE_2, None] 
        tts_success = False
        for voice_id in tts_voices_to_try:
            try:
                options_params = {"speed": 1.05, "lang": "en-us"}
                if voice_id: options_params["voice"] = voice_id
                
                tts_options = KokoroTTSOptions(**options_params)
                
                for tts_output in tts_model.stream_tts_sync(assistant_response_text, tts_options):
                    # Check if tts_output is already (sample_rate, audio_chunk_array)
                    if isinstance(tts_output, tuple) and len(tts_output) == 2 and isinstance(tts_output[1], np.ndarray):
                        sr, chunk_array = tts_output
                        if chunk_array.size > 0:
                             yield (sr, chunk_array), additional_outputs
                    elif isinstance(tts_output, np.ndarray): # If it's just the numpy array
                        if tts_output.size > 0:
                            yield (sample_rate, tts_output), additional_outputs # Assume sample_rate from STT/preprocessing
                    else:
                        print_status(f"⚠️ TTS stream yielded unexpected type: {type(tts_output)}. Skipping this chunk.")
                        continue # Skip this problematic chunk
                tts_success = True
                if voice_id != KOKORO_PREFERRED_VOICE and voice_id is not None: 
                    print_status(f"TTS successful with fallback voice: {voice_id}")
                break 
            except AssertionError as e: 
                print_status(f"⚠️ Kokoro TTS voice '{voice_id}' not found: {e}. Trying next fallback.")
            except Exception as e: 
                print_status(f"❌ Kokoro TTS error with voice '{voice_id}': {e}")
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
            for chunk_array in tts_model.stream_tts_sync(error_msg, tts_options):
                # Adapt to what tts_model.stream_tts_sync yields for error message
                if isinstance(chunk_array, tuple) and len(chunk_array) == 2 and isinstance(chunk_array[1], np.ndarray):
                     sr_err, arr_err = chunk_array
                     if arr_err.size > 0: yield (sr_err, arr_err), AdditionalOutputs()
                elif isinstance(chunk_array, np.ndarray) and chunk_array.size > 0:
                    yield (AUDIO_SAMPLE_RATE, chunk_array), AdditionalOutputs()
        except Exception as tts_err:
            print_status(f"❌ Failed to TTS critical error message: {tts_err}")
            yield EMPTY_AUDIO_YIELD_OUTPUT

if __name__ == "__main__":
    setup_async_environment() 

    print_status("🌐 Creating FastRTC stream with Bluetooth optimization...")
    try:
        stream = Stream(
            ReplyOnPause(
                smart_voice_assistant_callback_rt, 
                can_interrupt=True,
                algo_options=AlgoOptions(
                    audio_chunk_duration=2.0, started_talking_threshold=0.15, speech_threshold=0.08
                ),
                model_options=SileroVadOptions(
                    threshold=0.30, min_speech_duration_ms=300, 
                    min_silence_duration_ms=1800, speech_pad_ms=300, window_size_samples=1024
                )
            ), 
            modality="audio", mode="send-receive",
            track_constraints={ 
                "echoCancellation": True, "noiseSuppression": True, "autoGainControl": True,
                "sampleRate": {"ideal": AUDIO_SAMPLE_RATE}, "sampleSize": {"ideal": 16},
                "channelCount": {"exact": 1}, "latency": {"ideal": 0.05},
            }
        )
        print("=" * 70)
        print("🧠 SMART MEMORY + BLUETOOTH Voice Assistant Ready!")
        print("="*70)
        print("💡 Test Commands:")
        print("   • 'My name is [Your Name]'")
        print("   • 'No, my name is [Corrected Name]'")
        print("   • 'What is my name?' / 'Who am I?'")
        print("   • 'I like [something interesting]'")
        print("   • 'What do you remember about me?'")
        print("   • Ask a general question.")
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
        
        print_status("👋 Voice assistant shutdown process complete.")
        sys.exit(0)
