#!/usr/bin/env python3
"""
FastRTC Voice Assistant with A-MEM (Agentic Memory) + Bluetooth Optimization
Replaces fastrtc-whisper-cpp with a Hugging Face Transformers pipeline for STT,
based on the sofi444/realtime-transcription-fastrtc project.
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
    get_tts_model,
    AlgoOptions,
    SileroVadOptions,
    KokoroTTSOptions,
    audio_to_bytes, ### ADDED ### Utility from fastrtc
)
from fastrtc.utils import AdditionalOutputs
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import logging
from collections import deque
import hashlib

# ### ADDED ### Imports for Hugging Face STT
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)
from transformers.utils import is_flash_attn_2_available


# Import A-MEM components
from a_mem.memory_system import AgenticMemorySystem
from qdrant_client.models import Distance, VectorParams
from qdrant_client import QdrantClient
from a_mem.memory_system import AgenticMemorySystem, MemoryNote


# ### ADDED ### Import for audio processing
from scipy import signal


# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aioice.ice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)
logging.getLogger("phonemizer").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

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

# ### MODIFIED ### Configuration for Hugging Face STT model
# See https://huggingface.co/models?pipeline_tag=automatic-speech-recognition
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "openai/whisper-large-v3") # Changed from WHISPER_CPP_MODEL




# Language code mappings for Kokoro TTS
WHISPER_TO_KOKORO_LANG = {
    'en': 'a',    # American English (default)
    'it': 'i',    # Italian  
    'es': 'e',    # Spanish
    'fr': 'f',    # French
    'de': 'a',    # German -> fallback to English (not natively supported)
    'pt': 'p',    # Portuguese -> Brazilian Portuguese
    'ja': 'j',    # Japanese
    'ko': 'a',    # Korean -> fallback to English (not natively supported)
    'zh': 'z',    # Chinese -> Mandarin Chinese
    'hi': 'h',    # Hindi
}

# Language to voice mapping with official Kokoro voice names
KOKORO_VOICE_MAP = {
    'a': ['af_heart', 'af_bella', 'af_sarah'],                    # American English (best quality voices)
    'b': ['bf_emma', 'bf_isabella', 'bm_george'],                 # British English  
    'i': ['if_sara', 'im_nicola'],                                # Italian
    'e': ['ef_dora', 'em_alex', 'em_santa'],                      # Spanish
    'f': ['ff_siwis'],                                             # French (only one voice)
    'p': ['pf_dora', 'pm_alex', 'pm_santa'],                      # Brazilian Portuguese
    'j': ['jf_alpha', 'jf_gongitsune', 'jm_kumo'],               # Japanese
    'z': ['zf_xiaobei', 'zf_xiaoni', 'zm_yunjian', 'zm_yunxi'],  # Mandarin Chinese
    'h': ['hf_alpha', 'hf_beta', 'hm_omega'],                     # Hindi
}

# TTS language mapping for Kokoro
KOKORO_TTS_LANG_MAP = {
    'a': 'en-us',  # American English
    'b': 'en-gb',  # British English
    'i': 'it',     # Italian
    'e': 'es-es',  # Spanish
    'f': 'fr-fr',  # French
    'p': 'pt-br',  # Brazilian Portuguese
    'j': 'ja-jp',  # Japanese
    'z': 'zh-cn',  # Mandarin Chinese
    'h': 'hi-in',  # Hindi
}

DEFAULT_LANGUAGE = 'a'  # American English



print("🧠 FastRTC Voice Assistant - A-MEM + HF Transformers STT")
print("=" * 75)

def print_status(message):
    timestamp = time.strftime("%H:%M:%S")
    logger.info(f"{message}")

# ### ADDED ### Device selection utilities from sofi444/realtime-transcription-fastrtc
def get_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        return "mps"
    else:
        return "cpu"

def get_torch_and_np_dtypes(device, use_bfloat16=False):
    if device == "cuda":
        torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        np_dtype = np.float16
    elif device == "mps":
        torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        np_dtype = np.float16
    else:
        torch_dtype = torch.float32
        np_dtype = np.float32
    return torch_dtype, np_dtype

# ### MODIFIED ### STT model loading to use Hugging Face Transformers
print_status(f"🧠 Loading STT model (Hugging Face: {HF_MODEL_ID})...")

device = get_device(force_cpu=False)
torch_dtype, np_dtype = get_torch_and_np_dtypes(device, use_bfloat16=False)
print_status(f"Using device: {device}, torch_dtype: {torch_dtype}, np_dtype: {np_dtype}")

attention = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
print_status(f"Using attention implementation: {attention}")

try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation=attention
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(HF_MODEL_ID)

    transcribe_pipeline = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    print_status(f"✅ STT model ({HF_MODEL_ID}) loaded!")
except Exception as e:
    print_status(f"❌ STT model ({HF_MODEL_ID}) failed to load: {e}")
    print_status(f"Please ensure the model ID is correct and you have an internet connection.")
    sys.exit(1)

# Warm up the model
print_status("Warming up STT model with dummy input...")
warmup_audio = np.zeros((16000,), dtype=np_dtype)
transcribe_pipeline(warmup_audio)
print_status("✅ STT model warmup complete.")


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




class BluetoothAudioProcessor:
    def __init__(self):
        self.audio_buffer = deque(maxlen=10)
        self.noise_floor = None
        self.calibration_frames = 0
        self.min_calibration_frames = 15
        self.voice_detection_stats = deque(maxlen=30)
        
        # Audio healing components
        self.previous_frame = None
        self.corruption_history = deque(maxlen=5)
        self.dc_offset_filter = 0.0
        self.dc_filter_alpha = 0.999  # High-pass filter for DC removal
        
        print_status("🧠 Intelligent Audio Processor initialized")

    def detect_and_heal_corruption(self, audio_array):
        """Intelligently detect and fix common audio corruptions"""
        if audio_array.size == 0:
            return audio_array
        
        healed_audio = audio_array.copy()
        healing_applied = []
        
        # 1. FIX: Remove DC offset (hardware/bluetooth issue)
        if len(healed_audio) > 10:
            # High-pass filter to remove DC component
            for i in range(len(healed_audio)):
                self.dc_offset_filter = self.dc_filter_alpha * self.dc_offset_filter + healed_audio[i]
                healed_audio[i] = healed_audio[i] - self.dc_offset_filter
            
            if abs(np.mean(audio_array)) > 0.02:
                healing_applied.append("dc_removal")
        
        # 2. FIX: Smooth repetitive patterns (feedback/echo)
        if len(healed_audio) > 50:
            diff_std = np.std(np.diff(healed_audio))
            if diff_std < 0.001:  # Highly repetitive
                # Apply gentle smoothing to break repetitive patterns
                window_size = min(5, len(healed_audio) // 10)
                if window_size >= 3:
                    # Simple moving average to smooth repetitive spikes
                    kernel = np.ones(window_size) / window_size
                    if len(healed_audio) >= window_size:
                        smoothed = np.convolve(healed_audio, kernel, mode='same')
                        # Blend original and smoothed (preserve some original character)
                        healed_audio = 0.3 * healed_audio + 0.7 * smoothed
                        healing_applied.append("repetition_smoothing")
        
        # 3. FIX: Remove high-frequency noise/corruption
        rms = np.sqrt(np.mean(healed_audio**2))
        if rms > 0.3:  # Likely contains noise
            # Low-pass filter to remove high-frequency artifacts
            if len(healed_audio) >= 8:  # Minimum length for filtering
                try:
                    # Design a gentle low-pass filter
                    nyquist = 0.5 * AUDIO_SAMPLE_RATE
                    low_cutoff = min(4000, nyquist * 0.8)  # 4kHz cutoff or 80% of Nyquist
                    b, a = signal.butter(2, low_cutoff / nyquist, btype='low')
                    filtered = signal.filtfilt(b, a, healed_audio)
                    
                    # Only apply if it significantly reduces noise
                    filtered_rms = np.sqrt(np.mean(filtered**2))
                    if filtered_rms < rms * 0.8:  # At least 20% noise reduction
                        healed_audio = filtered
                        healing_applied.append("noise_filtering")
                except:
                    pass  # Skip filtering if it fails
        
        # 4. FIX: Intelligent gain control (prevent clipping, boost quiet audio)
        current_max = np.max(np.abs(healed_audio))
        current_rms = np.sqrt(np.mean(healed_audio**2))
        
        if current_max > 0.95:  # Near clipping
            # Soft limiting instead of hard clipping
            compression_ratio = 0.8 / current_max
            healed_audio = healed_audio * compression_ratio
            # Apply gentle compression curve for natural sound
            healed_audio = np.sign(healed_audio) * np.sqrt(np.abs(healed_audio))
            healing_applied.append("soft_limiting")
            
        elif current_rms < 0.01 and current_rms > 1e-6:  # Too quiet but not silent
            # Intelligent gain boost
            target_rms = 0.05
            gain = min(3.0, target_rms / current_rms)  # Max 3x gain
            healed_audio = healed_audio * gain
            healing_applied.append("gain_boost")
        
        # 5. FIX: Frame continuity (smooth transitions between frames)
        if self.previous_frame is not None and len(self.previous_frame) > 0 and len(healed_audio) > 0:
            # Check for sudden jumps between frames
            frame_jump = abs(healed_audio[0] - self.previous_frame[-1])
            if frame_jump > 0.5:  # Large discontinuity
                # Smooth the transition
                fade_length = min(10, len(healed_audio) // 4)
                if fade_length > 0:
                    fade_in = np.linspace(0, 1, fade_length)
                    target_start = self.previous_frame[-1] * 0.8  # Gentle transition
                    for i in range(fade_length):
                        healed_audio[i] = healed_audio[i] * fade_in[i] + target_start * (1 - fade_in[i])
                    healing_applied.append("frame_smoothing")
        
        # 6. FIX: Outlier removal (random spikes)
        if len(healed_audio) > 20:
            # Remove extreme outliers that are likely corruption
            median_val = np.median(np.abs(healed_audio))
            if median_val > 0:
                threshold = median_val * 10  # Values 10x above median are outliers
                outlier_mask = np.abs(healed_audio) > threshold
                if np.any(outlier_mask):
                    # Replace outliers with interpolated values
                    outlier_indices = np.where(outlier_mask)[0]
                    for idx in outlier_indices:
                        # Simple interpolation from neighbors
                        left_val = healed_audio[max(0, idx-1)]
                        right_val = healed_audio[min(len(healed_audio)-1, idx+1)]
                        healed_audio[idx] = (left_val + right_val) / 2
                    if len(outlier_indices) > 0:
                        healing_applied.append("outlier_removal")
        
        # Store frame for next iteration
        self.previous_frame = healed_audio[-10:] if len(healed_audio) >= 10 else healed_audio.copy()
        
        # Log healing if any was applied
        if healing_applied:
            healing_str = "+".join(healing_applied)
            print_status(f"🩺 Audio healed: {healing_str}")
        
        return healed_audio

    def calibrate_noise_floor(self, audio_data):
        """Existing noise floor calibration (unchanged)"""
        if self.calibration_frames < self.min_calibration_frames:
            rms = np.sqrt(np.mean(audio_data**2)) if audio_data.size > 0 else 0.0
            if self.noise_floor is None: 
                self.noise_floor = rms
            else: 
                self.noise_floor = 0.9 * self.noise_floor + 0.1 * rms
            self.calibration_frames += 1
            if self.calibration_frames == self.min_calibration_frames:
                print_status(f"🎙️ Bluetooth calibration complete: noise_floor={self.noise_floor:.6f}")

    def preprocess_bluetooth_audio(self, audio_data):
        """Enhanced preprocessing with intelligent healing"""
        # Parse input (unchanged)
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_array = audio_data
            if not isinstance(sample_rate, int) or not isinstance(audio_array, np.ndarray):
                return AUDIO_SAMPLE_RATE, np.array([], dtype=np.float32)
        elif isinstance(audio_data, np.ndarray):
            sample_rate, audio_array = AUDIO_SAMPLE_RATE, audio_data
        else:
            return AUDIO_SAMPLE_RATE, np.array([], dtype=np.float32)

        if audio_array.size == 0:
            return sample_rate, np.array([], dtype=np.float32)

        # Convert to float32
        if audio_array.dtype != np.float32: 
            audio_array = audio_array.astype(np.float32)

        # STEP 1: Apply intelligent healing FIRST
        healed_audio = self.detect_and_heal_corruption(audio_array)
        
        # STEP 2: Standard processing on healed audio
        self.calibrate_noise_floor(healed_audio)

        # STEP 3: Gentle adaptive processing (reduced from original)
        if self.noise_floor is not None:
            current_rms = np.sqrt(np.mean(healed_audio**2)) if healed_audio.size > 0 else 0.0
            if current_rms > self.noise_floor * 1.5 and current_rms > 1e-6:
                # Only apply minimal gain if really needed
                target_rms = 0.06
                if current_rms < target_rms * 0.6:
                    gain = min(2.0, (target_rms * 0.8 / current_rms) if current_rms > 1e-6 else 1.0)
                    healed_audio = healed_audio * gain

        # STEP 4: Final safety normalization
        max_abs_processed = np.max(np.abs(healed_audio))
        if max_abs_processed > 1.0 and max_abs_processed > 1e-6:
            healed_audio = healed_audio / max_abs_processed

        # Update stats
        self.voice_detection_stats.append({
            'rms': np.sqrt(np.mean(healed_audio**2)) if healed_audio.size > 0 else 0.0, 
            'timestamp': time.time()
        })
        
        return sample_rate, healed_audio

    def get_detection_stats(self):
        """Existing stats method (unchanged)"""
        if not self.voice_detection_stats: 
            return None
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
                model_name='all-MiniLM-L6-v2', # This is for embeddings, whisper uses its own models
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
                op, user_text, assistant_text, category = await self.memory_queue.get()
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self.executor, self._store_memory_background,
                        user_text, assistant_text, category
                    )
                except Exception as processing_error:
                    print_status(f"❌ Error during background A-MEM storage execution: {processing_error}")
                    import traceback
                    traceback.print_exc()
                finally:
                    self.memory_queue.task_done()
            except asyncio.CancelledError:
                print_status("Background A-MEM processor task was cancelled.")
                break
            except Exception as e:
                print_status(f"❌ Unexpected error in A-MEM queue processing loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)


    def _parse_timestamp(self, timestamp_str: Optional[str]) -> datetime:
        if not timestamp_str:
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            for fmt in ('%Y%m%d%H%M',
                        '%Y-%m-%d %H:%M:%S.%f%z', '%Y-%m-%d %H:%M:%S%z',
                        '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                try:
                    dt = datetime.strptime(timestamp_str, fmt)
                    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except ValueError:
                    continue
            logger.warning(f"Could not parse timestamp string: '{timestamp_str}' with any known format. Returning datetime.min.")
            return datetime.min.replace(tzinfo=timezone.utc)

    def _extract_name_from_memory_text(self, memory_text: str) -> Optional[str]:
        if "Assistant:" in memory_text:
            user_parts = []
            for part in memory_text.split('\n'):
                if part.startswith("User:"):
                    user_parts.append(part[5:].strip())
            text_to_analyze = " ".join(user_parts)
        else:
            text_to_analyze = memory_text

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
            match = re.search(r'\b' + pattern + r'\b', text_lower)
            if match:
                start_pos = match.start(1)
                end_pos = match.end(1)
                name = text_to_analyze[start_pos:end_pos].strip()
                if name and name.lower() not in invalid_names and len(name) > 1:
                    return name
        return None

    def _load_existing_memories(self):
        try:
                all_memories = list(self.amem_system.memories.values())
                if not all_memories:
                    print_status(f"📚 No memories in A-MEM system, checking ChromaDB...")
                    try:
                        collection = self.amem_system.retriever.collection
                        results = collection.get()
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
                        self.memory_cache['_currently_processing_content'] = memory_text
                        item_timestamp = self._parse_timestamp(timestamp_str)
                        del self.memory_cache['_currently_processing_content']

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

                    if personal_entries:
                        personal_entries.sort(key=lambda x: x['timestamp'], reverse=True)
                        top_personal_entry_name = personal_entries[0]['name']
                        self.memory_cache['user_name'] = top_personal_entry_name
                        logger.info(f"DEBUG _load_existing_memories: Sorted personal_entries (Top 3): {personal_entries[:3]}")
                        print_status(f"👤 Restored user name from A-MEM (personal_info): {self.memory_cache['user_name']}")
                    elif fallback_entries:
                        fallback_entries.sort(key=lambda x: x['timestamp'], reverse=True)
                        logger.info(f"DEBUG _load_existing_memories: Sorted fallback_entries (Top 3): {fallback_entries[:3]}")
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
            conversation_content = f"User: {user_text}\nAssistant: {assistant_text}"
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
        try:
            self.amem_system.memories.clear()
            self.amem_system.consolidate_memories()
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

        qclient = QdrantClient(host="localhost", port=6333)
        collections_response = qclient.get_collections()
        collection_names = [c.name for c in collections_response.collections]

        if "amem_voice_collection" in collection_names:
            print_status("✅ Qdrant collection 'amem_voice_collection' already exists.")
        else:
            print_status("Creating Qdrant collection 'amem_voice_collection'...")
            qclient.create_collection(
                collection_name="amem_voice_collection",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        self.user_id = "amem_voice_user_01"
        self.session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        self.amem_memory = AMemMemoryManager(self.user_id)
        
        # MISSING: Audio processor initialization
        self.audio_processor = BluetoothAudioProcessor()
        
        self.conversation_buffer = deque(maxlen=100)
        self.response_cache = {}
        self.cache_ttl_seconds = 180
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.turn_count = 0
        self.total_response_time = deque(maxlen=20)
        self.voice_detection_successes = 0
        
        # MISSING: Language tracking for dynamic voice switching
        self.current_language = DEFAULT_LANGUAGE  # Track current conversation language
        
        print_status(f"👤 User ID: {self.user_id}, Session: {self.session_id}")
        if USE_OLLAMA_FOR_CONVERSATION:
            print_status(f"🗣️ Conversational LLM: Ollama ({OLLAMA_CONVERSATIONAL_MODEL} via {OLLAMA_URL})")
        else:
            print_status(f"🗣️ Conversational LLM: LM Studio ({LM_STUDIO_MODEL} via {LM_STUDIO_URL})")
        print_status(f"🧠 A-MEM System: {AMEM_LLM_MODEL} with {AMEM_EMBEDDER_MODEL} embeddings")
        print_status(f"🎤 STT System: Hugging Face Transformers (Model: {HF_MODEL_ID})")

    # MISSING: Async initialization method
    async def initialize_async(self):
        print_status("Initializing async components for SmartVoiceAssistant...")
        connector = aiohttp.TCPConnector(limit_per_host=5, ssl=False)
        timeout = aiohttp.ClientTimeout(total=20, connect=5, sock_read=15)
        self.http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        print_status("✅ aiohttp ClientSession created.")
        await self.amem_memory.start_background_processor()
        print_status("✅ Async components initialized for SmartVoiceAssistant.")

    # MISSING: Async cleanup method
    async def cleanup_async(self):
        print_status("🧹 Starting async cleanup for SmartVoiceAssistant...")
        if self.http_session:
            await self.http_session.close()
            print_status("aiohttp ClientSession closed.")
        await self.amem_memory.shutdown()
        print_status("🧹 Async cleanup completed for SmartVoiceAssistant.")

    # MISSING: Audio processing method
    def process_audio_array(self, audio_data):
        return self.audio_processor.preprocess_bluetooth_audio(audio_data)

    # MISSING: Response caching methods
    def get_cached_response(self, text: str) -> Optional[str]:
        text_hash = hashlib.md5(text.lower().encode()).hexdigest()
        item = self.response_cache.get(text_hash)
        if item and (datetime.now(timezone.utc) - item['timestamp'] < timedelta(seconds=self.cache_ttl_seconds)):
            return item['response']
        return None

    def cache_response(self, text: str, response: str):
        text_hash = hashlib.md5(text.lower().encode()).hexdigest()
        self.response_cache[text_hash] = {'response': response, 'timestamp': datetime.now(timezone.utc)}



    def detect_language_from_text(self, text: str) -> str:
        """Enhanced language detection with more comprehensive word lists"""
        text_lower = text.lower()
        
        # Expanded Italian indicators - include more common words
        italian_words = [
            # Existing words
            'ciao', 'grazie', 'prego', 'bene', 'come stai', 'buongiorno', 'buonasera', 'molto', 'sono', 'dove',
            # NEW: More common Italian words
            'voglio', 'che', 'parli', 'italiano', 'parlare', 'posso', 'puoi', 'sì', 'no',
            'questo', 'quello', 'quando', 'perché', 'come', 'cosa', 'chi', 'quale',
            'anche', 'ancora', 'dopo', 'prima', 'sempre', 'mai', 'già', 'oggi',
            'ieri', 'domani', 'casa', 'lavoro', 'famiglia', 'amico', 'tempo',
            'bello', 'buono', 'grande', 'piccolo', 'nuovo', 'vecchio',
            'fare', 'dire', 'andare', 'venire', 'vedere', 'sapere', 'dare',
            'volere', 'dovere', 'potere', 'stare', 'avere', 'essere',
            'mi', 'ti', 'ci', 'vi', 'lo', 'la', 'li', 'le', 'gli', 'ne',
            'con', 'per', 'da', 'in', 'su', 'di', 'del', 'della', 'dello'
        ]
        
        # Spanish indicators - expanded
        spanish_words = [
            'hola', 'gracias', 'por favor', 'bueno', 'como estas', 'buenos dias', 'muy', 'soy', 'donde',
            'quiero', 'que', 'hablar', 'español', 'puedo', 'puedes', 'sí', 'no',
            'este', 'ese', 'cuando', 'porque', 'como', 'que', 'quien', 'cual',
            'también', 'todavia', 'después', 'antes', 'siempre', 'nunca', 'ya', 'hoy',
            'ayer', 'mañana', 'casa', 'trabajo', 'familia', 'amigo', 'tiempo',
            'hacer', 'decir', 'ir', 'venir', 'ver', 'saber', 'dar',
            'querer', 'deber', 'poder', 'estar', 'tener', 'ser'
        ]
        
        # French indicators - expanded  
        french_words = [
            'bonjour', 'merci', 'comment allez', 'tres bien', 'bonsoir', 'je suis', 'tres', 'ou',
            'veux', 'que', 'parler', 'français', 'peux', 'pouvez', 'oui', 'non',
            'ce', 'cette', 'quand', 'pourquoi', 'comment', 'quoi', 'qui', 'quel',
            'aussi', 'encore', 'après', 'avant', 'toujours', 'jamais', 'déjà', 'aujourd',
            'hier', 'demain', 'maison', 'travail', 'famille', 'ami', 'temps',
            'faire', 'dire', 'aller', 'venir', 'voir', 'savoir', 'donner',
            'vouloir', 'devoir', 'pouvoir', 'être', 'avoir', 'aller'
        ]
        
        # Portuguese indicators - expanded
        portuguese_words = [
            'ola', 'obrigado', 'obrigada', 'por favor', 'bom dia', 'como vai', 'muito', 'sou', 'onde',
            'quero', 'que', 'falar', 'português', 'posso', 'pode', 'sim', 'não',
            'este', 'esse', 'quando', 'porque', 'como', 'que', 'quem', 'qual',
            'também', 'ainda', 'depois', 'antes', 'sempre', 'nunca', 'já', 'hoje',
            'ontem', 'amanhã', 'casa', 'trabalho', 'família', 'amigo', 'tempo',
            'fazer', 'dizer', 'ir', 'vir', 'ver', 'saber', 'dar',
            'querer', 'dever', 'poder', 'estar', 'ter', 'ser'
        ]

        # Count matches for each language
        italian_matches = sum(1 for word in italian_words if word in text_lower)
        spanish_matches = sum(1 for word in spanish_words if word in text_lower)
        french_matches = sum(1 for word in french_words if word in text_lower)
        portuguese_matches = sum(1 for word in portuguese_words if word in text_lower)
        
        # Debug logging
        print_status(f"🔍 Language detection - IT:{italian_matches}, ES:{spanish_matches}, FR:{french_matches}, PT:{portuguese_matches}")
        
        # Require at least 2 matches to be confident, return the highest
        max_matches = max(italian_matches, spanish_matches, french_matches, portuguese_matches)
        
        if max_matches >= 2:
            if italian_matches == max_matches:
                return 'i'
            elif spanish_matches == max_matches:
                return 'e'
            elif french_matches == max_matches:
                return 'f'
            elif portuguese_matches == max_matches:
                return 'p'
        
        # Single word detection for key phrases
        if any(word in text_lower for word in ['italiano', 'italiana']):
            return 'i'
        if any(word in text_lower for word in ['español', 'castellano']):
            return 'e'
        if any(word in text_lower for word in ['français', 'francais']):
            return 'f'
        if any(word in text_lower for word in ['português', 'portugues']):
            return 'p'
        
        # Japanese and other languages (existing logic)
        if any(word in text_lower for word in ['konnichiwa', 'arigatou', 'sumimasen', 'hajimemashite', 'sayonara', 'watashi', 'desu']):
            return 'j'
        if any(word in text_lower for word in ['nihao', 'xiexie', 'zaijian', 'duibuqi', 'wo', 'shi', 'nali']):
            return 'z'
        if any(word in text_lower for word in ['namaste', 'dhanyawad', 'kaise', 'aap', 'main', 'hoon', 'kahan']):
            return 'h'
        
        # Default to American English
        return 'a'

    def get_voices_for_language(self, kokoro_lang: str) -> list:
        """Get appropriate voices for the detected language"""
        voices = KOKORO_VOICE_MAP.get(kokoro_lang, KOKORO_VOICE_MAP[DEFAULT_LANGUAGE])
        return voices.copy()  # Return a copy to avoid modifying the original

    def get_kokoro_language(self, whisper_lang: str) -> str:
        """Convert Whisper language code to Kokoro language code"""
        return WHISPER_TO_KOKORO_LANG.get(whisper_lang, DEFAULT_LANGUAGE)

    # MISSING: LLM context and response methods
    def _get_llm_context_prompt(self) -> str:
        memory_context = self.amem_memory.get_user_context()
        recent_conv = ""
        if self.conversation_buffer:
            turns = []
            for turn in list(self.conversation_buffer)[-3:]:
                turns.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")
            recent_conv = "\n---\nRecent Conversation:\n" + "\n".join(turns) if turns else ""

        system_prompt = f"""You are Echo, a friendly and multilingual voice assistant with advanced memory capabilities.
        Keep responses concise and natural for voice interaction.

        {memory_context}

        Remember:
        - Your name is Echo (the assistant)
        - You have an advanced agentic memory system that learns and evolves
        - You can form connections between different memories and concepts
        - You answers the user always in the same language they used to ask
        - You can remember the user's name and preferences
        - You can forget all memories if the user requests it
        - Be warm and conversational
        {recent_conv}"""
        return system_prompt.strip()

    async def get_llm_response_smart(self, user_text: str) -> str:
        cached_llm_response = self.get_cached_response(user_text)
        if cached_llm_response:
            return cached_llm_response

        recall_phrases = ['what do you remember', 'what do you know about me', 'tell me about myself', 'what is my name', 'who am i']
        if any(phrase in user_text.lower() for phrase in recall_phrases):
            memory_search_result = await self.amem_memory.search_memories_smart(user_text)
            self.cache_response(user_text, memory_search_result)
            return memory_search_result

        if any(phrase in user_text.lower() for phrase in ["delete all your memory", "reset", "forget everything"]):
            result = self.amem_memory.forget_all_memories()
            response = "I've erased all my memories and reset my knowledge network." if result else "Sorry, I couldn't erase my memories due to an internal error."
            self.cache_response(user_text, response)
            return response

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
                    "model": OLLAMA_CONVERSATIONAL_MODEL, "messages": messages, "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 250}
                }
                async with self.http_session.post(f"{OLLAMA_URL}/api/chat", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        assistant_response_text = data.get("message", {}).get("content", "").strip() or "I'm not sure how to respond to that."
                    else:
                        error_body = await response.text()
                        print_status(f"⚠️ Ollama LLM request failed: Status {response.status}, Body: {error_body[:200]}")
                        assistant_response_text = "I'm sorry, I encountered an issue trying to respond via Ollama."
            else: # Use LM Studio
                payload = {
                    "model": LM_STUDIO_MODEL, "messages": messages, "max_tokens": 250,
                    "temperature": 0.7, "stream": False
                }
                async with self.http_session.post(f"{LM_STUDIO_URL}/chat/completions", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        assistant_response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or "I'm not sure how to respond to that."
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



def run_coro_from_sync_thread_with_timeout(coro, timeout: float = 4.0) -> any:
    """Run coroutine with timeout to prevent WebRTC disconnections"""
    global main_event_loop
    if main_event_loop and main_event_loop.is_running():
        import asyncio
        future = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(coro, timeout=timeout), 
            main_event_loop
        )
        try:
            return future.result(timeout=timeout + 1.0)  # Add 1s buffer
        except asyncio.TimeoutError:
            print_status(f"❌ Async task timed out after {timeout}s")
            raise TimeoutError(f"Operation timed out after {timeout}s")
        except Exception as e:
            print_status(f"❌ Error in async task: {e}")
            return "I encountered an error processing your request."
    else:
        print_status("❌ Event loop not available")
        return "My processing system is not ready."
        


# BUG FIX 2: Fixed callback function with proper variable scoping

def smart_voice_assistant_callback_rt(audio_data_tuple: tuple):
    global assistant_instance, tts_model, transcribe_pipeline

    if not assistant_instance:
        yield EMPTY_AUDIO_YIELD_OUTPUT
        return

    try:
        sample_rate, audio_array = assistant_instance.process_audio_array(audio_data_tuple)

        if audio_array.size == 0:
            yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
            return

        # FIXED: Define lang_names at the top so it's always available
        lang_names = {
            'a': 'American English', 'b': 'British English', 'i': 'Italian', 
            'e': 'Spanish', 'f': 'French', 'p': 'Portuguese', 
            'j': 'Japanese', 'z': 'Chinese', 'h': 'Hindi'
        }

        # Enhanced STT with language detection
        user_text = ""
        detected_language = DEFAULT_LANGUAGE
        
        if audio_array.size > 0:
            outputs = transcribe_pipeline(
                audio_to_bytes((sample_rate, audio_array)),
                chunk_length_s=30,
                batch_size=1,
                generate_kwargs={'task': 'transcribe'},
                return_timestamps=False,
            )
            
            user_text = outputs["text"].strip()
            print_status(f"📝 Transcribed: '{user_text}'")
            
            # Try multiple methods to detect language
            whisper_language = 'en'  # Default
            
            # Method 1: Check if Whisper provided language info
            if hasattr(outputs, 'get'):
                if 'language' in outputs:
                    whisper_language = outputs['language']
                    print_status(f"🎤 Whisper detected language: {whisper_language}")
                elif 'chunks' in outputs and outputs['chunks']:
                    for chunk in outputs['chunks']:
                        if 'language' in chunk:
                            whisper_language = chunk['language']
                            print_status(f"🎤 Whisper chunk language: {whisper_language}")
                            break
            
            # Method 2: Text-based detection (primary method)
            text_detected_lang = assistant_instance.detect_language_from_text(user_text)
            
            # Method 3: Combine detections with priority to text detection
            if text_detected_lang != 'a':  # If text detection found non-English
                detected_language = text_detected_lang
                print_status(f"🔤 Text-based detection: {detected_language}")
            else:
                # Fall back to Whisper detection
                detected_language = assistant_instance.get_kokoro_language(whisper_language)
                print_status(f"🎤 Using Whisper detection: {whisper_language} -> {detected_language}")
            
            # Update language if changed
            if detected_language != assistant_instance.current_language:
                assistant_instance.current_language = detected_language
                lang_name = lang_names.get(detected_language, 'Unknown')
                print_status(f"🌍 Language switched to: {lang_name} ({detected_language})")
            else:
                lang_name = lang_names.get(detected_language, 'Unknown')
                print_status(f"🌍 Language confirmed: {lang_name} ({detected_language})")

        if not user_text.strip():
            yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
            return

        assistant_instance.voice_detection_successes += 1
        print_status(f"👤 User: {user_text}")

        start_turn_time = time.monotonic()
        try:
            # FIXED: Use the correct function name (with timeout)
            assistant_response_text = run_coro_from_sync_thread_with_timeout(
                assistant_instance.get_llm_response_smart(user_text),
                timeout=4.0
            )
        except TimeoutError:
            assistant_response_text = "Let me think about that and get back to you quickly."
        
        turn_processing_time = time.monotonic() - start_turn_time
        print_status(f"🤖 Assistant: {assistant_response_text}")
        print_status(f"⏱️ Turn Processing Time: {turn_processing_time:.2f}s")

        # Update conversation buffer
        assistant_instance.conversation_buffer.append({
            'user': user_text, 'assistant': assistant_response_text,
            'timestamp': datetime.now(timezone.utc)
        })
        assistant_instance.turn_count += 1
        assistant_instance.total_response_time.append(turn_processing_time)

        # Stats display (every 5 turns to reduce overhead)
        if assistant_instance.turn_count % 5 == 0:
            avg_resp = np.mean(list(assistant_instance.total_response_time)) if assistant_instance.total_response_time else 0
            mem_s = assistant_instance.amem_memory.get_stats()
            audio_s = assistant_instance.audio_processor.get_detection_stats()
            audio_q_str = f" | AudioRMS: {audio_s['avg_rms']:.3f}" if audio_s and audio_s['calibrated'] else ""
            
            lang_abbr = {
                'a': 'EN-US', 'b': 'EN-GB', 'i': 'IT', 'e': 'ES', 'f': 'FR', 
                'p': 'PT-BR', 'j': 'JA', 'z': 'ZH', 'h': 'HI'
            }.get(assistant_instance.current_language, 'UNK')
            voice_count = len(assistant_instance.get_voices_for_language(assistant_instance.current_language))
            lang_str = f" | Lang: {lang_abbr}({voice_count}v)"
            
            print_status(f"📊 Turn {assistant_instance.turn_count}: AvgResp={avg_resp:.2f}s | MemOps={mem_s['mem_ops']} | User='{mem_s['user_name_cache']}'{audio_q_str}{lang_str}")

        # TTS with confirmed language
        additional_outputs = AdditionalOutputs()
        tts_voices_to_try = assistant_instance.get_voices_for_language(assistant_instance.current_language)
        tts_voices_to_try.append(None)
        
        print_status(f"🎤 TTS using language '{assistant_instance.current_language}' with voices: {tts_voices_to_try[:3]}")
        
        tts_success = False
        for voice_id in tts_voices_to_try:
            try:
                options_params = {"speed": 1.05}
                kokoro_tts_lang = KOKORO_TTS_LANG_MAP.get(assistant_instance.current_language, 'en-us')
                options_params["lang"] = kokoro_tts_lang
                
                if voice_id: 
                    options_params["voice"] = voice_id
                    
                tts_options = KokoroTTSOptions(**options_params)
                print_status(f"🔊 Trying TTS with voice '{voice_id}', lang '{kokoro_tts_lang}'")
                
                chunk_count = 0
                total_samples = 0
                for tts_output_item in tts_model.stream_tts_sync(assistant_response_text, tts_options):
                    if isinstance(tts_output_item, tuple) and len(tts_output_item) == 2 and isinstance(tts_output_item[1], np.ndarray):
                        current_sr, current_chunk_array = tts_output_item
                        if current_chunk_array.size > 0:
                            chunk_count += 1
                            total_samples += current_chunk_array.size
                            # Yield smaller chunks to prevent timeouts
                            chunk_size = min(1024, current_chunk_array.size)
                            for i in range(0, current_chunk_array.size, chunk_size):
                                mini_chunk = current_chunk_array[i:i+chunk_size]
                                if mini_chunk.size > 0:
                                    yield (current_sr, mini_chunk), additional_outputs
                    elif isinstance(tts_output_item, np.ndarray) and tts_output_item.size > 0:
                        chunk_count += 1
                        total_samples += tts_output_item.size
                        chunk_size = min(1024, tts_output_item.size)
                        for i in range(0, tts_output_item.size, chunk_size):
                            mini_chunk = tts_output_item[i:i+chunk_size]
                            if mini_chunk.size > 0:
                                yield (sample_rate, mini_chunk), additional_outputs
                
                tts_success = True
                logger.info(f"TTS stream completed. Voice: {voice_id}, Chunks: {chunk_count}, Samples: {total_samples}")
                
                # Success message with language confirmation
                if assistant_instance.current_language != 'a' and voice_id:
                    lang_name = {
                        'i': 'Italian', 'e': 'Spanish', 'f': 'French', 'p': 'Portuguese',
                        'j': 'Japanese', 'z': 'Chinese', 'h': 'Hindi'
                    }.get(assistant_instance.current_language, 'Unknown')
                    print_status(f"✅ TTS SUCCESS using {lang_name} voice: {voice_id}")
                break
                
            except Exception as e:
                print_status(f"❌ TTS failed with voice '{voice_id}': {e}")
                continue
                
        if not tts_success:
            print_status(f"❌ All TTS attempts failed")
            yield SILENT_AUDIO_FRAME_TUPLE, additional_outputs

    except Exception as e:
        print_status(f"❌ CRITICAL Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Graceful error recovery
        try:
            error_msg = "Sorry, I encountered an error. Please try again."
            tts_options = KokoroTTSOptions(speed=1.0, lang="en-us")
            for tts_err_chunk in tts_model.stream_tts_sync(error_msg, tts_options):
                if isinstance(tts_err_chunk, tuple) and len(tts_err_chunk) == 2 and isinstance(tts_err_chunk[1], np.ndarray):
                     sr_err, arr_err = tts_err_chunk
                     if arr_err.size > 0: 
                         yield (sr_err, arr_err), AdditionalOutputs()
                elif isinstance(tts_err_chunk, np.ndarray) and tts_err_chunk.size > 0:
                    yield (AUDIO_SAMPLE_RATE, tts_err_chunk), AdditionalOutputs()
        except Exception:
            yield EMPTY_AUDIO_YIELD_OUTPUT


if __name__ == "__main__":
    setup_async_environment()

    if not assistant_instance or not assistant_instance.audio_processor.noise_floor:
        threshold = 0.15
    else:
        threshold = assistant_instance.audio_processor.noise_floor * 15

    print_status("🌐 Creating FastRTC stream with A-MEM + HF Transformers STT...")
    try:
        stream = Stream(
            ReplyOnPause(
                smart_voice_assistant_callback_rt,
                can_interrupt=True,
                algo_options=AlgoOptions(
                    audio_chunk_duration=2.0,
                    started_talking_threshold=0.15,
                    speech_threshold=threshold
                ),
                model_options=SileroVadOptions(
                    threshold=0.3,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=2000,
                    speech_pad_ms=200,
                    window_size_samples=512
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
                "latency": {"ideal": 0.01},
            }
        )
        print("=" * 70)
        print("🚀 FastRTC HF Transformers Voice Assistant Ready!") ### MODIFIED ###
        print(f"🎤 Using Whisper model: {HF_MODEL_ID}")
        print("="*70)
        print("💡 Test Commands:")
        print("   • 'My name is [Your Name]'")
        print("   • 'What is my name?' / 'Who am I?'")
        print("   • 'I like [something interesting]'")
        print("   • Ask questions in supported languages.")
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