import numpy as np
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.utils import is_flash_attn_2_available
import soundfile as sf # For loading audio files (pip install soundfile)
import librosa # For resampling if needed for test audio (pip install librosa)

# --- Configuration (Simplified for test) ---
HF_MODEL_ID = "openai/whisper-base" # Use a smaller model for faster testing
AUDIO_SAMPLE_RATE = 16000 # Target for STT

# Re-use your LANGUAGE_TO_TTS_CONFIG and DEFAULT_TTS_CONFIG
LANGUAGE_TO_TTS_CONFIG = {
    "english": {"lang_code": "en-us", "voices": ["en_voice1", "en_voice2"]},
    "italian": {"lang_code": "it-it", "voices": ["it_voice1", "it_voice2"]},
    "spanish": {"lang_code": "es-es", "voices": ["es_voice1"]},
    # Add other languages you test
}
DEFAULT_TTS_CONFIG = LANGUAGE_TO_TTS_CONFIG["english"]

# --- Helper Functions (copied from your script) ---
def get_device(force_cpu=False):
    if force_cpu: return "cpu"
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache(); return "mps"
    return "cpu"

def get_torch_dtype(device_str, use_bfloat16=False):
    if device_str == "cuda":
        return torch.bfloat16 if use_bfloat16 and torch.cuda.is_bf16_supported() else torch.float16
    elif device_str == "mps":
        return torch.float16
    return torch.bfloat16 if use_bfloat16 else torch.float32

print(f"--- Test: STT Language Detection & Dynamic TTS Config ---")
print(f"Using HF Model: {HF_MODEL_ID}")

# 1. Load STT Pipeline
try:
    device_selected = get_device()
    model_dtype = get_torch_dtype(device_selected)
    attention = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    print(f"Test Device: {device_selected}, Model Dtype: {model_dtype}, Attention: {attention}")

    stt_pipeline = pipeline(
        task="automatic-speech-recognition",
        model=HF_MODEL_ID, # Can also pass pre-loaded model and processor
        torch_dtype=model_dtype,
        device=device_selected,
        # model_kwargs={"attn_implementation": attention} # Pass model_kwargs for pipeline
    )
    print("STT Pipeline loaded successfully.")
except Exception as e:
    print(f"Failed to load STT pipeline: {e}")
    exit()

# 2. Test Cases - Define a list of audio files and their expected languages
test_audio_files = [
    {"path": "path/to/your/english_audio_48khz.wav", "expected_lang_name": "english", "expected_text_contains": "hello"},
    {"path": "path/to/your/italian_audio_16khz.wav", "expected_lang_name": "italian", "expected_text_contains": "ciao"},
    {"path": "path/to/your/spanish_audio_16khz.wav", "expected_lang_name": "spanish", "expected_text_contains": "hola"},
    # Add a file in a language NOT in your LANGUAGE_TO_TTS_CONFIG to test fallback
    {"path": "path/to/your/unconfigured_lang_audio.wav", "expected_lang_name": "some_other_lang_name", "expected_text_contains": "..."}
]

for test_case in test_audio_files:
    print(f"\n--- Testing: {test_case['path']} ---")
    try:
        audio_input, original_sr = sf.read(test_case["path"], dtype='float32')
        print(f"Original SR: {original_sr}Hz, Duration: {len(audio_input)/original_sr:.2f}s")

        # Resample if necessary
        if original_sr != AUDIO_SAMPLE_RATE:
            print(f"Resampling from {original_sr}Hz to {AUDIO_SAMPLE_RATE}Hz...")
            audio_input = librosa.resample(audio_input, orig_sr=original_sr, target_sr=AUDIO_SAMPLE_RATE)

        # Transcribe
        # The key is `return_timestamps` (for chunks) or language detection kwargs
        # Forcing language detection (task : transcribe is default, no need to specify lang for detection)
        result = stt_pipeline(
            audio_input.copy(), # Important to pass a copy if reusing audio_input
            generate_kwargs={"task": "transcribe"}, # Ensure it transcribes
            return_timestamps="chunks" # To get language per chunk potentially
        )

        transcribed_text = result["text"].strip()
        print(f"Transcribed Text: {transcribed_text}")

        # Language Detection Logic (mirroring your main script)
        detected_language_name = "unknown"
        if chunks := result.get("chunks"):
            if len(chunks) > 0 and chunks[0].get("language"):
                detected_language_name = chunks[0]["language"].lower()
                print(f"Language from Chunks: '{detected_language_name}'")
        # Fallback or alternative: some pipelines might put 'language' at top level
        if detected_language_name == "unknown" and (direct_lang := result.get("language")):
             detected_language_name = direct_lang.lower()
             print(f"Language from Top-Level: '{detected_language_name}'")


        # If still unknown, some models embed it in the text like <|en|>
        if detected_language_name == "unknown" and transcribed_text.startswith("<|") and transcribed_text.count("|>") > 0:
            lang_tag_end = transcribed_text.find("|>", 2) # find after <|
            if lang_tag_end != -1:
                lang_code_in_tag = transcribed_text[2:lang_tag_end]
                # This mapping is crucial if the tag isn't the full language name
                lang_map = {"en": "english", "it": "italian", "es": "spanish", "fr": "french", "de": "german"} # etc.
                detected_language_name = lang_map.get(lang_code_in_tag, f"tag:{lang_code_in_tag}")
                print(f"Language from Text Tag: '{detected_language_name}' (original tag: '{lang_code_in_tag}')")


        print(f"==> Detected Language: '{detected_language_name}' (Expected: '{test_case['expected_lang_name']}')")
        assert detected_language_name == test_case["expected_lang_name"], "Language detection mismatch!"
        if test_case["expected_text_contains"].lower() not in transcribed_text.lower():
             print(f"WARNING: Expected text '{test_case['expected_text_contains']}' not found in transcription.")


        # Dynamic TTS Config Selection
        tts_config = LANGUAGE_TO_TTS_CONFIG.get(detected_language_name, DEFAULT_TTS_CONFIG)
        selected_lang_code = tts_config["lang_code"]
        selected_voices = tts_config["voices"]
        print(f"Selected TTS Lang Code: {selected_lang_code}")
        print(f"Selected TTS Voices: {selected_voices}")

        if detected_language_name not in LANGUAGE_TO_TTS_CONFIG:
            print(f"NOTE: Language '{detected_language_name}' not in TTS config, fell back to default ({DEFAULT_TTS_CONFIG['lang_code']}).")

    except FileNotFoundError:
        print(f"Audio file not found: {test_case['path']}. Skipping.")
    except Exception as e:
        print(f"Error during test case {test_case['path']}: {e}")
        import traceback
        traceback.print_exc()

print("\n--- Test Complete ---")

# How to run:
# 1. Save this as test_stt_tts_language_pipeline.py
# 2. Install soundfile and librosa: pip install soundfile librosa
# 3. Create dummy WAV files in the specified paths or update paths.
#    E.g., english_audio_48khz.wav, italian_audio_16khz.wav
# 4. Run: python test_stt_tts_language_pipeline.py


# #!/usr/bin/env python3
# """
# Test FastRTC Whisper with real audio recording
# """

# import numpy as np
# import pyaudio
# import time
# from fastrtc import get_stt_model

# print("=== FastRTC Whisper Audio Test ===")

# # Load both models
# print("\n1. Loading STT models...")
# models = {}

# try:
#     models['moonshine'] = get_stt_model("moonshine/base")
#     print("✅ Moonshine loaded")
# except Exception as e:
#     print(f"❌ Moonshine failed: {e}")

# try:
#     models['whisper'] = get_stt_model("whisper/base")
#     print("✅ Whisper loaded")
# except Exception as e:
#     print(f"❌ Whisper failed: {e}")

# # Function to record audio
# def record_audio(duration=3, sample_rate=16000):
#     p = pyaudio.PyAudio()
#     stream = p.open(format=pyaudio.paFloat32,
#                     channels=1,
#                     rate=sample_rate,
#                     input=True,
#                     frames_per_buffer=1024)
    
#     print(f"🎤 Recording for {duration} seconds...")
#     frames = []
#     for _ in range(0, int(sample_rate / 1024 * duration)):
#         data = stream.read(1024)
#         frames.append(data)
    
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
    
#     # Convert to numpy array
#     audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
#     return audio_data

# # Test with multiple recordings
# test_cases = [
#     ("English", "Say something in English (e.g., 'Hello, my name is...')"),
#     ("Italian", "Say something in Italian (e.g., 'Ciao, mi chiamo...')"),
#     ("Mixed", "Mix English and Italian (e.g., 'Hello, ciao, come stai?')"),
# ]

# for test_name, instruction in test_cases:
#     print(f"\n\nTest: {test_name}")
#     print(f"Instructions: {instruction}")
#     print("Press Enter when ready to record...")
#     input()
    
#     audio = record_audio(duration=4)
#     print(f"Recorded {len(audio)} samples, RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    
#     # Test with each model
#     for model_name, model in models.items():
#         if model:
#             try:
#                 # Test raw audio
#                 result = model.stt((16000, audio))
#                 print(f"\n{model_name} result: '{result}'")
                
#                 # Also test with your preprocessing
#                 # Simulate your bluetooth preprocessing
#                 processed = audio.copy()
                
#                 # Your preprocessing steps
#                 max_abs = np.max(np.abs(processed))
#                 if max_abs > 1.0 and max_abs > 1e-6:
#                     processed = processed / max_abs
                
#                 # Add diff
#                 if len(processed) > 1:
#                     diff = np.diff(processed, prepend=processed[0])
#                     processed = processed + 0.15 * diff
                
#                 # Test processed audio
#                 result_processed = model.stt((16000, processed))
#                 print(f"{model_name} (processed): '{result_processed}'")
                
#             except Exception as e:
#                 print(f"{model_name} error: {e}")

# # Test different audio chunk sizes
# print("\n\n=== Testing chunk sizes ===")
# print("Say 'Hello world' or 'Ciao mondo' when ready...")
# input()

# chunk_sizes = [0.5, 1.0, 2.0, 3.0]
# for chunk_size in chunk_sizes:
#     print(f"\nRecording {chunk_size}s chunk...")
#     audio = record_audio(duration=chunk_size)
    
#     for model_name, model in models.items():
#         if model:
#             try:
#                 result = model.stt((16000, audio))
#                 print(f"{model_name} ({chunk_size}s): '{result}'")
#             except Exception as e:
#                 print(f"{model_name} ({chunk_size}s) error: {e}")

# # Test whisper configuration
# print("\n\n=== Checking Whisper configuration ===")
# if 'whisper' in models and models['whisper']:
#     whisper_model = models['whisper']
    
#     # Check if we can access model properties
#     if hasattr(whisper_model, 'model'):
#         print(f"Whisper model type: {type(whisper_model.model)}")
        
#         # Try to find language settings
#         for attr in dir(whisper_model.model):
#             if 'lang' in attr.lower() or 'language' in attr.lower():
#                 print(f"Found attribute: {attr}")
#                 try:
#                     value = getattr(whisper_model.model, attr)
#                     print(f"  Value: {value}")
#                 except:
#                     pass

# print("\n=== Test Complete ===")

# #!/usr/bin/env python3
# """
# Test FastRTC Whisper with real audio recording
# """

# import numpy as np
# import pyaudio
# import time
# from fastrtc import get_stt_model

# print("=== FastRTC Whisper Audio Test ===")

# # Load both models
# print("\n1. Loading STT models...")
# models = {}

# try:
#     models['moonshine'] = get_stt_model("moonshine/base")
#     print("✅ Moonshine loaded")
# except Exception as e:
#     print(f"❌ Moonshine failed: {e}")

# try:
#     models['whisper'] = get_stt_model("whisper/base")
#     print("✅ Whisper loaded")
# except Exception as e:
#     print(f"❌ Whisper failed: {e}")

# # Function to record audio
# def record_audio(duration=3, sample_rate=16000):
#     p = pyaudio.PyAudio()
#     stream = p.open(format=pyaudio.paFloat32,
#                     channels=1,
#                     rate=sample_rate,
#                     input=True,
#                     frames_per_buffer=1024)
    
#     print(f"🎤 Recording for {duration} seconds...")
#     frames = []
#     for _ in range(0, int(sample_rate / 1024 * duration)):
#         data = stream.read(1024)
#         frames.append(data)
    
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
    
#     # Convert to numpy array
#     audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
#     return audio_data

# # Test with multiple recordings
# test_cases = [
#     ("English", "Say something in English (e.g., 'Hello, my name is...')"),
#     ("Italian", "Say something in Italian (e.g., 'Ciao, mi chiamo...')"),
#     ("Mixed", "Mix English and Italian (e.g., 'Hello, ciao, come stai?')"),
# ]

# for test_name, instruction in test_cases:
#     print(f"\n\nTest: {test_name}")
#     print(f"Instructions: {instruction}")
#     print("Press Enter when ready to record...")
#     input()
    
#     audio = record_audio(duration=4)
#     print(f"Recorded {len(audio)} samples, RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    
#     # Test with each model
#     for model_name, model in models.items():
#         if model:
#             try:
#                 # Test raw audio
#                 result = model.stt((16000, audio))
#                 print(f"\n{model_name} result: '{result}'")
                
#                 # Also test with your preprocessing
#                 # Simulate your bluetooth preprocessing
#                 processed = audio.copy()
                
#                 # Your preprocessing steps
#                 max_abs = np.max(np.abs(processed))
#                 if max_abs > 1.0 and max_abs > 1e-6:
#                     processed = processed / max_abs
                
#                 # Add diff
#                 if len(processed) > 1:
#                     diff = np.diff(processed, prepend=processed[0])
#                     processed = processed + 0.15 * diff
                
#                 # Test processed audio
#                 result_processed = model.stt((16000, processed))
#                 print(f"{model_name} (processed): '{result_processed}'")
                
#             except Exception as e:
#                 print(f"{model_name} error: {e}")

# # Test different audio chunk sizes
# print("\n\n=== Testing chunk sizes ===")
# print("Say 'Hello world' or 'Ciao mondo' when ready...")
# input()

# chunk_sizes = [0.5, 1.0, 2.0, 3.0]
# for chunk_size in chunk_sizes:
#     print(f"\nRecording {chunk_size}s chunk...")
#     audio = record_audio(duration=chunk_size)
    
#     for model_name, model in models.items():
#         if model:
#             try:
#                 result = model.stt((16000, audio))
#                 print(f"{model_name} ({chunk_size}s): '{result}'")
#             except Exception as e:
#                 print(f"{model_name} ({chunk_size}s) error: {e}")

# # Test whisper configuration
# print("\n\n=== Checking Whisper configuration ===")
# if 'whisper' in models and models['whisper']:
#     whisper_model = models['whisper']
    
#     # Check if we can access model properties
#     if hasattr(whisper_model, 'model'):
#         print(f"Whisper model type: {type(whisper_model.model)}")
        
#         # Try to find language settings
#         for attr in dir(whisper_model.model):
#             if 'lang' in attr.lower() or 'language' in attr.lower():
#                 print(f"Found attribute: {attr}")
#                 try:
#                     value = getattr(whisper_model.model, attr)
#                     print(f"  Value: {value}")
#                 except:
#                     pass

# print("\n=== Test Complete ===")


# #!/usr/bin/env python3
# """
# Debug audio pipeline with exact same parameters as main app
# """

# import numpy as np
# import time
# from pywhispercpp.model import Model
# from fastrtc import get_stt_model
# import sys

# print("=== Audio Pipeline Debug Test ===")

# # Test 1: Check available STT models in FastRTC
# print("\n1. Testing FastRTC STT model support...")
# models_to_test = [
#     "moonshine/base",
#     "moonshine/tiny", 
#     "whisper/base",
#     "whisper/tiny",
#     "whisper/small",
#     "whisper",
#     "whisper-base",
#     "whisper_base"
# ]

# available_models = []
# for model_name in models_to_test:
#     try:
#         print(f"  Trying {model_name}... ", end="")
#         stt = get_stt_model(model_name)
#         print("✅ Success!")
#         available_models.append(model_name)
#         del stt  # Clean up
#     except Exception as e:
#         print(f"❌ Failed: {str(e)[:50]}...")

# print(f"\nAvailable models: {available_models}")

# # Test 2: Compare Moonshine vs Whisper with same audio
# print("\n2. Comparing Moonshine vs Whisper outputs...")

# # Create test audio samples
# test_samples = {
#     "silence": np.zeros(16000 * 2, dtype=np.float32),
#     "quiet_noise": np.random.normal(0, 0.001, 16000 * 2).astype(np.float32),
#     "medium_noise": np.random.normal(0, 0.01, 16000 * 2).astype(np.float32),
#     "loud_noise": np.random.normal(0, 0.1, 16000 * 2).astype(np.float32),
# }

# # Simulate your preprocessing
# class BluetoothAudioSimulator:
#     def __init__(self):
#         self.noise_floor = 0.001
        
#     def preprocess_bluetooth_audio(self, audio_data):
#         if isinstance(audio_data, tuple):
#             sample_rate, audio_array = audio_data
#         else:
#             sample_rate, audio_array = 16000, audio_data
            
#         if audio_array.dtype != np.float32:
#             audio_array = audio_array.astype(np.float32)
        
#         # Your preprocessing logic
#         max_abs = np.max(np.abs(audio_array))
#         if max_abs > 1.0 and max_abs > 1e-6:
#             audio_array = audio_array / max_abs
        
#         # Add diff like in your code
#         if len(audio_array) > 1:
#             diff = np.diff(audio_array, prepend=audio_array[0])
#             audio_array = audio_array + 0.15 * diff
        
#         # Apply gain like in your code
#         current_rms = np.sqrt(np.mean(audio_array**2)) if audio_array.size > 0 else 0.0
#         if current_rms > self.noise_floor * 1.5 and current_rms > 1e-6:
#             target_rms = 0.08
#             if current_rms < target_rms * 0.4:
#                 gain = min(2.5, (target_rms * 0.6 / current_rms) if current_rms > 1e-6 else 1.0)
#                 audio_array = audio_array * gain
        
#         return sample_rate, audio_array

# bluetooth_processor = BluetoothAudioSimulator()

# # Test 3: Load both models if available
# moonshine_model = None
# whisper_model = None

# if "moonshine/base" in available_models:
#     try:
#         moonshine_model = get_stt_model("moonshine/base")
#         print("\n✅ Moonshine loaded")
#     except:
#         pass

# # Try pywhispercpp since FastRTC whisper might not work
# print("\nLoading pywhispercpp model...")
# try:
#     whisper_model = Model('base', n_threads=6)
#     print("✅ Pywhispercpp loaded")
    
#     # Create wrapper to match your code
#     class WhisperSTTWrapper:
#         def __init__(self, model):
#             self.model = model
            
#         def stt(self, audio_tuple):
#             sample_rate, audio_array = audio_tuple
#             if audio_array.dtype != np.float32:
#                 audio_array = audio_array.astype(np.float32)
#             segments = self.model.transcribe(audio_array)
#             text = ''.join([segment.text for segment in segments])
#             return text
    
#     whisper_wrapper = WhisperSTTWrapper(whisper_model)
# except Exception as e:
#     print(f"❌ Failed to load pywhispercpp: {e}")
#     whisper_wrapper = None

# # Test 4: Process each sample through both models
# print("\n3. Testing audio samples through pipeline...")
# print("-" * 60)

# for sample_name, audio in test_samples.items():
#     print(f"\nTesting: {sample_name}")
#     print(f"  Original - Size: {audio.size}, RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    
#     # Process through your bluetooth preprocessor
#     sr, processed = bluetooth_processor.preprocess_bluetooth_audio((16000, audio))
#     print(f"  Processed - RMS: {np.sqrt(np.mean(processed**2)):.6f}")
    
#     # Test with Moonshine
#     if moonshine_model:
#         try:
#             moonshine_result = moonshine_model.stt((sr, processed))
#             print(f"  Moonshine: '{moonshine_result}'")
#         except Exception as e:
#             print(f"  Moonshine error: {e}")
    
#     # Test with Whisper
#     if whisper_wrapper:
#         try:
#             whisper_result = whisper_wrapper.stt((sr, processed))
#             print(f"  Whisper: '{whisper_result}'")
#         except Exception as e:
#             print(f"  Whisper error: {e}")

# # Test 5: Simulate your exact audio chunk parameters
# print("\n\n4. Testing with your exact chunk parameters...")
# print("-" * 60)

# # Your parameters
# AUDIO_CHUNK_DURATION = 2.0  # seconds
# SAMPLE_RATE = 16000
# CHUNK_SAMPLES = int(SAMPLE_RATE * AUDIO_CHUNK_DURATION)

# print(f"Chunk duration: {AUDIO_CHUNK_DURATION}s")
# print(f"Samples per chunk: {CHUNK_SAMPLES}")

# # Create a chunk with brief speech in the middle
# chunk = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
# # Add a short burst of "speech" (sine wave) in the middle
# speech_start = CHUNK_SAMPLES // 3
# speech_duration = SAMPLE_RATE // 2  # 0.5 seconds
# t = np.linspace(0, 0.5, speech_duration)
# speech_signal = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone
# chunk[speech_start:speech_start + speech_duration] = speech_signal.astype(np.float32)

# print(f"\nTesting chunk with simulated speech burst:")
# sr, processed_chunk = bluetooth_processor.preprocess_bluetooth_audio((SAMPLE_RATE, chunk))

# if moonshine_model:
#     moon_result = moonshine_model.stt((sr, processed_chunk))
#     print(f"  Moonshine: '{moon_result}'")

# if whisper_wrapper:
#     whisper_result = whisper_wrapper.stt((sr, processed_chunk))
#     print(f"  Whisper: '{whisper_result}'")

# # Test 6: Check if it's a audio normalization issue
# print("\n\n5. Testing different audio levels...")
# print("-" * 60)

# base_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
# levels = [0.001, 0.01, 0.1, 0.3, 0.5, 1.0]

# for level in levels:
#     test_signal = base_signal * level
#     sr, processed = bluetooth_processor.preprocess_bluetooth_audio((16000, test_signal))
    
#     print(f"\nLevel {level}:")
#     print(f"  Original RMS: {np.sqrt(np.mean(test_signal**2)):.6f}")
#     print(f"  Processed RMS: {np.sqrt(np.mean(processed**2)):.6f}")
    
#     if whisper_wrapper:
#         result = whisper_wrapper.stt((sr, processed))
#         print(f"  Whisper: '{result}'")

# print("\n=== Test Complete ===")


# #!/usr/bin/env python3
# """
# Test script to debug Whisper STT issues
# """

# import numpy as np
# import time
# from pywhispercpp.model import Model
# import pyaudio

# print("=== Whisper STT Debug Test ===")

# # Test 1: Initialize Whisper
# print("\n1. Testing Whisper initialization...")
# try:
#     whisper_model = Model('base', n_threads=6)
#     print("✅ Whisper model loaded successfully")
# except Exception as e:
#     print(f"❌ Failed to load Whisper: {e}")
#     exit(1)

# # Test 2: Test with silence
# print("\n2. Testing with silence...")
# silence = np.zeros(16000 * 3, dtype=np.float32)  # 3 seconds of silence
# try:
#     segments = whisper_model.transcribe(silence)
#     text = ''.join([segment.text for segment in segments])
#     print(f"Silence transcription: '{text}'")
#     print(f"Number of segments: {len(segments)}")
#     if segments:
#         print(f"First segment: {segments[0].__dict__ if hasattr(segments[0], '__dict__') else segments[0]}")
# except Exception as e:
#     print(f"❌ Error transcribing silence: {e}")

# # Test 3: Test with noise
# print("\n3. Testing with random noise...")
# noise = np.random.normal(0, 0.01, 16000 * 3).astype(np.float32)
# try:
#     segments = whisper_model.transcribe(noise)
#     text = ''.join([segment.text for segment in segments])
#     print(f"Noise transcription: '{text}'")
# except Exception as e:
#     print(f"❌ Error transcribing noise: {e}")

# # Test 4: Record actual audio
# print("\n4. Testing with real audio (speak after the beep)...")
# print("Press Enter to start recording (3 seconds)...")
# input()

# p = pyaudio.PyAudio()
# stream = p.open(format=pyaudio.paFloat32,
#                 channels=1,
#                 rate=16000,
#                 input=True,
#                 frames_per_buffer=1024)

# print("🎤 Recording... (speak now!)")
# frames = []
# for _ in range(0, int(16000 / 1024 * 3)):
#     data = stream.read(1024)
#     frames.append(data)

# stream.stop_stream()
# stream.close()
# p.terminate()

# # Convert to numpy array
# audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
# print(f"Recorded {len(audio_data)} samples")

# # Test transcription
# try:
#     segments = whisper_model.transcribe(audio_data)
#     text = ''.join([segment.text for segment in segments])
#     print(f"Your speech: '{text}'")
    
#     # Debug segment structure
#     if segments:
#         print("\nSegment details:")
#         for i, seg in enumerate(segments[:3]):  # First 3 segments
#             print(f"Segment {i}: {seg.text}")
#             if hasattr(seg, '__dict__'):
#                 print(f"  Attributes: {list(seg.__dict__.keys())}")
# except Exception as e:
#     print(f"❌ Error transcribing real audio: {e}")
#     import traceback
#     traceback.print_exc()

# # Test 5: Test language detection
# print("\n5. Testing language detection...")
# test_phrases = [
#     "Hello, my name is John",
#     "Ciao, mi chiamo Giovanni",
#     "Buongiorno, come stai?"
# ]

# for phrase in test_phrases:
#     print(f"\nTesting phrase: '{phrase}'")
#     # We'll need to use text-to-speech or pre-recorded audio for this
#     # For now, just show what we'd detect from text
#     italian_words = ['ciao', 'grazie', 'sono', 'mi chiamo', 'buongiorno', 'come stai']
#     detected = "it" if any(word in phrase.lower() for word in italian_words) else "en"
#     print(f"Detected language: {detected}")

# print("\n=== Test Complete ===")



# Memory System Debugging and Testing Script
# #!/usr/bin/env python3
# """
# Test script for A-MEM memory system debugging
# Tests both name extraction and persistence issues
# """

# import asyncio
# import sys
# import os
# from pathlib import Path
# from datetime import datetime, timedelta, timezone
# import json
# import time

# # Add the parent directory to the path to import our modules
# sys.path.append(str(Path(__file__).parent))

# from a_mem.memory_system import AgenticMemorySystem, MemoryNote
# from a_mem.retrievers import ChromaRetriever
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Test configuration - same as main app
# AMEM_LLM_MODEL = "llama3.2:3b" 
# AMEM_EMBEDDER_MODEL = "nomic-embed-text"
# USER_ID = "amem_voice_user_01"

# class TestAMEMSystem:
#     def __init__(self):
#         self.results = []
        
#     def log_result(self, test_name, passed, details=""):
#         result = {
#             "test": test_name,
#             "passed": passed,
#             "details": details,
#             "timestamp": datetime.now().isoformat()
#         }
#         self.results.append(result)
#         print(f"{'✅' if passed else '❌'} {test_name}: {details}")
    
#     async def test_1_chromadb_persistence(self):
#         """Test if ChromaDB actually persists data"""
#         print("\n=== Test 1: ChromaDB Persistence ===")
        
#         # Create a test collection
#         from chromadb import Client
#         client = Client()
        
#         # Clean up any existing test collection
#         try:
#             client.delete_collection("test_persistence")
#         except:
#             pass
            
#         # Create collection and add data
#         collection = client.create_collection("test_persistence")
#         test_id = "test_doc_1"
#         test_content = "My name is TestUser"
#         test_metadata = {"name": "TestUser", "timestamp": "20240101"}
        
#         collection.add(
#             documents=[test_content],
#             metadatas=[test_metadata],
#             ids=[test_id]
#         )
        
#         # Retrieve immediately
#         results = collection.get(ids=[test_id])
#         immediate_found = len(results['ids']) > 0
#         self.log_result("ChromaDB immediate retrieval", immediate_found, 
#                        f"Found: {results['ids'] if immediate_found else 'None'}")
        
#         # Create a new client instance (simulating restart)
#         client2 = Client()
#         collection2 = client2.get_collection("test_persistence")
#         results2 = collection2.get(ids=[test_id])
#         persist_found = len(results2['ids']) > 0
#         self.log_result("ChromaDB persistence after new client", persist_found,
#                        f"Found: {results2['ids'] if persist_found else 'None'}")
        
#         # Clean up
#         client2.delete_collection("test_persistence")
        
#     async def test_2_amem_memory_storage(self):
#         """Test A-MEM system memory storage"""
#         print("\n=== Test 2: A-MEM Memory Storage ===")
        
#         # Initialize A-MEM system
#         amem = AgenticMemorySystem(
#             model_name='all-MiniLM-L6-v2',
#             llm_backend="ollama",
#             llm_model=AMEM_LLM_MODEL,
#             evo_threshold=50
#         )
        
#         # Add a memory
#         test_content = "User: My name is TestUser\nAssistant: Nice to meet you TestUser!"
#         memory_id = amem.add_note(
#             content=test_content,
#             tags=["personal_info", "conversation"],
#             category="personal_info",
#             timestamp=datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
#         )
        
#         self.log_result("A-MEM add_note", bool(memory_id), f"ID: {memory_id}")
        
#         # Check in-memory storage
#         in_memory = memory_id in amem.memories
#         self.log_result("A-MEM in-memory storage", in_memory, 
#                        f"Memory count: {len(amem.memories)}")
        
#         # Check ChromaDB storage
#         chroma_results = amem.retriever.collection.get(ids=[memory_id])
#         in_chromadb = len(chroma_results['ids']) > 0
#         self.log_result("A-MEM ChromaDB storage", in_chromadb,
#                        f"Found in ChromaDB: {in_chromadb}")
        
#         # Test search
#         search_results = amem.search_agentic("TestUser", k=5)
#         search_found = any("TestUser" in r.get('content', '') for r in search_results)
#         self.log_result("A-MEM search", search_found, 
#                        f"Results: {len(search_results)}")
        
#         return amem, memory_id
    
#     async def test_3_name_extraction(self):
#         """Test name extraction logic"""
#         print("\n=== Test 3: Name Extraction ===")
        
#         # Import the memory manager
#         from voice_assistant_with_memory_V3 import AMemMemoryManager
        
#         manager = AMemMemoryManager(USER_ID)
        
#         # Test cases
#         test_cases = [
#             ("My name is John", "John", True),
#             ("I'm Sarah", "Sarah", True),
#             ("Call me Mike", "Mike", True),
#             ("Shocked! I'm glad to meet you", None, False),  # Should NOT extract "glad"
#             ("Actually, my name is Jane", "Jane", True),
#             ("I'm going to the store", None, False),  # Should NOT extract "going"
#             ("User: My name is Bob\nAssistant: Hello Bob!", "Bob", True),
#             ("Assistant: Shocked! I'm happy", None, False),  # Should NOT extract from assistant
#         ]
        
#         for text, expected_name, should_extract in test_cases:
#             extracted = manager.extract_user_name(text)
#             passed = (extracted == expected_name) if should_extract else (extracted is None)
#             self.log_result(f"Extract from '{text[:30]}...'", passed,
#                            f"Expected: {expected_name}, Got: {extracted}")
    
#     async def test_4_memory_persistence(self):
#         """Test memory persistence across sessions"""
#         print("\n=== Test 4: Memory Persistence Across Sessions ===")
        
#         # Session 1: Create and store memories
#         print("\n--- Session 1: Creating memories ---")
#         from voice_assistant_with_memory_V3 import AMemMemoryManager
#         from a_mem.memory_system import MemoryNote # Import MemoryNote if not already
        
#         manager1 = AMemMemoryManager(USER_ID)
#         await manager1.start_background_processor()
        
#         # --- Create a timestamp guaranteed to be later ---
#         # Get current time, then advance it by a few minutes for the test note
#         # This ensures its timestamp string will be definitively later.
#         now = datetime.now(timezone.utc)
#         persist_user_time = now + timedelta(minutes=5) # Arbitrarily 5 minutes later
#         persist_user_timestamp_str = persist_user_time.strftime("%Y%m%d%H%M")

#         # When calling add_to_memory_smart, it eventually calls amem_system.add_note.
#         # We need to ensure this specific timestamp is used.
#         # The add_note method in AgenticMemorySystem takes 'time' as an optional kwarg,
#         # which it then passes to MemoryNote as 'timestamp'.
#         # AMemMemoryManager._store_memory_background calls amem_system.add_note
#         # and passes its own timestamp. We need to override this for the test.
#         #
#         # A simpler way for the test, if AMemMemoryManager.add_to_memory_smart
#         # doesn't allow passing a specific timestamp down, is to directly use
#         # manager1.amem_system.add_note for this critical test entry.

#         # Let's try direct add for full control in the test
#         user_text_persist = "My name is PersistUser"
#         assistant_text_persist = "Nice to meet you PersistUser!"
#         manager1.amem_system.add_note(
#             content=f"User: {user_text_persist}\nAssistant: {assistant_text_persist}",
#             tags=["personal_info", "conversation"],
#             category="personal_info",
#             timestamp=persist_user_timestamp_str # Use our controlled, later timestamp
#         )
#         # Also update the local cache in manager1 as add_to_memory_smart would have
#         manager1.update_local_cache(user_text_persist, "personal_info", is_current_turn_extraction=True)


#         # For the coffee note, let its timestamp be natural
#         await manager1.add_to_memory_smart(
#             "I like cold brew coffee",
#             "That's great! Cold brew is delicious."
#         )
        
#         # Wait for background processing (especially if add_to_memory_smart was used for coffee)
#         await asyncio.sleep(3) # Increased sleep just in case background processing is slow
        
#         self.log_result("Session 1 - Name in cache", 
#                        manager1.memory_cache.get('user_name') == 'PersistUser',
#                        f"Cache: {manager1.memory_cache.get('user_name')}")
        
#         amem_count = len(manager1.amem_system.memories)
#         self.log_result("Session 1 - A-MEM memories created", 
#                        amem_count > 0, # Should be at least 2 new ones + old ones
#                        f"Count: {amem_count}")
        
#         collection = manager1.amem_system.retriever.collection
#         all_docs = collection.get()
#         chromadb_count = len(all_docs['ids']) if 'ids' in all_docs else 0
#         self.log_result("Session 1 - ChromaDB documents", 
#                        chromadb_count > 0,
#                        f"Count: {chromadb_count}")
        
#         await manager1.shutdown()
        
#         # Session 2: Create new manager and check persistence
#         print("\n--- Session 2: Loading memories ---")
#         manager2 = AMemMemoryManager(USER_ID)
        
#         loaded_name = manager2.memory_cache.get('user_name')
#         self.log_result("Session 2 - Name loaded from storage", 
#                        loaded_name == 'PersistUser',
#                        f"Loaded: {loaded_name}")
        
#         amem_count2 = len(manager2.amem_system.memories)
#         self.log_result("Session 2 - A-MEM memories loaded", 
#                        amem_count2 > 0, # Should be same or more than amem_count
#                        f"Count: {amem_count2}")
        
#         search_result = await manager2.search_memories_smart("what is my name")
#         name_found = "PersistUser" in search_result
#         self.log_result("Session 2 - Search finds name", 
#                        name_found,
#                        f"Result: {search_result[:100]}...")
        
#         await manager2.shutdown()
    
#     async def test_5_chromadb_metadata(self):
#         """Test ChromaDB metadata handling"""
#         print("\n=== Test 5: ChromaDB Metadata Handling ===")
        
#         from a_mem.retrievers import ChromaRetriever
        
#         retriever = ChromaRetriever(collection_name="test_metadata")
        
#         # Add document with complex metadata
#         test_metadata = {
#             "id": "test_123",
#             "content": "Test content with user name John",
#             "keywords": ["name", "john"],
#             "tags": ["personal", "test"],
#             "timestamp": "20240101120000",
#             "links": ["link1", "link2"]
#         }
        
#         retriever.add_document(
#             document="Test content with user name John",
#             metadata=test_metadata,
#             doc_id="test_123"
#         )
        
#         # Retrieve and check
#         results = retriever.collection.get(ids=["test_123"])
#         if results['ids']:
#             metadata = results['metadatas'][0]
            
#             # Check each field
#             for key in ["content", "timestamp"]:
#                 self.log_result(f"Metadata {key} preserved", 
#                                metadata.get(key) == test_metadata[key],
#                                f"{key}: {metadata.get(key)}")
            
#             # Check JSON fields
#             for key in ["keywords", "tags", "links"]:
#                 stored_value = json.loads(metadata.get(key, "[]"))
#                 expected_value = test_metadata[key]
#                 self.log_result(f"Metadata {key} preserved", 
#                                stored_value == expected_value,
#                                f"{key}: {stored_value}")
        
#         # Cleanup
#         retriever.client.delete_collection("test_metadata")
    
#     async def run_all_tests(self):
#         """Run all tests"""
#         print("🧪 Starting A-MEM System Tests")
#         print("=" * 50)
        
#         await self.test_1_chromadb_persistence()
#         await self.test_2_amem_memory_storage()
#         await self.test_3_name_extraction()
#         await self.test_4_memory_persistence()
#         await self.test_5_chromadb_metadata()
        
#         # Summary
#         print("\n" + "=" * 50)
#         print("📊 Test Summary:")
#         passed = sum(1 for r in self.results if r['passed'])
#         total = len(self.results)
#         print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
        
#         # Failed tests
#         failed = [r for r in self.results if not r['passed']]
#         if failed:
#             print("\n❌ Failed tests:")
#             for r in failed:
#                 print(f"  - {r['test']}: {r['details']}")

# if __name__ == "__main__":
#     tester = TestAMEMSystem()
#     asyncio.run(tester.run_all_tests())
    
# SOUND TEST
# #!/usr/bin/env python3
# import sounddevice as sd
# import numpy as np
# from fastrtc import get_stt_model

# # Load STT model
# stt_model = get_stt_model("moonshine/base")

# print("🎤 Recording 3 seconds of audio... Say 'Hello world'")
# sample_rate = 16000
# duration = 3  # seconds

# # Record audio
# audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
# sd.wait()  # Wait for recording to complete

# print("🔄 Processing with STT...")
# result = stt_model.stt((sample_rate, audio_data.flatten()))
# print(f"STT Result: '{result}'")

# # Test TTS model
# #!/usr/bin/env python3
# """
# Comprehensive Audio Pipeline Test for FastRTC Voice Assistant
# Tests STT, TTS, Audio Processing, and WebRTC connectivity
# """

# import sys
# import time
# import os
# import numpy as np
# import asyncio
# import threading
# from pathlib import Path
# import logging
# from datetime import datetime
# import wave
# import tempfile

# # FastRTC imports
# try:
#     from fastrtc import get_stt_model, get_tts_model, KokoroTTSOptions
#     FASTRTC_AVAILABLE = True
# except ImportError as e:
#     print(f"❌ FastRTC not available: {e}")
#     FASTRTC_AVAILABLE = False

# # Audio processing imports
# try:
#     import sounddevice as sd
#     SOUNDDEVICE_AVAILABLE = True
# except ImportError:
#     print("⚠️ sounddevice not available - audio device tests will be skipped")
#     SOUNDDEVICE_AVAILABLE = False

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class AudioPipelineTest:
#     def __init__(self):
#         self.test_results = {}
#         self.stt_model = None
#         self.tts_model = None
#         self.sample_rate = 16000
        
#     def log_test_result(self, test_name: str, success: bool, message: str = "", details: dict = None):
#         """Log test results for summary"""
#         self.test_results[test_name] = {
#             'success': success,
#             'message': message,
#             'details': details or {},
#             'timestamp': datetime.now().isoformat()
#         }
#         status = "✅" if success else "❌"
#         print(f"{status} {test_name}: {message}")
#         if details:
#             for key, value in details.items():
#                 print(f"   └─ {key}: {value}")

#     def test_fastrtc_availability(self):
#         """Test if FastRTC is properly installed and accessible"""
#         print("\n🔍 Testing FastRTC Availability...")
        
#         if not FASTRTC_AVAILABLE:
#             self.log_test_result("FastRTC Import", False, "FastRTC module not available")
#             return False
            
#         try:
#             # Test basic imports
#             from fastrtc import Stream, ReplyOnPause, AlgoOptions, SileroVadOptions
#             from fastrtc.utils import AdditionalOutputs
#             self.log_test_result("FastRTC Import", True, "All FastRTC components imported successfully")
#             return True
#         except Exception as e:
#             self.log_test_result("FastRTC Import", False, f"Failed to import FastRTC components: {e}")
#             return False

#     def test_stt_model_loading(self):
#         """Test STT model loading and basic functionality"""
#         print("\n🎤 Testing STT Model...")
        
#         try:
#             # Load STT model
#             start_time = time.time()
#             self.stt_model = get_stt_model("moonshine/base")
#             load_time = time.time() - start_time
            
#             self.log_test_result("STT Model Loading", True, f"Loaded in {load_time:.2f}s", 
#                                {"model_type": "moonshine/base", "load_time": f"{load_time:.2f}s"})
#             return True
            
#         except Exception as e:
#             self.log_test_result("STT Model Loading", False, f"Failed to load STT model: {e}")
#             return False

#     def test_tts_model_loading(self):
#         """Test TTS model loading and voice availability"""
#         print("\n🔊 Testing TTS Model...")
        
#         try:
#             # Load TTS model
#             start_time = time.time()
#             self.tts_model = get_tts_model("kokoro")
#             load_time = time.time() - start_time
            
#             # Check available voices
#             available_voices = []
#             if hasattr(self.tts_model, 'model') and hasattr(self.tts_model.model, 'voices'):
#                 available_voices = list(getattr(self.tts_model.model, 'voices', []))
            
#             self.log_test_result("TTS Model Loading", True, f"Loaded in {load_time:.2f}s", 
#                                {"model_type": "kokoro", "load_time": f"{load_time:.2f}s", 
#                                 "available_voices": len(available_voices),
#                                 "sample_voices": available_voices[:5]})
#             return True
            
#         except Exception as e:
#             self.log_test_result("TTS Model Loading", False, f"Failed to load TTS model: {e}")
#             return False

#     def test_stt_with_synthetic_audio(self):
#         """Test STT with synthetic audio data"""
#         print("\n🎵 Testing STT with Synthetic Audio...")
        
#         if not self.stt_model:
#             self.log_test_result("STT Synthetic Test", False, "STT model not loaded")
#             return False
            
#         try:
#             # Generate synthetic audio (sine wave)
#             duration = 2.0  # seconds
#             frequency = 440  # Hz (A note)
#             samples = int(self.sample_rate * duration)
#             t = np.linspace(0, duration, samples, False)
#             synthetic_audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
#             # Test STT
#             start_time = time.time()
#             result = self.stt_model.stt((self.sample_rate, synthetic_audio))
#             process_time = time.time() - start_time
            
#             self.log_test_result("STT Synthetic Test", True, f"Processed synthetic audio", 
#                                {"result": result or "(empty)", "process_time": f"{process_time:.3f}s",
#                                 "audio_duration": f"{duration}s", "audio_samples": samples})
#             return True
            
#         except Exception as e:
#             self.log_test_result("STT Synthetic Test", False, f"STT failed with synthetic audio: {e}")
#             return False

#     def test_stt_with_silence(self):
#         """Test STT with silence"""
#         print("\n🔇 Testing STT with Silence...")
        
#         if not self.stt_model:
#             self.log_test_result("STT Silence Test", False, "STT model not loaded")
#             return False
            
#         try:
#             # Generate silence
#             duration = 1.0
#             samples = int(self.sample_rate * duration)
#             silence = np.zeros(samples, dtype=np.float32)
            
#             # Test STT
#             start_time = time.time()
#             result = self.stt_model.stt((self.sample_rate, silence))
#             process_time = time.time() - start_time
            
#             self.log_test_result("STT Silence Test", True, f"Processed silence", 
#                                {"result": result or "(empty)", "process_time": f"{process_time:.3f}s"})
#             return True
            
#         except Exception as e:
#             self.log_test_result("STT Silence Test", False, f"STT failed with silence: {e}")
#             return False

#     def test_tts_basic_synthesis(self):
#         """Test basic TTS synthesis"""
#         print("\n🗣️ Testing TTS Basic Synthesis...")
        
#         if not self.tts_model:
#             self.log_test_result("TTS Basic Test", False, "TTS model not loaded")
#             return False
            
#         try:
#             test_text = "Hello, this is a test of the text to speech system."
            
#             # Test different voices
#             voices_to_test = ["af_heart", "af_alloy", "af_bella", None]  # None = default
            
#             for voice in voices_to_test:
#                 try:
#                     options_params = {"speed": 1.0, "lang": "en-us"}
#                     if voice:
#                         options_params["voice"] = voice
                    
#                     tts_options = KokoroTTSOptions(**options_params)
                    
#                     # Test synthesis
#                     start_time = time.time()
#                     audio_chunks = []
#                     total_samples = 0
                    
#                     for chunk in self.tts_model.stream_tts_sync(test_text, tts_options):
#                         if isinstance(chunk, tuple) and len(chunk) == 2:
#                             sr, audio_data = chunk
#                             if isinstance(audio_data, np.ndarray) and audio_data.size > 0:
#                                 audio_chunks.append(audio_data)
#                                 total_samples += audio_data.size
#                         elif isinstance(chunk, np.ndarray) and chunk.size > 0:
#                             audio_chunks.append(chunk)
#                             total_samples += chunk.size
                    
#                     synthesis_time = time.time() - start_time
                    
#                     if audio_chunks:
#                         full_audio = np.concatenate(audio_chunks)
#                         duration = len(full_audio) / self.sample_rate
                        
#                         self.log_test_result(f"TTS Voice Test ({voice or 'default'})", True, 
#                                            f"Generated {duration:.2f}s of audio", 
#                                            {"synthesis_time": f"{synthesis_time:.3f}s",
#                                             "total_samples": total_samples,
#                                             "audio_duration": f"{duration:.2f}s",
#                                             "chunks_generated": len(audio_chunks)})
                        
#                         # Test only one working voice for now
#                         return True
#                     else:
#                         self.log_test_result(f"TTS Voice Test ({voice or 'default'})", False, 
#                                            "No audio generated")
                        
#                 except Exception as voice_error:
#                     self.log_test_result(f"TTS Voice Test ({voice or 'default'})", False, 
#                                        f"Voice failed: {voice_error}")
#                     continue
            
#             return False
            
#         except Exception as e:
#             self.log_test_result("TTS Basic Test", False, f"TTS synthesis failed: {e}")
#             return False

#     def test_audio_device_availability(self):
#         """Test audio device availability"""
#         print("\n🎧 Testing Audio Device Availability...")
        
#         if not SOUNDDEVICE_AVAILABLE:
#             self.log_test_result("Audio Device Test", False, "sounddevice module not available")
#             return False
            
#         try:
#             # Get available audio devices
#             devices = sd.query_devices()
            
#             input_devices = []
#             output_devices = []
            
#             for i, device in enumerate(devices):
#                 if device['max_input_channels'] > 0:
#                     input_devices.append({"id": i, "name": device['name'], "channels": device['max_input_channels']})
#                 if device['max_output_channels'] > 0:
#                     output_devices.append({"id": i, "name": device['name'], "channels": device['max_output_channels']})
            
#             # Test default devices
#             try:
#                 default_input = sd.query_devices(kind='input')
#                 default_output = sd.query_devices(kind='output')
                
#                 self.log_test_result("Audio Device Test", True, 
#                                    f"Found {len(input_devices)} input and {len(output_devices)} output devices",
#                                    {"default_input": default_input['name'],
#                                     "default_output": default_output['name'],
#                                     "sample_input_devices": [d['name'] for d in input_devices[:3]],
#                                     "sample_output_devices": [d['name'] for d in output_devices[:3]]})
#                 return True
                
#             except Exception as default_error:
#                 self.log_test_result("Audio Device Test", False, f"No default audio devices: {default_error}")
#                 return False
                
#         except Exception as e:
#             self.log_test_result("Audio Device Test", False, f"Failed to query audio devices: {e}")
#             return False

#     def test_webrtc_compatibility(self):
#         """Test WebRTC related functionality"""
#         print("\n🌐 Testing WebRTC Compatibility...")
        
#         try:
#             # Test creating basic FastRTC components
#             from fastrtc import Stream, ReplyOnPause, AlgoOptions, SileroVadOptions
#             from fastrtc.utils import AdditionalOutputs
            
#             # Test creating algorithm options
#             algo_options = AlgoOptions(
#                 audio_chunk_duration=3.5,
#                 started_talking_threshold=0.2,
#                 speech_threshold=0.15
#             )
            
#             # Test creating VAD options
#             vad_options = SileroVadOptions(
#                 threshold=0.5,
#                 min_speech_duration_ms=300,
#                 min_silence_duration_ms=3000,
#                 speech_pad_ms=300,
#                 window_size_samples=1024
#             )
            
#             # Test creating AdditionalOutputs
#             additional_outputs = AdditionalOutputs()
            
#             self.log_test_result("WebRTC Compatibility", True, "All WebRTC components created successfully",
#                                {"algo_options": "✓", "vad_options": "✓", "additional_outputs": "✓"})
#             return True
            
#         except Exception as e:
#             self.log_test_result("WebRTC Compatibility", False, f"WebRTC component creation failed: {e}")
#             return False

#     def test_bluetooth_audio_processor(self):
#         """Test the Bluetooth audio processor from your code"""
#         print("\n📱 Testing Bluetooth Audio Processor...")
        
#         try:
#             # Import your BluetoothAudioProcessor
#             sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
#             # Create a simple version for testing
#             class TestBluetoothAudioProcessor:
#                 def __init__(self):
#                     self.noise_floor = None
#                     self.calibration_frames = 0
#                     self.min_calibration_frames = 15
                    
#                 def preprocess_bluetooth_audio(self, audio_data):
#                     if isinstance(audio_data, tuple) and len(audio_data) == 2:
#                         sample_rate, audio_array = audio_data
#                     else:
#                         return 16000, np.array([], dtype=np.float32)
                    
#                     if audio_array.size == 0:
#                         return sample_rate, np.array([], dtype=np.float32)
                    
#                     # Simple preprocessing
#                     if audio_array.dtype != np.float32:
#                         audio_array = audio_array.astype(np.float32)
                    
#                     # Normalize
#                     max_abs = np.max(np.abs(audio_array))
#                     if max_abs > 1.0 and max_abs > 1e-6:
#                         audio_array = audio_array / max_abs
                    
#                     return sample_rate, audio_array
            
#             processor = TestBluetoothAudioProcessor()
            
#             # Test with different audio types
#             test_cases = [
#                 ("Empty array", (16000, np.array([], dtype=np.float32))),
#                 ("Normal audio", (16000, np.random.randn(1600).astype(np.float32) * 0.5)),
#                 ("Loud audio", (16000, np.random.randn(1600).astype(np.float32) * 2.0)),
#                 ("Integer audio", (16000, (np.random.randn(1600) * 32767).astype(np.int16))),
#             ]
            
#             results = {}
#             for test_name, test_audio in test_cases:
#                 try:
#                     sr, processed = processor.preprocess_bluetooth_audio(test_audio)
#                     results[test_name] = {
#                         "success": True,
#                         "output_samples": processed.size,
#                         "max_amplitude": np.max(np.abs(processed)) if processed.size > 0 else 0
#                     }
#                 except Exception as e:
#                     results[test_name] = {"success": False, "error": str(e)}
            
#             success_count = sum(1 for r in results.values() if r.get("success", False))
            
#             self.log_test_result("Bluetooth Audio Processor", success_count == len(test_cases),
#                                f"Passed {success_count}/{len(test_cases)} test cases",
#                                results)
            
#             return success_count == len(test_cases)
            
#         except Exception as e:
#             self.log_test_result("Bluetooth Audio Processor", False, f"Processor test failed: {e}")
#             return False

#     def test_end_to_end_pipeline(self):
#         """Test the complete STT -> TTS pipeline"""
#         print("\n🔄 Testing End-to-End Pipeline...")
        
#         if not (self.stt_model and self.tts_model):
#             self.log_test_result("End-to-End Pipeline", False, "STT or TTS model not loaded")
#             return False
            
#         try:
#             # Create test audio with known content
#             test_phrase = "Hello world"
            
#             # Step 1: Generate audio with TTS
#             print("   Step 1: Generating audio with TTS...")
#             tts_options = KokoroTTSOptions(speed=1.0, lang="en-us", voice="af_heart")
            
#             audio_chunks = []
#             for chunk in self.tts_model.stream_tts_sync(test_phrase, tts_options):
#                 if isinstance(chunk, tuple) and len(chunk) == 2:
#                     sr, audio_data = chunk
#                     if isinstance(audio_data, np.ndarray) and audio_data.size > 0:
#                         audio_chunks.append(audio_data)
#                 elif isinstance(chunk, np.ndarray) and chunk.size > 0:
#                     audio_chunks.append(chunk)
            
#             if not audio_chunks:
#                 self.log_test_result("End-to-End Pipeline", False, "TTS generated no audio")
#                 return False
            
#             generated_audio = np.concatenate(audio_chunks)
            
#             # Step 2: Process audio back through STT
#             print("   Step 2: Processing audio through STT...")
#             stt_result = self.stt_model.stt((self.sample_rate, generated_audio))
            
#             # Step 3: Analyze results
#             print("   Step 3: Analyzing results...")
            
#             # Check if we got any result
#             has_result = bool(stt_result and stt_result.strip())
            
#             # Basic similarity check (simple word matching)
#             original_words = set(test_phrase.lower().split())
#             result_words = set((stt_result or "").lower().split())
#             word_overlap = len(original_words.intersection(result_words))
#             similarity_score = word_overlap / len(original_words) if original_words else 0
            
#             self.log_test_result("End-to-End Pipeline", has_result,
#                                f"Pipeline completed with {'good' if similarity_score > 0.5 else 'poor'} results",
#                                {"original_text": test_phrase,
#                                 "stt_result": stt_result or "(empty)",
#                                 "audio_duration": f"{len(generated_audio) / self.sample_rate:.2f}s",
#                                 "word_similarity": f"{similarity_score:.2f}",
#                                 "has_result": has_result})
            
#             return has_result
            
#         except Exception as e:
#             self.log_test_result("End-to-End Pipeline", False, f"Pipeline test failed: {e}")
#             return False

#     def test_callback_function_format(self):
#         """Test the callback function format expected by FastRTC"""
#         print("\n⚙️ Testing Callback Function Format...")
        
#         try:
#             def test_callback(audio_data_tuple):
#                 """Test callback that mimics your voice assistant callback"""
#                 sample_rate = 16000
#                 silent_samples = int(sample_rate * 0.02)  # 20ms
#                 silent_audio = np.zeros(silent_samples, dtype=np.float32)
                
#                 try:
#                     # Check input format
#                     if not isinstance(audio_data_tuple, tuple) or len(audio_data_tuple) != 2:
#                         yield (sample_rate, silent_audio), None
#                         return
                    
#                     input_sr, audio_array = audio_data_tuple
                    
#                     if not isinstance(audio_array, np.ndarray) or audio_array.size == 0:
#                         yield (sample_rate, silent_audio), None
#                         return
                    
#                     # Simple echo - return the same audio
#                     yield (input_sr, audio_array), None
                    
#                 except Exception as e:
#                     print(f"Callback error: {e}")
#                     yield (sample_rate, silent_audio), None
            
#             # Test the callback with different inputs
#             test_inputs = [
#                 (16000, np.random.randn(160).astype(np.float32)),  # Normal
#                 (16000, np.array([], dtype=np.float32)),           # Empty
#                 (22050, np.random.randn(220).astype(np.float32)),  # Different SR
#             ]
            
#             callback_results = {}
#             for i, test_input in enumerate(test_inputs):
#                 try:
#                     results = list(test_callback(test_input))
#                     callback_results[f"test_{i}"] = {
#                         "success": True,
#                         "output_count": len(results),
#                         "output_format_valid": all(isinstance(r, tuple) and len(r) == 2 for r in results)
#                     }
#                 except Exception as e:
#                     callback_results[f"test_{i}"] = {"success": False, "error": str(e)}
            
#             success_count = sum(1 for r in callback_results.values() if r.get("success", False))
            
#             self.log_test_result("Callback Function Format", success_count == len(test_inputs),
#                                f"Callback passed {success_count}/{len(test_inputs)} tests",
#                                callback_results)
            
#             return success_count == len(test_inputs)
            
#         except Exception as e:
#             self.log_test_result("Callback Function Format", False, f"Callback test failed: {e}")
#             return False

#     def run_all_tests(self):
#         """Run all tests and provide a summary"""
#         print("🚀 Starting Comprehensive Audio Pipeline Tests")
#         print("=" * 80)
        
#         test_functions = [
#             self.test_fastrtc_availability,
#             self.test_stt_model_loading,
#             self.test_tts_model_loading,
#             self.test_stt_with_synthetic_audio,
#             self.test_stt_with_silence,
#             self.test_tts_basic_synthesis,
#             self.test_audio_device_availability,
#             self.test_webrtc_compatibility,
#             self.test_bluetooth_audio_processor,
#             self.test_callback_function_format,
#             self.test_end_to_end_pipeline,
#         ]
        
#         start_time = time.time()
        
#         for test_func in test_functions:
#             try:
#                 test_func()
#             except Exception as e:
#                 test_name = test_func.__name__.replace('test_', '').replace('_', ' ').title()
#                 self.log_test_result(test_name, False, f"Test crashed: {e}")
            
#             time.sleep(0.5)  # Brief pause between tests
        
#         total_time = time.time() - start_time
        
#         # Print summary
#         print("\n" + "=" * 80)
#         print("📊 TEST SUMMARY")
#         print("=" * 80)
        
#         passed = sum(1 for result in self.test_results.values() if result['success'])
#         total = len(self.test_results)
        
#         print(f"Total Tests: {total}")
#         print(f"Passed: {passed}")
#         print(f"Failed: {total - passed}")
#         print(f"Success Rate: {(passed/total*100):.1f}%")
#         print(f"Total Time: {total_time:.2f}s")
        
#         print("\n📋 DETAILED RESULTS:")
#         for test_name, result in self.test_results.items():
#             status = "✅" if result['success'] else "❌"
#             print(f"{status} {test_name}: {result['message']}")
            
#             if not result['success']:
#                 print("   🔍 RECOMMENDATION:")
#                 if "fastrtc" in test_name.lower():
#                     print("   └─ Check FastRTC installation: pip install fastrtc")
#                 elif "stt" in test_name.lower():
#                     print("   └─ STT model issue - check model files and dependencies")
#                 elif "tts" in test_name.lower():
#                     print("   └─ TTS model issue - check Kokoro model installation")
#                 elif "audio device" in test_name.lower():
#                     print("   └─ Audio hardware issue - check microphone/speaker connections")
#                 elif "webrtc" in test_name.lower():
#                     print("   └─ WebRTC issue - check browser compatibility and network")
#                 else:
#                     print("   └─ Check error message above for specific details")
        
#         print("\n🎯 NEXT STEPS:")
#         if passed == total:
#             print("✅ All tests passed! Your audio pipeline should be working.")
#             print("   If you're still having issues, check:")
#             print("   • Browser permissions for microphone access")
#             print("   • Network connectivity for WebRTC")
#             print("   • Browser console for JavaScript errors")
#         else:
#             print("❌ Some tests failed. Focus on fixing the failed components first.")
#             print("   Priority order: FastRTC → Models → Audio Devices → Pipeline")
        
#         return self.test_results

# if __name__ == "__main__":
#     tester = AudioPipelineTest()
#     results = tester.run_all_tests()
    
#     # Exit with error code if any tests failed
#     failed_tests = sum(1 for result in results.values() if not result['success'])
#     sys.exit(failed_tests)