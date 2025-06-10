#!/usr/bin/env python3
"""
Gradio4.py - Enhanced FastRTC Voice Assistant
Uses 100% refactored components with sophisticated error handling and memory tracking
"""

import sys
import time
import os
import numpy as np
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import traceback
from datetime import datetime, timezone

# FastRTC imports
from fastrtc import (
    ReplyOnPause, Stream, AlgoOptions, SileroVadOptions, audio_to_bytes
)
from fastrtc.utils import AdditionalOutputs

# Colorama for enhanced terminal output
from colorama import init, Fore, Back, Style
init(autoreset=True)

# Add src to path for refactored components
sys.path.insert(0, 'src')

# Import 100% refactored components
from src.core.voice_assistant import VoiceAssistant
from src.core.interfaces import AudioData, TranscriptionResult
from src.integration.fastrtc_bridge import FastRTCBridge
from src.utils.logging import setup_logging, get_logger
from src.config.settings import DEFAULT_LANGUAGE


# Set up logging
setup_logging()
logger = get_logger(__name__)

# Constants
AUDIO_SAMPLE_RATE = 16000
MINIMAL_SILENT_FRAME_DURATION_MS = 20
MINIMAL_SILENT_SAMPLES = int(AUDIO_SAMPLE_RATE * (MINIMAL_SILENT_FRAME_DURATION_MS / 1000.0))
SILENT_AUDIO_CHUNK_ARRAY = np.zeros(MINIMAL_SILENT_SAMPLES, dtype=np.float32)
SILENT_AUDIO_FRAME_TUPLE = (AUDIO_SAMPLE_RATE, SILENT_AUDIO_CHUNK_ARRAY)
EMPTY_AUDIO_YIELD_OUTPUT = (SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs())


class ColorfulLogger:
    """Enhanced colorful logging system with categorized output."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def print_colorful(self, message: str, color=Fore.WHITE, style=Style.NORMAL):
        """Print colorful messages with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"{Fore.CYAN}[{timestamp}]{Style.RESET_ALL} {style}{color}{message}{Style.RESET_ALL}")
        self.logger.info(message)
    
    def print_user(self, message: str):
        """Print user messages in blue."""
        self.print_colorful(f"👤 User: {message}", Fore.BLUE, Style.BRIGHT)
    
    def print_assistant(self, message: str):
        """Print assistant messages in green."""
        self.print_colorful(f"🤖 Assistant: {message}", Fore.GREEN, Style.BRIGHT)
    
    def print_memory(self, message: str):
        """Print memory-related messages in magenta."""
        self.print_colorful(f"🧠 Memory: {message}", Fore.MAGENTA, Style.BRIGHT)
    
    def print_language(self, message: str):
        """Print language-related messages in yellow."""
        self.print_colorful(f"🌍 Language: {message}", Fore.YELLOW, Style.BRIGHT)
    
    def print_tts(self, message: str):
        """Print TTS-related messages in cyan."""
        self.print_colorful(f"🎤 TTS: {message}", Fore.CYAN, Style.BRIGHT)
    
    def print_error(self, message: str):
        """Print error messages in red."""
        self.print_colorful(f"❌ Error: {message}", Fore.RED, Style.BRIGHT)
    
    def print_success(self, message: str):
        """Print success messages in green."""
        self.print_colorful(f"✅ Success: {message}", Fore.GREEN, Style.BRIGHT)
    
    def print_info(self, message: str):
        """Print info messages in white."""
        self.print_colorful(f"ℹ️  Info: {message}", Fore.WHITE, Style.NORMAL)
    
    def print_timing(self, message: str):
        """Print timing messages in light blue."""
        self.print_colorful(f"⏱️  Timing: {message}", Fore.LIGHTBLUE_EX, Style.NORMAL)


class EnhancedErrorHandler:
    """Sophisticated error handling with graceful fallbacks."""
    
    def __init__(self, logger: ColorfulLogger):
        self.logger = logger
    
    def handle_audio_processing_error(self, error: Exception) -> Tuple[int, np.ndarray]:
        """Handle audio processing errors with graceful fallback."""
        self.logger.print_error(f"Audio processing failed: {error}")
        return SILENT_AUDIO_FRAME_TUPLE
    
    def handle_stt_error(self, error: Exception) -> str:
        """Handle STT errors with graceful fallback."""
        self.logger.print_error(f"STT failed: {error}")
        return ""
    
    def handle_llm_timeout(self, timeout_duration: float) -> str:
        """Handle LLM timeout with fallback response."""
        self.logger.print_error(f"LLM request timed out after {timeout_duration:.2f}s")
        return "Let me think about that and get back to you quickly."
    
    def handle_llm_error(self, error: Exception) -> str:
        """Handle LLM errors with fallback response."""
        self.logger.print_error(f"LLM error: {error}")
        return "I encountered an error processing your request."
    
    def handle_tts_error(self, voice_id: str, error: Exception):
        """Handle TTS errors and log for fallback chain."""
        self.logger.print_error(f"TTS failed with voice '{voice_id}': {error}")
    
    def handle_critical_error(self, error: Exception):
        """Handle critical errors with full traceback."""
        self.logger.print_error(f"Critical error: {error}")
        self.logger.print_error(f"Traceback: {traceback.format_exc()}")


class MemoryTracker:
    """Enhanced memory tracking with before/after context comparison."""
    
    def __init__(self, logger: ColorfulLogger):
        self.logger = logger
        self.memory_operations = []
    
    async def track_memory_context_before_llm(self, voice_assistant: VoiceAssistant) -> Optional[Any]:
        """Track memory context before LLM call."""
        try:
            if hasattr(voice_assistant, 'memory_manager') and voice_assistant.memory_manager:
                context = await voice_assistant.memory_manager.get_conversation_context()
                if context:
                    self.logger.print_memory(f"Context loaded ({len(str(context))} chars)")
                    return context
                else:
                    self.logger.print_memory("No context available")
            else:
                self.logger.print_memory("No memory manager available")
        except Exception as e:
            self.logger.print_memory(f"Context load failed: {e}")
        return None
    
    async def track_memory_context_after_llm(self, voice_assistant: VoiceAssistant, context_before: Optional[Any]):
        """Track memory context after LLM call and compare changes."""
        try:
            if hasattr(voice_assistant, 'memory_manager') and voice_assistant.memory_manager:
                context_after = await voice_assistant.memory_manager.get_conversation_context()
                if context_before and context_after:
                    if str(context_before) != str(context_after):
                        self.logger.print_memory("Context updated with new conversation")
                    else:
                        self.logger.print_memory("Context unchanged")
                elif context_after and not context_before:
                    self.logger.print_memory("New context created")
        except Exception as e:
            self.logger.print_memory(f"Context check failed: {e}")
    
    def log_memory_operation(self, operation_type: str, details: Dict[str, Any]):
        """Log memory operations for debugging."""
        operation = {
            'timestamp': datetime.now().isoformat(),
            'operation_type': operation_type,
            'details': details
        }
        self.memory_operations.append(operation)


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self, logger: ColorfulLogger):
        self.logger = logger
        self.turn_times = []
    
    def start_turn_timer(self) -> float:
        """Start timing a conversation turn."""
        return time.monotonic()
    
    def track_turn_processing_time(self, start_time: float) -> float:
        """Track and log turn processing time."""
        processing_time = time.monotonic() - start_time
        self.turn_times.append(processing_time)
        self.logger.print_timing(f"Response generated in {processing_time:.2f}s")
        return processing_time
    
    def get_audio_duration(self, audio_data_tuple: tuple) -> float:
        """Calculate audio duration for logging."""
        if isinstance(audio_data_tuple, tuple) and len(audio_data_tuple) == 2:
            sample_rate, audio_array = audio_data_tuple
            if hasattr(audio_array, 'size') and sample_rate > 0:
                return audio_array.size / sample_rate
        return 0.0
    
    def create_optimized_chunks(self, audio_array: np.ndarray, chunk_size: int = 1024):
        """Create optimized audio chunks for streaming."""
        for i in range(0, audio_array.size, chunk_size):
            mini_chunk = audio_array[i:i+chunk_size]
            if mini_chunk.size > 0:
                yield mini_chunk


class EnhancedVoiceAssistant:
    """Enhanced wrapper around VoiceAssistant with sophisticated error handling and memory tracking."""
    
    def __init__(self):
        """Initialize enhanced voice assistant with all tracking components."""
        self.voice_assistant = VoiceAssistant()
        self.logger = ColorfulLogger()
        self.error_handler = EnhancedErrorHandler(self.logger)
        self.memory_tracker = MemoryTracker(self.logger)
        self.performance_monitor = PerformanceMonitor(self.logger)
        
        # Language names mapping
        self.lang_names = {
            'a': 'American English', 'b': 'British English', 'i': 'Italian', 
            'e': 'Spanish', 'f': 'French', 'p': 'Portuguese', 
            'j': 'Japanese', 'z': 'Chinese', 'h': 'Hindi'
        }
    
    async def initialize_async(self):
        """Initialize async components."""
        self.logger.print_info("Initializing enhanced voice assistant...")
        await self.voice_assistant.initialize_async()
        
        # Check memory manager initialization
        if hasattr(self.voice_assistant, 'memory_manager') and self.voice_assistant.memory_manager:
            self.logger.print_memory("Memory manager initialized successfully")
        else:
            self.logger.print_memory("Memory manager not available")
        
        self.logger.print_success("Enhanced voice assistant ready")
    
    async def cleanup_async(self):
        """Clean up async resources."""
        self.logger.print_info("Cleaning up enhanced voice assistant...")
        await self.voice_assistant.cleanup_async()
        self.logger.print_info("Enhanced voice assistant cleanup complete")
    
    def process_audio_safely(self, audio_data_tuple: tuple) -> Tuple[int, np.ndarray]:
        """Process audio bypassing audio processor - pass raw audio directly to STT."""
        try:
            duration = self.performance_monitor.get_audio_duration(audio_data_tuple)
            self.logger.print_info(f"Processing audio ({duration:.1f}s)")
            
            # CRITICAL FIX: Bypass audio processor - use raw audio directly like working callback
            if isinstance(audio_data_tuple, tuple) and len(audio_data_tuple) == 2:
                sample_rate, raw_audio_array = audio_data_tuple
                
                # CRITICAL FIX: Proper audio array handling for shape (1, N) -> (N,)
                if isinstance(raw_audio_array, np.ndarray) and len(raw_audio_array.shape) > 1:
                    if raw_audio_array.shape[0] == 1:  # Shape like (1, 384000) - single channel
                        audio_array = raw_audio_array[0]  # Extract the actual audio data: (384000,)
                    elif raw_audio_array.shape[1] == 1:  # Shape like (384000, 1) - single column
                        audio_array = raw_audio_array[:, 0]  # Extract: (384000,)
                    elif raw_audio_array.shape[1] > 1:  # Multi-channel like (384000, 2)
                        audio_array = np.mean(raw_audio_array, axis=1)  # Convert to mono: (384000,)
                    else:
                        audio_array = raw_audio_array.flatten()
                else:
                    audio_array = raw_audio_array if isinstance(raw_audio_array, np.ndarray) else np.array(raw_audio_array, dtype=np.float32)
                
                # Convert to float32 (same as working callback)
                if isinstance(audio_array, np.ndarray):
                    if audio_array.dtype == np.int16:
                        # Convert int16 to float32 properly
                        audio_array = audio_array.astype(np.float32) / 32768.0
                    else:
                        audio_array = audio_array.astype(np.float32)
            else:
                audio_array = np.array([], dtype=np.float32)
                sample_rate = 16000
            
            if audio_array.size == 0:
                self.logger.print_info("Empty audio array, yielding silent frame")
                return SILENT_AUDIO_FRAME_TUPLE
                
            return sample_rate, audio_array
            
        except Exception as e:
            return self.error_handler.handle_audio_processing_error(e)
    
    async def transcribe_with_tracking(self, audio_data: Tuple[int, np.ndarray]) -> str:
        """Transcribe audio using refactored STT engine with enhanced tracking."""
        sample_rate, audio_array = audio_data
        
        if audio_array.size == 0:
            return ""
        
        try:
            # Use refactored STT engine directly
            transcription_result = await self.voice_assistant.stt_engine.transcribe_with_sample_rate(
                audio_array, sample_rate
            )
            
            if hasattr(transcription_result, 'text'):
                user_text = transcription_result.text.strip()
                # Use refactored language detection
                detected_language = self.voice_assistant.detect_language_from_text(user_text)
                
                # Enhanced language tracking
                self._track_language_change(detected_language)
                
                return user_text
            else:
                return str(transcription_result).strip()
                
        except Exception as e:
            return self.error_handler.handle_stt_error(e)
    
    def _track_language_change(self, detected_language: str):
        """Track language changes with enhanced logging."""
        if detected_language != self.voice_assistant.current_language:
            self.voice_assistant.current_language = detected_language
            lang_name = self.lang_names.get(detected_language, 'Unknown')
            self.logger.print_language(f"Switched to {lang_name} ({detected_language})")
        else:
            lang_name = self.lang_names.get(detected_language, 'Unknown')
            self.logger.print_language(f"Confirmed {lang_name} ({detected_language})")
    
    async def get_llm_response_with_monitoring(self, user_text: str) -> str:
        """Get LLM response using refactored service with enhanced monitoring."""
        start_turn_time = self.performance_monitor.start_turn_timer()
        
        try:
            # Enhanced memory tracking before LLM call
            memory_context_before = await self.memory_tracker.track_memory_context_before_llm(
                self.voice_assistant
            )
            
            # Use refactored LLM service through VoiceAssistant
            response = await self.voice_assistant.get_llm_response_smart(user_text)
            
            # Track performance
            self.performance_monitor.track_turn_processing_time(start_turn_time)
            
            # Enhanced memory tracking after LLM call
            await self.memory_tracker.track_memory_context_after_llm(
                self.voice_assistant, memory_context_before
            )
            
            self.logger.print_assistant(response)
            return response
            
        except TimeoutError:
            return self.error_handler.handle_llm_timeout(time.monotonic() - start_turn_time)
        except Exception as e:
            return self.error_handler.handle_llm_error(e)
    
    def stream_tts_with_fallback(self, text: str, language: str):
        """Stream TTS using refactored engine with enhanced fallback logic."""
        
        # Language conversion for TTS
        if len(language) > 1:
            kokoro_language = self.voice_assistant.convert_to_kokoro_language(language)
            self.logger.print_language(f"Converted to Kokoro format: {kokoro_language}")
            language = kokoro_language
        
        # Use refactored voice mapping
        voices_to_try = self.voice_assistant.get_voices_for_language(language)
        voices_to_try.append(None)  # Default voice fallback
        
        self.logger.print_tts(f"Using language '{language}' with {len(voices_to_try)} voice options")
        
        for voice_id in voices_to_try:
            try:
                self.logger.print_tts(f"Trying voice: {voice_id or 'default'}")
                
                chunk_count = 0
                total_samples = 0
                
                # Use refactored TTS streaming
                for current_sr, current_chunk_array in self.voice_assistant.stream_tts_synthesis(
                    text, voice_id, language
                ):
                    if isinstance(current_chunk_array, np.ndarray) and current_chunk_array.size > 0:
                        chunk_count += 1
                        total_samples += current_chunk_array.size
                        
                        # Enhanced chunking for performance
                        for mini_chunk in self.performance_monitor.create_optimized_chunks(current_chunk_array):
                            yield (current_sr, mini_chunk), AdditionalOutputs()
                
                self.logger.print_success(
                    f"TTS completed with voice '{voice_id or 'default'}' "
                    f"({chunk_count} chunks, {total_samples} samples)"
                )
                return  # Success, exit the loop
                
            except Exception as e:
                self.error_handler.handle_tts_error(voice_id, e)
                continue
        
        # All TTS attempts failed
        self.logger.print_error("All TTS attempts failed")
        yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()


# Global instances
enhanced_assistant: Optional[EnhancedVoiceAssistant] = None
main_event_loop: Optional[asyncio.AbstractEventLoop] = None
async_worker_thread: Optional[threading.Thread] = None


def enhanced_voice_assistant_callback_rt(audio_data_tuple: tuple):
    """
    Enhanced callback using 100% refactored components with sophisticated tracking.
    Maintains exact same functionality as original callback but uses refactored architecture.
    """
    global enhanced_assistant
    
    if not enhanced_assistant:
        print("❌ Enhanced voice assistant not initialized!")
        yield EMPTY_AUDIO_YIELD_OUTPUT
        return
    
    try:
        # 1. Audio Processing (100% Refactored)
        sample_rate, audio_array = enhanced_assistant.process_audio_safely(audio_data_tuple)
        
        if audio_array.size == 0:
            yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
            return
        
        # 2. STT with Enhanced Error Handling (100% Refactored)
        user_text = run_coro_from_sync_thread_with_timeout(
            enhanced_assistant.transcribe_with_tracking((sample_rate, audio_array)),
            timeout=8.0
        )
        
        if not user_text.strip():
            yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
            return
        
        enhanced_assistant.logger.print_user(user_text)
        
        # 3. LLM Response with Enhanced Monitoring (100% Refactored)
        assistant_response_text = run_coro_from_sync_thread_with_timeout(
            enhanced_assistant.get_llm_response_with_monitoring(user_text),
            timeout=10.0
        )
        
        # 4. TTS Streaming with Enhanced Fallback (100% Refactored)
        additional_outputs = AdditionalOutputs()
        
        for audio_chunk, outputs in enhanced_assistant.stream_tts_with_fallback(
            assistant_response_text, enhanced_assistant.voice_assistant.current_language
        ):
            yield audio_chunk, outputs
    
    except Exception as e:
        enhanced_assistant.error_handler.handle_critical_error(e)
        yield EMPTY_AUDIO_YIELD_OUTPUT


def setup_enhanced_async_environment():
    """Setup enhanced async environment with 100% refactored components."""
    global main_event_loop, enhanced_assistant, async_worker_thread
    
    # Create enhanced voice assistant
    enhanced_assistant = EnhancedVoiceAssistant()
    enhanced_assistant.logger.print_info("Creating enhanced voice assistant instance...")

    def run_async_loop_in_thread():
        global main_event_loop, enhanced_assistant
        main_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(main_event_loop)
        
        if enhanced_assistant:
            enhanced_assistant.logger.print_info("Initializing enhanced async components...")
            main_event_loop.run_until_complete(enhanced_assistant.initialize_async())
        else:
            print("❌ Enhanced voice assistant instance is None in async thread")
            return

        try:
            main_event_loop.run_forever()
        except KeyboardInterrupt:
            enhanced_assistant.logger.print_info("Async loop interrupted")
        finally:
            if enhanced_assistant and main_event_loop and not main_event_loop.is_closed():
                enhanced_assistant.logger.print_info("Cleaning up enhanced assistant resources...")
                main_event_loop.run_until_complete(enhanced_assistant.cleanup_async())
            if main_event_loop and not main_event_loop.is_closed():
                 main_event_loop.close()
            enhanced_assistant.logger.print_info("Enhanced async event loop closed")

    async_worker_thread = threading.Thread(target=run_async_loop_in_thread, daemon=True, name="EnhancedAsyncWorkerThread")
    async_worker_thread.start()

    # Wait for initialization
    for _ in range(100):
        if main_event_loop and main_event_loop.is_running() and enhanced_assistant:
            enhanced_assistant.logger.print_success("Enhanced async environment ready")
            return
        time.sleep(0.1)
    
    print("⚠️ Enhanced async environment setup timeout")


def run_coro_from_sync_thread_with_timeout(coro, timeout: float = 4.0) -> any:
    """Run coroutine with timeout using enhanced error handling."""
    global main_event_loop
    if main_event_loop and main_event_loop.is_running():
        start_time = time.monotonic()
        
        future = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(coro, timeout=timeout),
            main_event_loop
        )
        try:
            result = future.result(timeout=timeout + 1.0)
            elapsed = time.monotonic() - start_time
            return result
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start_time
            if enhanced_assistant:
                enhanced_assistant.logger.print_error(f"Async task timed out after {timeout}s")
            raise TimeoutError(f"Operation timed out after {timeout}s")
        except Exception as e:
            elapsed = time.monotonic() - start_time
            if enhanced_assistant:
                enhanced_assistant.logger.print_error(f"Async task error: {e}")
            return "I encountered an error processing your request."
    else:
        if enhanced_assistant:
            enhanced_assistant.logger.print_error("Event loop not available")
        return "My processing system is not ready."


def setup_websocket_bridge():
    global websocket_bridge
    if WEBSOCKET_AVAILABLE:
        try:
            websocket_bridge = initialize_websocket_bridge(port=7861)
            websocket_bridge.start_in_background()
            print(f"{Fore.GREEN}🌐 WebSocket bridge started on ws://localhost:7861{Style.RESET_ALL}")
            print(f"{Fore.CYAN}💻 Modern UI available at http://localhost:3005{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}❌ WebSocket bridge failed: {e}{Style.RESET_ALL}")
            return False
    return False


def main():
    """Enhanced main function with 100% refactored architecture."""
    print(f"{Fore.CYAN}{Style.BRIGHT}🚀 Gradio4 - Enhanced FastRTC Voice Assistant{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✨ Using 100% Refactored Components with Enhanced Features{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 70}{Style.RESET_ALL}")
    
    # Setup enhanced async environment
    setup_enhanced_async_environment()
    
    # Create and configure FastRTC bridge
    bridge = FastRTCBridge()
    bridge.create_stream(enhanced_voice_assistant_callback_rt)
    
    try:
        print(f"{Fore.GREEN}{Style.BRIGHT}💡 Test the enhanced assistant with:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   • 'My name is [Your Name]'{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   • 'What is my name?'{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   • 'I like [something]'{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   • 'What do you know about me?'{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   • 'Tell me about myself'{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   • Ask questions in multiple languages{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'=' * 70}{Style.RESET_ALL}")
        
        # Launch the enhanced stream
        bridge.launch_stream()
        
    except KeyboardInterrupt:
        if enhanced_assistant:
            enhanced_assistant.logger.print_info("Shutting down enhanced assistant...")
    except Exception as e:
        if enhanced_assistant:
            enhanced_assistant.logger.print_error(f"Launch error: {e}")
        else:
            print(f"❌ Launch error: {e}")
    finally:
        # Enhanced cleanup
        if enhanced_assistant:
            enhanced_assistant.logger.print_info("Enhanced session ended")
        print(f"\n{Fore.CYAN}{Style.BRIGHT}👋 Enhanced Voice Assistant Session Complete{Style.RESET_ALL}")


if __name__ == "__main__":
    main()