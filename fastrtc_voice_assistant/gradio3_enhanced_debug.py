#!/usr/bin/env python3
"""
Enhanced Gradio3 with Memory Debugging

This version includes comprehensive memory debugging to identify why memory
functionality differs from gradio2.py.
"""

import sys
import time
import os
import numpy as np
import asyncio
import threading
from pathlib import Path
from fastrtc import (
    ReplyOnPause,
    Stream,
    get_tts_model,
    AlgoOptions,
    SileroVadOptions,
    KokoroTTSOptions,
    audio_to_bytes,
)
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import deque

# Add src to path for imports
sys.path.insert(0, 'src')

from src.core.voice_assistant import VoiceAssistant
from src.core.interfaces import AudioData, TranscriptionResult
from src.utils.logging import setup_logging, get_logger
from src.config.settings import DEFAULT_LANGUAGE
import traceback
from fastrtc.utils import AdditionalOutputs
from src.integration.fastrtc_bridge import FastRTCBridge

# Import our memory debugger
from debug_memory_comparison import MemoryDebugger

AUDIO_SAMPLE_RATE = 16000
MINIMAL_SILENT_FRAME_DURATION_MS = 20
MINIMAL_SILENT_SAMPLES = int(AUDIO_SAMPLE_RATE * (MINIMAL_SILENT_FRAME_DURATION_MS / 1000.0))
SILENT_AUDIO_CHUNK_ARRAY = np.zeros(MINIMAL_SILENT_SAMPLES, dtype=np.float32)
SILENT_AUDIO_FRAME_TUPLE = (AUDIO_SAMPLE_RATE, SILENT_AUDIO_CHUNK_ARRAY)
EMPTY_AUDIO_YIELD_OUTPUT = (SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs())

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Global instances
voice_assistant: Optional[VoiceAssistant] = None
main_event_loop: Optional[asyncio.AbstractEventLoop] = None
async_worker_thread: Optional[threading.Thread] = None
memory_debugger = MemoryDebugger()

def print_status(message):
    """Enhanced print_status with memory debugging."""
    timestamp = time.strftime("%H:%M:%S")
    logger.info(f"{message}")
    
    # Log memory-related messages
    if any(keyword in message.lower() for keyword in ['memory', 'remember', 'context', 'conversation']):
        memory_debugger.log_memory_operation('status_message', {'message': message})

def voice_assistant_callback_rt(audio_data_tuple: tuple):
    """Enhanced callback with comprehensive memory debugging."""
    global voice_assistant

    logger.info(f"🔧 ENHANCED DEBUG: Callback entry - audio_data_tuple type={type(audio_data_tuple)}")
    
    if not voice_assistant:
        logger.error("🔧 ENHANCED DEBUG: voice_assistant is None!")
        memory_debugger.log_memory_operation('callback_error', {'error': 'voice_assistant is None'})
        yield EMPTY_AUDIO_YIELD_OUTPUT
        return

    try:
        # Log callback start
        memory_debugger.log_memory_operation('callback_start', {
            'timestamp': datetime.now().isoformat(),
            'audio_data_type': str(type(audio_data_tuple))
        })

        # Process audio (same as original)
        if isinstance(audio_data_tuple, tuple) and len(audio_data_tuple) == 2:
            sample_rate, raw_audio_array = audio_data_tuple
            
            if isinstance(raw_audio_array, np.ndarray) and len(raw_audio_array.shape) > 1:
                if raw_audio_array.shape[0] == 1:
                    audio_array = raw_audio_array[0]
                elif raw_audio_array.shape[1] == 1:
                    audio_array = raw_audio_array[:, 0]
                elif raw_audio_array.shape[1] > 1:
                    audio_array = np.mean(raw_audio_array, axis=1)
                else:
                    audio_array = raw_audio_array.flatten()
            else:
                audio_array = raw_audio_array if isinstance(raw_audio_array, np.ndarray) else np.array(raw_audio_array, dtype=np.float32)
            
            if isinstance(audio_array, np.ndarray):
                if audio_array.dtype == np.int16:
                    audio_array = audio_array.astype(np.float32) / 32768.0
                else:
                    audio_array = audio_array.astype(np.float32)
        else:
            audio_array = np.array([], dtype=np.float32)
            sample_rate = 16000

        if audio_array.size == 0:
            memory_debugger.log_memory_operation('empty_audio', {'reason': 'audio_array.size == 0'})
            yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
            return

        # Language names (same as V4)
        lang_names = {
            'a': 'American English', 'b': 'British English', 'i': 'Italian', 
            'e': 'Spanish', 'f': 'French', 'p': 'Portuguese', 
            'j': 'Japanese', 'z': 'Chinese', 'h': 'Hindi'
        }

        # STT processing with enhanced memory debugging
        user_text = ""
        detected_language = DEFAULT_LANGUAGE
        
        if audio_array.size > 0:
            logger.info(f"🔧 ENHANCED DEBUG: Processing audio for STT...")
            memory_debugger.log_memory_operation('stt_start', {
                'audio_size': audio_array.size,
                'sample_rate': sample_rate,
                'audio_duration': len(audio_array) / sample_rate
            })
            
            try:
                transcription_result = run_coro_from_sync_thread_with_timeout(
                    voice_assistant.stt_engine.transcribe_with_sample_rate(audio_array, sample_rate),
                    timeout=8.0
                )
                
                if hasattr(transcription_result, 'text'):
                    user_text = transcription_result.text.strip()
                    detected_language = voice_assistant.detect_language_from_text(user_text)
                    logger.info(f"🌍 Language detected by MediaPipe: {detected_language}")
                else:
                    user_text = str(transcription_result).strip()
                    detected_language = DEFAULT_LANGUAGE
                
                memory_debugger.log_memory_operation('stt_complete', {
                    'transcribed_text': user_text,
                    'detected_language': detected_language,
                    'text_length': len(user_text)
                })
                
                print_status(f"📝 Transcribed: '{user_text}'")
                
                # Update current language
                if detected_language != voice_assistant.current_language:
                    voice_assistant.current_language = detected_language
                    lang_name = lang_names.get(detected_language, 'Unknown')
                    print_status(f"🌍 Language switched to: {lang_name} ({detected_language})")
                    memory_debugger.log_memory_operation('language_switch', {
                        'from': voice_assistant.current_language,
                        'to': detected_language,
                        'language_name': lang_name
                    })
                else:
                    lang_name = lang_names.get(detected_language, 'Unknown')
                    print_status(f"🌍 Language confirmed: {lang_name} ({detected_language})")
                
            except Exception as e:
                logger.error(f"🔧 ENHANCED DEBUG: STT failed: {e}")
                memory_debugger.log_memory_operation('stt_error', {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                user_text = ""
                detected_language = DEFAULT_LANGUAGE

        if not user_text.strip():
            memory_debugger.log_memory_operation('no_text_transcribed', {})
            yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
            return

        print_status(f"👤 User: {user_text}")

        # LLM response with comprehensive memory debugging
        start_turn_time = time.monotonic()
        memory_debugger.log_memory_operation('llm_request_start', {
            'user_input': user_text,
            'turn_start_time': start_turn_time,
            'voice_assistant_id': id(voice_assistant)
        })
        
        try:
            # Check memory state before LLM call
            memory_context_before = None
            if hasattr(voice_assistant, 'memory_manager') and voice_assistant.memory_manager:
                try:
                    memory_context_before = run_coro_from_sync_thread_with_timeout(
                        voice_assistant.memory_manager.get_conversation_context(),
                        timeout=2.0
                    )
                    memory_debugger.log_memory_operation('memory_context_before_llm', {
                        'context_exists': memory_context_before is not None,
                        'context_length': len(str(memory_context_before)) if memory_context_before else 0,
                        'context_preview': str(memory_context_before)[:200] if memory_context_before else None,
                        'memory_manager_type': str(type(voice_assistant.memory_manager))
                    })
                except Exception as e:
                    memory_debugger.log_memory_operation('memory_context_error_before', {
                        'error': str(e),
                        'has_memory_manager': hasattr(voice_assistant, 'memory_manager'),
                        'memory_manager_none': voice_assistant.memory_manager is None if hasattr(voice_assistant, 'memory_manager') else 'no_attr'
                    })
            else:
                memory_debugger.log_memory_operation('no_memory_manager', {
                    'has_memory_manager_attr': hasattr(voice_assistant, 'memory_manager'),
                    'memory_manager_value': getattr(voice_assistant, 'memory_manager', 'NO_ATTR')
                })
            
            # Make LLM call
            assistant_response_text = run_coro_from_sync_thread_with_timeout(
                voice_assistant.get_llm_response_smart(user_text),
                timeout=4.0
            )
            
            turn_processing_time = time.monotonic() - start_turn_time
            
            # Check memory state after LLM call
            memory_context_after = None
            if hasattr(voice_assistant, 'memory_manager') and voice_assistant.memory_manager:
                try:
                    memory_context_after = run_coro_from_sync_thread_with_timeout(
                        voice_assistant.memory_manager.get_conversation_context(),
                        timeout=2.0
                    )
                    memory_debugger.log_memory_operation('memory_context_after_llm', {
                        'context_exists': memory_context_after is not None,
                        'context_length': len(str(memory_context_after)) if memory_context_after else 0,
                        'context_changed': str(memory_context_before) != str(memory_context_after),
                        'context_preview': str(memory_context_after)[:200] if memory_context_after else None
                    })
                except Exception as e:
                    memory_debugger.log_memory_operation('memory_context_error_after', {'error': str(e)})
            
            memory_debugger.log_memory_operation('llm_response_complete', {
                'response': assistant_response_text,
                'response_length': len(assistant_response_text),
                'processing_time': turn_processing_time,
                'memory_context_changed': str(memory_context_before) != str(memory_context_after) if memory_context_before and memory_context_after else None
            })
            
            # Log the conversation turn
            memory_debugger.log_conversation_turn(user_text, assistant_response_text, memory_context_after)
            
        except TimeoutError:
            assistant_response_text = "Let me think about that and get back to you quickly."
            memory_debugger.log_memory_operation('llm_timeout', {
                'timeout_duration': time.monotonic() - start_turn_time,
                'fallback_response': assistant_response_text
            })
        except Exception as e:
            assistant_response_text = "I encountered an error processing your request."
            memory_debugger.log_memory_operation('llm_error', {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'fallback_response': assistant_response_text
            })
        
        turn_processing_time = time.monotonic() - start_turn_time
        print_status(f"🤖 Assistant: {assistant_response_text}")
        print_status(f"⏱️ Turn Processing Time: {turn_processing_time:.2f}s")

        # TTS processing with logging
        additional_outputs = AdditionalOutputs()
        
        # Language conversion logging
        logger.info(f"🔧 Language Debug: Current language before conversion: '{voice_assistant.current_language}'")
        
        if len(voice_assistant.current_language) > 1:
            kokoro_language = voice_assistant.convert_to_kokoro_language(voice_assistant.current_language)
            logger.info(f"🔧 Language Debug: Converting '{voice_assistant.current_language}' → '{kokoro_language}'")
            voice_assistant.current_language = kokoro_language
            memory_debugger.log_memory_operation('language_conversion', {
                'original': voice_assistant.current_language,
                'converted': kokoro_language
            })
        else:
            logger.info(f"🔧 Language Debug: Language '{voice_assistant.current_language}' already in Kokoro format")
        
        tts_voices_to_try = voice_assistant.get_voices_for_language(voice_assistant.current_language)
        tts_voices_to_try.append(None)
        
        print_status(f"🎤 TTS using language '{voice_assistant.current_language}' with voices: {tts_voices_to_try[:3]}")
        
        memory_debugger.log_memory_operation('tts_start', {
            'text_to_synthesize': assistant_response_text,
            'text_length': len(assistant_response_text),
            'language': voice_assistant.current_language,
            'voices_to_try': tts_voices_to_try[:3]
        })
        
        tts_success = False
        for voice_id in tts_voices_to_try:
            try:
                logger.info(f"🔧 TTS Debug: Attempting TTS with voice='{voice_id}', language='{voice_assistant.current_language}'")
                logger.info(f"🔧 TTS Debug: TTS engine available: {voice_assistant.tts_engine.is_available()}")
                logger.info(f"🔧 TTS Debug: TTS engine type: {type(voice_assistant.tts_engine)}")
                
                chunk_count = 0
                total_samples = 0
                
                for current_sr, current_chunk_array in voice_assistant.stream_tts_synthesis(
                    assistant_response_text, voice_id, voice_assistant.current_language
                ):
                    if isinstance(current_chunk_array, np.ndarray) and current_chunk_array.size > 0:
                        chunk_count += 1
                        total_samples += current_chunk_array.size
                        chunk_size = min(1024, current_chunk_array.size)
                        for i in range(0, current_chunk_array.size, chunk_size):
                            mini_chunk = current_chunk_array[i:i+chunk_size]
                            if mini_chunk.size > 0:
                                yield (current_sr, mini_chunk), additional_outputs
                
                tts_success = True
                logger.info(f"✅ TTS stream completed using refactored engine. Voice: {voice_id}, Chunks: {chunk_count}, Samples: {total_samples}")
                print_status(f"✅ TTS SUCCESS using refactored engine with voice: {voice_id}")
                
                memory_debugger.log_memory_operation('tts_success', {
                    'voice_id': voice_id,
                    'chunk_count': chunk_count,
                    'total_samples': total_samples,
                    'engine_type': str(type(voice_assistant.tts_engine))
                })
                break
                    
            except Exception as e:
                logger.error(f"❌ TTS failed with voice '{voice_id}': {e}")
                print_status(f"❌ TTS failed with voice '{voice_id}': {e}")
                memory_debugger.log_memory_operation('tts_error', {
                    'voice_id': voice_id,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                continue
                
        if not tts_success:
            print_status(f"❌ All TTS attempts failed")
            memory_debugger.log_memory_operation('tts_all_failed', {})
            yield SILENT_AUDIO_FRAME_TUPLE, additional_outputs

        # Log callback completion
        memory_debugger.log_memory_operation('callback_complete', {
            'total_processing_time': time.monotonic() - start_turn_time,
            'tts_success': tts_success
        })

    except Exception as e:
        print_status(f"❌ CRITICAL Error: {e}")
        memory_debugger.log_memory_operation('critical_error', {
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        yield EMPTY_AUDIO_YIELD_OUTPUT

# Setup function to initialize async environment
def setup_async_environment():
    """Setup async environment with memory debugging."""
    global main_event_loop, voice_assistant, async_worker_thread
    
    logger.info("🧠 Creating VoiceAssistant instance with memory debugging...")
    voice_assistant = VoiceAssistant()
    memory_debugger.voice_assistant = voice_assistant
    memory_debugger.log_memory_operation('voice_assistant_created', {
        'voice_assistant_id': id(voice_assistant),
        'voice_assistant_type': str(type(voice_assistant))
    })

    def run_async_loop_in_thread():
        global main_event_loop, voice_assistant
        main_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(main_event_loop)
        
        if voice_assistant:
            logger.info("⚡ Initializing async components...")
            memory_debugger.log_memory_operation('async_init_start', {})
            main_event_loop.run_until_complete(voice_assistant.initialize_async())
            memory_debugger.log_memory_operation('async_init_complete', {
                'has_memory_manager': hasattr(voice_assistant, 'memory_manager'),
                'memory_manager_type': str(type(getattr(voice_assistant, 'memory_manager', None)))
            })
        else:
            print_status("🚨 Voice assistant instance is None in async thread. Cannot initialize.")
            return

        try:
            main_event_loop.run_forever()
        except KeyboardInterrupt:
            print_status("Async loop interrupted in thread.")
        finally:
            if voice_assistant and main_event_loop and not main_event_loop.is_closed():
                print_status("Cleaning up assistant resources in async thread...")
                main_event_loop.run_until_complete(voice_assistant.cleanup_async())
            if main_event_loop and not main_event_loop.is_closed():
                 main_event_loop.close()
            print_status("Async event loop closed.")

    async_worker_thread = threading.Thread(target=run_async_loop_in_thread, daemon=True, name="AsyncWorkerThread")
    async_worker_thread.start()

    # Wait for initialization
    for _ in range(100):
        if main_event_loop and main_event_loop.is_running() and voice_assistant:
            print_status("✅ Async environment and refactored components are ready.")
            memory_debugger.log_memory_operation('setup_complete', {
                'initialization_successful': True
            })
            return
        time.sleep(0.1)
    print_status("⚠️ Async environment or components did not confirm readiness in time.")
    memory_debugger.log_memory_operation('setup_timeout', {
        'initialization_successful': False
    })

def run_coro_from_sync_thread_with_timeout(coro, timeout: float = 4.0) -> any:
    """Run coroutine with timeout and memory debugging."""
    global main_event_loop
    if main_event_loop and main_event_loop.is_running():
        import asyncio
        import time
        
        start_time = time.monotonic()
        logger.info(f"🔧 Async Debug: Starting coroutine with timeout={timeout}s")
        
        future = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(coro, timeout=timeout),
            main_event_loop
        )
        try:
            result = future.result(timeout=timeout + 1.0)
            elapsed = time.monotonic() - start_time
            logger.info(f"🔧 Async Debug: Coroutine completed in {elapsed:.2f}s")
            return result
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start_time
            logger.error(f"🔧 Async Debug: Timeout after {elapsed:.2f}s (limit was {timeout}s)")
            print_status(f"❌ Async task timed out after {timeout}s")
            raise TimeoutError(f"Operation timed out after {timeout}s")
        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error(f"🔧 Async Debug: Error after {elapsed:.2f}s: {e}")
            print_status(f"❌ Error in async task: {e}")
            return "I encountered an error processing your request."
    else:
        print_status("❌ Event loop not available")
        return "My processing system is not ready."

def main():
    """Enhanced main function with comprehensive memory debugging."""
    print("🧠 Enhanced Gradio3 with Comprehensive Memory Debugging")
    print("=" * 70)
    
    setup_async_environment()

    bridge = FastRTCBridge()
    bridge.create_stream(voice_assistant_callback_rt)
    
    try:
        print("🔍 Memory debugging enabled - detailed logs will be generated")
        print("💡 Test memory by saying things like:")
        print("   • 'My name is [Your Name]'")
        print("   • 'What is my name?'")
        print("   • 'I like [something]'")
        print("   • 'What do you know about me?'")
        print("   • 'I work as a [profession]'")
        print("   • 'Tell me about myself'")
        print("=" * 70)
        
        bridge.launch_stream()
        
    except KeyboardInterrupt:
        print_status("🛑 Shutting down...")
    except Exception as e:
        print_status(f"❌ Launch error: {e}")
        memory_debugger.log_memory_operation('launch_error', {
            'error': str(e),
            'traceback': traceback.format_exc()
        })
    finally:
        # Save comprehensive memory debug report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = memory_debugger.save_debug_report(f"gradio3_enhanced_memory_report_{timestamp}.json")
        analysis = memory_debugger.analyze_memory_performance()
        
        print("\n" + "="*70)
        print("🧠 COMPREHENSIVE MEMORY DEBUG SUMMARY")
        print("="*70)
        print(f"Total Conversation Turns: {analysis['total_conversation_turns']}")
        print(f"Total Memory Operations: {analysis['total_memory_operations']}")
        print(f"Memory Ops per Turn: {analysis['memory_ops_per_turn']:.2f}")
        print(f"Continuity Score: {analysis['continuity_score']:.2f}")
        print(f"Session Duration: {analysis['session_duration']:.1f}s")
        print(f"Debug Report: {report_file}")
        print("="*70)
        
        # Print recent memory operations for quick analysis
        if analysis['recent_memory_ops']:
            print("\n🔍 RECENT MEMORY OPERATIONS:")
            for op in analysis['recent_memory_ops'][-5:]:
                print(f"   {op['timestamp']}: {op['operation_type']} - {op['details']}")
        
        print("\n💡 DEBUGGING TIPS:")
        print("   • Check the JSON report for detailed memory operation logs")
        print("   • Compare with gradio2.py memory performance")
        print("   • Look for differences in memory_manager initialization")
        print("   • Analyze conversation continuity patterns")
        print("="*70)

if __name__ == "__main__":
    main()