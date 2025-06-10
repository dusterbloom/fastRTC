#!/usr/bin/env python3
"""
Memory Debugging Tool for Gradio2 vs Gradio3 Comparison

This script helps debug memory functionality differences between the two versions.
"""

import sys
import time
import os
import numpy as np
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, 'src')

from src.core.voice_assistant import VoiceAssistant
from src.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

class MemoryDebugger:
    """Debug memory functionality and conversation flow."""
    
    def __init__(self):
        self.conversation_log = []
        self.memory_operations = []
        self.voice_assistant = None
        self.session_start = datetime.now()
        
    def log_conversation_turn(self, user_input: str, assistant_response: str, memory_context: Any = None):
        """Log a conversation turn with memory context."""
        turn_data = {
            'timestamp': datetime.now().isoformat(),
            'turn_number': len(self.conversation_log) + 1,
            'user_input': user_input,
            'assistant_response': assistant_response,
            'memory_context': str(memory_context) if memory_context else None,
            'memory_operations_count': len(self.memory_operations)
        }
        self.conversation_log.append(turn_data)
        logger.info(f"🧠 MEMORY DEBUG - Turn {turn_data['turn_number']}: User='{user_input}' | Assistant='{assistant_response[:50]}...'")
        
    def log_memory_operation(self, operation_type: str, details: Dict[str, Any]):
        """Log memory operations (store, retrieve, etc.)."""
        operation_data = {
            'timestamp': datetime.now().isoformat(),
            'operation_type': operation_type,
            'details': details,
            'turn_context': len(self.conversation_log)
        }
        self.memory_operations.append(operation_data)
        logger.info(f"🔍 MEMORY OP - {operation_type}: {details}")
        
    def analyze_memory_performance(self) -> Dict[str, Any]:
        """Analyze memory performance metrics."""
        total_turns = len(self.conversation_log)
        total_memory_ops = len(self.memory_operations)
        
        # Count memory operation types
        op_types = {}
        for op in self.memory_operations:
            op_type = op['operation_type']
            op_types[op_type] = op_types.get(op_type, 0) + 1
            
        # Analyze conversation continuity
        continuity_score = self._calculate_continuity_score()
        
        analysis = {
            'session_duration': (datetime.now() - self.session_start).total_seconds(),
            'total_conversation_turns': total_turns,
            'total_memory_operations': total_memory_ops,
            'memory_ops_per_turn': total_memory_ops / max(total_turns, 1),
            'operation_types': op_types,
            'continuity_score': continuity_score,
            'conversation_log': self.conversation_log[-5:],  # Last 5 turns
            'recent_memory_ops': self.memory_operations[-10:]  # Last 10 ops
        }
        
        return analysis
        
    def _calculate_continuity_score(self) -> float:
        """Calculate a score for conversation continuity (0-1)."""
        if len(self.conversation_log) < 2:
            return 1.0
            
        # Simple heuristic: check if assistant responses reference previous context
        continuity_indicators = 0
        total_responses = 0
        
        for i, turn in enumerate(self.conversation_log[1:], 1):
            total_responses += 1
            response = turn['assistant_response'].lower()
            
            # Look for continuity indicators
            if any(indicator in response for indicator in [
                'you mentioned', 'as we discussed', 'earlier you said', 
                'your name', 'you told me', 'remember', 'recall'
            ]):
                continuity_indicators += 1
                
        return continuity_indicators / max(total_responses, 1)
        
    def save_debug_report(self, filename: str = None):
        """Save debug report to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"memory_debug_report_{timestamp}.json"
            
        report = self.analyze_memory_performance()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"📊 Debug report saved to: {filename}")
        return filename

# Global debugger instance
memory_debugger = MemoryDebugger()

async def test_memory_functionality():
    """Test memory functionality directly."""
    logger.info("🧪 Testing memory functionality directly...")
    
    # Create voice assistant
    voice_assistant = VoiceAssistant()
    await voice_assistant.initialize_async()
    memory_debugger.voice_assistant = voice_assistant
    
    # Test conversation scenarios
    test_scenarios = [
        "My name is Alice and I love programming",
        "What is my name?",
        "I also enjoy reading science fiction books",
        "What do you know about my interests?",
        "I work as a software engineer at Google",
        "Tell me about myself"
    ]
    
    logger.info("🎯 Running memory test scenarios...")
    
    for i, user_input in enumerate(test_scenarios, 1):
        logger.info(f"\n--- Test Scenario {i} ---")
        logger.info(f"User: {user_input}")
        
        try:
            # Get LLM response
            response = await voice_assistant.get_llm_response_smart(user_input)
            logger.info(f"Assistant: {response}")
            
            # Log the conversation turn
            memory_debugger.log_conversation_turn(user_input, response)
            
            # Check memory state
            if hasattr(voice_assistant, 'memory_manager'):
                memory_state = await voice_assistant.memory_manager.get_conversation_context()
                memory_debugger.log_memory_operation('context_check', {
                    'context_length': len(str(memory_state)) if memory_state else 0,
                    'scenario': i
                })
            
            # Small delay between scenarios
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ Error in scenario {i}: {e}")
            memory_debugger.log_memory_operation('error', {
                'scenario': i,
                'error': str(e)
            })
    
    # Generate final report
    report_file = memory_debugger.save_debug_report()
    
    # Print summary
    analysis = memory_debugger.analyze_memory_performance()
    print("\n" + "="*60)
    print("🧠 MEMORY FUNCTIONALITY TEST RESULTS")
    print("="*60)
    print(f"Total Conversation Turns: {analysis['total_conversation_turns']}")
    print(f"Total Memory Operations: {analysis['total_memory_operations']}")
    print(f"Memory Ops per Turn: {analysis['memory_ops_per_turn']:.2f}")
    print(f"Continuity Score: {analysis['continuity_score']:.2f}")
    print(f"Session Duration: {analysis['session_duration']:.1f}s")
    print(f"Report saved to: {report_file}")
    print("="*60)
    
    await voice_assistant.cleanup_async()

def create_enhanced_gradio3_debug():
    """Create an enhanced version of gradio3.py with memory debugging."""
    
    enhanced_code = '''#!/usr/bin/env python3
"""
Enhanced Gradio3 with Memory Debugging
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
    """Enhanced callback with memory debugging."""
    global voice_assistant

    logger.info(f"🔧 ENHANCED DEBUG: Callback entry - audio_data_tuple type={type(audio_data_tuple)}")
    
    if not voice_assistant:
        logger.error("🔧 ENHANCED DEBUG: voice_assistant is None!")
        yield EMPTY_AUDIO_YIELD_OUTPUT
        return

    try:
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
            yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
            return

        # STT processing with memory debugging
        user_text = ""
        detected_language = DEFAULT_LANGUAGE
        
        if audio_array.size > 0:
            logger.info(f"🔧 ENHANCED DEBUG: Processing audio for STT...")
            memory_debugger.log_memory_operation('stt_start', {
                'audio_size': audio_array.size,
                'sample_rate': sample_rate
            })
            
            try:
                transcription_result = run_coro_from_sync_thread_with_timeout(
                    voice_assistant.stt_engine.transcribe_with_sample_rate(audio_array, sample_rate),
                    timeout=8.0
                )
                
                if hasattr(transcription_result, 'text'):
                    user_text = transcription_result.text.strip()
                    detected_language = voice_assistant.detect_language_from_text(user_text)
                else:
                    user_text = str(transcription_result).strip()
                    detected_language = DEFAULT_LANGUAGE
                
                memory_debugger.log_memory_operation('stt_complete', {
                    'transcribed_text': user_text,
                    'detected_language': detected_language
                })
                
            except Exception as e:
                logger.error(f"🔧 ENHANCED DEBUG: STT failed: {e}")
                memory_debugger.log_memory_operation('stt_error', {'error': str(e)})
                user_text = ""
                detected_language = DEFAULT_LANGUAGE

        if not user_text.strip():
            yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
            return

        print_status(f"👤 User: {user_text}")

        # LLM response with enhanced memory debugging
        start_turn_time = time.monotonic()
        memory_debugger.log_memory_operation('llm_request_start', {
            'user_input': user_text,
            'turn_start_time': start_turn_time
        })
        
        try:
            # Check memory state before LLM call
            if hasattr(voice_assistant, 'memory_manager'):
                try:
                    memory_context = run_coro_from_sync_thread_with_timeout(
                        voice_assistant.memory_manager.get_conversation_context(),
                        timeout=2.0
                    )
                    memory_debugger.log_memory_operation('memory_context_retrieved', {
                        'context_length': len(str(memory_context)) if memory_context else 0,
                        'context_preview': str(memory_context)[:200] if memory_context else None
                    })
                except Exception as e:
                    memory_debugger.log_memory_operation('memory_context_error', {'error': str(e)})
            
            assistant_response_text = run_coro_from_sync_thread_with_timeout(
                voice_assistant.get_llm_response_smart(user_text),
                timeout=4.0
            )
            
            turn_processing_time = time.monotonic() - start_turn_time
            memory_debugger.log_memory_operation('llm_response_complete', {
                'response': assistant_response_text,
                'processing_time': turn_processing_time
            })
            
            # Log the conversation turn
            memory_debugger.log_conversation_turn(user_text, assistant_response_text)
            
        except TimeoutError:
            assistant_response_text = "Let me think about that and get back to you quickly."
            memory_debugger.log_memory_operation('llm_timeout', {
                'timeout_duration': time.monotonic() - start_turn_time
            })
        
        turn_processing_time = time.monotonic() - start_turn_time
        print_status(f"🤖 Assistant: {assistant_response_text}")
        print_status(f"⏱️ Turn Processing Time: {turn_processing_time:.2f}s")

        # TTS processing (same as original but with logging)
        additional_outputs = AdditionalOutputs()
        
        if len(voice_assistant.current_language) > 1:
            kokoro_language = voice_assistant.convert_to_kokoro_language(voice_assistant.current_language)
            voice_assistant.current_language = kokoro_language
        
        tts_voices_to_try = voice_assistant.get_voices_for_language(voice_assistant.current_language)
        tts_voices_to_try.append(None)
        
        memory_debugger.log_memory_operation('tts_start', {
            'text_to_synthesize': assistant_response_text,
            'language': voice_assistant.current_language,
            'voices_to_try': tts_voices_to_try[:3]
        })
        
        tts_success = False
        for voice_id in tts_voices_to_try:
            try:
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
                memory_debugger.log_memory_operation('tts_success', {
                    'voice_id': voice_id,
                    'chunk_count': chunk_count,
                    'total_samples': total_samples
                })
                break
                    
            except Exception as e:
                memory_debugger.log_memory_operation('tts_error', {
                    'voice_id': voice_id,
                    'error': str(e)
                })
                continue
                
        if not tts_success:
            memory_debugger.log_memory_operation('tts_all_failed', {})
            yield SILENT_AUDIO_FRAME_TUPLE, additional_outputs

    except Exception as e:
        print_status(f"❌ CRITICAL Error: {e}")
        memory_debugger.log_memory_operation('critical_error', {
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        yield EMPTY_AUDIO_YIELD_OUTPUT

# Rest of the functions remain the same as gradio3.py
def setup_async_environment():
    """Setup async environment (same as original)."""
    global main_event_loop, voice_assistant, async_worker_thread
    
    voice_assistant = VoiceAssistant()
    memory_debugger.voice_assistant = voice_assistant

    def run_async_loop_in_thread():
        global main_event_loop, voice_assistant
        main_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(main_event_loop)
        
        if voice_assistant:
            main_event_loop.run_until_complete(voice_assistant.initialize_async())
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

    for _ in range(100):
        if main_event_loop and main_event_loop.is_running() and voice_assistant:
            print_status("✅ Async environment and refactored components are ready.")
            return
        time.sleep(0.1)
    print_status("⚠️ Async environment or components did not confirm readiness in time.")

def run_coro_from_sync_thread_with_timeout(coro, timeout: float = 4.0) -> any:
    """Run coroutine with timeout (same as original)."""
    global main_event_loop
    if main_event_loop and main_event_loop.is_running():
        import asyncio
        import time
        
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
            print_status(f"❌ Async task timed out after {timeout}s")
            raise TimeoutError(f"Operation timed out after {timeout}s")
        except Exception as e:
            elapsed = time.monotonic() - start_time
            print_status(f"❌ Error in async task: {e}")
            return "I encountered an error processing your request."
    else:
        print_status("❌ Event loop not available")
        return "My processing system is not ready."

def main():
    """Enhanced main function with memory debugging."""
    print("🧠 Enhanced Gradio3 with Memory Debugging")
    print("=" * 60)
    
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
        print("=" * 60)
        
        bridge.launch_stream()
        
    except KeyboardInterrupt:
        print_status("🛑 Shutting down...")
    finally:
        # Save memory debug report
        report_file = memory_debugger.save_debug_report("gradio3_enhanced_memory_report.json")
        analysis = memory_debugger.analyze_memory_performance()
        
        print("\\n" + "="*60)
        print("🧠 MEMORY DEBUG SUMMARY")
        print("="*60)
        print(f"Total Conversation Turns: {analysis['total_conversation_turns']}")
        print(f"Total Memory Operations: {analysis['total_memory_operations']}")
        print(f"Memory Ops per Turn: {analysis['memory_ops_per_turn']:.2f}")
        print(f"Continuity Score: {analysis['continuity_score']:.2f}")
        print(f"Debug Report: {report_file}")
        print("="*60)

if __name__ == "__main__":
    main()
'''
    
    with open('fastrtc_voice_assistant/gradio3_enhanced_debug.py', 'w') as f:
        f.write(enhanced_code)
    
    logger.info("✅ Created enhanced gradio3 debug version: gradio3_enhanced_debug.py")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Debugging Tool")
    parser.add_argument("--test-memory", action="store_true", 
                       help="Run direct memory functionality test")
    parser.add_argument("--create-enhanced", action="store_true",
                       help="Create enhanced gradio3 debug version")
    
    args = parser.parse_args()
    
    if args.test_memory:
        asyncio.run(test_memory_functionality())
    elif args.create_enhanced:
        create_enhanced_gradio3_debug()
    else:
        print("Memory Debugging Tool")
        print("Usage:")
        print("  python debug_memory_comparison.py --test-memory")
        print("  python debug_memory_comparison.py --create-enhanced")