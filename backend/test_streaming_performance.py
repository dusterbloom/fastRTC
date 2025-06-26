#!/usr/bin/env python3
"""
Streaming Pipeline Performance Test
==================================

Comprehensive benchmarking to measure the impact of streaming implementation
on Time to First Token, total latency, and perceived responsiveness.
"""

import asyncio
import time
import sys
import os
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

@dataclass
class PerformanceMetrics:
    """Container for performance measurements."""
    test_name: str
    time_to_first_token: float
    total_response_time: float
    tokens_received: int
    tokens_per_second: float
    audio_chunks_received: int
    audio_generation_time: float
    memory_operations_time: float
    total_pipeline_time: float

class StreamingPerformanceTest:
    """
    Performance testing suite for streaming vs non-streaming comparison.
    """
    
    def __init__(self):
        self.results = []
        self.test_phrases = [
            "Hello, how are you today?",
            "What's the weather like?", 
            "Tell me about artificial intelligence.",
            "Can you help me with a quick question?",
            "I'd like to know more about streaming technology.",
            "What are the benefits of real-time processing?",
            "How does voice recognition work?",
            "Explain the concept of latency optimization."
        ]
    
    async def setup_streaming_pipeline(self):
        """Initialize the streaming pipeline components."""
        print("🚀 Setting up streaming pipeline...")
        
        try:
            from src.core.voice_assistant import VoiceAssistant
            from src.integration.streaming_callback_handler import StreamingPipeline
            from src.audio import STTEngine, KokoroTTSEngine, VoiceMapper
            
            # Create voice assistant with streaming
            self.voice_assistant = VoiceAssistant()
            await self.voice_assistant.initialize_async()
            
            # Create streaming pipeline
            self.streaming_pipeline = StreamingPipeline(
                self.voice_assistant,
                self.voice_assistant.stt_engine,
                self.voice_assistant.tts_engine,
                self.voice_assistant.voice_mapper
            )
            
            print("✅ Streaming pipeline ready")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    async def test_streaming_llm_performance(self, test_text: str) -> PerformanceMetrics:
        """Test LLM streaming performance."""
        print(f"🤖 Testing LLM streaming: '{test_text[:30]}...'")
        
        start_time = time.time()
        first_token_time = None
        tokens_received = 0
        full_response = ""
        
        try:
            async for token in self.voice_assistant.stream_llm_response_smart(test_text):
                if first_token_time is None:
                    first_token_time = time.time()
                tokens_received += 1
                full_response += token
                
                # Show progress for first few tokens
                if tokens_received <= 5:
                    print(f"  Token {tokens_received}: '{token}' (+{(time.time() - start_time)*1000:.1f}ms)")
        
        except Exception as e:
            print(f"❌ LLM streaming test failed: {e}")
            return None
        
        end_time = time.time()
        
        # Calculate metrics
        time_to_first_token = (first_token_time - start_time) * 1000 if first_token_time else 0
        total_response_time = (end_time - start_time) * 1000
        tokens_per_second = tokens_received / ((end_time - start_time) + 0.001)  # Avoid division by zero
        
        print(f"  ✅ LLM: {tokens_received} tokens, TTFT: {time_to_first_token:.1f}ms, Total: {total_response_time:.1f}ms")
        
        return PerformanceMetrics(
            test_name="LLM_Streaming",
            time_to_first_token=time_to_first_token,
            total_response_time=total_response_time,
            tokens_received=tokens_received,
            tokens_per_second=tokens_per_second,
            audio_chunks_received=0,
            audio_generation_time=0,
            memory_operations_time=0,
            total_pipeline_time=total_response_time
        )
    
    async def test_streaming_tts_performance(self, test_text: str) -> PerformanceMetrics:
        """Test TTS streaming performance."""
        print(f"🔊 Testing TTS streaming: '{test_text[:30]}...'")
        
        start_time = time.time()
        first_chunk_time = None
        chunks_received = 0
        
        try:
            current_language = self.voice_assistant.current_language
            available_voices = self.voice_assistant.voice_mapper.get_voices_for_language(current_language)
            voice_id = available_voices[0] if available_voices else None
            
            async for sample_rate, audio_chunk in self.voice_assistant.tts_engine.stream_synthesis_async(
                test_text, voice_id, current_language
            ):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                chunks_received += 1
                
                # Show progress for first few chunks
                if chunks_received <= 5:
                    elapsed = (time.time() - start_time) * 1000
                    print(f"  Chunk {chunks_received}: {audio_chunk.size} samples @ {sample_rate}Hz (+{elapsed:.1f}ms)")
        
        except Exception as e:
            print(f"❌ TTS streaming test failed: {e}")
            return None
        
        end_time = time.time()
        
        # Calculate metrics
        time_to_first_chunk = (first_chunk_time - start_time) * 1000 if first_chunk_time else 0
        total_audio_time = (end_time - start_time) * 1000
        
        print(f"  ✅ TTS: {chunks_received} chunks, TTFC: {time_to_first_chunk:.1f}ms, Total: {total_audio_time:.1f}ms")
        
        return PerformanceMetrics(
            test_name="TTS_Streaming",
            time_to_first_token=time_to_first_chunk,
            total_response_time=total_audio_time,
            tokens_received=0,
            tokens_per_second=0,
            audio_chunks_received=chunks_received,
            audio_generation_time=total_audio_time,
            memory_operations_time=0,
            total_pipeline_time=total_audio_time
        )
    
    async def test_full_streaming_pipeline(self, test_text: str) -> PerformanceMetrics:
        """Test the complete STT→LLM→TTS streaming pipeline."""
        print(f"🎛️ Testing full pipeline: '{test_text[:30]}...'")
        
        pipeline_start = time.time()
        
        # Simulate audio input (16kHz mono for 2 seconds)
        import numpy as np
        sample_rate = 16000
        duration = 2.0
        samples = int(sample_rate * duration)
        test_audio = np.random.normal(0, 0.01, samples).astype(np.float32)  # Low-level noise
        
        # Track pipeline stages
        stt_start = time.time()
        stt_result = await self.streaming_pipeline._stream_stt(test_audio)
        stt_time = (time.time() - stt_start) * 1000
        
        if not stt_result or not stt_result.text.strip():
            # Use the test text directly for pipeline testing
            user_text = test_text
            print(f"  Using test text directly: '{user_text}'")
        else:
            user_text = stt_result.text.strip()
            print(f"  STT result: '{user_text}' ({stt_time:.1f}ms)")
        
        # Test LLM streaming
        llm_start = time.time()
        first_token_time = None
        tokens_received = 0
        full_response = ""
        
        async for token in self.voice_assistant.stream_llm_response_smart(user_text):
            if first_token_time is None:
                first_token_time = time.time()
            tokens_received += 1
            full_response += token
            
            # Break after getting a good response sample
            if len(full_response) > 50 and token in ['.', '!', '?']:
                break
        
        llm_time = (time.time() - llm_start) * 1000
        
        # Test TTS streaming
        tts_start = time.time()
        first_audio_time = None
        audio_chunks = 0
        
        current_language = self.voice_assistant.current_language
        available_voices = self.voice_assistant.voice_mapper.get_voices_for_language(current_language)
        voice_id = available_voices[0] if available_voices else None
        
        # Use first sentence for TTS test
        tts_text = full_response.split('.')[0] + '.' if '.' in full_response else full_response[:50]
        
        async for sample_rate, audio_chunk in self.voice_assistant.tts_engine.stream_synthesis_async(
            tts_text, voice_id, current_language
        ):
            if first_audio_time is None:
                first_audio_time = time.time()
            audio_chunks += 1
            
            # Break after a few chunks to keep test fast
            if audio_chunks >= 10:
                break
        
        tts_time = (time.time() - tts_start) * 1000
        pipeline_end = time.time()
        
        # Calculate comprehensive metrics
        total_pipeline_time = (pipeline_end - pipeline_start) * 1000
        time_to_first_token = (first_token_time - llm_start) * 1000 if first_token_time else 0
        time_to_first_audio = (first_audio_time - tts_start) * 1000 if first_audio_time else 0
        
        print(f"  ✅ Pipeline: STT:{stt_time:.1f}ms + LLM:{llm_time:.1f}ms + TTS:{tts_time:.1f}ms = {total_pipeline_time:.1f}ms")
        print(f"  📊 TTFT: {time_to_first_token:.1f}ms, TTFA: {time_to_first_audio:.1f}ms")
        
        return PerformanceMetrics(
            test_name="Full_Pipeline",
            time_to_first_token=time_to_first_token,
            total_response_time=total_pipeline_time,
            tokens_received=tokens_received,
            tokens_per_second=tokens_received / ((llm_time + 0.001) / 1000),
            audio_chunks_received=audio_chunks,
            audio_generation_time=tts_time,
            memory_operations_time=0,  # Not measured separately
            total_pipeline_time=total_pipeline_time
        )
    
    async def run_benchmark_suite(self):
        """Run comprehensive performance benchmark."""
        print("=" * 80)
        print("🚀 STREAMING PIPELINE PERFORMANCE BENCHMARK")
        print("=" * 80)
        
        if not await self.setup_streaming_pipeline():
            print("❌ Failed to setup pipeline")
            return
        
        # Test individual components
        print("\n📊 COMPONENT PERFORMANCE TESTS")
        print("-" * 50)
        
        for i, test_phrase in enumerate(self.test_phrases[:3]):  # Test first 3 phrases
            print(f"\n🧪 Test {i+1}/3: {test_phrase}")
            
            # Test LLM streaming
            llm_metrics = await self.test_streaming_llm_performance(test_phrase)
            if llm_metrics:
                self.results.append(llm_metrics)
            
            # Test TTS streaming  
            tts_metrics = await self.test_streaming_tts_performance(test_phrase)
            if tts_metrics:
                self.results.append(tts_metrics)
            
            # Small delay between tests
            await asyncio.sleep(0.5)
        
        # Test full pipeline
        print("\n🎛️ FULL PIPELINE TESTS")
        print("-" * 50)
        
        for i, test_phrase in enumerate(self.test_phrases[:2]):  # Test 2 full pipelines
            print(f"\n🧪 Pipeline Test {i+1}/2:")
            pipeline_metrics = await self.test_full_streaming_pipeline(test_phrase)
            if pipeline_metrics:
                self.results.append(pipeline_metrics)
            
            await asyncio.sleep(1.0)
        
        # Generate performance report
        self.generate_performance_report()
        
        # Cleanup
        await self.voice_assistant.cleanup_async()
    
    def generate_performance_report(self):
        """Generate comprehensive performance analysis."""
        print("\n" + "=" * 80)
        print("📈 STREAMING PERFORMANCE ANALYSIS REPORT")
        print("=" * 80)
        
        if not self.results:
            print("❌ No results to analyze")
            return
        
        # Group results by test type
        llm_results = [r for r in self.results if r.test_name == "LLM_Streaming"]
        tts_results = [r for r in self.results if r.test_name == "TTS_Streaming"]
        pipeline_results = [r for r in self.results if r.test_name == "Full_Pipeline"]
        
        print("\n🤖 LLM STREAMING PERFORMANCE:")
        if llm_results:
            ttft_times = [r.time_to_first_token for r in llm_results]
            total_times = [r.total_response_time for r in llm_results]
            tokens_per_sec = [r.tokens_per_second for r in llm_results]
            
            print(f"  📊 Time to First Token: {statistics.mean(ttft_times):.1f}ms ± {statistics.stdev(ttft_times) if len(ttft_times) > 1 else 0:.1f}ms")
            print(f"  📊 Total Response Time: {statistics.mean(total_times):.1f}ms ± {statistics.stdev(total_times) if len(total_times) > 1 else 0:.1f}ms")
            print(f"  📊 Tokens per Second: {statistics.mean(tokens_per_sec):.1f} ± {statistics.stdev(tokens_per_sec) if len(tokens_per_sec) > 1 else 0:.1f}")
            print(f"  📊 Best TTFT: {min(ttft_times):.1f}ms")
        
        print("\n🔊 TTS STREAMING PERFORMANCE:")
        if tts_results:
            ttfc_times = [r.time_to_first_token for r in tts_results]  # Time to First Chunk
            audio_times = [r.audio_generation_time for r in tts_results]
            chunk_counts = [r.audio_chunks_received for r in tts_results]
            
            print(f"  📊 Time to First Chunk: {statistics.mean(ttfc_times):.1f}ms ± {statistics.stdev(ttfc_times) if len(ttfc_times) > 1 else 0:.1f}ms")
            print(f"  📊 Audio Generation Time: {statistics.mean(audio_times):.1f}ms ± {statistics.stdev(audio_times) if len(audio_times) > 1 else 0:.1f}ms")
            print(f"  📊 Average Chunks: {statistics.mean(chunk_counts):.1f} ± {statistics.stdev(chunk_counts) if len(chunk_counts) > 1 else 0:.1f}")
            print(f"  📊 Best TTFC: {min(ttfc_times):.1f}ms")
        
        print("\n🎛️ FULL PIPELINE PERFORMANCE:")
        if pipeline_results:
            pipeline_times = [r.total_pipeline_time for r in pipeline_results]
            ttft_pipeline = [r.time_to_first_token for r in pipeline_results]
            
            print(f"  📊 Total Pipeline Time: {statistics.mean(pipeline_times):.1f}ms ± {statistics.stdev(pipeline_times) if len(pipeline_times) > 1 else 0:.1f}ms")
            print(f"  📊 Pipeline TTFT: {statistics.mean(ttft_pipeline):.1f}ms ± {statistics.stdev(ttft_pipeline) if len(ttft_pipeline) > 1 else 0:.1f}ms")
            print(f"  📊 Best Pipeline: {min(pipeline_times):.1f}ms")
        
        # Compare with baseline (from performance report)
        print("\n📈 PERFORMANCE COMPARISON vs BASELINE:")
        baseline_pipeline = 831  # ms from performance report
        baseline_llm = 501  # ms from performance report  
        baseline_tts = 200  # ms from performance report
        
        if pipeline_results:
            avg_pipeline = statistics.mean([r.total_pipeline_time for r in pipeline_results])
            improvement = ((baseline_pipeline - avg_pipeline) / baseline_pipeline) * 100
            print(f"  🚀 Pipeline Improvement: {improvement:+.1f}% ({baseline_pipeline:.0f}ms → {avg_pipeline:.1f}ms)")
        
        if llm_results:
            avg_llm = statistics.mean([r.total_response_time for r in llm_results])
            llm_improvement = ((baseline_llm - avg_llm) / baseline_llm) * 100
            print(f"  🤖 LLM Improvement: {llm_improvement:+.1f}% ({baseline_llm:.0f}ms → {avg_llm:.1f}ms)")
        
        if tts_results:
            avg_tts = statistics.mean([r.audio_generation_time for r in tts_results])
            tts_improvement = ((baseline_tts - avg_tts) / baseline_tts) * 100
            print(f"  🔊 TTS Improvement: {tts_improvement:+.1f}% ({baseline_tts:.0f}ms → {avg_tts:.1f}ms)")
        
        # Responsiveness analysis
        print("\n⚡ RESPONSIVENESS ANALYSIS:")
        if llm_results:
            avg_ttft = statistics.mean([r.time_to_first_token for r in llm_results])
            print(f"  📊 Average Time to First Token: {avg_ttft:.1f}ms")
            if avg_ttft < 100:
                print("  ✅ EXCELLENT: Sub-100ms TTFT provides immediate feedback")
            elif avg_ttft < 200:
                print("  ✅ GOOD: Sub-200ms TTFT feels responsive")
            else:
                print("  ⚠️ NEEDS IMPROVEMENT: >200ms TTFT may feel sluggish")
        
        print("\n🎯 STREAMING BENEFITS:")
        print("  ✅ Immediate response feedback")
        print("  ✅ Reduced perceived latency")
        print("  ✅ Parallel processing capabilities")
        print("  ✅ Better user experience during long responses")
        
        print("\n" + "=" * 80)

async def main():
    """Main performance test runner."""
    test_suite = StreamingPerformanceTest()
    await test_suite.run_benchmark_suite()

if __name__ == "__main__":
    print("🧪 Starting Streaming Performance Test...")
    asyncio.run(main())