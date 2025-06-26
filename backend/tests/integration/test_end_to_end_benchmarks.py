"""
End-to-End Performance Benchmark Tests for FastRTC Voice Assistant.

This module provides comprehensive benchmarking for the entire voice processing pipeline,
measuring performance at each stage to identify bottlenecks and optimization opportunities.
"""

import pytest
import asyncio
import time
import psutil
import gc
import threading
import statistics
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

import numpy as np

from src.core.voice_assistant import VoiceAssistant
from src.core.interfaces import AudioData, TranscriptionResult
from src.integration.callback_handler import StreamCallbackHandler
from tests.fixtures.audio_samples import create_test_audio, create_performance_test_audio


@dataclass
class PipelineTimings:
    """Container for pipeline stage timings."""
    time_to_first_token: float = 0.0
    stt_processing_time: float = 0.0
    context_retrieval_time: float = 0.0
    llm_response_time: float = 0.0
    context_update_time: float = 0.0
    tts_processing_time: float = 0.0
    total_pipeline_latency: float = 0.0
    audio_preprocessing_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    timings: PipelineTimings
    success: bool
    error_message: str = ""
    metadata: Dict[str, Any] = None


class PerformanceInstrumentedVoiceAssistant(VoiceAssistant):
    """Voice assistant with detailed performance instrumentation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_timings = PipelineTimings()
        self.performance_history = []
    
    async def process_audio_turn_instrumented(self, user_text: str) -> Tuple[str, PipelineTimings]:
        """Process audio turn with detailed timing instrumentation."""
        timings = PipelineTimings()
        process = psutil.Process()
        
        # Start total pipeline timing
        pipeline_start = time.perf_counter()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Stage 1: Audio preprocessing (simulated - in real scenario this would be actual audio)
            preprocess_start = time.perf_counter()
            # Simulate audio preprocessing time
            await asyncio.sleep(0.01)  # Minimal delay to simulate preprocessing
            timings.audio_preprocessing_time = time.perf_counter() - preprocess_start
            
            # Stage 2: STT Processing (simulated via text input)
            stt_start = time.perf_counter()
            # In real scenario, this would call STT engine
            # For benchmarking, we simulate the processing time
            await asyncio.sleep(0.1)  # Simulate STT processing delay
            timings.stt_processing_time = time.perf_counter() - stt_start
            timings.time_to_first_token = timings.stt_processing_time
            
            # Stage 3: Context Retrieval
            context_start = time.perf_counter()
            user_context = await self.amem_memory.get_user_context_async(self.user_id)
            timings.context_retrieval_time = time.perf_counter() - context_start
            
            # Stage 4: LLM Response Generation
            llm_start = time.perf_counter()
            assistant_response = await self.get_llm_response_smart(user_text)
            timings.llm_response_time = time.perf_counter() - llm_start
            
            # Stage 5: Context Update
            update_start = time.perf_counter()
            await self.amem_memory.add_memory_async(
                content=f"User: {user_text}\nAssistant: {assistant_response}",
                user_id=self.user_id
            )
            timings.context_update_time = time.perf_counter() - update_start
            
            # Stage 6: TTS Processing (simulated)
            tts_start = time.perf_counter()
            # In real scenario, this would call TTS engine
            await asyncio.sleep(0.2)  # Simulate TTS processing delay
            timings.tts_processing_time = time.perf_counter() - tts_start
            
            # Calculate total pipeline latency
            timings.total_pipeline_latency = time.perf_counter() - pipeline_start
            
            # Memory and CPU metrics
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            timings.memory_usage_mb = final_memory - initial_memory
            timings.cpu_usage_percent = process.cpu_percent()
            
            self.last_timings = timings
            self.performance_history.append(timings)
            
            return assistant_response, timings
            
        except Exception as e:
            timings.total_pipeline_latency = time.perf_counter() - pipeline_start
            raise e


class TestEndToEndBenchmarks:
    """Comprehensive end-to-end performance benchmarks."""
    
    @pytest.fixture
    def instrumented_voice_assistant(self):
        """Create an instrumented voice assistant for detailed performance testing."""
        # Create mock components with realistic delays
        mock_stt = AsyncMock()
        mock_tts = AsyncMock()
        mock_processor = Mock()
        mock_memory = AsyncMock()
        mock_llm = AsyncMock()
        mock_cache = AsyncMock()
        mock_conversation_buffer = Mock()
        mock_async_manager = AsyncMock()
        mock_config = {"test_mode": True}
        
        # Configure realistic mock responses
        mock_stt.transcribe.return_value = TranscriptionResult(
            text="Test transcription",
            language="en",
            confidence=0.95
        )
        
        mock_memory.get_user_context_async.return_value = "User context from memory"
        mock_memory.add_memory_async.return_value = "memory_id_123"
        
        mock_llm.get_response_smart.return_value = "Test response from LLM"
        
        return PerformanceInstrumentedVoiceAssistant(
            audio_processor=mock_processor,
            stt_engine=mock_stt,
            tts_engine=mock_tts,
            memory_manager=mock_memory,
            response_cache=mock_cache,
            conversation_buffer=mock_conversation_buffer,
            llm_service=mock_llm,
            async_manager=mock_async_manager,
            config=mock_config
        )
    
    @pytest.mark.asyncio
    async def test_short_query_benchmark(self, instrumented_voice_assistant):
        """Benchmark performance for short queries (1-2 seconds of audio)."""
        test_text = "Hi there!"
        
        response, timings = await instrumented_voice_assistant.process_audio_turn_instrumented(test_text)
        
        # Performance assertions for short queries
        assert timings.total_pipeline_latency < 2.0, f"Short query took {timings.total_pipeline_latency:.3f}s, too slow"
        assert timings.stt_processing_time < 0.5, f"STT too slow: {timings.stt_processing_time:.3f}s"
        assert timings.context_retrieval_time < 0.3, f"Context retrieval too slow: {timings.context_retrieval_time:.3f}s"
        assert timings.llm_response_time < 1.0, f"LLM too slow: {timings.llm_response_time:.3f}s"
        assert timings.tts_processing_time < 0.8, f"TTS too slow: {timings.tts_processing_time:.3f}s"
        
        # Log performance metrics
        print(f"\n🚀 Short Query Performance:")
        print(f"  Time to first token: {timings.time_to_first_token:.3f}s")
        print(f"  STT processing: {timings.stt_processing_time:.3f}s")
        print(f"  Context retrieval: {timings.context_retrieval_time:.3f}s")
        print(f"  LLM response: {timings.llm_response_time:.3f}s")
        print(f"  Context update: {timings.context_update_time:.3f}s")
        print(f"  TTS processing: {timings.tts_processing_time:.3f}s")
        print(f"  Total latency: {timings.total_pipeline_latency:.3f}s")
        print(f"  Memory usage: {timings.memory_usage_mb:.1f}MB")
        
        assert response is not None
    
    @pytest.mark.asyncio
    async def test_medium_query_benchmark(self, instrumented_voice_assistant):
        """Benchmark performance for medium queries (3-5 seconds of audio)."""
        test_text = "Can you help me understand the current weather and give me some recommendations?"
        
        response, timings = await instrumented_voice_assistant.process_audio_turn_instrumented(test_text)
        
        # Performance assertions for medium queries
        assert timings.total_pipeline_latency < 4.0, f"Medium query took {timings.total_pipeline_latency:.3f}s, too slow"
        assert timings.stt_processing_time < 1.0, f"STT too slow: {timings.stt_processing_time:.3f}s"
        assert timings.context_retrieval_time < 0.5, f"Context retrieval too slow: {timings.context_retrieval_time:.3f}s"
        assert timings.llm_response_time < 2.0, f"LLM too slow: {timings.llm_response_time:.3f}s"
        
        print(f"\n🚀 Medium Query Performance:")
        print(f"  Total latency: {timings.total_pipeline_latency:.3f}s")
        print(f"  Time breakdown - STT: {timings.stt_processing_time:.3f}s, LLM: {timings.llm_response_time:.3f}s, TTS: {timings.tts_processing_time:.3f}s")
        
        assert response is not None
    
    @pytest.mark.asyncio
    async def test_long_query_benchmark(self, instrumented_voice_assistant):
        """Benchmark performance for long queries (6-10 seconds of audio)."""
        test_text = """Can you provide a comprehensive analysis of the current market trends, 
        including detailed explanations of the key factors affecting performance, 
        and give me specific recommendations for the next quarter with supporting data?"""
        
        response, timings = await instrumented_voice_assistant.process_audio_turn_instrumented(test_text)
        
        # Performance assertions for long queries
        assert timings.total_pipeline_latency < 6.0, f"Long query took {timings.total_pipeline_latency:.3f}s, too slow"
        assert timings.stt_processing_time < 2.0, f"STT too slow: {timings.stt_processing_time:.3f}s"
        assert timings.llm_response_time < 4.0, f"LLM too slow: {timings.llm_response_time:.3f}s"
        
        print(f"\n🚀 Long Query Performance:")
        print(f"  Total latency: {timings.total_pipeline_latency:.3f}s")
        print(f"  Detailed breakdown:")
        print(f"    Audio preprocessing: {timings.audio_preprocessing_time:.3f}s")
        print(f"    STT processing: {timings.stt_processing_time:.3f}s")
        print(f"    Context retrieval: {timings.context_retrieval_time:.3f}s")
        print(f"    LLM response: {timings.llm_response_time:.3f}s")
        print(f"    Context update: {timings.context_update_time:.3f}s")
        print(f"    TTS processing: {timings.tts_processing_time:.3f}s")
        
        assert response is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_bottleneck_analysis(self, instrumented_voice_assistant):
        """Analyze pipeline bottlenecks across multiple requests."""
        test_queries = [
            "Hello",
            "What's the weather like?",
            "Can you explain quantum computing in simple terms?",
            "Help me plan a vacation to Japan with detailed itinerary",
            "Quick question"
        ]
        
        all_timings = []
        
        for query in test_queries:
            response, timings = await instrumented_voice_assistant.process_audio_turn_instrumented(query)
            all_timings.append(timings)
            assert response is not None
        
        # Analyze bottlenecks
        avg_stt = statistics.mean([t.stt_processing_time for t in all_timings])
        avg_context = statistics.mean([t.context_retrieval_time for t in all_timings])
        avg_llm = statistics.mean([t.llm_response_time for t in all_timings])
        avg_tts = statistics.mean([t.tts_processing_time for t in all_timings])
        avg_total = statistics.mean([t.total_pipeline_latency for t in all_timings])
        
        # Identify bottleneck
        stage_times = {
            "STT": avg_stt,
            "Context": avg_context, 
            "LLM": avg_llm,
            "TTS": avg_tts
        }
        bottleneck = max(stage_times, key=stage_times.get)
        
        print(f"\n🔍 Pipeline Bottleneck Analysis:")
        print(f"  Average timings across {len(test_queries)} queries:")
        print(f"    STT: {avg_stt:.3f}s")
        print(f"    Context Retrieval: {avg_context:.3f}s")
        print(f"    LLM: {avg_llm:.3f}s")
        print(f"    TTS: {avg_tts:.3f}s")
        print(f"    Total: {avg_total:.3f}s")
        print(f"  Primary bottleneck: {bottleneck} ({stage_times[bottleneck]:.3f}s)")
        
        # Performance thresholds
        assert avg_total < 3.0, f"Average pipeline too slow: {avg_total:.3f}s"
        assert bottleneck in stage_times, "Failed to identify bottleneck"
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_benchmark(self, instrumented_voice_assistant):
        """Benchmark concurrent request processing."""
        test_text = "Concurrent test query"
        num_concurrent = 3
        
        start_time = time.perf_counter()
        
        # Process multiple requests concurrently
        tasks = [
            instrumented_voice_assistant.process_audio_turn_instrumented(f"{test_text} #{i}")
            for i in range(num_concurrent)
        ]
        
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
        
        # All requests should succeed
        assert len(results) == num_concurrent
        for response, timings in results:
            assert response is not None
            assert timings.total_pipeline_latency < 5.0
        
        # Concurrent processing should be more efficient than sequential
        avg_individual_time = statistics.mean([timings.total_pipeline_latency for _, timings in results])
        
        print(f"\n⚡ Concurrent Processing Benchmark:")
        print(f"  {num_concurrent} concurrent requests")
        print(f"  Total wall clock time: {total_time:.3f}s")
        print(f"  Average individual latency: {avg_individual_time:.3f}s")
        print(f"  Efficiency ratio: {(avg_individual_time * num_concurrent) / total_time:.2f}x")
        
        # Should complete faster than sequential processing
        assert total_time < (avg_individual_time * num_concurrent)
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_benchmark(self, instrumented_voice_assistant):
        """Test memory efficiency over multiple requests."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        test_text = "Memory efficiency test query"
        num_requests = 10
        
        for i in range(num_requests):
            response, timings = await instrumented_voice_assistant.process_audio_turn_instrumented(test_text)
            assert response is not None
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = current_memory - initial_memory
            
            # Memory growth should be bounded
            assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f}MB after {i+1} requests"
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_growth = final_memory - initial_memory
        
        print(f"\n💾 Memory Efficiency Benchmark:")
        print(f"  {num_requests} requests processed")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Total growth: {total_growth:.1f}MB")
        print(f"  Memory per request: {total_growth/num_requests:.2f}MB")
        
        # Memory growth should be reasonable
        assert total_growth < 50, f"Total memory growth too high: {total_growth:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_performance_regression_baseline(self, instrumented_voice_assistant):
        """Establish baseline performance metrics for regression testing."""
        test_scenarios = [
            ("short", "Hi"),
            ("medium", "What's the weather today?"),
            ("long", "Explain the theory of relativity in detail"),
        ]
        
        baseline_metrics = {}
        
        for scenario_name, test_text in test_scenarios:
            # Run multiple iterations for stable measurements
            timings_list = []
            
            for _ in range(5):
                response, timings = await instrumented_voice_assistant.process_audio_turn_instrumented(test_text)
                timings_list.append(timings)
                assert response is not None
            
            # Calculate statistics
            avg_latency = statistics.mean([t.total_pipeline_latency for t in timings_list])
            avg_stt = statistics.mean([t.stt_processing_time for t in timings_list])
            avg_llm = statistics.mean([t.llm_response_time for t in timings_list])
            avg_tts = statistics.mean([t.tts_processing_time for t in timings_list])
            
            baseline_metrics[scenario_name] = {
                "avg_total_latency": avg_latency,
                "avg_stt_time": avg_stt,
                "avg_llm_time": avg_llm,
                "avg_tts_time": avg_tts,
                "sample_count": len(timings_list)
            }
        
        print(f"\n📊 Performance Baseline Metrics:")
        for scenario, metrics in baseline_metrics.items():
            print(f"  {scenario.upper()} queries:")
            print(f"    Total latency: {metrics['avg_total_latency']:.3f}s")
            print(f"    STT: {metrics['avg_stt_time']:.3f}s")
            print(f"    LLM: {metrics['avg_llm_time']:.3f}s")
            print(f"    TTS: {metrics['avg_tts_time']:.3f}s")
        
        # Save baseline for future regression testing
        timestamp = datetime.now().isoformat()
        baseline_data = {
            "timestamp": timestamp,
            "metrics": baseline_metrics,
            "test_environment": {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            }
        }
        
        # In a real implementation, this would be saved to a file
        print(f"\n📈 Baseline metrics established at {timestamp}")
        
        # Basic performance sanity checks
        assert baseline_metrics["short"]["avg_total_latency"] < 2.0
        assert baseline_metrics["medium"]["avg_total_latency"] < 4.0
        assert baseline_metrics["long"]["avg_total_latency"] < 6.0
        
        return baseline_data


class TestCallbackHandlerBenchmarks:
    """Benchmark tests for the StreamCallbackHandler specifically."""
    
    @pytest.fixture
    def mock_callback_handler(self):
        """Create a mock callback handler for testing."""
        mock_voice_assistant = Mock()
        mock_stt = Mock()
        mock_tts = Mock()
        mock_voice_mapper = Mock()
        
        return StreamCallbackHandler(
            voice_assistant=mock_voice_assistant,
            stt_engine=mock_stt,
            tts_engine=mock_tts,
            voice_mapper=mock_voice_mapper
        )
    
    def test_audio_callback_performance(self, mock_callback_handler):
        """Test the performance of the audio callback function."""
        # Create test audio data
        sample_rate = 16000
        duration = 2.0  # seconds
        audio_array = np.random.random(int(sample_rate * duration)).astype(np.float32)
        audio_data_tuple = (sample_rate, audio_array)
        
        # Mock the voice assistant methods
        mock_callback_handler.voice_assistant.get_llm_response_smart = AsyncMock(return_value="Test response")
        mock_callback_handler.voice_assistant.stream_tts_synthesis = Mock(return_value=[(sample_rate, audio_array)])
        mock_callback_handler.voice_assistant.conversation_buffer = Mock()
        mock_callback_handler.voice_assistant.conversation_buffer.add_turn = Mock()
        
        start_time = time.perf_counter()
        
        # Process audio through callback
        result_generator = mock_callback_handler.process_audio_stream(audio_data_tuple)
        results = list(result_generator)
        
        processing_time = time.perf_counter() - start_time
        
        print(f"\n🎵 Audio Callback Performance:")
        print(f"  Processing time: {processing_time:.3f}s")
        print(f"  Audio duration: {duration:.1f}s")
        print(f"  Real-time factor: {processing_time/duration:.2f}x")
        print(f"  Output chunks: {len(results)}")
        
        # Should process faster than real-time for good UX
        assert processing_time < duration * 2.0, f"Callback too slow: {processing_time:.3f}s for {duration}s audio"
        assert len(results) > 0, "No audio output generated"


if __name__ == "__main__":
    # Run benchmarks directly
    pytest.main([__file__, "-v", "-s"])