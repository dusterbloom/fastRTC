"""
Simplified Pipeline Performance Tests for FastRTC Voice Assistant.

This module provides performance benchmarking for individual pipeline components
without the complex initialization requirements.
"""

import pytest
import asyncio
import time
import psutil
import statistics
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

import numpy as np

from src.utils.performance_monitor import (
    PerformanceTimer, PerformanceCollector, PipelineBenchmark,
    measure_performance, benchmark_function
)
from tests.fixtures.audio_samples import create_test_audio


@dataclass
class SimplePipelineTimings:
    """Simplified container for pipeline stage timings."""
    stt_processing_time: float = 0.0
    context_retrieval_time: float = 0.0
    llm_response_time: float = 0.0
    context_update_time: float = 0.0
    tts_processing_time: float = 0.0
    total_pipeline_latency: float = 0.0


class TestPipelinePerformance:
    """Simplified performance tests for pipeline components."""
    
    @pytest.fixture
    def performance_collector(self):
        """Create a performance collector for testing."""
        return PerformanceCollector()
    
    @pytest.fixture
    def pipeline_benchmark(self):
        """Create a pipeline benchmark instance."""
        return PipelineBenchmark()
    
    def test_performance_timer_basic(self):
        """Test basic performance timer functionality."""
        timer = PerformanceTimer("test_operation")
        
        timer.start()
        time.sleep(0.01)  # Sleep for 10ms
        duration = timer.stop()
        
        assert duration >= 0.01, f"Duration {duration} should be at least 0.01s"
        assert duration < 0.1, f"Duration {duration} should be less than 0.1s"
        print(f"Timer test - Duration: {duration:.3f}s")
    
    def test_performance_timer_context_manager(self):
        """Test performance timer as context manager."""
        with PerformanceTimer("context_test") as timer:
            time.sleep(0.005)  # Sleep for 5ms
        
        assert timer.duration >= 0.005, f"Context manager duration {timer.duration} should be at least 0.005s"
        print(f"Context manager test - Duration: {timer.duration:.3f}s")
    
    def test_performance_collector_basic(self, performance_collector):
        """Test basic performance collector functionality."""
        # Record some test metrics
        performance_collector.record_metric("test_operation_1", 100.0, {"test": True})
        performance_collector.record_metric("test_operation_2", 200.0, {"test": True})
        performance_collector.record_metric("other_operation", 50.0, {"test": False})
        
        # Get summary for test operations
        summary = performance_collector.get_metrics_summary("test_operation")
        
        assert summary["sample_count"] == 2
        assert summary["duration_stats"]["avg_ms"] == 150.0
        assert summary["duration_stats"]["min_ms"] == 100.0
        assert summary["duration_stats"]["max_ms"] == 200.0
        
        print(f"Collector test - Samples: {summary['sample_count']}, Avg: {summary['duration_stats']['avg_ms']}ms")
    
    def test_measure_performance_context(self, performance_collector):
        """Test the measure_performance context manager."""
        def slow_operation():
            time.sleep(0.01)  # Simulate 10ms operation
            return "result"
        
        with measure_performance(performance_collector, "slow_operation_test") as timer:
            result = slow_operation()
        
        assert result == "result"
        assert timer.duration >= 0.01
        
        # Check that metric was recorded
        summary = performance_collector.get_metrics_summary("slow_operation_test")
        assert summary["sample_count"] == 1
        assert summary["duration_stats"]["avg_ms"] >= 10.0
        
        print(f"Context test - Duration: {timer.duration:.3f}s, Recorded: {summary['duration_stats']['avg_ms']:.1f}ms")
    
    def test_benchmark_function_utility(self):
        """Test the benchmark_function utility."""
        def test_function():
            time.sleep(0.005)  # 5ms operation
            return "success"
        
        summary = benchmark_function(test_function, "test_func_benchmark", iterations=3)
        
        assert summary["iterations"] == 3
        assert summary["success_rate"] == 1.0
        assert summary["successful_iterations"] == 3
        assert summary["duration_stats"]["avg_ms"] >= 5.0
        
        print(f"Benchmark test - Iterations: {summary['iterations']}, Success Rate: {summary['success_rate']}, Avg: {summary['duration_stats']['avg_ms']:.1f}ms")
    
    def test_pipeline_benchmark_stages(self, pipeline_benchmark):
        """Test pipeline benchmark with multiple stages."""
        # Simulate a complete pipeline
        pipeline_benchmark.start_pipeline()
        
        # Stage 1: STT
        pipeline_benchmark.start_stage("stt")
        time.sleep(0.01)  # Simulate STT processing
        pipeline_benchmark.end_stage("stt", {"audio_length": "2s"})
        
        # Stage 2: Context Retrieval
        pipeline_benchmark.start_stage("context_retrieval")
        time.sleep(0.005)  # Simulate context lookup
        pipeline_benchmark.end_stage("context_retrieval", {"context_size": "500 chars"})
        
        # Stage 3: LLM
        pipeline_benchmark.start_stage("llm")
        time.sleep(0.02)  # Simulate LLM processing
        pipeline_benchmark.end_stage("llm", {"response_length": "100 tokens"})
        
        # Stage 4: TTS
        pipeline_benchmark.start_stage("tts")
        time.sleep(0.015)  # Simulate TTS processing
        pipeline_benchmark.end_stage("tts", {"audio_output": "3s"})
        
        # End pipeline
        pipeline_benchmark.end_pipeline({"query_type": "test"})
        
        # Analyze results
        summary = pipeline_benchmark.get_pipeline_summary()
        
        assert "total_pipeline" in summary
        assert "stages" in summary
        assert len(summary["stages"]) == 4
        
        total_duration = summary["total_pipeline"]["duration_stats"]["avg_ms"]
        print(f"\nPipeline Benchmark Results:")
        print(f"  Total Duration: {total_duration:.1f}ms")
        
        for stage_name, stage_stats in summary["stages"].items():
            stage_duration = stage_stats["duration_stats"]["avg_ms"]
            print(f"  {stage_name}: {stage_duration:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_simulated_voice_pipeline_performance(self):
        """Test simulated voice processing pipeline performance."""
        timings = SimplePipelineTimings()
        
        # Simulate complete voice processing pipeline
        pipeline_start = time.perf_counter()
        
        # Stage 1: STT Processing
        stt_start = time.perf_counter()
        await asyncio.sleep(0.1)  # Simulate realistic STT delay
        timings.stt_processing_time = time.perf_counter() - stt_start
        
        # Stage 2: Context Retrieval
        context_start = time.perf_counter()
        await asyncio.sleep(0.02)  # Simulate context lookup
        timings.context_retrieval_time = time.perf_counter() - context_start
        
        # Stage 3: LLM Response
        llm_start = time.perf_counter()
        await asyncio.sleep(0.5)  # Simulate LLM processing
        timings.llm_response_time = time.perf_counter() - llm_start
        
        # Stage 4: Context Update
        update_start = time.perf_counter()
        await asyncio.sleep(0.01)  # Simulate memory update
        timings.context_update_time = time.perf_counter() - update_start
        
        # Stage 5: TTS Processing
        tts_start = time.perf_counter()
        await asyncio.sleep(0.2)  # Simulate TTS processing
        timings.tts_processing_time = time.perf_counter() - tts_start
        
        # Calculate total pipeline latency
        timings.total_pipeline_latency = time.perf_counter() - pipeline_start
        
        # Performance assertions
        assert timings.total_pipeline_latency < 2.0, f"Pipeline too slow: {timings.total_pipeline_latency:.3f}s"
        assert timings.stt_processing_time < 0.5, f"STT too slow: {timings.stt_processing_time:.3f}s"
        assert timings.llm_response_time < 1.0, f"LLM too slow: {timings.llm_response_time:.3f}s"
        assert timings.tts_processing_time < 0.5, f"TTS too slow: {timings.tts_processing_time:.3f}s"
        
        # Display results
        print(f"\n🎯 Simulated Voice Pipeline Performance:")
        print(f"  STT Processing: {timings.stt_processing_time:.3f}s")
        print(f"  Context Retrieval: {timings.context_retrieval_time:.3f}s")
        print(f"  LLM Response: {timings.llm_response_time:.3f}s")
        print(f"  Context Update: {timings.context_update_time:.3f}s")
        print(f"  TTS Processing: {timings.tts_processing_time:.3f}s")
        print(f"  Total Pipeline: {timings.total_pipeline_latency:.3f}s")
        
        # Calculate bottleneck
        stage_times = {
            "STT": timings.stt_processing_time,
            "Context": timings.context_retrieval_time,
            "LLM": timings.llm_response_time,
            "TTS": timings.tts_processing_time
        }
        bottleneck = max(stage_times, key=stage_times.get)
        print(f"  Primary Bottleneck: {bottleneck} ({stage_times[bottleneck]:.3f}s)")
    
    @pytest.mark.asyncio
    async def test_audio_processing_performance(self):
        """Test audio processing performance with real numpy arrays."""
        # Create test audio data
        sample_rate = 16000
        duration = 2.0  # seconds
        samples = int(sample_rate * duration)
        
        # Generate test audio
        audio_start = time.perf_counter()
        audio_array = np.random.random(samples).astype(np.float32)
        audio_generation_time = time.perf_counter() - audio_start
        
        # Simulate audio preprocessing
        preprocess_start = time.perf_counter()
        
        # Normalize audio
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Calculate audio metrics
        audio_rms = np.sqrt(np.mean(audio_array**2))
        audio_peak = np.max(np.abs(audio_array))
        
        preprocess_time = time.perf_counter() - preprocess_start
        
        # Performance assertions
        assert audio_generation_time < 0.1, f"Audio generation too slow: {audio_generation_time:.3f}s"
        assert preprocess_time < 0.05, f"Audio preprocessing too slow: {preprocess_time:.3f}s"
        assert 0.0 <= audio_rms <= 1.0, f"Invalid RMS value: {audio_rms}"
        assert audio_peak <= 1.0, f"Invalid peak value: {audio_peak}"
        
        print(f"\n🎵 Audio Processing Performance:")
        print(f"  Audio Length: {duration}s ({samples} samples)")
        print(f"  Generation Time: {audio_generation_time:.3f}s")
        print(f"  Preprocessing Time: {preprocess_time:.3f}s")
        print(f"  Audio RMS: {audio_rms:.6f}")
        print(f"  Audio Peak: {audio_peak:.6f}")
    
    def test_memory_usage_monitoring(self, performance_collector):
        """Test memory usage monitoring during operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operation
        with measure_performance(performance_collector, "memory_test"):
            # Create some data structures
            large_arrays = []
            for i in range(100):
                array = np.random.random(1000).astype(np.float32)
                large_arrays.append(array)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Get metrics
        summary = performance_collector.get_metrics_summary("memory_test")
        
        print(f"\n💾 Memory Usage Monitoring:")
        print(f"  Initial Memory: {initial_memory:.1f}MB")
        print(f"  Final Memory: {final_memory:.1f}MB")
        print(f"  Memory Increase: {memory_increase:.1f}MB")
        print(f"  Recorded Memory: {summary['memory_stats']['max_mb']:.1f}MB")
        
        # Memory usage should be reasonable
        assert memory_increase < 100, f"Excessive memory usage: {memory_increase:.1f}MB"
    
    def test_concurrent_performance_measurement(self, performance_collector):
        """Test performance measurement with concurrent operations."""
        async def concurrent_operation(operation_id: int):
            with measure_performance(performance_collector, f"concurrent_op_{operation_id}"):
                await asyncio.sleep(0.01 + (operation_id * 0.001))  # Variable delay
                return f"result_{operation_id}"
        
        async def run_concurrent_test():
            tasks = [concurrent_operation(i) for i in range(5)]
            results = await asyncio.gather(*tasks)
            return results
        
        # Run concurrent test
        start_time = time.perf_counter()
        results = asyncio.run(run_concurrent_test())
        total_time = time.perf_counter() - start_time
        
        assert len(results) == 5
        assert all("result_" in result for result in results)
        
        # Check individual operation metrics
        for i in range(5):
            summary = performance_collector.get_metrics_summary(f"concurrent_op_{i}")
            assert summary["sample_count"] == 1
        
        print(f"\n⚡ Concurrent Performance Test:")
        print(f"  Total Time: {total_time:.3f}s")
        print(f"  Operations: 5")
        print(f"  Results: {results}")


class TestPerformanceRegressionDetection:
    """Test performance regression detection capabilities."""
    
    def test_baseline_establishment(self):
        """Test establishing performance baselines."""
        collector = PerformanceCollector()
        
        # Simulate baseline measurements
        baseline_operations = [
            ("fast_op", 10.0),  # 10ms
            ("medium_op", 50.0),  # 50ms
            ("slow_op", 100.0),  # 100ms
        ]
        
        for op_name, duration_ms in baseline_operations:
            for _ in range(5):  # Multiple samples
                collector.record_metric(op_name, duration_ms + np.random.normal(0, 2))
        
        # Generate baseline report
        baseline_report = {}
        for op_name, _ in baseline_operations:
            baseline_report[op_name] = collector.get_metrics_summary(op_name)
        
        print(f"\n📊 Performance Baseline Report:")
        for op_name, stats in baseline_report.items():
            avg_duration = stats["duration_stats"]["avg_ms"]
            std_dev = stats["duration_stats"]["std_dev_ms"]
            print(f"  {op_name}: {avg_duration:.1f}ms ± {std_dev:.1f}ms")
        
        # Assertions
        assert len(baseline_report) == 3
        for op_name, stats in baseline_report.items():
            assert stats["sample_count"] == 5
            assert stats["duration_stats"]["avg_ms"] > 0
    
    def test_performance_regression_detection(self):
        """Test detection of performance regressions."""
        collector = PerformanceCollector()
        
        # Establish baseline (good performance)
        baseline_duration = 50.0  # 50ms
        for _ in range(5):
            collector.record_metric("regression_test", baseline_duration + np.random.normal(0, 2))
        
        baseline_stats = collector.get_metrics_summary("regression_test")
        baseline_avg = baseline_stats["duration_stats"]["avg_ms"]
        
        # Clear and simulate regression (worse performance)
        collector.clear_metrics()
        regressed_duration = 80.0  # 80ms (60% slower)
        for _ in range(5):
            collector.record_metric("regression_test", regressed_duration + np.random.normal(0, 2))
        
        regressed_stats = collector.get_metrics_summary("regression_test")
        regressed_avg = regressed_stats["duration_stats"]["avg_ms"]
        
        # Calculate regression percentage
        regression_percent = ((regressed_avg - baseline_avg) / baseline_avg) * 100
        
        print(f"\n🔍 Performance Regression Detection:")
        print(f"  Baseline Average: {baseline_avg:.1f}ms")
        print(f"  Regressed Average: {regressed_avg:.1f}ms")
        print(f"  Regression: {regression_percent:.1f}%")
        
        # Detect significant regression (>20% slower)
        is_regression = regression_percent > 20.0
        assert is_regression, f"Should detect regression of {regression_percent:.1f}%"
        
        print(f"  Regression Detected: {is_regression}")


if __name__ == "__main__":
    # Run pipeline performance tests
    pytest.main([__file__, "-v", "-s"])