"""
Performance Monitoring Utilities for FastRTC Voice Assistant.

Provides tools for measuring and analyzing performance across the voice processing pipeline.
"""

import time
import psutil
import threading
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: str
    operation: str
    duration_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class PerformanceTimer:
    """High-precision timer for measuring operation durations."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self):
        """Stop the timer and calculate duration."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        return self.duration
    
    def elapsed(self) -> float:
        """Get elapsed time without stopping the timer."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        return time.perf_counter() - self.start_time
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class SystemMonitor:
    """Monitor system resources during operations."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
        self._stop_event = threading.Event()
    
    def start_monitoring(self, interval: float = 0.1):
        """Start continuous monitoring of system resources."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._stop_event.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return summary statistics."""
        if not self.monitoring:
            return {}
        
        self._stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        self.monitoring = False
        
        if not self.metrics_history:
            return {}
        
        # Calculate summary statistics
        cpu_values = [m['cpu_percent'] for m in self.metrics_history]
        memory_values = [m['memory_mb'] for m in self.metrics_history]
        
        summary = {
            "avg_cpu_percent": statistics.mean(cpu_values),
            "max_cpu_percent": max(cpu_values),
            "avg_memory_mb": statistics.mean(memory_values),
            "max_memory_mb": max(memory_values),
            "sample_count": len(self.metrics_history)
        }
        
        # Clear history for next monitoring session
        self.metrics_history.clear()
        
        return summary
    
    def _monitor_loop(self, interval: float):
        """Internal monitoring loop."""
        while not self._stop_event.wait(interval):
            try:
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()
                
                self.metrics_history.append({
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_info.rss / 1024 / 1024,
                    "memory_vms_mb": memory_info.vms / 1024 / 1024
                })
            except Exception as e:
                logger.warning(f"Error in monitoring loop: {e}")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics snapshot."""
        try:
            memory_info = self.process.memory_info()
            return {
                "cpu_percent": self.process.cpu_percent(),
                "memory_mb": memory_info.rss / 1024 / 1024,
                "memory_vms_mb": memory_info.vms / 1024 / 1024,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {}


class PerformanceCollector:
    """Collect and aggregate performance metrics."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.system_monitor = SystemMonitor()
    
    def record_metric(
        self,
        operation: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a performance metric."""
        current_metrics = self.system_monitor.get_current_metrics()
        
        metric = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            duration_ms=duration_ms,
            memory_usage_mb=current_metrics.get("memory_mb", 0),
            cpu_usage_percent=current_metrics.get("cpu_percent", 0),
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        logger.debug(f"Recorded metric: {operation} - {duration_ms:.2f}ms")
    
    def get_metrics_summary(self, operation_filter: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for collected metrics."""
        filtered_metrics = self.metrics
        if operation_filter:
            filtered_metrics = [m for m in self.metrics if operation_filter in m.operation]
        
        if not filtered_metrics:
            return {}
        
        durations = [m.duration_ms for m in filtered_metrics]
        memory_usage = [m.memory_usage_mb for m in filtered_metrics]
        cpu_usage = [m.cpu_usage_percent for m in filtered_metrics]
        
        return {
            "operation_filter": operation_filter,
            "sample_count": len(filtered_metrics),
            "duration_stats": {
                "avg_ms": statistics.mean(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "median_ms": statistics.median(durations),
                "std_dev_ms": statistics.stdev(durations) if len(durations) > 1 else 0
            },
            "memory_stats": {
                "avg_mb": statistics.mean(memory_usage),
                "max_mb": max(memory_usage)
            },
            "cpu_stats": {
                "avg_percent": statistics.mean(cpu_usage),
                "max_percent": max(cpu_usage)
            }
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        try:
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_metrics": len(self.metrics),
                "metrics": [m.to_dict() for m in self.metrics],
                "summary": self.get_metrics_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported {len(self.metrics)} metrics to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
    
    def clear_metrics(self):
        """Clear all collected metrics."""
        self.metrics.clear()
        logger.debug("Cleared all metrics")


@contextmanager
def measure_performance(
    collector: PerformanceCollector,
    operation: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Context manager for measuring operation performance."""
    timer = PerformanceTimer(operation)
    collector.system_monitor.start_monitoring()
    
    try:
        timer.start()
        yield timer
    finally:
        duration_ms = timer.stop() * 1000  # Convert to milliseconds
        system_stats = collector.system_monitor.stop_monitoring()
        
        # Add system stats to metadata
        final_metadata = metadata or {}
        final_metadata.update(system_stats)
        
        collector.record_metric(operation, duration_ms, final_metadata)


def benchmark_function(
    func: Callable,
    operation_name: str,
    iterations: int = 5,
    collector: Optional[PerformanceCollector] = None
) -> Dict[str, Any]:
    """Benchmark a function with multiple iterations."""
    if collector is None:
        collector = PerformanceCollector()
    
    results = []
    
    for i in range(iterations):
        with measure_performance(collector, f"{operation_name}_iter_{i}"):
            try:
                result = func()
                results.append({"success": True, "result": result})
            except Exception as e:
                results.append({"success": False, "error": str(e)})
    
    # Calculate statistics
    successful_results = [r for r in results if r["success"]]
    success_rate = len(successful_results) / len(results)
    
    summary = collector.get_metrics_summary(operation_name)
    summary.update({
        "iterations": iterations,
        "success_rate": success_rate,
        "successful_iterations": len(successful_results)
    })
    
    return summary


class PipelineBenchmark:
    """Specialized benchmark for voice processing pipeline."""
    
    def __init__(self):
        self.collector = PerformanceCollector()
        self.stage_timers = {}
        self.current_pipeline_start = None
    
    def start_pipeline(self):
        """Start timing a complete pipeline run."""
        self.current_pipeline_start = time.perf_counter()
        self.stage_timers.clear()
    
    def start_stage(self, stage_name: str):
        """Start timing a pipeline stage."""
        self.stage_timers[stage_name] = time.perf_counter()
    
    def end_stage(self, stage_name: str, metadata: Optional[Dict[str, Any]] = None):
        """End timing a pipeline stage."""
        if stage_name not in self.stage_timers:
            logger.warning(f"Stage '{stage_name}' not started")
            return
        
        duration_ms = (time.perf_counter() - self.stage_timers[stage_name]) * 1000
        self.collector.record_metric(f"pipeline_stage_{stage_name}", duration_ms, metadata)
    
    def end_pipeline(self, metadata: Optional[Dict[str, Any]] = None):
        """End timing the complete pipeline."""
        if self.current_pipeline_start is None:
            logger.warning("Pipeline not started")
            return
        
        total_duration_ms = (time.perf_counter() - self.current_pipeline_start) * 1000
        self.collector.record_metric("pipeline_total", total_duration_ms, metadata)
        self.current_pipeline_start = None
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline performance."""
        total_stats = self.collector.get_metrics_summary("pipeline_total")
        stage_stats = {}
        
        # Get stats for each stage
        for metric in self.collector.metrics:
            if metric.operation.startswith("pipeline_stage_"):
                stage_name = metric.operation.replace("pipeline_stage_", "")
                if stage_name not in stage_stats:
                    stage_stats[stage_name] = self.collector.get_metrics_summary(f"pipeline_stage_{stage_name}")
        
        return {
            "total_pipeline": total_stats,
            "stages": stage_stats,
            "total_samples": len([m for m in self.collector.metrics if m.operation == "pipeline_total"])
        }


# Global performance collector instance
global_collector = PerformanceCollector()


def get_global_collector() -> PerformanceCollector:
    """Get the global performance collector instance."""
    return global_collector