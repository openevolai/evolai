"""
Metrics Collection for EvolAI Validator

Provides metrics collection, aggregation, and export for monitoring validator performance.
Compatible with Prometheus, W&B, and custom monitoring systems.
"""

import time
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """Thread-safe counter metric"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()
    
    def inc(self, amount: float = 1.0):
        """Increment counter"""
        with self._lock:
            self._value += amount
    
    def get(self) -> float:
        """Get current value"""
        with self._lock:
            return self._value
    
    def reset(self):
        """Reset counter to zero"""
        with self._lock:
            self._value = 0.0


class Gauge:
    """Thread-safe gauge metric (can go up or down)"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()
    
    def set(self, value: float):
        """Set gauge value"""
        with self._lock:
            self._value = value
    
    def inc(self, amount: float = 1.0):
        """Increment gauge"""
        with self._lock:
            self._value += amount
    
    def dec(self, amount: float = 1.0):
        """Decrement gauge"""
        with self._lock:
            self._value -= amount
    
    def get(self) -> float:
        """Get current value"""
        with self._lock:
            return self._value


class Histogram:
    """Thread-safe histogram for tracking distributions"""
    
    def __init__(self, name: str, description: str = "", buckets: Optional[List[float]] = None):
        self.name = name
        self.description = description
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._observations: List[float] = []
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()
    
    def observe(self, value: float):
        """Record an observation"""
        with self._lock:
            self._observations.append(value)
            self._sum += value
            self._count += 1
            
            # Keep only last 10000 observations to prevent memory issues
            if len(self._observations) > 10000:
                self._observations = self._observations[-10000:]
    
    def get_stats(self) -> Dict[str, float]:
        """Get histogram statistics"""
        with self._lock:
            if not self._observations:
                return {
                    "count": 0,
                    "sum": 0.0,
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                }
            
            sorted_obs = sorted(self._observations)
            count = len(sorted_obs)
            
            return {
                "count": count,
                "sum": self._sum,
                "mean": self._sum / count,
                "min": sorted_obs[0],
                "max": sorted_obs[-1],
                "p50": sorted_obs[int(count * 0.5)],
                "p95": sorted_obs[int(count * 0.95)],
                "p99": sorted_obs[int(count * 0.99)],
            }


class MetricsCollector:
    """
    Central metrics collection system
    
    Provides counters, gauges, and histograms for monitoring validator operations.
    """
    
    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()
        
        # Initialize standard validator metrics
        self._init_standard_metrics()
    
    def _init_standard_metrics(self):
        """Initialize standard validator metrics"""
        # Evaluation metrics
        self.register_counter("evaluations_total", "Total number of evaluations")
        self.register_counter("evaluations_success", "Successful evaluations")
        self.register_counter("evaluations_failed", "Failed evaluations")
        self.register_histogram("evaluation_duration_seconds", "Evaluation duration")
        
        # Model loading metrics
        self.register_counter("model_loads_total", "Total model loads")
        self.register_counter("model_loads_success", "Successful model loads")
        self.register_counter("model_loads_failed", "Failed model loads")
        self.register_histogram("model_load_duration_seconds", "Model load duration")
        
        # Error metrics
        self.register_counter("errors_total", "Total errors")
        self.register_counter("oom_errors_total", "GPU OOM errors")
        self.register_counter("network_errors_total", "Network errors")
        self.register_counter("retries_total", "Total retries")
        
        # Circuit breaker metrics
        self.register_counter("circuit_breaker_opens", "Circuit breaker opens")
        self.register_gauge("circuit_breaker_state", "Circuit breaker state (0=closed, 1=open, 2=half-open)")
        
        # Resource metrics
        self.register_gauge("gpu_memory_used_bytes", "GPU memory used")
        self.register_gauge("gpu_memory_available_bytes", "GPU memory available")
        self.register_gauge("disk_space_free_bytes", "Disk space free")
        
        # Weight setting metrics
        self.register_counter("weight_updates_total", "Total weight updates")
        self.register_counter("weight_updates_success", "Successful weight updates")
        self.register_counter("weight_updates_failed", "Failed weight updates")
        
        logger.info("Initialized standard validator metrics")
    
    def register_counter(self, name: str, description: str = "") -> Counter:
        """Register a new counter metric"""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description)
            return self._counters[name]
    
    def register_gauge(self, name: str, description: str = "") -> Gauge:
        """Register a new gauge metric"""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description)
            return self._gauges[name]
    
    def register_histogram(self, name: str, description: str = "", buckets: Optional[List[float]] = None) -> Histogram:
        """Register a new histogram metric"""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, description, buckets)
            return self._histograms[name]
    
    def get_counter(self, name: str) -> Optional[Counter]:
        """Get a counter by name"""
        return self._counters.get(name)
    
    def get_gauge(self, name: str) -> Optional[Gauge]:
        """Get a gauge by name"""
        return self._gauges.get(name)
    
    def get_histogram(self, name: str) -> Optional[Histogram]:
        """Get a histogram by name"""
        return self._histograms.get(name)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary"""
        metrics = {}
        
        with self._lock:
            # Counters
            for name, counter in self._counters.items():
                metrics[name] = {
                    "type": "counter",
                    "value": counter.get(),
                    "description": counter.description,
                }
            
            # Gauges
            for name, gauge in self._gauges.items():
                metrics[name] = {
                    "type": "gauge",
                    "value": gauge.get(),
                    "description": gauge.description,
                }
            
            # Histograms
            for name, histogram in self._histograms.items():
                metrics[name] = {
                    "type": "histogram",
                    "stats": histogram.get_stats(),
                    "description": histogram.description,
                }
        
        return metrics
    
    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format
        
        Returns:
            Prometheus-formatted metrics string
        """
        lines = []
        
        with self._lock:
            # Export counters
            for name, counter in self._counters.items():
                if counter.description:
                    lines.append(f"# HELP {name} {counter.description}")
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {counter.get()}")
            
            # Export gauges
            for name, gauge in self._gauges.items():
                if gauge.description:
                    lines.append(f"# HELP {name} {gauge.description}")
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {gauge.get()}")
            
            # Export histograms
            for name, histogram in self._histograms.items():
                if histogram.description:
                    lines.append(f"# HELP {name} {histogram.description}")
                lines.append(f"# TYPE {name} histogram")
                
                stats = histogram.get_stats()
                lines.append(f"{name}_count {stats['count']}")
                lines.append(f"{name}_sum {stats['sum']}")
        
        return "\n".join(lines) + "\n"


class Timer:
    """
    Context manager for timing operations and recording to histogram
    
    Usage:
        with Timer(metrics.get_histogram("operation_duration")):
            do_operation()
    """
    
    def __init__(self, histogram: Optional[Histogram] = None, callback: Optional[callable] = None):
        """
        Initialize timer
        
        Args:
            histogram: Histogram to record duration to
            callback: Optional callback to call with duration
        """
        self.histogram = histogram
        self.callback = callback
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        """Start timing"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record"""
        self.duration = time.time() - self.start_time
        
        if self.histogram:
            self.histogram.observe(self.duration)
        
        if self.callback:
            self.callback(self.duration)
        
        return False
    
    def get_duration(self) -> Optional[float]:
        """Get measured duration"""
        return self.duration


# Global metrics instance
_global_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics
