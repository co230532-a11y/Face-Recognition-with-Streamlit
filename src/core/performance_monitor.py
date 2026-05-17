"""
Performance Monitoring Module
Track CPU, Memory, and Processing Time
"""

import psutil
import time
import logging
from pathlib import Path
import sys
import json
from datetime import datetime

# Add parent to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.config import LOGS_DIR, RESULTS_DIR, MEMORY_THRESHOLD_MB, CPU_THRESHOLD_PERCENT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'performance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system performance during face recognition"""
    
    def __init__(self):
        self.measurements = []
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.start_cpu_percent = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        try:
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.start_cpu_percent = self.process.cpu_percent(interval=0.1)
            
            logger.info(f"Monitoring started - Memory: {self.start_memory:.2f}MB, CPU: {self.start_cpu_percent:.2f}%")
            return True
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
    
    def stop_monitoring(self):
        """Stop performance monitoring and return metrics"""
        try:
            if self.start_time is None:
                logger.warning("Monitoring not started")
                return None
            
            elapsed_time = time.time() - self.start_time
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_used = current_memory - self.start_memory
            current_cpu_percent = self.process.cpu_percent(interval=0.1)
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'elapsed_time_seconds': elapsed_time,
                'elapsed_time_ms': elapsed_time * 1000,
                'memory_start_mb': self.start_memory,
                'memory_end_mb': current_memory,
                'memory_used_mb': memory_used,
                'cpu_percent': current_cpu_percent,
                'process_id': self.process.pid
            }
            
            # Check thresholds
            if memory_used > MEMORY_THRESHOLD_MB:
                logger.warning(f"Memory usage exceeded threshold: {memory_used:.2f}MB > {MEMORY_THRESHOLD_MB}MB")
            
            if current_cpu_percent > CPU_THRESHOLD_PERCENT:
                logger.warning(f"CPU usage exceeded threshold: {current_cpu_percent:.2f}% > {CPU_THRESHOLD_PERCENT}%")
            
            logger.info(f"Monitoring stopped - Time: {elapsed_time:.2f}s, Memory: {memory_used:.2f}MB, CPU: {current_cpu_percent:.2f}%")
            
            self.measurements.append(metrics)
            return metrics
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            return None
    
    def get_system_info(self):
        """Get overall system information"""
        try:
            system_stats = {
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_total_mb': psutil.virtual_memory().total / 1024 / 1024,
                'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
                'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
            
            logger.info(f"System Info - CPU: {system_stats['cpu_percent']:.2f}%, "
                       f"Memory: {system_stats['memory_percent']:.2f}%")
            
            return system_stats
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return None
    
    def save_measurements(self, filename="performance_metrics.json"):
        """Save performance measurements to file"""
        try:
            filepath = RESULTS_DIR / filename
            
            with open(filepath, 'w') as f:
                json.dump(self.measurements, f, indent=2)
            
            logger.info(f"Performance metrics saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save measurements: {e}")
            return None
    
    def get_average_metrics(self):
        """Calculate average metrics from all measurements"""
        try:
            if not self.measurements:
                logger.warning("No measurements recorded")
                return None
            
            avg_time = sum(m['elapsed_time_ms'] for m in self.measurements) / len(self.measurements)
            avg_memory = sum(m['memory_used_mb'] for m in self.measurements) / len(self.measurements)
            avg_cpu = sum(m['cpu_percent'] for m in self.measurements) / len(self.measurements)
            
            return {
                'num_measurements': len(self.measurements),
                'average_time_ms': avg_time,
                'average_memory_mb': avg_memory,
                'average_cpu_percent': avg_cpu,
                'max_memory_mb': max(m['memory_used_mb'] for m in self.measurements),
                'max_cpu_percent': max(m['cpu_percent'] for m in self.measurements)
            }
        except Exception as e:
            logger.error(f"Failed to calculate average metrics: {e}")
            return None


class TimingDecorator:
    """Decorator to measure function execution time"""
    
    def __init__(self, name="Operation"):
        self.name = name
        self.monitor = PerformanceMonitor()
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            self.monitor.start_monitoring()
            
            try:
                result = func(*args, **kwargs)
            finally:
                metrics = self.monitor.stop_monitoring()
                if metrics:
                    logger.info(f"{self.name} completed in {metrics['elapsed_time_ms']:.2f}ms, "
                               f"Memory: {metrics['memory_used_mb']:.2f}MB")
            
            return result
        
        return wrapper


# Context manager for timing
class Timer:
    """Context manager for timing code blocks"""
    
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.monitor = PerformanceMonitor()
    
    def __enter__(self):
        self.monitor.start_monitoring()
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        metrics = self.monitor.stop_monitoring()
        if metrics:
            logger.info(f"{self.name} completed in {metrics['elapsed_time_ms']:.2f}ms")
        
        return False


if __name__ == "__main__":
    print("Performance monitoring module loaded successfully")
