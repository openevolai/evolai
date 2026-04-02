"""
Resource Management for Validator

Handles GPU memory, disk space, and cleanup with proper error recovery.
Based on Google DeepMind and OpenAI best practices.
"""

import os
import shutil
import tempfile
import psutil
import torch
from contextlib import contextmanager
from typing import Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """Resource limits for validator operations"""
    max_gpu_memory_fraction: float = 0.75  # Conservative 75% instead of 90%
    min_free_disk_gb: float = 50.0  # Require 50GB free disk
    max_model_load_timeout_seconds: int = 600  # 10 minutes max
    cleanup_on_oom: bool = True
    

class GPUMemoryManager:
    """Manages GPU memory with OOM recovery"""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.allocated_memory_gb = 0.0
    
    def check_available_memory(self, required_gb: float) -> bool:
        """
        Check if enough GPU memory is available
        
        Args:
            required_gb: Required memory in GB
            
        Returns:
            True if memory available
        """
        if not torch.cuda.is_available():
            return False
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            free_memory_gb = (total_memory - allocated) / 1e9
            
            logger.debug(f"GPU memory: free={free_memory_gb:.2f}GB, required={required_gb:.2f}GB")
            
            return free_memory_gb >= required_gb
        except Exception as e:
            logger.error(f"Failed to check GPU memory: {e}")
            return False
    
    def cleanup_gpu_memory(self, aggressive: bool = False):
        """
        Clean up GPU memory
        
        Args:
            aggressive: If True, also empty CUDA cache and collect garbage
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
                if aggressive:
                    import gc
                    gc.collect()
                    torch.cuda.synchronize()
                    
                allocated_after = torch.cuda.memory_allocated(0) / 1e9
                logger.info(f"GPU memory after cleanup: {allocated_after:.2f}GB allocated")
        except Exception as e:
            logger.error(f"GPU cleanup failed: {e}")
    
    @contextmanager
    def allocate_gpu_memory(self, model_name: str, estimated_size_gb: float):
        """
        Context manager for GPU memory allocation with automatic cleanup
        
        Args:
            model_name: Name of model being loaded
            estimated_size_gb: Estimated model size
            
        Yields:
            None if successful
            
        Raises:
            RuntimeError: If OOM or allocation fails
        """
        try:
            # Check if we have enough memory
            if not self.check_available_memory(estimated_size_gb):
                # Try aggressive cleanup first
                self.cleanup_gpu_memory(aggressive=True)
                
                if not self.check_available_memory(estimated_size_gb):
                    raise RuntimeError(
                        f"Insufficient GPU memory for {model_name}. "
                        f"Required: {estimated_size_gb:.2f}GB"
                    )
            
            logger.info(f"Allocating GPU memory for {model_name}")
            self.allocated_memory_gb += estimated_size_gb
            
            yield
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM loading {model_name}: {e}")
            self.cleanup_gpu_memory(aggressive=True)
            raise RuntimeError(f"GPU out of memory loading {model_name}") from e
            
        except Exception as e:
            logger.error(f"GPU allocation failed for {model_name}: {e}")
            raise
            
        finally:
            # Always cleanup on exit
            self.allocated_memory_gb -= estimated_size_gb
            self.cleanup_gpu_memory(aggressive=False)


class DiskSpaceManager:
    """Manages disk space with cleanup"""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.temp_dirs = []
    
    def check_disk_space(self, path: str = "/") -> float:
        """
        Check available disk space in GB
        
        Args:
            path: Path to check (default: root)
            
        Returns:
            Free space in GB
        """
        try:
            stat = psutil.disk_usage(path)
            free_gb = stat.free / 1e9
            return free_gb
        except Exception as e:
            logger.error(f"Failed to check disk space: {e}")
            return 0.0
    
    def ensure_disk_space(self, required_gb: float, path: str = "/"):
        """
        Ensure sufficient disk space is available
        
        Args:
            required_gb: Required space in GB
            path: Path to check
            
        Raises:
            RuntimeError: If insufficient space
        """
        free_gb = self.check_disk_space(path)
        
        if free_gb < self.limits.min_free_disk_gb:
            raise RuntimeError(
                f"Insufficient disk space. "
                f"Free: {free_gb:.2f}GB, Minimum required: {self.limits.min_free_disk_gb}GB"
            )
        
        if free_gb < required_gb:
            logger.warning(
                f"Low disk space. Free: {free_gb:.2f}GB, Requested: {required_gb:.2f}GB"
            )
    
    @contextmanager
    def temporary_directory(self, prefix: str = "evolai_"):
        """
        Create temporary directory with guaranteed cleanup
        
        Args:
            prefix: Prefix for temp directory name
            
        Yields:
            Path to temporary directory
        """
        temp_dir = None
        try:
            # Check disk space before creating
            self.ensure_disk_space(required_gb=10.0)  # Require 10GB for models
            
            temp_dir = tempfile.mkdtemp(prefix=prefix)
            self.temp_dirs.append(temp_dir)
            logger.debug(f"Created temp directory: {temp_dir}")
            
            yield temp_dir
            
        finally:
            # Always cleanup, even on exception
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    if temp_dir in self.temp_dirs:
                        self.temp_dirs.remove(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.error(f"Failed to cleanup temp directory {temp_dir}: {e}")
    
    def cleanup_all_temp_dirs(self):
        """Emergency cleanup of all tracked temp directories"""
        for temp_dir in self.temp_dirs[:]:  # Copy list to avoid modification during iteration
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"Emergency cleanup: {temp_dir}")
            except Exception as e:
                logger.error(f"Failed to cleanup {temp_dir}: {e}")
            finally:
                self.temp_dirs.remove(temp_dir)


class ResourceManager:
    """
    Unified resource manager for GPU, disk, and memory
    
    This is the main interface validators should use.
    """
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.gpu_manager = GPUMemoryManager(self.limits)
        self.disk_manager = DiskSpaceManager(self.limits)
    
    @contextmanager
    def managed_model_loading(
        self,
        model_name: str,
        estimated_size_gb: float = 20.0,
        temp_dir_prefix: str = "model_"
    ):
        """
        Comprehensive resource management for model loading
        
        Handles:
        - GPU memory allocation with OOM recovery
        - Disk space verification
        - Temporary directory with auto-cleanup
        - Error recovery and logging
        
        Args:
            model_name: Name of model to load
            estimated_size_gb: Estimated model size in GB
            temp_dir_prefix: Prefix for temp directory
            
        Yields:
            temp_dir: Path to temporary directory for model cache
            
        Example:
            with resource_mgr.managed_model_loading("user/model", 9.0) as temp_dir:
                model = AutoModel.from_pretrained(
                    "user/model",
                    cache_dir=temp_dir
                )
                # ... use model ...
            # GPU and disk automatically cleaned up
        """
        try:
            with self.disk_manager.temporary_directory(prefix=temp_dir_prefix) as temp_dir:
                with self.gpu_manager.allocate_gpu_memory(model_name, estimated_size_gb):
                    yield temp_dir
        except RuntimeError as e:
            logger.error(f"Resource allocation failed for {model_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in resource management: {e}")
            # Attempt emergency cleanup
            self.emergency_cleanup()
            raise
    
    def emergency_cleanup(self):
        """
        Emergency cleanup when something goes wrong
        
        Called automatically on exceptions, but can be called manually
        """
        logger.warning("Performing emergency resource cleanup")
        self.gpu_manager.cleanup_gpu_memory(aggressive=True)
        self.disk_manager.cleanup_all_temp_dirs()
    
    def get_resource_stats(self) -> dict:
        """
        Get current resource usage statistics
        
        Returns:
            Dict with GPU memory, disk space, etc.
        """
        stats = {}
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            stats['gpu_allocated_gb'] = round(allocated, 2)
            stats['gpu_reserved_gb'] = round(reserved, 2)
        
        disk_free = self.disk_manager.check_disk_space()
        stats['disk_free_gb'] = round(disk_free, 2)
        stats['temp_dirs_count'] = len(self.disk_manager.temp_dirs)
        
        return stats
