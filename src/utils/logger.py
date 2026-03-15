"""
Logging utilities for phenomobile project.
Provides consistent logging configuration across the application.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format: [LEVEL] message
        formatted = f"{log_color}[{record.levelname}]{reset} {record.getMessage()}"
        return formatted


def setup_logger(
    name: str = 'phenomobile',
    level: str = 'INFO',
    log_file: Optional[str] = None,
    verbose: bool = False
) -> logging.Logger:
    """
    Setup logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        verbose: Enable verbose output (DEBUG level)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set log level
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'phenomobile') -> logging.Logger:
    """
    Get existing logger or create new one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up with default configuration
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


def log_execution_time(func):
    """Decorator to log function execution time."""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise
    
    return wrapper


def log_step(step_name: str, step_num: Optional[int] = None, total_steps: Optional[int] = None):
    """Log a step in a multi-step process."""
    logger = get_logger()
    
    if step_num is not None and total_steps is not None:
        logger.info(f"Step {step_num}/{total_steps}: {step_name}")
    else:
        logger.info(f"Step: {step_name}")


def log_progress(current: int, total: int, description: str = "Processing"):
    """Log progress for long-running operations."""
    logger = get_logger()
    percentage = (current / total) * 100
    logger.info(f"{description}: {current}/{total} ({percentage:.1f}%)")


def log_data_info(df, name: str = "Dataset"):
    """Log basic information about a DataFrame."""
    logger = get_logger()
    
    if hasattr(df, 'shape'):
        logger.info(f"{name} shape: {df.shape}")
        logger.info(f"{name} columns: {list(df.columns)}")
    else:
        logger.info(f"{name}: {type(df)} - {len(df) if hasattr(df, '__len__') else 'unknown size'}")
