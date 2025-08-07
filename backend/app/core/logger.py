import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
import json
import os
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def setup_logger(name: str = "rag_app") -> logging.Logger:
    """Configure and return a logger with both file and console handlers"""
    
    # Create logs directory if not exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Base logger configuration
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # File handler (rotating logs)
    file_handler = RotatingFileHandler(
        filename=logs_dir / "rag_app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    console_handler.setLevel(logging.DEBUG if os.getenv("DEBUG") else logging.INFO)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Configure third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logger

# Global logger instance
logger = setup_logger()