"""
Logging setup for AFET
"""

import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime


class AFETLogger:
    """
    Custom logger for AFET
    """
    
    def __init__(self, 
                 name: str = 'afet',
                 level: str = 'INFO',
                 log_dir: Optional[str] = None):
        """
        Initialize logger
        """
        self.name = name
        self.level = level.upper()
        self.log_dir = log_dir or str(Path.home() / '.afet' / 'logs')
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Add handlers
        self._add_handlers()
    
    def _add_handlers(self) -> None:
        """
        Add logging handlers
        """
        # File handler
        log_file = os.path.join(
            self.log_dir,
            f'{self.name}_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self._get_formatter())
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(self._get_formatter())
        self.logger.addHandler(console_handler)
    
    def _get_formatter(self) -> logging.Formatter:
        """
        Get logging formatter
        """
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger
        """
        return self.logger

# Initialize global logger
logger = AFETLogger('afet').get_logger()
