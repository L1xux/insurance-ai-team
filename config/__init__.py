"""
Configuration package for the Data Analysis Project.

This package provides configuration management, logging setup,
and environment-specific settings.
"""

from .settings import settings
from .logging_config import setup_logging

__all__ = [
    "settings", 
    "setup_logging"
]
