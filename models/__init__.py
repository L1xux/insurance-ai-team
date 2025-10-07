"""
Data models package for the Data Analysis Project.

This package provides data models with type hints, validation,
and serialization capabilities.
"""

__version__ = "1.0.0"

from .review_model import ReviewData, ReviewDataBatch

__all__ = [
    "ReviewData",
    "ReviewDataBatch"
]