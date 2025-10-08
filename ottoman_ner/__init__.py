"""
Ottoman Turkish Named Entity Recognition Toolkit

A simple and modern toolkit for Ottoman Turkish NER tasks.
"""

from ._version import __version__

__author__ = "Fatih Burak Karagöz"
__email__ = "fatihburak@protonmail.com"

# Main interface
from .core import OttomanNER, DEFAULT_PRETRAINED_MODEL

# Essential utilities
from .utils import setup_logging

__all__ = [
    "OttomanNER",
    "setup_logging",
    "DEFAULT_PRETRAINED_MODEL",
    "__version__"
]
