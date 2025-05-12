"""
Polyframe: A Python library for 3D transformations and geometry using homogenous matrices, with a focus on performance 
and ease of use, while allowing users to define their own coordinate frames.

More information is available at https://github.com/nsspencer/polyframe
"""

__version__ = version = "0.1.6"

# exposing the public API of the package
from polyframe._polyframe import (
    Direction,
    define_convention,
)

__all__ = [
    "Direction",
    "define_convention"
]
