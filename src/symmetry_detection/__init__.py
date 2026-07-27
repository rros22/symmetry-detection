"""
symmetry_detection: data-driven detection of point symmetries in ODEs.
"""

from . import basis_functions, data_matrices, debugging, generate_trajectories

__all__ = [
    "basis_functions",
    "data_matrices",
    "debugging",
    "generate_trajectories",
]

__version__ = "0.1.0"
