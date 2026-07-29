"""
symmetry_detection: data-driven detection of point symmetries in ODEs.
"""

from . import basis_functions, data_matrices, defaults, generate_trajectories, plotting, reporting, solver, manifold

__all__ = [
    "basis_functions",
    "data_matrices",
    "defaults",
    "generate_trajectories",
    "plotting",
    "reporting",
    "solver",
    "manifold",
]

__version__ = "0.1.0"
