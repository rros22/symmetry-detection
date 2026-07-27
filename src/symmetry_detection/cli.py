"""
Console-script entry points for the symmetry_detection package (see [project.scripts]
in pyproject.toml). These are intentionally headless (no plotting) so they behave well
in CI / non-interactive environments. For visualizations, see the scripts in examples/.
"""

import argparse

import numpy as np
from scipy.linalg import svd

from . import basis_functions as bf
from . import data_matrices as dm
from . import generate_trajectories as gtr


def generate_main():
    """Entry point for `symdet-generate`: generate and summarize a synthetic ODE trajectory dataset."""
    args = gtr.resolve_ode_args(gtr.build_ode_parser().parse_args())

    X = gtr.generate_equation_manifold(
        ode_name=args.ode,
        x_start=args.start,
        x_end=args.end,
        initial_conditions=args.initial_conditions,
        num_points=args.num_points,
        method=args.method,
    )
    print(
        f"Generated {X.shape[0]} trajectories x {X.shape[2]} points "
        f"for ODE '{args.ode}' (array shape {X.shape})."
    )


def detect_main():
    """Entry point for `symdet-detect`: run symmetry detection for a built-in demo ODE."""
    parser = argparse.ArgumentParser(
        description="Run symmetry detection for a built-in demo ODE.",
        parents=[gtr.build_ode_parser(add_help=False), dm.build_basis_parser(add_help=False)],
    )
    args = parser.parse_args()
    args = gtr.resolve_ode_args(args)
    args = dm.resolve_basis_args(args)

    X = gtr.generate_equation_manifold(
        ode_name=args.ode,
        x_start=args.start,
        x_end=args.end,
        initial_conditions=args.initial_conditions,
        num_points=args.num_points,
        method=args.method,
    )
    X_stacked = gtr.concatenate_trajectories(X)
    N = gtr.NORMALS[args.ode](X_stacked[0], X_stacked[1])

    G, L, L_x, L_u, P = dm.G_matrix(
        X_stacked, N,
        basis_type=args.basis_type,
        param_range=args.param_range,
        param_list=args.param_list,
    )

    _, S, Vt = svd(G.T)
    idx = np.argmin(S)
    xi, eta, zeta = bf.characteristic_functions(L, L_x, L_u, P, Vt[idx, :])

    print(f"Smallest singular value: {S[idx]:.3e} (out of {len(S)})")
    print("For a visualization of the result, see examples/run_odes_demo.py")
