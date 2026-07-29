"""
Console-script entry points for the symmetry_detection package (see [project.scripts]
in pyproject.toml). These are intentionally headless (no plotting) so they behave well
in CI / non-interactive environments. For visualizations, see the scripts in examples/.
"""

import argparse

from . import data_matrices as dm
from . import generate_trajectories as gtr
from . import reporting as rpt
from . import solver


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
        rtol=args.rtol,
        atol=args.atol,
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
    parser.add_argument("--sv_choice", type=int, default=0,
                         help="Which of the small singular values to use (0 = largest of the "
                              "small ones, increasing towards the globally smallest)")
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
        rtol=args.rtol,
        atol=args.atol,
    )
    X_stacked = gtr.concatenate_trajectories(X)
    N = gtr.NORMALS[args.ode](X_stacked[0], X_stacked[1])

    svd_result = solver.solve_svd(
        X_stacked, N,
        basis_type=args.basis_type,
        param_range=args.param_range,
        param_list=args.param_list,
    )
    small_sv_idx = solver.small_singular_value_indices(svd_result.S)
    rpt.print_svd_summary(svd_result.S, small_sv_idx)

    idx = solver.pick_small_singular_value(small_sv_idx, sv_choice=args.sv_choice)
    result = solver.select_singular_vector(svd_result, idx)
    rpt.print_selection_summary(result.S, result.idx)
    print("For a visualization of the result, see examples/run_odes_demo.py")
