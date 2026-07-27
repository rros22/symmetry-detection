"""
Visualize a generated ODE trajectory manifold together with its analytic normal field.

Requires the package to be installed (e.g. `pip install -e .` from the repo root).

Usage:
    python examples/plot_ode_manifold.py --ode bernoulli
"""

import numpy as np
from matplotlib import pyplot as plt

from symmetry_detection import debugging as db
from symmetry_detection import generate_trajectories as gtr

# Default case run when this script is invoked with no CLI arguments. Edit any
# value here to change what runs by default; every field can still be overridden
# individually from the command line (e.g. `--ode bernoulli`). Leave fields as
# None to fall back via resolve_ode_args to ODE_DEFAULTS[ode].
DEFAULT_CASE = dict(
    ode="bernoulli",
    start=None,
    end=None,
    initial_conditions=None,
    num_points=None,
    method=None,
)


def get_args():
    parser = gtr.build_ode_parser()
    parser.set_defaults(**DEFAULT_CASE)
    args = parser.parse_args()
    args = gtr.resolve_ode_args(args)
    return args


def main():
    args = get_args()

    X = gtr.generate_equation_manifold(
        ode_name=args.ode,
        x_start=args.start,
        x_end=args.end,
        initial_conditions=args.initial_conditions,
        num_points=args.num_points,
        method=args.method,
    )

    fig = plt.figure(figsize=(6, 6))
    ax0 = fig.add_subplot(131, projection='3d')
    ax1 = fig.add_subplot(132, projection='3d')
    ax2 = fig.add_subplot(133)

    ax2.set_box_aspect(1)
    ax0.set_aspect('equal')

    # Define grid of values of surface manifold over [x_min,x_max] x [u_min,u_max].
    x_min = np.min(X[:, 0, :])
    x_max = np.max(X[:, 0, :])
    u_min = np.min(X[:, 1, :])
    u_max = np.max(X[:, 1, :])

    res = 20
    X_grid = np.linspace(x_min, x_max, res)
    U_grid = np.linspace(u_min, u_max, res)
    X_grid, U_grid = np.meshgrid(X_grid, U_grid)
    P_grid = gtr.ODES[args.ode](X_grid, U_grid)

    p_grid_min = np.min(P_grid)
    p_grid_max = np.max(P_grid)

    # Evaluate analytic normal field on the surface
    fig2, ax3 = db.scaled_3D_quiver_surface(
        X_grid, U_grid, P_grid, gtr.NORMALS[args.ode](X_grid, U_grid),
        (x_min, x_max), (u_min, u_max), (p_grid_min, p_grid_max),
    )
    X_stacked = gtr.concatenate_trajectories(X)
    fig3, ax4 = db.scaled_3D_quiver_trajectories(
        X_stacked, gtr.NORMALS[args.ode], style='lines', num_points=args.num_points)

    # Plotting
    ax0.plot_surface(X_grid, U_grid, P_grid, cmap='coolwarm')

    for trajectory in X:
        ax1.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :])
        ax2.plot(trajectory[0, :], trajectory[1, :])

    plt.show()


if __name__ == "__main__":
    main()
