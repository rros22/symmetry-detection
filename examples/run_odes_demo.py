import argparse

import numpy as np
from matplotlib import pyplot as plt

from symmetry_detection import data_matrices as dm
from symmetry_detection import generate_trajectories as gtr
from symmetry_detection import plotting as plot
from symmetry_detection import reporting as rpt
from symmetry_detection import solver

# Top-level defaults for this script, applied before CLI parsing. Any value
# passed explicitly on the command line (e.g. `--ode bernoulli`) always takes
# precedence over what's set here. Leave a field as None to fall back to the
# package-wide defaults in symmetry_detection.defaults instead (integration
# fields via resolve_ode_args, basis fields via resolve_basis_args).
DEFAULT_CASE = dict(
    ode="bernoulli",
    basis_type="monomial",
    param_range=(0, 3),
    param_list=None,
    start=None,
    end=None,
    initial_conditions=None,
    num_points=None,
    method=None,
    sv_choice=2,
)

def get_args():
    parser = argparse.ArgumentParser(
        description="Run symmetry detection for the given ODE.",
        parents=[gtr.build_ode_parser(add_help=False), dm.build_basis_parser(add_help=False)],
    )
    parser.add_argument("--sv_choice", type=int,
                         help="Which of the small singular values to use (0 = largest of the "
                              "small ones, increasing towards the globally smallest)")
    parser.set_defaults(**DEFAULT_CASE)
    args = parser.parse_args()
    args = gtr.resolve_ode_args(args)
    args = dm.resolve_basis_args(args)
    return args

def main():
    """
    Run symmetry detection for the given ODE
    """

    # 0. Read in CLI arguments
    args = get_args()
    print(f"Running symmetry detection for {args.ode}")

    # 1. Generate trajectory
    X = gtr.generate_equation_manifold(
        ode_name=args.ode,
        x_start=args.start,
        x_end=args.end,
        initial_conditions=args.initial_conditions,
        num_points=args.num_points,
        method=args.method)

    X_stacked = gtr.concatenate_trajectories(X)
    N = gtr.NORMALS[args.ode](X_stacked[0], X_stacked[1])

    # 2. Build the homogeneous system and solve its null space via SVD.
    svd_result = solver.solve_svd(
        X_stacked, N,
        basis_type=args.basis_type,
        param_range=args.param_range,
        param_list=args.param_list,
    )

    # 3. Identifying the "small" singular values and picking one to
    # reconstruct from are both explicit steps done here, after the SVD is
    # already solved - solve_svd itself makes no judgment about either.
    drop_ratio = 100
    small_sv_idx = solver.small_singular_value_indices(svd_result.S, drop_ratio=drop_ratio)
    rpt.print_svd_summary(svd_result.S, small_sv_idx)

    idx = solver.pick_small_singular_value(small_sv_idx, sv_choice=args.sv_choice)
    result = solver.select_singular_vector(svd_result, idx)
    rpt.print_selection_summary(result.S, result.idx)

    # Every figure below is derived from this one chosen singular vector, so
    # each one is titled/suptitled with its index for traceability.
    sigma_label = rf"$\sigma_{{{result.idx}}}$"

    # 4. Plot singular value spectrum; highlight post-drop values in red and
    # mark the one actually selected (relevant when the null space has
    # dimension > 1, i.e. more than one small singular value).
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(f"Selected singular vector: {sigma_label}")
    S_norm = result.S / result.S[0]
    all_idx = np.arange(len(result.S))
    large_sv_idx = np.setdiff1d(all_idx, small_sv_idx, assume_unique=True)

    ax[0].semilogy(large_sv_idx, S_norm[large_sv_idx], marker='o', linestyle='None', color='C0')
    ax[0].semilogy(small_sv_idx, S_norm[small_sv_idx], marker='o', linestyle='None', color='C3')
    ax[0].semilogy(result.idx, S_norm[result.idx], marker='*', markersize=14, linestyle='None', color='k', zorder=3)
    ax[0].set_xlabel("Index $i$")
    ax[0].set_ylabel(r"$\sigma_i / \sigma_1$")
    ax[0].set_title("Singular value spectrum (normalized)")
    ax[0].set_box_aspect(1)

    n_basis = len(result.param_list)

    # 4b. Bar charts of learnt xi / eta coefficients for the selected singular vector
    fig2, (ax_xi, ax_eta) = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle(f"Selected singular vector: {sigma_label}")
    plot.plot_basis_coefficient_bars(
        result.singular_vector[:n_basis], result.param_list, args.basis_type, ax=ax_xi, title=r"$\xi$ coefficients", log_scale=False
    )
    plot.plot_basis_coefficient_bars(
        result.singular_vector[n_basis:], result.param_list, args.basis_type, ax=ax_eta, title=r"$\eta$ coefficients", log_scale=False
    )
    fig2.tight_layout()

    """Xi and eta are not normalized, so the arrow length is not a good indicator of the local magnitude."""
    xy_norm = np.sqrt(result.xi**2 + result.eta**2)
    xy_norm[xy_norm == 0] = 1

    # 5. Plot the trajectories and the reconstructed vector field
    ax[1].quiver(X_stacked[0], X_stacked[1], result.xi / xy_norm, result.eta / xy_norm, angles='xy', scale=20)
    ax[1].set_box_aspect(1)


    for trajectory in X:
        ax[1].plot(trajectory[0], trajectory[1])

    ax[1].set_xlabel("x")
    ax[1].set_ylabel("u")
    ax[1].set_title("Trajectories and reconstructed infinitesimal generator (normalized)")
    ax[1].set_box_aspect(1)

    # 6. Plot the jet space reconstructed tangent and analytic normal fields side by side
    V = np.stack((result.xi, result.eta, result.zeta))

    fig3, ax3 = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))
    fig3.suptitle(f"Selected singular vector: {sigma_label}")
    plot.scaled_3D_quiver(X_stacked, V, mode="tangent", ax=ax3[0])
    ax3[0].set_title("Reconstructed infinitesimal generator (jet space)")
    plot.scaled_3D_quiver(X_stacked, N, mode="normal", ax=ax3[1])
    ax3[1].set_title("Analytic normal field")
    fig3.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
