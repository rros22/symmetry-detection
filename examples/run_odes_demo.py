import argparse

import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import svd

from symmetry_detection import basis_functions as bf
from symmetry_detection import data_matrices as dm
from symmetry_detection import debugging as db
from symmetry_detection import generate_trajectories as gtr

# Default case run when this script is invoked with no CLI arguments. Edit any
# value here to change what runs by default; every field can still be overridden
# individually from the command line (e.g. `--ode bernoulli`). Leave a field as
# None to fall back to the shared defaults in generate_trajectories/data_matrices
# (e.g. ODE_DEFAULTS for that --ode, or (0, 3) for --param_range).
DEFAULT_CASE = dict(
    ode="bernoulli",
    basis_type="monomial",
    param_range=None,
    param_list=None,
    start=None,
    end=None,
    initial_conditions=None,
    num_points=None,
    method=None,
)

def get_args():
    parser = argparse.ArgumentParser(
        description="Run symmetry detection for the given ODE.",
        parents=[gtr.build_ode_parser(add_help=False), dm.build_basis_parser(add_help=False)],
    )
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

    # 2. Construct the G matrix
    G, L, L_x, L_u, P  = dm.G_matrix(X_stacked, N, basis_type=args.basis_type, param_range= args.param_range, param_list=args.param_list)

    # 3. Solve the null space problem (on G transposed) using the SVD
    U,S,Vt = svd(G.T)
    idx = np.argmin(S)

    # 4. Plot singular value spectrum
    fig, ax = plt.subplots(1,2)
    ax[0].semilogy(S,marker='o', linestyle='None')

    # 5. Plot the trajectories and the reconstructed vector field
    xi, eta, zeta = bf.characteristic_functions(L, L_x, L_u, P, Vt[idx, :])

    ax[1].quiver(X_stacked[0], X_stacked[1], xi, eta, angles='xy',scale=20,scale_units='xy')
    ax[1].set_box_aspect(1)

    for trajectory in X:
        ax[1].plot(trajectory[0], trajectory[1])

    # 6. Plot the jet space reconstructed tangent field
    V = np.stack((xi,eta,zeta))
    print(f"The shape of V is {V.shape}")
    print(f"The shape of X_stacked is {X_stacked.shape}")

    fig2, ax2 = db.scaled_3D_quiver(X_stacked, V, mode="tangent")
    fig2, ax2 = db.scaled_3D_quiver(X_stacked, N, mode="normal")
    
    plt.show()
    

if __name__ == "__main__":
    main()