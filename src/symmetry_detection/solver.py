"""
Core symmetry-detection pipeline, deliberately split into small independent
steps so the caller controls each decision explicitly rather than the
solver making judgment calls on their behalf:

    svd_result = solve_svd(X_stacked, N, basis_type, param_range, param_list)
    small_sv_idx = small_singular_value_indices(svd_result.S)   # your cutoff heuristic
    idx = pick_small_singular_value(small_sv_idx, sv_choice=0)  # your choice of which one
    result = select_singular_vector(svd_result, idx)            # reconstruct xi, eta, zeta

Shared by the CLI entry points and the example scripts so they can't drift
apart on how a symmetry is actually solved for.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd

from . import basis_functions as bf
from . import data_matrices as dm


@dataclass
class SVDResult:
    """Outcome of building the homogeneous system G and solving its null space via SVD."""

    G: np.ndarray
    L: np.ndarray
    L_x: np.ndarray
    L_u: np.ndarray
    P: np.ndarray
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    param_list: list


def solve_svd(X_stacked, N, basis_type, param_range=None, param_list=None):
    """
    Build the homogeneous system G for the given trajectories/normal field
    and solve its null space via SVD. This is pure linear algebra: it makes
    no judgment about which singular value(s) are "small" or which one to
    reconstruct from - see `small_singular_value_indices` and
    `select_singular_vector` for those, applied explicitly by the caller.
    """
    G, L, L_x, L_u, P = dm.G_matrix(
        X_stacked, N, basis_type=basis_type, param_range=param_range, param_list=param_list,
    )
    validated_param_list = dm.validate_parameters(basis_type, param_range, param_list)

    U, S, Vt = svd(G.T)

    return SVDResult(G=G, L=L, L_x=L_x, L_u=L_u, P=P, U=U, S=S, Vt=Vt, param_list=validated_param_list)


def small_singular_value_indices(S, drop_ratio=100.0):
    """
    Return indices of singular values at/after the first adjacent drop of
    drop_ratio (default: two orders of magnitude). If no such drop exists,
    return only the index of the smallest singular value.

    This is a standalone heuristic over S, not something `solve_svd` applies
    automatically - call it explicitly on `SVDResult.S` when you want it.
    """
    if len(S) <= 1:
        return np.array([np.argmin(S)])

    steep = np.flatnonzero(S[:-1] / S[1:] >= drop_ratio)
    if steep.size:
        return np.arange(steep[0] + 1, len(S))
    return np.array([np.argmin(S)])


def pick_small_singular_value(small_sv_idx, sv_choice=0):
    """
    Map a user-facing sv_choice to an absolute index into S/Vt: 0 selects the
    largest of the "small" singular values (as identified by
    `small_singular_value_indices`), up to len(small_sv_idx) - 1 for the
    globally smallest. An out-of-range sv_choice is clamped to the nearest
    valid index with a warning rather than raising, since it's a "which one"
    tuning knob, not a hard precondition.
    """
    max_choice = len(small_sv_idx) - 1
    if not (0 <= sv_choice <= max_choice):
        clamped_choice = min(max(sv_choice, 0), max_choice)
        print(
            f"Warning: sv_choice={sv_choice} is out of range ({len(small_sv_idx)} small "
            f"singular value(s) available, valid range 0 to {max_choice}). "
            f"Using sv_choice={clamped_choice} instead."
        )
        sv_choice = clamped_choice
    return small_sv_idx[sv_choice]


@dataclass
class SymmetryResult:
    """Everything needed to inspect or plot one chosen singular vector's reconstruction."""

    G: np.ndarray
    L: np.ndarray
    L_x: np.ndarray
    L_u: np.ndarray
    P: np.ndarray
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    param_list: list
    idx: int
    singular_vector: np.ndarray
    xi: np.ndarray
    eta: np.ndarray
    zeta: np.ndarray


def select_singular_vector(svd_result, idx):
    """
    Reconstruct the characteristic functions (xi, eta, zeta) of the
    generator associated with the singular vector at the given absolute
    index into svd_result.S / svd_result.Vt. idx is plain and explicit - the
    caller decides what it means (e.g. via `small_singular_value_indices` +
    `pick_small_singular_value`, or any other criterion).
    """
    if not (0 <= idx < len(svd_result.S)):
        raise ValueError(f"idx={idx} is out of range for {len(svd_result.S)} singular values.")

    singular_vector = svd_result.Vt[idx, :]
    xi, eta, zeta = bf.characteristic_functions(
        svd_result.L, svd_result.L_x, svd_result.L_u, svd_result.P, singular_vector,
    )

    return SymmetryResult(
        G=svd_result.G, L=svd_result.L, L_x=svd_result.L_x, L_u=svd_result.L_u, P=svd_result.P,
        U=svd_result.U, S=svd_result.S, Vt=svd_result.Vt, param_list=svd_result.param_list,
        idx=idx, singular_vector=singular_vector, xi=xi, eta=eta, zeta=zeta,
    )
