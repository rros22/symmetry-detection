"""
    Methods for the numerical estimation of trajectory derivatives, and
    the normal space of the equation manifold.
"""

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

def estimate_first_derivative(solution, gaussian_sigma=0, window_length=11, polyorder=3):
    """
        Estimate the first derivative of a (x,u) trajectory using Gaussian smoothing to
        pre-process the data + Savitzky-Golay differentiation.
    """

    # 1. Extract the time series from the solution object
    x = solution.t
    u = solution.y

    # 2. Check if the sampling is uniform. Otherwise SG filter will fail
    dx = np.diff(x)
    if not np.allclose(dx, dx[0]):
        raise ValueError("Savitzky-Golay requires uniformly sampled data")

    h = dx[0]

    # 3. Apply gaussian smoothing
    if gaussian_sigma > 0:
        u_smoothed = gaussian_filter1d(
            u,
            sigma=gaussian_sigma,
            axis=1,
            mode="nearest"
        )
    else:
        u_smoothed = u

    # 4. Compute derivative
    du_dx = savgol_filter(
        u_smoothed,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=h,
        axis=1,
        mode="interp"
    )

    return du_dx


"""
    2D (scattered-data) generalization of Savitzky-Golay, used to estimate the
    normal field N = (-f_x, -f_u, 1) of the equation manifold u_x = f(x, u)
    directly from a multi-trajectory point cloud.

    A single trajectory only samples the 1D curve u(x); differentiating its own
    u_x along x gives u_xx = D_x(u_x) = f_x + u_x * f_u - one scalar equation in
    the two unknowns f_x, f_u at every point, which is not enough to recover
    them separately. Doing so requires genuine variation in u at (approximately)
    fixed x, i.e. information from *neighboring trajectories*, not just further
    points along the same one - hence the differentiation has to be done in the
    (x, u) plane (jointly across trajectories), not along a single trajectory.

    generate_equation_manifold builds every trajectory with the identical
    t_eval = np.linspace(x_start, x_end, num_points), so X[i, 0, j] = x_j is
    shared across all trajectories i. That means index-adjacency in (i, j)
    space (trajectory index, point-along-x index) already tracks spatial
    adjacency in (x, u): moving in j moves along one trajectory (fixed x
    spacing), and moving in i at fixed j moves across trajectories at exactly
    the same x (only u changes). So local neighborhoods for a 2D analogue of
    Savitzky-Golay can be built by plain array-index slicing - no KD-tree or
    generic nearest-neighbor search needed.
"""


@dataclass
class NormalFieldEstimate:
    """
    Result of estimate_normal_field. N is aligned with
    X[order][traj_slice, :, point_slice] - use .apply(X) to extract the
    matching (and, if discard_boundary was used, cropped) subset of X (or any
    other array sharing X's trajectory/point axes, e.g. X itself before
    concatenate_trajectories).
    """

    N: np.ndarray
    order: np.ndarray
    traj_slice: slice
    point_slice: slice
    cond: np.ndarray

    def apply(self, X):
        """Return X reordered/cropped to align with self.N."""
        return X[self.order][self.traj_slice, ..., self.point_slice]


def _local_design_matrix(dx, du, degree):
    """
    Build the local polynomial design-matrix columns for a scalar field fit
    around a reference point, from coordinate offsets dx = x - x0, du = u - u0.

        degree=1: [1, dx, du]                       (local plane / gradient only)
        degree=2: [1, dx, du, dx**2, dx*du, du**2]   (adds curvature terms)

    dx, du must broadcast against each other. The linear coefficients (index 1
    and 2) are the estimated partial derivatives w.r.t. x and u respectively,
    for either degree - this mirrors `polyorder` in estimate_first_derivative:
    a wider window with degree=2 trades a larger effective footprint for less
    bias from curvature within the window.
    """
    ones = np.ones_like(dx)
    if degree == 1:
        cols = [ones, dx, du]
    elif degree == 2:
        cols = [ones, dx, du, dx * dx, dx * du, du * du]
    else:
        raise ValueError(f"Unsupported degree={degree}; only 1 and 2 are implemented.")
    return np.stack(cols, axis=-1)


def _warn_if_ill_conditioned(cond, threshold=1e8):
    max_cond = np.nanmax(cond) if np.any(np.isfinite(cond)) else np.nan
    if np.isfinite(max_cond) and max_cond > threshold:
        n_bad = int(np.nansum(cond > threshold))
        print(
            f"Warning: estimate_normal_field has {n_bad} point(s) with a poorly "
            f"conditioned local fit (max condition number {max_cond:.2e} > "
            f"{threshold:.0e}). Consider a larger window, a lower degree, or "
            f"denser/more spread-out trajectories."
        )


def estimate_normal_field(X, window=(2, 5), degree=1, discard_boundary=True, boundary_margin=0):
    """
        Estimate the normal field N = (-f_x, -f_u, 1) directly from a
        multi-trajectory point cloud, via local (per-point) polynomial
        regression over a structured neighborhood in (trajectory-index,
        point-index) space - the scattered-data generalization of
        estimate_first_derivative's 1D Savitzky-Golay differentiation.

        X: array of shape (num_traj, 3, num_points), as returned by
           generate_equation_manifold. X[:, 0, :] must be identical across
           trajectories (every trajectory sampled on the same x grid).
        window: (di, dj) half-widths of the local neighborhood in
           (trajectory-index, point-index) space, i.e. neighbors of point
           (i0, j0) are X[i0-di:i0+di+1, :, j0-dj:j0+dj+1].
        degree: 1 (local plane / gradient only) or 2 (adds curvature terms;
           see _local_design_matrix).
        discard_boundary: the manifold sampled by the trajectories is not
           compact, so the first/last di trajectories and first/last dj
           points along x only have one-sided (truncated) windows and thus
           structurally less reliable gradient estimates. When True
           (default), those boundary points are excluded from the
           returned estimate entirely (N only covers the interior
           rectangle) - they are still used as neighbors/context for the
           interior points' windows, just not themselves reported as valid
           estimates. When False, boundary points instead get a
           best-effort estimate from their truncated window (a per-point
           least-squares fallback).
        boundary_margin: int or (margin_i, margin_j) - discard this many
           *additional* trajectories/points beyond what discard_boundary
           already removes, in case even fully-windowed fits near the edge
           are still untrustworthy (e.g. trajectories fanning in/out makes
           the physical neighborhood size vary, or you simply want a wider
           window for the fit itself - larger `window` - than the region
           you actually report estimates for). Applied identically
           regardless of discard_boundary: it always further *crops* the
           returned N (rather than masking with NaN), so N/apply() stay a
           dense rectangle. Note that with discard_boundary=False, the
           within-margin boundary fallback fits are still computed and then
           discarded by the crop; use discard_boundary=True instead if you
           don't need them for anything else, to skip that wasted work.

        Returns a NormalFieldEstimate:
            N: (3, num_traj', num_points') - N[0]=-f_x, N[1]=-f_u, N[2]=1.
               num_traj'=num_traj-2*(di+margin_i), num_points'=num_points-2*
               (dj+margin_j) when discard_boundary=True, otherwise
               num_traj'=num_traj-2*margin_i, num_points'=num_points-2*
               margin_j (with NaNs only at points where even the truncated
               boundary fit had too few neighbors).
            order: permutation of trajectory indices that sorts
               trajectories by u at the first shared x sample - required
               because index-adjacency across trajectories is only a valid
               proxy for spatial adjacency in u once they are ordered by u.
               (defaults.ODE_DEFAULTS initial_conditions are already
               ascending, so this is a no-op there, but this makes the
               function correct regardless of input order.)
            traj_slice, point_slice: slices into X[order] along the
               trajectory/point axes that N is aligned with - use
               `est.apply(X)` rather than applying these directly.
            cond: (num_traj', num_points') condition number of each local
               fit's design matrix, for diagnosing unreliable estimates
               (e.g. where trajectories bunch up locally in u).

        Typical usage, to keep N column-aligned with X_stacked downstream:

            est = estimate_normal_field(X, window=(2, 5), degree=1)
            X_trim = est.apply(X)
            X_stacked_trim = concatenate_trajectories(X_trim)
            N = est.N.reshape(3, -1)
    """
    if X.ndim != 3 or X.shape[1] != 3:
        raise ValueError(f"X must have shape (num_traj, 3, num_points); got {X.shape}")

    num_traj, _, num_points = X.shape
    di, dj = window
    if isinstance(boundary_margin, (int, np.integer)):
        margin_i = margin_j = int(boundary_margin)
    else:
        margin_i, margin_j = boundary_margin
    if margin_i < 0 or margin_j < 0:
        raise ValueError(f"boundary_margin must be non-negative, got {boundary_margin}.")

    if degree == 1:
        n_coeffs = 3
    elif degree == 2:
        n_coeffs = 6
    else:
        raise ValueError(f"Unsupported degree={degree}; only 1 and 2 are implemented.")

    if num_traj < 2 * di + 1:
        raise ValueError(
            f"window trajectory half-width di={di} needs at least {2 * di + 1} "
            f"trajectories, got {num_traj}."
        )
    if num_points < 2 * dj + 1:
        raise ValueError(
            f"window point half-width dj={dj} needs at least {2 * dj + 1} points "
            f"per trajectory, got {num_points}."
        )

    window_size = (2 * di + 1) * (2 * dj + 1)
    if window_size < n_coeffs + 2:
        raise ValueError(
            f"window={window} gives only {window_size} neighbors per fit, too few "
            f"to reliably fit {n_coeffs} coefficients (degree={degree}). Increase "
            f"window or reduce degree."
        )

    x = X[:, 0, :]
    if not np.allclose(x, x[0]):
        raise ValueError(
            "estimate_normal_field requires every trajectory to share the same x "
            "grid (same t_eval), so that trajectory-index adjacency is a valid "
            "proxy for spatial adjacency in u."
        )
    x_shared = x[0]

    # Sort trajectories by u at the first shared x sample, so index-adjacency
    # across trajectories tracks spatial adjacency in u (a no-op for the
    # already-ascending initial_conditions used throughout this package).
    order = np.argsort(X[:, 1, 0])
    u = X[order, 1, :]
    u_x = X[order, 2, :]

    # --- Interior points: fully vectorized, batched least-squares solve ---
    u_windows = sliding_window_view(u, (2 * di + 1, 2 * dj + 1))
    y_windows = sliding_window_view(u_x, (2 * di + 1, 2 * dj + 1))
    x_windows = sliding_window_view(x_shared, 2 * dj + 1)  # (num_points-2*dj, 2*dj+1)

    n_i, n_j = u_windows.shape[:2]  # num_traj-2*di, num_points-2*dj

    du = u_windows - u_windows[:, :, di:di + 1, dj:dj + 1]
    dx = (x_windows - x_windows[:, dj:dj + 1])[None, :, None, :]
    dx = np.broadcast_to(dx, du.shape)

    A = _local_design_matrix(dx, du, degree).reshape(n_i * n_j, window_size, n_coeffs)
    y = y_windows.reshape(n_i * n_j, window_size)

    AtA = np.einsum("nki,nkj->nij", A, A)
    Aty = np.einsum("nki,nk->ni", A, y)
    coeffs = np.linalg.solve(AtA, Aty)
    cond_interior = np.linalg.cond(A).reshape(n_i, n_j)

    f_x_interior = coeffs[:, 1].reshape(n_i, n_j)
    f_u_interior = coeffs[:, 2].reshape(n_i, n_j)

    if discard_boundary:
        if 2 * margin_i >= n_i or 2 * margin_j >= n_j:
            raise ValueError(
                f"boundary_margin={boundary_margin} leaves no interior points given "
                f"window={window} and X's shape {X.shape} (the window's own interior "
                f"is {n_i}x{n_j} trajectories/points); reduce boundary_margin or "
                f"window, or increase the number of trajectories/points in X."
            )
        traj_sel = slice(margin_i, n_i - margin_i) if margin_i else slice(None)
        point_sel = slice(margin_j, n_j - margin_j) if margin_j else slice(None)
        f_x_final = f_x_interior[traj_sel, point_sel]
        f_u_final = f_u_interior[traj_sel, point_sel]
        cond_final = cond_interior[traj_sel, point_sel]
        N = np.stack([-f_x_final, -f_u_final, np.ones_like(f_x_final)], axis=0)
        _warn_if_ill_conditioned(cond_final)
        return NormalFieldEstimate(
            N=N, order=order,
            traj_slice=slice(di + margin_i, num_traj - di - margin_i),
            point_slice=slice(dj + margin_j, num_points - dj - margin_j),
            cond=cond_final,
        )

    # --- Boundary points: per-point least-squares fallback over truncated windows ---
    f_x = np.full((num_traj, num_points), np.nan)
    f_u = np.full((num_traj, num_points), np.nan)
    cond = np.full((num_traj, num_points), np.nan)
    f_x[di:num_traj - di, dj:num_points - dj] = f_x_interior
    f_u[di:num_traj - di, dj:num_points - dj] = f_u_interior
    cond[di:num_traj - di, dj:num_points - dj] = cond_interior

    for i0 in range(num_traj):
        for j0 in range(num_points):
            if di <= i0 < num_traj - di and dj <= j0 < num_points - dj:
                continue  # already filled in from the vectorized interior solve

            i_lo, i_hi = max(0, i0 - di), min(num_traj, i0 + di + 1)
            j_lo, j_hi = max(0, j0 - dj), min(num_points, j0 + dj + 1)

            du_local = (u[i_lo:i_hi, j_lo:j_hi] - u[i0, j0]).ravel()
            dx_local = np.broadcast_to(
                (x_shared[j_lo:j_hi] - x_shared[j0])[None, :], (i_hi - i_lo, j_hi - j_lo)
            ).ravel()
            y_local = u_x[i_lo:i_hi, j_lo:j_hi].ravel()

            if du_local.size < n_coeffs + 2:
                continue  # too few neighbors at this corner/edge - leave as NaN

            A_local = _local_design_matrix(dx_local, du_local, degree)
            coeffs_local, *_ = np.linalg.lstsq(A_local, y_local, rcond=None)
            f_x[i0, j0] = coeffs_local[1]
            f_u[i0, j0] = coeffs_local[2]
            cond[i0, j0] = np.linalg.cond(A_local)

    if 2 * margin_i >= num_traj or 2 * margin_j >= num_points:
        raise ValueError(
            f"boundary_margin={boundary_margin} leaves no points given X's shape "
            f"{X.shape}; reduce boundary_margin."
        )
    traj_slice = slice(margin_i, num_traj - margin_i) if margin_i else slice(None)
    point_slice = slice(margin_j, num_points - margin_j) if margin_j else slice(None)

    N_full = np.stack([-f_x, -f_u, np.ones_like(f_x)], axis=0)
    N = N_full[:, traj_slice, point_slice]
    cond = cond[traj_slice, point_slice]
    _warn_if_ill_conditioned(cond)
    return NormalFieldEstimate(
        N=N, order=order, traj_slice=traj_slice, point_slice=point_slice, cond=cond,
    )


def _subsample_indices(n, target, stride):
    """Evenly-spaced index selection: exactly `target` indices in [0, n) if given
    (falling back to fewer only if rounding collides), else plain `::stride` decimation."""
    if target is not None:
        if target < 1:
            raise ValueError(f"target count must be >= 1, got {target}.")
        if target >= n:
            return np.arange(n)
        return np.unique(np.round(np.linspace(0, n - 1, target)).astype(int))
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")
    return np.arange(0, n, stride)


def subsample_trajectories(X, N, num_traj=None, num_points=None, traj_stride=1, point_stride=1):
    """
    Subsample the trajectory/point grid of an aligned (X, N) pair, so a large,
    dense point cloud can be used to fit a good normal field - estimate_normal_field
    benefits from many trajectories/points, since it is only their density that
    lets f_x and f_u be resolved independently and locally - while feeding a
    much smaller, evenly-spaced subset into the symmetry-detection linear
    algebra (solver.solve_svd/data_matrices.G_matrix): that algorithm itself
    works well, and stays computationally cheap, with relatively few
    trajectories/points; it is only the numerical derivative/normal-field
    estimation that needs density.

    X: (num_traj, 3, num_points), e.g. NormalFieldEstimate.apply(X_full).
    N: (3, num_traj, num_points) - same trajectory/point axes as X (in N's own
       axis order), already column-aligned with X (e.g. NormalFieldEstimate.N,
       or gtr.NORMALS[ode_name](...) reshaped like N in the analytic case).

    num_traj, num_points: if given, keep exactly this many trajectories/points,
       evenly spaced by index (not interpolated) across the available range;
       takes precedence over traj_stride/point_stride. May return slightly
       fewer than requested if rounding produces duplicate indices.
    traj_stride, point_stride: otherwise, simple decimation - keep every
       k-th trajectory/point (equivalent to X[::traj_stride, :, ::point_stride]).

    Returns X_sub, N_sub, subsampled with the *same* selected trajectory/point
    indices for both, so concatenate_trajectories(X_sub) stays column-aligned
    with N_sub.reshape(3, -1).
    """
    if X.ndim != 3 or X.shape[1] != 3:
        raise ValueError(f"X must have shape (num_traj, 3, num_points); got {X.shape}")
    if N.ndim != 3 or N.shape[0] != 3:
        raise ValueError(f"N must have shape (3, num_traj, num_points); got {N.shape}")
    if X.shape[0] != N.shape[1] or X.shape[2] != N.shape[2]:
        raise ValueError(
            f"X and N are not aligned: X implies (num_traj, num_points)="
            f"{(X.shape[0], X.shape[2])}, N implies {(N.shape[1], N.shape[2])}."
        )

    num_traj_in, _, num_points_in = X.shape
    traj_idx = _subsample_indices(num_traj_in, num_traj, traj_stride)
    point_idx = _subsample_indices(num_points_in, num_points, point_stride)

    X_sub = X[np.ix_(traj_idx, np.arange(3), point_idx)]
    N_sub = N[np.ix_(np.arange(3), traj_idx, point_idx)]
    return X_sub, N_sub
