"""
Matplotlib plotting helpers shared by the example scripts: trajectory/vector
field quiver plots in the (x, u, p) jet space, and bar charts of learnt basis
coefficients.
"""

import numpy as np
from matplotlib import pyplot as plt


def compute_data_range(X):
    if len(X.shape) == 2:
        if X.shape[0] != 3:
            raise ValueError(
                f"Stacked X must have shape (3, n_points), got {X.shape}."
            )
        x_range = (X[0].min(), X[0].max())
        u_range = (X[1].min(), X[1].max())
        p_range = (X[2].min(), X[2].max())
    elif len(X.shape) == 3:
        x_range = (X[:, 0].min(), X[:, 0].max())
        u_range = (X[:, 1].min(), X[:, 1].max())
        p_range = (X[:, 2].min(), X[:, 2].max())
    else:
        raise ValueError(
            f"The dimension of the provided array is {len(X.shape)}. "
            "Expected 2 (stacked) or 3 (trajectory) dimensions."
        )

    return x_range, u_range, p_range


def _unit_cube_transform(X, N, x_range, u_range, p_range, mode="normal"):
    """
    Mode can be "normal" or "tangent". First check if mode is any of the two.
    """
    if mode not in ["normal", "tangent"]:
        raise ValueError(f"Mode must be 'normal' or 'tangent', got {mode!r}")


    mins = np.array([x_range[0], u_range[0], p_range[0]])
    spans = np.array([
        x_range[1] - x_range[0],
        u_range[1] - u_range[0],
        p_range[1] - p_range[0],
    ])
    broadcast = (3,) + (1,) * (X.ndim - 1)
    Xn = (X - mins.reshape(broadcast)) / spans.reshape(broadcast)
    if mode == "normal":
        Nn = N * spans.reshape(broadcast)
    elif mode == "tangent":
        Nn = N / spans.reshape(broadcast)
    return Xn, Nn


def _finish_3d_plot(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_zlabel('p')


def scaled_3D_quiver_surface(X_grid, U_grid, P_grid, N, x_range, u_range, p_range, mode="normal"):
    """Plot an equation manifold surface with its normal field."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    X_stacked = np.stack([X_grid, U_grid, P_grid], axis=0)
    Xn, Nn = _unit_cube_transform(X_stacked, N, x_range, u_range, p_range, mode=mode)
    ax.plot_surface(Xn[0], Xn[1], Xn[2], cmap='coolwarm')
    ax.quiver(Xn[0], Xn[1], Xn[2], Nn[0], Nn[1], Nn[2], normalize=True, length=0.08, color='k', alpha=0.3)
    _finish_3d_plot(ax)

    return fig, ax


def scaled_3D_quiver(X_stacked, V, style='scatter', num_points=None, mode="normal", ax=None):
    """
    Plot stacked trajectory points with a vector field.

    X_stacked and V have shape (3, n_points) with rows (x, u, p) and vector components.
    style: "lines" plots each trajectory as a curve; "scatter" plots all sample points.
    num_points: points per trajectory; required when style='lines'.
    ax: optional existing 3D axes to draw on; if None, a new figure is created.
    """
    if X_stacked.ndim != 2 or X_stacked.shape[0] != 3:
        raise ValueError(f"X_stacked must have shape (3, n_points), got {X_stacked.shape}.")
    if V.shape != X_stacked.shape:
        raise ValueError(f"V must have the same shape as X_stacked, got {V.shape} vs {X_stacked.shape}.")
    if style not in ("lines", "scatter"):
        raise ValueError(f"style must be 'lines' or 'scatter', got {style!r}")
    if style == "lines" and num_points is None:
        raise ValueError("num_points is required when style='lines'.")

    x_range, u_range, p_range = compute_data_range(X_stacked)

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    Xn, Vn = _unit_cube_transform(X_stacked, V, x_range, u_range, p_range, mode=mode)
    if style == "lines":
        num_traj = X_stacked.shape[1] // num_points
        Xn_traj = Xn.reshape(3, num_traj, num_points)
        for i in range(num_traj):
            ax.plot(Xn_traj[0, i], Xn_traj[1, i], Xn_traj[2, i], color='C0', linewidth=1.0)
    else:
        ax.scatter(Xn[0], Xn[1], Xn[2], color='C0', s=8, alpha=0.6)
    ax.quiver(Xn[0], Xn[1], Xn[2], Vn[0], Vn[1], Vn[2], normalize=True, length=0.08, color='k', alpha=0.3)
    _finish_3d_plot(ax)

    return fig, ax


def scaled_3D_quiver_trajectories(X_stacked, normal_fn, style="lines", num_points=None, mode="normal", ax=None):
    """
    Plot integrated trajectories with their normal field.

    X_stacked has shape (3, n_points) with rows (x, u, p), as produced by
    _concatenate_trajectories. The normal field is computed via normal_fn(x, u).
    """
    V = normal_fn(X_stacked[0], X_stacked[1])
    return scaled_3D_quiver(X_stacked, V, style=style, num_points=num_points, mode=mode, ax=ax)


def format_basis_label(basis_type, m, n):
    """Return a matplotlib/LaTeX label for basis function parameters (m, n)."""
    if basis_type == "monomial":
        parts = []
        if m != 0:
            parts.append("x" if m == 1 else f"x^{{{m}}}")
        if n != 0:
            parts.append("u" if n == 1 else f"u^{{{n}}}")
        if not parts:
            return r"$1$"
        return "$" + " ".join(parts) + "$"

    if basis_type == "chebyshev":
        def _cheb(var, k):
            return "1" if k == 0 else f"T_{{{k}}}({var})"

        return f"${_cheb('x', m)} {_cheb('u', n)}$"

    raise ValueError(f"Unknown basis type: {basis_type!r}")


def plot_basis_coefficient_bars(
    coeffs,
    param_list,
    basis_type,
    *,
    sort_by_magnitude=True,
    ax=None,
    title=None,
    log_scale=False,
):
    """
    Bar chart of basis functions versus learnt coefficients for one component
    (e.g. xi or eta coefficients from a single SVD singular vector).

    Parameters
    ----------
    coeffs : array-like, shape (num_basis,)
        Coefficients aligned with param_list.
    param_list : list of (m, n) tuples
        Basis function parameter pairs.
    basis_type : str
        "monomial" or "chebyshev".
    sort_by_magnitude : bool, default True
        If True, sort bars by descending |coefficient|; otherwise use param_list order.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; if None, a new figure/axes is created.
    title : str, optional
        Axes title.

    Returns
    -------
    fig, ax
    """
    coeffs = np.asarray(coeffs)
    if len(coeffs) != len(param_list):
        raise ValueError(
            f"coeffs length {len(coeffs)} does not match param_list length {len(param_list)}."
        )

    order = np.argsort(-np.abs(coeffs)) if sort_by_magnitude else np.arange(len(coeffs))
    sorted_coeffs = coeffs[order]
    labels = [format_basis_label(basis_type, m, n) for m, n in param_list]
    sorted_labels = [labels[i] for i in order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(coeffs) * 0.4), 4))
    else:
        fig = ax.figure

    ax.bar(np.arange(len(sorted_coeffs)), sorted_coeffs, color="C0", log=log_scale)
    ax.set_xticks(np.arange(len(sorted_labels)))
    ax.set_xticklabels(sorted_labels, rotation=45, ha="right")
    ax.set_ylabel("Coefficient")
    ax.axhline(0, color="k", linewidth=0.5)
    if title is not None:
        ax.set_title(title)

    return fig, ax
