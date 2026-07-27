import numpy as np
from rich.console import Console
from rich.table import Table
from matplotlib import pyplot as plt

# Printing for debbuging
def print_configuration(ode_name, x_start, x_end, initial_conditions, num_points, method):
    # Format array preview nicely
    ic_arr = np.asarray(initial_conditions)
    if len(ic_arr) > 4:
        ic_str = f"[{ic_arr[0]:.2f}, {ic_arr[1]:.2f}, ..., {ic_arr[-1]:.2f}] ({len(ic_arr)} trajectories)"
    else:
        ic_str = f"{np.round(ic_arr, 2).tolist()}"

    title = f" INTEGRATION CONFIGURATION: {ode_name.upper()} "
    width = 60

    print("=" * width)
    print(f"{title.center(width, ' ')}")
    print("=" * width)
    print(f"  {'Domain (X Span)':<25} : [{x_start}, {x_end}]")
    print(f"  {'Points per Trajectory':<25} : {num_points}")
    print(f"  {'Initial Conditions':<25} : {ic_str}")
    print(f"  {'Solver Method':<25} : {method}")
    print("-" * width)

def print_configuration_rich(ode_name, x_start, x_end, initial_conditions, num_points, method):
    console = Console()
    ic_arr = np.asarray(initial_conditions)
    
    table = Table(title=f"ODE Configuration — {ode_name.upper()}", title_style="bold cyan", border_style="dim")
    table.add_column("Parameter", style="bold white", justify="left")
    table.add_column("Value", style="green", justify="left")

    table.add_row("X Range", f"[{x_start}, {x_end}]")
    table.add_row("Resolution", f"{num_points} points/trajectory")
    table.add_row("Trajectory Count", f"{len(ic_arr)} initial conditions")
    table.add_row("IC Range", f"[{ic_arr.min():.2f} ... {ic_arr.max():.2f}]")
    table.add_row("Integrator", method)

    console.print(table)

# Plots
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


def scaled_3D_quiver(X_stacked, V, style='scatter', num_points=None, mode="normal"):
    """
    Plot stacked trajectory points with a vector field.

    X_stacked and V have shape (3, n_points) with rows (x, u, p) and vector components.
    style: "lines" plots each trajectory as a curve; "scatter" plots all sample points.
    num_points: points per trajectory; required when style='lines'.
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

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

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

# Small wrapper for the scaled_3D_quiver function to plot integrated trajectories.
def scaled_3D_quiver_trajectories(X_stacked, normal_fn, style="lines", num_points=None, mode="normal"):
    """
    Plot integrated trajectories with their normal field.

    X_stacked has shape (3, n_points) with rows (x, u, p), as produced by
    _concatenate_trajectories. The normal field is computed via normal_fn(x, u).
    """
    V = normal_fn(X_stacked[0], X_stacked[1])
    return scaled_3D_quiver(X_stacked, V, style=style, num_points=num_points, mode=mode)



