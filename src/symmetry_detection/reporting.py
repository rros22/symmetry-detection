"""
Console output helpers, used by both the CLI entry points and the example
scripts so that reported information is formatted consistently everywhere.
"""

import numpy as np
from rich.console import Console
from rich.table import Table


def print_ode_configuration(ode_name, x_start, x_end, initial_conditions, num_points, method):
    """Pretty-print the integration configuration for one ODE run."""
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


def print_svd_summary(S, small_sv_idx):
    """
    Print the outcome of `solver.solve_svd`: the estimated null-space
    dimension and the candidate small singular value indices, before any of
    them has been picked via `solver.select_singular_vector`.
    """
    print(f"Estimated null-space dimension: {len(small_sv_idx)}")
    print(f"Small singular value indices: {np.asarray(small_sv_idx).tolist()}")


def print_selection_summary(S, idx):
    """
    Print which singular value was selected for reconstruction (i.e.
    `SymmetryResult.idx`, as returned by `solver.select_singular_vector`).
    """
    print(f"Selected singular value: sigma_{idx} = {S[idx]:.3e} (out of {len(S)})")
