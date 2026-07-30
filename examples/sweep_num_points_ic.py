"""
Grid-search `num_points` (points/trajectory) x `num_ic` (number of initial
conditions) for the bernoulli example from `notebooks/ode_symmetry_explorer.ipynb`,
looking for the combination that maximizes the drop between two consecutive
singular values of the symmetry-detection SVD (`solver.solve_svd(...).S`).

This mirrors the notebook's pipeline exactly (same ODE, integrator, normal-field
estimation, subsampling, and basis settings) so the sweep result is directly
applicable there. Only `num_points`/`num_ic` are varied; everything else is held
fixed at the notebook's current values.

The "drop" between consecutive singular values sigma_i, sigma_{i+1} is measured
as the ratio sigma_i / sigma_{i+1} (>= 1, since S is sorted descending) - the same
quantity solver.small_singular_value_indices thresholds against. We report, for
each (num_points, num_ic), the single largest such ratio anywhere in the spectrum
(max_drop_ratio) and where it occurs (drop_at_idx, meaning between sigma_i and
sigma_{i+1}).

Usage:
    python examples/sweep_num_points_ic.py
    python examples/sweep_num_points_ic.py --num_points 150 300 600 1000 --num_ic 30 60 100
"""

import argparse
import contextlib
import io
import time

import numpy as np
from matplotlib import pyplot as plt

from symmetry_detection import generate_trajectories as gtr
from symmetry_detection import manifold as mfd
from symmetry_detection import solver

# --- Fixed pipeline settings, copied from notebooks/ode_symmetry_explorer.ipynb ---
ODE = "bernoulli"
METHOD = "DOP853"
ANALYTIC_DERIVATIVE = False
RTOL = 1e-12
ATOL = 1e-12

NORMAL_WINDOW = (5, 5)
NORMAL_DEGREE = 1
NORMAL_BOUNDARY_MARGIN = 0

SUBSAMPLE_NUM_TRAJ = 10
SUBSAMPLE_NUM_POINTS = 100

BASIS_TYPE = "monomial"
PARAM_RANGE = (0, 1)
CHARACTERISTIC = False

DEFAULT_NUM_POINTS_GRID = [150, 300, 600, 1000, 1500, 2000, 3000]
DEFAULT_NUM_IC_GRID = [30, 60, 100, 200, 350, 500, 750]


def get_args():
    parser = argparse.ArgumentParser(
        description="Sweep num_points x num_ic to maximize the singular-value drop."
    )
    parser.add_argument("--num_points", type=int, nargs="+", default=DEFAULT_NUM_POINTS_GRID,
                         help="Grid of points/trajectory to try")
    parser.add_argument("--num_ic", type=int, nargs="+", default=DEFAULT_NUM_IC_GRID,
                         help="Grid of number-of-initial-conditions to try")
    parser.add_argument("--out_csv", default="examples/sweep_results.csv",
                         help="Where to write the full results table")
    parser.add_argument("--out_png", default="examples/sweep_results.png",
                         help="Where to write the heatmap figure")
    return parser.parse_args()


def run_one(ode_args, num_points, num_ic):
    """Run the notebook's pipeline for one (num_points, num_ic) and return S (sorted desc).

    Stdout from the library (rich config tables, progress prints) is
    suppressed here so the sweep's own per-combination log line stays
    readable; nothing informational is lost since every input is already
    echoed in that log line.
    """
    initial_conditions = np.linspace(0.75, 2, num_ic)

    with contextlib.redirect_stdout(io.StringIO()):
        X = gtr.generate_equation_manifold(
            ode_name=ode_args.ode,
            x_start=ode_args.start,
            x_end=ode_args.end,
            initial_conditions=initial_conditions,
            num_points=num_points,
            method=METHOD,
            rtol=RTOL,
            atol=ATOL,
            analytic_derivative=ANALYTIC_DERIVATIVE,
        )

        normal_est = mfd.estimate_normal_field(
            X, window=NORMAL_WINDOW, degree=NORMAL_DEGREE,
            discard_boundary=True, boundary_margin=NORMAL_BOUNDARY_MARGIN,
        )
        X_dense, N_dense = normal_est.apply(X), normal_est.N

        X_sub, N_sub = mfd.subsample_trajectories(
            X_dense, N_dense, num_traj=SUBSAMPLE_NUM_TRAJ, num_points=SUBSAMPLE_NUM_POINTS,
        )
        X_stacked = gtr.concatenate_trajectories(X_sub)
        N = N_sub.reshape(3, -1)

        svd_result = solver.solve_svd(
            X_stacked, N, basis_type=BASIS_TYPE, param_range=PARAM_RANGE, param_list=None,
            characteristic=CHARACTERISTIC,
        )
    return svd_result.S


def main():
    args = get_args()

    ode_args = argparse.Namespace(
        ode=ODE, start=None, end=None, initial_conditions=None, num_points=None,
        method=None, rtol=None, atol=None,
    )
    ode_args = gtr.resolve_ode_args(ode_args)

    n_pts, n_ic = len(args.num_points), len(args.num_ic)
    ratio_grid = np.full((n_ic, n_pts), np.nan)
    idx_grid = np.full((n_ic, n_pts), -1, dtype=int)
    rows = []

    total = n_pts * n_ic
    done = 0
    t_start = time.time()

    for i, num_ic in enumerate(args.num_ic):
        for j, num_points in enumerate(args.num_points):
            done += 1
            t0 = time.time()
            try:
                S = run_one(ode_args, num_points, num_ic)
                ratios = S[:-1] / S[1:]
                drop_idx = int(np.argmax(ratios))
                max_ratio = float(ratios[drop_idx])
                error = ""
            except Exception as exc:  # noqa: BLE001 - record and keep sweeping
                S = None
                drop_idx = -1
                max_ratio = np.nan
                error = str(exc)

            elapsed = time.time() - t0
            ratio_grid[i, j] = max_ratio
            idx_grid[i, j] = drop_idx

            rows.append(dict(
                num_points=num_points, num_ic=num_ic,
                max_drop_ratio=max_ratio, drop_at_idx=drop_idx,
                num_singular_values=(len(S) if S is not None else 0),
                singular_values=(",".join(f"{s:.6e}" for s in S) if S is not None else ""),
                elapsed_s=elapsed, error=error,
            ))

            status = f"max_drop_ratio={max_ratio:.3g} @ idx {drop_idx}" if error == "" else f"FAILED: {error}"
            print(f"[{done}/{total}] num_points={num_points:5d} num_ic={num_ic:4d}  "
                  f"({elapsed:5.1f}s)  {status}")

    print(f"\nTotal sweep time: {time.time() - t_start:.1f}s")

    # --- Save full results table ---
    import csv
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote full results to {args.out_csv}")

    # --- Identify and report the best combination ---
    valid_rows = [r for r in rows if r["error"] == ""]
    if not valid_rows:
        print("No successful runs; cannot identify a best combination.")
        return

    best = max(valid_rows, key=lambda r: r["max_drop_ratio"])
    print("\nBest combination (maximizes drop between two consecutive singular values):")
    print(f"  num_points = {best['num_points']}")
    print(f"  num_ic     = {best['num_ic']}")
    print(f"  max_drop_ratio (sigma_i / sigma_{{i+1}}) = {best['max_drop_ratio']:.4g}, "
          f"between sigma_{best['drop_at_idx']} and sigma_{best['drop_at_idx'] + 1}")
    print(f"  singular values = [{best['singular_values']}]")

    top10 = sorted(valid_rows, key=lambda r: r["max_drop_ratio"], reverse=True)[:10]
    print("\nTop 10 combinations by max_drop_ratio:")
    for r in top10:
        print(f"  num_points={r['num_points']:5d}  num_ic={r['num_ic']:4d}  "
              f"max_drop_ratio={r['max_drop_ratio']:.4g}  (idx {r['drop_at_idx']})")

    # --- Heatmap of max_drop_ratio over the (num_ic, num_points) grid ---
    fig, ax = plt.subplots(figsize=(1.2 * n_pts + 2, 1.0 * n_ic + 2))
    log_ratio = np.log10(ratio_grid)
    im = ax.imshow(log_ratio, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(n_pts))
    ax.set_xticklabels(args.num_points)
    ax.set_yticks(range(n_ic))
    ax.set_yticklabels(args.num_ic)
    ax.set_xlabel("num_points (points/trajectory)")
    ax.set_ylabel("num_ic (initial conditions)")
    ax.set_title("log10(max drop ratio between consecutive singular values)")

    for i in range(n_ic):
        for j in range(n_pts):
            if np.isfinite(ratio_grid[i, j]):
                ax.text(j, i, f"{ratio_grid[i, j]:.2g}", ha="center", va="center",
                         color="white", fontsize=8)

    best_i = args.num_ic.index(best["num_ic"])
    best_j = args.num_points.index(best["num_points"])
    ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1, fill=False,
                                edgecolor="red", linewidth=3))

    fig.colorbar(im, ax=ax, label=r"$\log_{10}(\max_i\ \sigma_i/\sigma_{i+1})$")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"\nWrote heatmap to {args.out_png}")


if __name__ == "__main__":
    main()
