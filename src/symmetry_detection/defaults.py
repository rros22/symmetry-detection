"""
Centralized fallback default values for the package.

These are the last-resort values used when a parameter is not supplied
explicitly. Keeping them in one place means the CLI entry points
(`symdet-generate`, `symdet-detect`) and the example scripts can never
silently disagree about what "default" means.

Precedence when running a script, highest wins:
    1. Explicit CLI flags (e.g. `--ode riccati`)
    2. A script's own top-level `DEFAULT_CASE` dict (set via
       `parser.set_defaults(**DEFAULT_CASE)` before `parse_args()`), if it
       sets a value
    3. The package-wide defaults defined here

A field is only filled in from a lower level when it is still `None` after
the levels above it have been applied (see `resolve_ode_args` in
`generate_trajectories.py` and `resolve_basis_args` in `data_matrices.py`).
"""

import numpy as np

DEFAULT_ODE = "bernoulli"

ODE_DEFAULTS = {
    "rational": {
        "x_start": 0.75,
        "x_end": 2.0,
        "initial_conditions": np.linspace(1, 6, 1000),
        "initial_condition": 1.0,
        "num_points": 3000,
        "method": "RK45",
    },
    "bernoulli": {
        "x_start": 0.2,
        "x_end": 1.2,
        "initial_conditions": np.linspace(0.75, 2, 1000),
        "initial_condition": 1.0,
        "num_points": 3000,
        "method": "RK45",
    },
    "riccati": {
        "x_start": 0.1,
        "x_end": 0.5,
        "initial_conditions": np.linspace(-50, 106, 1000),
        "initial_condition": 1.0,
        "num_points": 30,
        "method": "RK45",
    },
    "scaling": {
        "x_start": 0.1,
        "x_end": 2,
        "initial_conditions": np.linspace(0.1, 2, 1000),
        "initial_condition": 1.0,
        "num_points": 3000,
        "method": "RK45",
    },
    "abel": {
        "x_start": 0.0,
        "x_end": 0.09,
        "initial_conditions": np.linspace(-2, 2, 1000),
        "initial_condition": 0.0,
        "num_points": 3000,
        "method": "RK45",
    },
}

DEFAULT_BASIS_TYPE = "monomial"
DEFAULT_PARAM_RANGE = (0, 1)

# scipy.integrate.solve_ivp defaults to rtol=1e-3, atol=1e-6. That is loose
# enough that the produced trajectory (and its dense-output interpolant) sits
# ~1e-3 away from an exact solution of the ODE, *independent of num_points*:
# t_eval only resamples the interpolant built from the adaptive-step
# integration, it does not itself refine the integration. When the jet-space
# embedding uses a numerically estimated derivative (see manifold.py), that
# fixed ~1e-3 defect dominates the derivative-estimation error at essentially
# any sampling density, which is why increasing num_points alone does not
# make the numerical embedding converge towards the analytic one. Tightening
# the tolerance here (instead of relying on num_points) is what actually buys
# convergence.
DEFAULT_RTOL = 1e-10
DEFAULT_ATOL = 1e-12
