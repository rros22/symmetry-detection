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
        "initial_conditions": np.linspace(1, 7, 20),
        "initial_condition": 1.0,
        "num_points": 50,
        "method": "RK45",
    },
    "bernoulli": {
        "x_start": 0.2,
        "x_end": 1.2,
        "initial_conditions": np.linspace(0.75, 2, 10),
        "initial_condition": 1.0,
        "num_points": 30,
        "method": "RK45",
    },
    "riccati": {
        "x_start": 0.1,
        "x_end": 0.5,
        "initial_conditions": np.linspace(-50, 106, 10),
        "initial_condition": 1.0,
        "num_points": 30,
        "method": "RK45",
    },
    "scaling": {
        "x_start": 0.1,
        "x_end": 2,
        "initial_conditions": np.linspace(0.1, 2, 15),
        "initial_condition": 1.0,
        "num_points": 30,
        "method": "RK45",
    },
    "abel": {
        "x_start": 0.0,
        "x_end": 0.09,
        "initial_conditions": np.linspace(-2, 2, 200),
        "initial_condition": 0.0,
        "num_points": 50,
        "method": "RK45",
    },
}

DEFAULT_BASIS_TYPE = "monomial"
DEFAULT_PARAM_RANGE = (0, 1)
