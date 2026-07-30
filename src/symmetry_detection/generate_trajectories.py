"""
    - Five ODEs with point symmetries
    - Trajectory generation
"""

import argparse
import numpy as np
from scipy.integrate import solve_ivp

from . import defaults as dflt
from . import reporting as rpt
from . import manifold as mfd

# ODEs
def bernoulli_ode(x,u):
     """
        Bernoulli equation du/dx = 2*u/x - u^2*x^2
     """

     return 2*u/x - (u**2)*(x**2)

def rational_ode(x,u):
    """
        Rational equation with rotation symmetry du/dx = (u^3 + u*x^2 - u - x)/(x*u^2 + x^3 + u - x)
    """

    return (u**3 + u*x**2 - u - x)/(x*u**2 + x**3 + u - x)

def riccati_ode(x,u):
    """
        Riccati equation du/dx = x*u^2 - 2*u/x - 1/x^3
    """

    return x*u**2 - 2*u/x - 1/x**3

def scaling_ode(x, u):
    """
        Non-linear ODE with scaling symmetry du/dx = u/x + x/u
    """

    return u/x + x/u

def abel_ode(x,u):
    """
        Abel equation of the first kind
    """

    return u**3 - x

def grad_bernoulli_ode(x, u):
    """
        Gradient [-df/dx, -df/du, 1] for du/dx = 2*u/x - u^2*x^2
    """
    df_dx = -2 * u / (x**2) - 2 * (u**2) * x
    df_du = 2 / x - 2 * u * (x**2)

    return np.stack([-df_dx, -df_du, np.ones_like(x)], axis=0)


def grad_rational_ode(x, u):
    """
        Gradient [-df/dx, -df/du, 1] for du/dx = (u^3 + u*x^2 - u - x) / (x*u^2 + x^3 + u - x)
    """
    N = u**3 + u * (x**2) - u - x
    D = x * (u**2) + x**3 + u - x

    dN_dx = 2 * u * x - 1
    dN_du = 3 * (u**2) + x**2 - 1

    dD_dx = u**2 + 3 * (x**2) - 1
    dD_du = 2 * x * u + 1

    df_dx = (dN_dx * D - N * dD_dx) / (D**2)
    df_du = (dN_du * D - N * dD_du) / (D**2)

    return np.stack([-df_dx, -df_du, np.ones_like(x)], axis=0)


def grad_riccati_ode(x, u):
    """
        Gradient [-df/dx, -df/du, 1] for du/dx = x*u^2 - 2*u/x - 1/x^3
    """
    df_dx = u**2 + 2 * u / (x**2) + 3 / (x**4)
    df_du = 2 * x * u - 2 / x

    return np.stack([-df_dx, -df_du, np.ones_like(x)], axis=0)


def grad_scaling_ode(x, u):
    """
        Gradient [-df/dx, -df/du, 1] for du/dx = u/x + x/u
    """
    df_dx = -u / (x**2) + 1 / u
    df_du = 1 / x - x / (u**2)

    return np.stack([-df_dx, -df_du, np.ones_like(x)], axis=0)


def grad_abel_ode(x, u):
    """
        Gradient [-df/dx, -df/du, 1] for du/dx = u^3 - x
    """
    df_dx = -1.0 * np.ones_like(x)
    df_du = 3 * (u**2)

    return np.stack([-df_dx, -df_du, np.ones_like(x)], axis=0)

# Examples and default parameters
ODES = {
    "bernoulli": bernoulli_ode,
    "rational": rational_ode,
    "riccati": riccati_ode,
    "scaling": scaling_ode,
    "abel": abel_ode
}

NORMALS = {
    "bernoulli": grad_bernoulli_ode,
    "rational": grad_rational_ode,
    "riccati": grad_riccati_ode,
    "scaling": grad_scaling_ode,
    "abel": grad_abel_ode
}

INTEGRATORS = {
    "RK45": "RK45",
    "RK23": "RK23",
    "DOP853": "DOP853",
    "Radau": "Radau",
    "BDF": "BDF",
    "LSODA": "LSODA",
    "DOP853": "DOP853",
}

# Generate a full trajectory
def _generate_trajectory(ode_name, x_start, x_end, u0, num_points, method, rtol, atol):
    """
        Integrate one of the differential equations in the examples.

        rtol/atol are passed straight through to solve_ivp. They matter far more than
        num_points for the *accuracy* of the trajectory: num_points (via t_eval) only
        controls how densely the already-computed adaptive-step solution is resampled,
        not how accurately it was computed. See defaults.DEFAULT_RTOL/DEFAULT_ATOL.
    """

    if ode_name not in ODES:
        raise ValueError(f"Unknown ODE: {ode_name}. Choose from {list(ODES.keys())}")
    
    if method not in INTEGRATORS:
        raise ValueError(f"Unknown Mehtod: {method}. Choose from {list(INTEGRATORS.keys())}")
    
    ode_rhs = ODES[ode_name]
    x_eval = np.linspace(x_start, x_end, num_points)
    solution = solve_ivp(ode_rhs, t_span=(x_start, x_end), t_eval=x_eval, y0 = [u0], method=method, rtol=rtol, atol=atol)

    return solution

def generate_equation_manifold(ode_name, x_start, x_end, initial_conditions, num_points, method, rtol, atol, analytic_derivative=True):
    """ 
        o = len(initial_conditions): is the number of trajectories.
        n: is the dimension of the embedding (x, u, u'). In this example n = 3 since we are embbeding ODE trajectories into the first order Jet Space.
        t = num_points: is the number of datapoints per trajectory.

        1. Integrates one of the differential equations in the examples for "o" trajectories.
        2. Constructs a jet space embbeding of the trajectories, by appending the time, derivative wrt time coordinate to the state coordinate.
        3. Returns a np.array of dimensions (o,n,t)

        Integration parameters are not defaulted here; use resolve_ode_args at the CLI
        boundary or pass values explicitly when calling programmatically.

        When analytic_derivative=False, the trajectory's own integration accuracy (rtol,
        atol) becomes a hard floor on how well the numerically-estimated derivative can
        agree with the analytic one, independent of num_points - see the note on
        defaults.DEFAULT_RTOL/DEFAULT_ATOL and _generate_trajectory. If rtol/atol are too
        loose, increasing num_points will not converge the numerical embedding towards the
        analytic one, since it just resamples the same, fixed-accuracy dense-output solution.
    """

    # Input data
    rpt.print_ode_configuration(ode_name, x_start, x_end, initial_conditions, num_points, method, rtol, atol)

    # Iterate over all initial conditions
    trajectories = []

    if analytic_derivative:
        print("Constructing embedding with analytic derivative")
    else:
        print("Constructing embedding with numerical derivative")

    for ic in initial_conditions:
        solution = _generate_trajectory(ode_name, x_start=x_start, x_end=x_end, u0=ic, num_points=num_points, method=method, rtol=rtol, atol=atol)

        if solution.success:
            if analytic_derivative:
                # Compute analytic derivative
                du_dx = ODES[ode_name](solution.t, solution.y[0])
            else:
                # Compute numerical derivative
                du_dx= mfd.estimate_first_derivative(solution)[0]  
            embedding = np.array([solution.t, solution.y[0], du_dx])
            trajectories.append(embedding)
        
        else:
            raise RuntimeError(f"Integration failed for initial condition {ic}: {solution.message}")

    return np.array(trajectories)

# For synthetically generated trajectories, the X matrix needs to be flattened into a 2D array of shape (num_dimension, num_traj * num_points).
def concatenate_trajectories(X):
    """
    Flattens 3D trajectories (num_traj, state_dim, num_points) into a 2D matrix (state_dim, num_traj * num_points).
    """
    num_traj, state_dim, num_pts = X.shape
    return X.transpose(1, 0, 2).reshape(state_dim, -1)

# CLI tooling
def build_ode_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Generate and test ODE trajectories.", add_help=add_help
    )
    parser.add_argument("--ode", choices=ODES.keys(), default=None, help="Name of the ODE to solve")
    parser.add_argument("--start", type=float, default=None, help="Start x value")
    parser.add_argument("--end", type=float, default=None, help="End x value")
    parser.add_argument("--initial_conditions", type=float, nargs="+", default=None,
                         help="Initial conditions, e.g. --initial_conditions 1.0 1.1 1.2")
    parser.add_argument("--num_points", type=int, default=None, help="Number of points per trajectory")
    parser.add_argument("--method", choices=INTEGRATORS.keys(), default=None, help="Numerical integrator choice")
    parser.add_argument("--rtol", type=float, default=None, help="solve_ivp relative tolerance")
    parser.add_argument("--atol", type=float, default=None, help="solve_ivp absolute tolerance")
    return parser

def resolve_ode_args(args):
    """
    Fill in the ODE name and integration parameters from
    symmetry_detection.defaults when not provided on the CLI. This is the
    only place that applies default integration settings.
    """
    args.ode = args.ode if args.ode is not None else dflt.DEFAULT_ODE
    if args.ode not in dflt.ODE_DEFAULTS:
        raise ValueError(
            f"No default integration parameters for ODE {args.ode!r}. "
            f"Choose from {list(dflt.ODE_DEFAULTS.keys())} or pass --start, --end, "
            f"--initial_conditions, --num_points, and --method explicitly."
        )
    defaults = dflt.ODE_DEFAULTS[args.ode]
    args.start = args.start if args.start is not None else defaults["x_start"]
    args.end = args.end if args.end is not None else defaults["x_end"]
    if args.initial_conditions is not None:
        args.initial_conditions = np.array(args.initial_conditions)
    else:
        args.initial_conditions = defaults["initial_conditions"]
    args.num_points = args.num_points if args.num_points is not None else defaults["num_points"]
    args.method = args.method if args.method is not None else defaults["method"]
    args.rtol = args.rtol if args.rtol is not None else defaults.get("rtol", dflt.DEFAULT_RTOL)
    args.atol = args.atol if args.atol is not None else defaults.get("atol", dflt.DEFAULT_ATOL)
    return args
