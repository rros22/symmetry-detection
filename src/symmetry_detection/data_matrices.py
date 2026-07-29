import numpy as np
import argparse

from . import basis_functions as bf
from . import defaults as dflt

"""
    Contains functions that generate matrices, which evaluate the basis functions and their derivatives at the points of the trajectories.
    There are three types of matrices:
        - 1. Basis function and derivative evaluation matrices: Each row corresponds to a basis function and each column correponds to a point. The dimensions of the matrices are (num_basis_functions, num_points).
        - 2. Diagonal matrices: Rows and columns correspond to points. The dimensions of the matrices are (num_points, num_points).
        - 3. The homogeneous system (untranspose) matrix: This matrix is constructed using the previous two types of matrices. And has the dimensions of the matrix are (2*num_basis_functions, num_points).  
"""

"""
    Type 1 matrices:
        - The parameter range is the range of the parameters of the basis functions. For example, if the parameter range is (0, 3), then the number of basis functions is 4 (0, 1, 2, 3).
        - The parameter list is a list of tuples, where each tuple contains the parameters of the basis functions. For example, if the parameter list is [(0, 0), (1, 0), (0, 1), (1, 1)], then the number of basis functions is 4.
          The parameter list overrides the parameter range. If the parameter list is not None, then the parameter range is ignored.
        - The basis type is the type of basis functions to use. Current options are "monomial" and "chebyshev". Some options have invalid parameters. For example the chevyshev basis functions are only defined for non-negative parameters. 
          If the parameter range or list contains invalid parameters, then the function will raise an error.
"""

# Parse CLI arguments

# data_matrices.py
def _parse_pair(s):
    """Parses 'm,n' into the (m, n) tuple expected by param_list."""
    try:
        m, n = s.split(",")
        return (int(m), int(n))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid pair '{s}', expected format 'm,n'")

def build_basis_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Basis function options.", add_help=add_help)
    parser.add_argument("--basis_type", choices=bf.BASIS_FUNCTIONS.keys(), default=None,
                         help="Basis function family to use")
    parser.add_argument("--param_range", type=int, nargs=2, default=None, metavar=("MIN", "MAX"),
                         help="Parameter range (m, n) for basis functions, e.g. --param_range 0 3")
    parser.add_argument("--param_list", type=_parse_pair, nargs="+", default=None,
                         help="Explicit list of (m,n) pairs, e.g. --param_list 0,0 1,0 0,1. Overrides --param_range")
    return parser

def resolve_basis_args(args):
    """
    Fill basis parameters from symmetry_detection.defaults when not provided
    on the CLI. This is the only place that applies default basis_type /
    param_range.
    """
    args.basis_type = args.basis_type if args.basis_type is not None else dflt.DEFAULT_BASIS_TYPE
    if args.param_range is not None:
        args.param_range = tuple(args.param_range)
    elif args.param_list is None:
        args.param_range = dflt.DEFAULT_PARAM_RANGE
    return args

def validate_parameters(basis_type, param_range=None, param_list=None):
    if basis_type not in bf.BASIS_FUNCTIONS:
        raise ValueError(
            f"Invalid basis type: {basis_type}. Valid options are: {list(bf.BASIS_FUNCTIONS.keys())}"
        )

    if param_list is None and param_range is None:
        raise ValueError("Either param_range or param_list must be provided.")
    
    is_chebyshev = (basis_type == "chebyshev")

    # 2 Case A: Custom parameter list provided
    if param_list is not None:
        if not isinstance(param_list, list):
            raise ValueError(f"Invalid parameter list: {param_list}. Must be a list of 2-tuples.")

        processed_params = []
        for param in param_list:
            # Check tuple structure AND numeric types inside
            if (
                not isinstance(param, tuple)
                or len(param) != 2
                or not all(isinstance(val, (int, float)) for val in param)
            ):
                raise ValueError(
                    f"Invalid parameter: {param}. Each item must be a 2-tuple of numbers (ints or floats)."
                )
            # Case A Rule: Drop negative parameters completely for Chebyshev
            if is_chebyshev and (param[0] < 0 or param[1] < 0):
                print(
                    f"Warning: Dropped parameter {param}. Chebyshev basis functions "
                    f"require non-negative indices."
                )
                continue  # Skip/drop this pair
            
            processed_params.append(param)

        # Remove duplicates while preserving original insertion order
        validated_param_list = list(dict.fromkeys(processed_params))

        if not validated_param_list:
            raise ValueError("The provided parameter list is empty after removing invalid pairs.")

    # 3. Case B: Generate parameter list from range
    else:
        # Validate param_range: must be a 2-tuple of numbers (int/float)
        if (
            not isinstance(param_range, tuple) 
            or len(param_range) != 2 
            or not all(isinstance(val, (int, float)) for val in param_range)
        ):
            raise ValueError(
                f"Invalid parameter range: {param_range}. "
                f"Must be a 2-tuple of numbers, e.g., (0, 3)."
            )
        
        # Standardize range order: (min, max)
        start, stop = min(param_range), max(param_range)

        # Pre-trim range for Chebyshev to avoid generating invalid pairs
        if is_chebyshev:
            trimmed_start = max(0, start)
            if trimmed_start != start:
                print(
                    f"Warning: The parameter range {param_range} was trimmed to "
                    f"({trimmed_start}, {stop}) for Chebyshev basis functions."
                )
            start = trimmed_start

        # Check for invalid range bounds (e.g., if full range was negative)
        if start > stop:
            raise ValueError(f"Invalid range bounds: ({start}, {stop}) after basis restrictions.")

        # Generate unique pairs directly
        validated_param_list = [(m, n) for m in range(start, stop + 1) for n in range(start, stop + 1)]
    
    return validated_param_list

def bf_matrices(X_stacked, basis_type, param_range=None, param_list=None):
    # Check shape of X_stacked
    if len(X_stacked.shape) != 2:
        raise ValueError(f"The shape of the input data matrix is {X_stacked.shape}, with dimension = {len(X_stacked.shape)}. It should be a 2D array")

    # Validate parameters
    validated_param_list = validate_parameters(basis_type, param_range, param_list)

    # Extract repeated variables once
    x, u = X_stacked[0].T, X_stacked[1].T
    funcs = bf.BASIS_FUNCTIONS[basis_type]

    # Evaluates each basis function on vector inputs x and u
    # Resulting shape: (num_basis_functions, total_points)
    def _build_matrix(func):
        return np.array([func(x, u, p[0], p[1]) for p in validated_param_list])

    # Generate the matrices
    L   = _build_matrix(funcs[0])
    L_x = _build_matrix(funcs[1])
    L_u = _build_matrix(funcs[2])

    return L, L_x, L_u

def normal_vec_diag_matrices(N):
    N_x = np.diag(N[0])
    N_u = np.diag(N[1])
    N_ux = np.diag(N[2])

    return N_x, N_u, N_ux

def p_diag_matrix(X_stacked):
    return np.diag(X_stacked[2])

# Point symmetry
def G_matrix(X_stacked, N, basis_type, param_range=None, param_list=None, characteristic=False):
    # Evaluate basis function matrices and diagonals
    L, L_x, L_u  = bf_matrices(X_stacked, basis_type, param_range, param_list)
    N_x, N_u, N_ux = normal_vec_diag_matrices(N)
    P = p_diag_matrix(X_stacked)

    # Construct parts of G (characteristic or regular symmetry)
    if characteristic:
        G_1 = - L_x@P - L_u@P@P + L@N_x
        G_2 = L_x + L_u@P + L@N_u
    else:
        G_1 = L@N_x - (L_x + L_u@P)@P@N_ux
        G_2 = L@N_u + (L_x + L_u@P)@N_ux


    # Stack parts
    G = np.vstack((G_1,G_2))

    return G, L, L_x, L_u, P 
