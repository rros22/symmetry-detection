"""
    Basis functions to construct the characteristic functions of the symmetry generators.
"""

import numpy as np
from numpy.polynomial.chebyshev import chebder, chebval


"""
    Monomial basis functions and their derivatives.
"""

def monomial_bf(x, u, m, n):
    return x**m * u**n

def monomial_bf_dx(x, u, m, n):
    if m == 0:
        return np.zeros_like(x * u, dtype=float)
    return m * x**(m - 1) * u**n

def monomial_bf_du(x, u, m, n):
    if n == 0:
        return np.zeros_like(x * u, dtype=float)
    return n * x**m * u**(n - 1)

"""
    Chebyshev basis functions and their derivatives.
"""

def T(n, x):
    c = np.zeros(n + 1)
    c[n] = 1
    return chebval(x, c)


def dT(n, x):
    if n == 0:
        return np.zeros_like(x)

    c = np.zeros(n + 1)
    c[n] = 1

    dc = chebder(c)

    return chebval(x, dc)


def chebyshev_bf(x, u, m, n):
    return T(m, x) * T(n, u)


def chebyshev_bf_dx(x, u, m, n):
    return dT(m, x) * T(n, u)


def chebyshev_bf_du(x, u, m, n):
    return T(m, x) * dT(n, u)

"""
    Dictionary of basis functions and their derivatives. The keys are the names of the basis functions, and the values are tuples of the form (basis_function, derivative_wrt_x, derivative_wrt_u).
"""

BASIS_FUNCTIONS = {
    "monomial": (monomial_bf, monomial_bf_dx, monomial_bf_du),
    "chebyshev": (chebyshev_bf, chebyshev_bf_dx, chebyshev_bf_du),
}

# Reconstruct characteristic functions of the learnt vector field from basis function family, list of basis function params, and list of learnt coefficients.
def characteristic_functions(L, L_x, L_u, P, coeffs):
    # No testing for now
    # The first half of the coefficient list belongs to the first characteristic function
    coeff_no = len(coeffs)
    if coeff_no % 2 != 0:
        raise ValueError(f"The number of coefficients is {coeff_no} which is not an even number.")

    xi_coeffs = coeffs[0:int(coeff_no/2)]
    eta_coeffs = coeffs[int(coeff_no/2):]
    
    # Compute the characteristic functions. P is the diagonal field u_x
    # evaluated at every point (one scalar per point), passed as a plain
    # vector rather than a dense diag(P) matrix: `diag(P) @ M` becomes
    # `P[:, None] * M` and `diag(P) @ vec` becomes `P * vec`, exact but
    # O(num_points) instead of O(num_points^2). The scaling is applied to M
    # *before* reducing over coeffs (not to the already-reduced M @ coeffs),
    # to match the original diag-matmul evaluation order bit-for-bit -
    # `a * sum_j(M_j * c_j)` and `sum_j((a * M_j) * c_j)` are only equal up
    # to floating-point rounding. np.ascontiguousarray forces the same
    # (C-order) memory layout the original diag-matmul produced, since BLAS
    # can otherwise pick a different (still correct, but differently-rounded)
    # summation order for the matmul below depending on layout.
    M = np.ascontiguousarray(P[:, None] * L_u.T + L_x.T)
    xi = L.T@xi_coeffs
    eta = L.T@eta_coeffs
    zeta = (-P[:, None] * M) @ xi_coeffs + M @ eta_coeffs

    return xi, eta, zeta

# Reconstruct the characteristic vector field. We use the same (xi, eta, zeta) API as
# characteristic_functions so that the same downstream (plotting) functions can be reused,
# even though this generator is by definition vertical (xi is identically zero).
#
# Unlike characteristic_functions, this needs the gradient field N = (-f_x, -f_u, 1) of the
# ODE manifold F = u_x - f(x, u), in addition to L, L_x, L_u, P. This is because reconstructing
# zeta requires u_xx = D_x(u_x), which is not derivable from the basis matrices alone - on the
# equation manifold u_xx = f_x + f*f_u = -(N_x + N_u * u_x), using N_x = -f_x, N_u = -f_u, and
# u_x = f(x, u) = P (see characteristic_symmetries.pdf, Sec. 2).
def reconstruct_characteristic(L, L_x, L_u, P, N, coeffs):
    coeff_no = len(coeffs)
    if coeff_no % 2 != 0:
        raise ValueError(f"The number of coefficients is {coeff_no} which is not an even number.")

    xi_coeffs = coeffs[0:int(coeff_no/2)]
    eta_coeffs = coeffs[int(coeff_no/2):]

    # N_x/N_u and P are diagonal fields (one scalar per point); kept as plain
    # vectors rather than dense diag(...) matrices - see characteristic_functions.
    n_x, n_u = N[0], N[1]

    # The reconstructed generator is vertical by definition (Eq. 3).
    xi = np.zeros_like(L.T@xi_coeffs)

    Lxi = L.T@xi_coeffs

    # The characteristic: eta_hat = eta - xi * u_x (Eq. 4/10).
    eta = L.T@eta_coeffs - P*Lxi

    # zeta_hat = D_x(eta_hat) (Eq. 7/11). Expanding via the product rule, D_x(eta_hat) equals
    # the ordinary point-symmetry prolongation zeta_point (as in characteristic_functions),
    # plus a -xi * u_xx correction from the u_x-dependence of eta_hat itself.
    # (See characteristic_functions for why M is scaled before, not after,
    # reducing over coeffs, and forced C-contiguous.)
    M = np.ascontiguousarray(P[:, None] * L_u.T + L_x.T)
    zeta_point = (-P[:, None] * M) @ xi_coeffs + M @ eta_coeffs
    zeta = zeta_point + (n_x + n_u*P)*Lxi

    return xi, eta, zeta


# Test chevyshev polynomials
if __name__ == "__main__":
    """
        Plot chevyshev basis ranging with m, n ranging from 0 to 3. Make a subplot 3d surface for each combination of m, n.
    """
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1, 100)
    u = np.linspace(-1, 1, 100)
    X, U = np.meshgrid(x, u)

    fig, axs = plt.subplots(4, 4, figsize=(16, 16), subplot_kw={"projection": "3d"})
    for m in range(4):
        for n in range(4):
            ax = axs[m, n]
            Z = chebyshev_bf(X, U, m, n)
            ax.plot_surface(X, U, Z, cmap="viridis")
            ax.set_title(f"T({m}, x) * T({n}, u)")
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.set_zlabel("T(m, x) * T(n, u)")


    """
     In a new figure, plot the monomial basis functions with m, n ranging from 0 to 3. Make a subplot 3d surface for each combination of m, n.
    """
    fig, axs = plt.subplots(4, 4, figsize=(16, 16), subplot_kw={"projection": "3d"})
    for m in range(4):
        for n in range(4):
            ax = axs[m, n]
            Z = monomial_bf(X, U, m, n)
            ax.plot_surface(X, U, Z, cmap="viridis")
            ax.set_title(f"x^{m} * u^{n}")
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.set_zlabel("x^m * u^n")
    plt.tight_layout()
    plt.show()

