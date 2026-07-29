"""
    Methods for the numerical estimation of trajectory derivatives, and
    the normal space of the equation manifold.
"""

import numpy as np
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
