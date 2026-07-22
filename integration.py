"""
integration.py
===============
Piecewise integration coefficients (density-weighted trapezoidal rule).
"""

import numpy as np


# ============================================================
# PIECEWISE INTEGRATION (trapezoidal rule, interface corrections)
# ============================================================
def compute_piecwise_integral(i_int, M1, M2, dz):
    """
    Trapezoidal integral of M1 @ M2 with a correction at the interface i_int.

    Parameters
    ----------
    i_int : int
        Index of the water/sediment interface.
    M1 : np.ndarray, shape (n, N)
        Left-hand side operand.
    M2 : np.ndarray, shape (N, m)
        Right-hand side operand.
    dz : float
        Vertical step of the mesh.

    Returns
    -------
    np.ndarray, shape (n, m)
        Trapezoidal integral of M1 @ M2.
    """
    I = M1 @ M2
    I -= 0.5 * np.outer(M1[:, 0], M2[0, :])
    I -= 0.5 * np.outer(M1[:, -1], M2[-1, :])
    I -= 0.5 * np.outer(M1[:, i_int], M2[i_int, :])
    I += 0.5 * np.outer(M1[:, i_int - 1], M2[i_int - 1, :])
    return dz * I


def compute_piecewise_scalarpdct(i_int, M1, M2, dz, rho_w, rho_s):
    """
    Trapezoidal integral weighted by 1/rho (water/sediment split at i_int).

    Parameters
    ----------
    i_int : int
        Index of the water/sediment interface.
    M1 : np.ndarray, shape (n, N)
        Left-hand side operand.
    M2 : np.ndarray, shape (N, m)
        Right-hand side operand.
    dz : float
        Vertical step of the mesh.
    rho_w : float
        Density in the water column.
    rho_s : float
        Density in the sediment.

    Returns
    -------
    np.ndarray, shape (n, m)
        Density-weighted trapezoidal integral of M1 @ M2.
    """
    N = M1.shape[1]
    inv_rho = np.empty(N)
    inv_rho[:i_int] = 1.0 / rho_w
    inv_rho[i_int:] = 1.0 / rho_s

    I = M1 @ (inv_rho[:, np.newaxis] * M2)
    I -= 0.5 * inv_rho[0] * np.outer(M1[:, 0], M2[0, :])
    I -= 0.5 * inv_rho[-1] * np.outer(M1[:, -1], M2[-1, :])
    I += 0.5 * inv_rho[i_int - 1] * np.outer(M1[:, i_int - 1], M2[i_int - 1, :])
    I -= 0.5 * inv_rho[i_int] * np.outer(M1[:, i_int], M2[i_int, :])

    return dz * I