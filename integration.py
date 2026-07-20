"""
integration.py
======================
Coefficients d'integration piecewise (trapezes ponderes par la densite)
"""

import numpy as np


# ============================================================
# INTEGRATION PIECEWISE (regle des trapezes, corrections d'interface)
# ============================================================
def compute_piecwise_integral(i_int, M1, M2, dz):
    """Integrale trapezoidale de M1 @ M2 avec correction a l'interface i_int."""
    I = M1 @ M2
    I -= 0.5 * np.outer(M1[:, 0], M2[0, :])
    I -= 0.5 * np.outer(M1[:, -1], M2[-1, :])
    I -= 0.5 * np.outer(M1[:, i_int], M2[i_int, :])
    I += 0.5 * np.outer(M1[:, i_int - 1], M2[i_int - 1, :])
    return dz * I


def compute_piecewise_scalarpdct(i_int, M1, M2, dz, rho_w, rho_s):
    """Integrale trapezoidale ponderee par 1/rho (eau/sediment separes par i_int)."""
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


