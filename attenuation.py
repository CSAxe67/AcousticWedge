"""
attenution.py
======================
Inter-mode coupling coefficients:
    - attenuation_terms : attenuation-induced coupling (T_jl)
    - ComputeAllCjl     : bathymetric coupling via finite difference on beta (V_jl)
    - compute_all_C1ij  : analytical bathymetric coupling at the interface (C1ij)
"""

import numpy as np

from config import SimulationConfig


# ============================================================
# ATTENUATION-INDUCED COUPLING (T_jl)
# ============================================================
def attenuation_terms(cfg: SimulationConfig, all_phis, DZ_W, N_w):
    """
    Coupling matrix T_jl = omega^2 * eta * <phi_j | f_z | phi_l>, where
    f_z(z) carries the attenuation profile (water/sediment, with an
    optional gradient beyond cfg.z_transition).

    Vectorized over (j, l): the computation reduces to a matrix product
    phi^T @ diag(f_z) @ phi with an endpoint correction (equivalent to N
    calls to `integration.compute_piecwise_integral(0, ...)`, but without
    a Python loop).

    Parameters
    ----------
    cfg : SimulationConfig
        Global configuration. Uses cfg.p_w, cfg.p_s, cfg.c_w, cfg.c_s,
        cfg.omega, cfg.eta, cfg.beta_water, cfg.beta_sediment_1,
        cfg.beta_sediment_2, cfg.z_transition.
    all_phis : list of np.ndarray, each (N_tot, nmod)
        Normalized modes at each position x.
    DZ_W : list of float
        Vertical mesh step at each position x.
    N_w : list of int
        Water/sediment interface index at each position x.

    Returns
    -------
    list of np.ndarray, each (nmod, nmod)
        Coupling matrix T_jl at each position x.
    """
    if not (len(all_phis) == len(DZ_W) == len(N_w)):
        raise ValueError(
            "all_phis, DZ_W and N_w must have the same length "
            f"(got {len(all_phis)}, {len(DZ_W)}, {len(N_w)})."
        )

    p_w, p_s = cfg.p_w, cfg.p_s
    c_w, c_s = cfg.c_w, cfg.c_s
    omega, eta = cfg.omega, cfg.eta
    beta_water = cfg.beta_water
    beta_sediment_1, beta_sediment_2 = cfg.beta_sediment_1, cfg.beta_sediment_2
    z_transition = cfg.z_transition

    all_Coupling = []
    for i in range(len(all_phis)):
        phi = all_phis[i]      # (N_tot, nmod)
        dz = DZ_W[i]
        nw = N_w[i]
        N_tot = phi.shape[0]

        f_z = np.zeros(N_tot)
        f_z[:nw] = beta_water / (p_w * c_w**2)
        if z_transition is not None:
            h_eau = nw * dz
            z_fond = N_tot * dz
            z_coords = np.arange(N_tot) * dz
            sed_mask = z_coords >= h_eau
            t = np.clip((z_coords - z_transition) / (z_fond - z_transition), 0.0, 1.0)
            beta_z = (1.0 - t) * beta_sediment_1 + t * beta_sediment_2
            f_z[sed_mask] = beta_z[sed_mask] / (p_s * c_s**2)
        else:
            f_z[nw:] = beta_sediment_1 / (p_s * c_s**2)

        # Vectorized equivalent of integration.compute_piecwise_integral(0, f_z_row, phi_j*phi_l, dz)
        # for all (j, l) pairs simultaneously: the interface correction at
        # i_int=0 fully cancels the contribution of the z=0 point.
        integral_jl = dz * (phi.T @ (f_z[:, np.newaxis] * phi)
                            - np.outer(f_z[0] * phi[0, :], phi[0, :]))

        Coupling = (omega**2 * eta) * integral_jl
        all_Coupling.append(Coupling)

    return all_Coupling