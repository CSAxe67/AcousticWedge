"""
bathymetric_coupling.py
========================
Bathymetric coupling coefficients:
    - compute_coupling_terms_1 : finite-difference coupling on beta (V_jl)
    - compute_couplingterms_2  : analytical coupling at the interface (C1ij)
"""

import numpy as np

from config import SimulationConfig
from integration import compute_piecewise_scalarpdct


# ============================================================
# BATHYMETRIC COUPLING — FINITE DIFFERENCE ON BETA (V_jl)
# ============================================================
def compute_coupling_terms_1(cfg: SimulationConfig, all_phis, all_phis_der, DZ_W, N_w):
    """
    Coupling coefficients V_jl = <dphi_j/dh | phi_l>_rho, where dphi/dh is
    estimated by finite difference between the modes at h and h + beta*dz.

    Parameters
    ----------
    cfg : SimulationConfig
        Global configuration. Uses cfg.Deriv_step (beta) and cfg.p_w, cfg.p_s.
    all_phis : list of np.ndarray, each (N_tot, nmod)
        Normalized modes at h, at each position x.
    all_phis_der : list of np.ndarray, each (N_tot, nmod)
        Normalized modes at h + beta*dz, at each position x.
    DZ_W : list of float
        Vertical mesh step at each position x.
    N_w : list of int
        Water/sediment interface index at each position x.

    Returns
    -------
    list of np.ndarray, each (nmod, nmod)
        Coupling matrix V_jl at each position x.
    """
    if not (len(all_phis) == len(all_phis_der) == len(DZ_W) == len(N_w)):
        raise ValueError(
            "all_phis, all_phis_der, DZ_W and N_w must have the same length "
            f"(got {len(all_phis)}, {len(all_phis_der)}, {len(DZ_W)}, {len(N_w)})."
        )

    beta = cfg.Deriv_step
    rho_w, rho_s = cfg.p_w, cfg.p_s
    dhdx = cfg.dh_dx

    all_Cjl = []

    for i in range(len(all_phis)):
        phi0 = all_phis[i]       # modes at h
        phi1 = all_phis_der[i]   # modes at h + dh
        dz = DZ_W[i]
        nw = N_w[i]
        dh = beta * dz

        dphi_dh = (phi1 - phi0) / dh

        M1 = dphi_dh.T   # (nmod, N_tot)
        M2 = phi0        # (N_tot, nmod)

        Cjl = compute_piecewise_scalarpdct(nw, M1, M2, dz, rho_w, rho_s)
        all_Cjl.append(-Cjl*dhdx)

    return all_Cjl


# ============================================================
# BATHYMETRIC COUPLING — ANALYTICAL FORMULA AT THE INTERFACE (C1ij)
# ============================================================
def compute_couplingterms_x(cfg: SimulationConfig, modes, dmodes, wnum, nw):
    """
    C1ij matrix (bathymetric coupling, cf. reference [2] Eq. 18-20) for
    ONE given position x.

    Vectorized over (j, l) via numpy broadcasting (avoids the O(nmod^2)
    Python double loop, which is costly for nmod ~ 100-150).

    Parameters
    ----------
    cfg : SimulationConfig
        Global configuration. Uses cfg.p_w, cfg.p_s, cfg.c_w, cfg.c_s, cfg.omega.
    modes : np.ndarray, shape (N_tot, nmod)
        Normalized modes at this position (phi_j).
    dmodes : np.ndarray, shape (N_tot, nmod)
        Derivatives dphi_j/dz at this position.
    wnum : np.ndarray, shape (nmod,)
        Wavenumbers k_j at this position.
    nw : int
        Water/sediment interface index.

    Returns
    -------
    np.ndarray, shape (nmod, nmod)
        C1ij coupling matrix.
    """
    p_w, p_s = cfg.p_w, cfg.p_s
    c_w, c_s = cfg.c_w, cfg.c_s
    omega = cfg.omega

    ih = nw
    gammaw = 1.0 / p_w
    gammab = 1.0 / p_s

    phi_ih = modes[ih, :]     # (nmod,)
    dphi_ih = dmodes[ih, :]   # (nmod,)
    kj2 = wnum**2

    # Term that only depends on j (the "row" index)
    term_j = gammaw * (omega / c_w) ** 2 - gammab * (omega / c_s) ** 2 + (gammab - gammaw) * kj2

    # b1[j, l] = phi_ih[j]*phi_ih[l]*term_j[j] - (p_s - p_w)*gammab^2*dphi_ih[j]*dphi_ih[l]
    b1 = np.outer(phi_ih * term_j, phi_ih) - (p_s - p_w) * gammab**2 * np.outer(dphi_ih, dphi_ih)

    # denom[j, l] = kl^2 - kj^2
    denom = kj2[np.newaxis, :] - kj2[:, np.newaxis]
    np.fill_diagonal(denom, 1.0)  # avoids division by 0 (diagonal handled separately)

    C1ij = -b1 / denom
    np.fill_diagonal(C1ij, phi_ih**2 * (gammab - gammaw) / 2.0)  # Eq. (18), case j == l

    return C1ij


def compute_couplingterms_2(cfg: SimulationConfig, all_modes, all_dmodes, all_roots, N_w):
    """
    Computes C1ij for all positions x (see compute_couplingterms_x).

    Parameters
    ----------
    cfg : SimulationConfig
        Global configuration, forwarded to compute_couplingterms_x.
    all_modes : list of np.ndarray, each (N_tot, nmod)
        Normalized modes at each position x.
    all_dmodes : list of np.ndarray, each (N_tot, nmod)
        Derivatives dphi/dz at each position x.
    all_roots : list of np.ndarray, each (nmod,)
        Wavenumbers at each position x.
    N_w : list of int
        Water/sediment interface index at each position x.

    Returns
    -------
    list of np.ndarray, each (nmod, nmod)
        C1ij coupling matrix at each position x.
    """
    if not (len(all_modes) == len(all_dmodes) == len(all_roots) == len(N_w)):
        raise ValueError(
            "all_modes, all_dmodes, all_roots and N_w must have the same length "
            f"(got {len(all_modes)}, {len(all_dmodes)}, {len(all_roots)}, {len(N_w)})."
        )

    all_C1ij = []
    for i in range(len(all_modes)):
        C1 = compute_couplingterms_x(
            cfg,
            modes=all_modes[i],
            dmodes=all_dmodes[i],
            wnum=all_roots[i],
            nw=N_w[i],
        )
        all_C1ij.append(-C1)

    return all_C1ij