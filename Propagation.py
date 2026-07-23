"""
Propagation.py
==============
Coupled multi-mode acoustic propagation (wedge) computation.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.special import hankel1

from config import SimulationConfig
from geometry import build_medium_mesh
from Modescompute import (
    Modes_compute,
    compute_imaginary_wavenumbers,
    compute_mode_derivatives_z,
)
from attenuation import attenuation_terms
from coupling import compute_couplingterms_2


def _coupled_modes_rhs(A, kj, dkj, Cjl, Tjl, theta):
    """Right-hand side dA/dx for the coupled modal amplitudes, at fixed x."""
    delta_theta = -theta[:, None] + theta[None, :]
    V_term = (Cjl - Cjl.T) * np.exp(1j * delta_theta)
    T_term = Tjl * np.exp(1j * delta_theta)

    return (
        - 0.5 * (dkj / kj) * A
        - 0.5 * (V_term @ (kj * A)) / kj       # K^{-1} V K B
        - (T_term @ A) / kj                     # K^{-1} T B
    )


def compute_propagation(cfg: SimulationConfig):
    """
    Computes the coupled modal propagation field over the fine grid X_fine.

    Parameters
    ----------
    cfg : SimulationConfig
        Global configuration.

    Returns
    -------
    dict
        X, Z            : coarse grid and vertical profiles
        X_fin, D_fin    : fine horizontal grid / interpolated bathymetry
        TL              : 1D Transmission Loss at receiver depth zr (dB)
        TL_2d, Z_fine   : 2D TL field (x, z) and associated vertical grid
        phis_array      : modes on the coarse grid (N_pos, N_tot, nmod)
        kjx_array       : complex wavenumbers (N_pos, nmod)
        dx              : fine grid step
        iz, izs         : receiver / source depth indices
    """
    # --- Medium profile (coarse grid) ---
    mesh = build_medium_mesh(cfg, cfg.R_factor[-1])
    D, X, DZ_W, N_w, N_tot = mesh.D, mesh.X, mesh.dz, mesh.N_w, mesh.N_tot
    Z = np.array(mesh.Z)

    # --- Fine interpolation grids ---
    X_fine = np.linspace(cfg.X_0, cfg.X_fin, cfg.N_Grilles_fines)
    D_fine = np.linspace(cfg.D_0, cfg.D_fin, cfg.N_Grilles_fines)
    dx = X_fine[1] - X_fine[0]

    # --- Receiver / source indices ---
    iz = np.argmin(np.abs(Z[0] - cfg.zr))
    izs = np.argmin(np.abs(Z[0] - cfg.zs))

    # --- Eigenmodes and wavenumbers (real part) ---
    all_roots, all_phis = Modes_compute(cfg)

    # --- Attenuation-induced coupling ---
    T = attenuation_terms(cfg, all_phis, DZ_W, N_w)

   

    # --- Bathymetric coupling (bottom slope) ---
    all_dmodes = compute_mode_derivatives_z(all_phis, DZ_W, N_w)
    all_C1ij = compute_couplingterms_2(cfg, all_phis, all_dmodes, all_roots, N_w)

    kjx_array = np.array(all_roots)     # (N_pos, nmod)
    phis_array = np.array(all_phis)     # (N_pos, N_tot, nmod)
    Cjl_array = np.array(all_C1ij)     # (N_pos, nmod, nmod)
    Tjl_array = np.array(T)             # (N_pos, nmod, nmod)
    dhdx = np.gradient(D_fine, dx)

    # --- Interpolation onto the fine grid ---
    interp_kj_re = interp1d(X, np.real(kjx_array), axis=0, kind="linear", fill_value="extrapolate")
    interp_phi = interp1d(X, phis_array, axis=0, kind="linear", fill_value="extrapolate")
    interp_Cjl = interp1d(X, Cjl_array, axis=0, kind="linear", fill_value="extrapolate")
    interp_Tjl = interp1d(X, Tjl_array, axis=0, kind="linear", fill_value="extrapolate")

    kj_fine_re = interp_kj_re(X_fine)
    dkjx = np.zeros_like(kj_fine_re)
    dkjx[1:-1, :] = (kj_fine_re[2:, :] - kj_fine_re[:-2, :]) / (2 * dx)
    interp_dkj = interp1d(X_fine, dkjx, axis=0, kind="linear", fill_value="extrapolate")

    thetajx = cumulative_trapezoid(kj_fine_re, X_fine, axis=0, initial=0)

    # --- Initial condition (source) ---
    phi0 = phis_array[0, izs, :]
    k0 = kjx_array[0, :]
    A = 0.5 * np.conj(phi0) / k0

    nx = len(X_fine)
    Ajx = np.zeros((nx, cfg.nmod), dtype=complex)
    Ajx[0, :] = A

    # --- RK4 integration of the coupled modal amplitudes ---
    for ii in range(1, nx):
        x0, x1 = X_fine[ii - 1], X_fine[ii]
        xm = x0 + 0.5 * dx
        theta0, theta1 = thetajx[ii - 1], thetajx[ii]
        theta_mid = 0.5 * (theta0 + theta1)

        kj_m, dkj_m = interp_kj_re(xm), interp_dkj(xm)
        Cjl_m = interp_Cjl(xm) * dhdx[ii]
        Tjl_m = interp_Tjl(xm)

        K1 = _coupled_modes_rhs(
            A, interp_kj_re(x0), interp_dkj(x0),
            interp_Cjl(x0) * dhdx[ii - 1], interp_Tjl(x0), theta0,
        )
        K2 = _coupled_modes_rhs(A + 0.5 * dx * K1, kj_m, dkj_m, Cjl_m, Tjl_m, theta_mid)
        K3 = _coupled_modes_rhs(A + 0.5 * dx * K2, kj_m, dkj_m, Cjl_m, Tjl_m, theta_mid)
        K4 = _coupled_modes_rhs(
            A + dx * K3, interp_kj_re(x1), interp_dkj(x1),
            interp_Cjl(x1) * dhdx[ii], interp_Tjl(x1), theta1,
        )

        A = A + (dx / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)
        Ajx[ii, :] = A

    Ajx *= np.exp(1j * thetajx)

    # --- Pressure field / TL at zr (1D) ---
    phizrx = interp_phi(X_fine)[:, iz, :]
    # P = np.sum(Ajx * phizrx, axis=1) / np.sqrt(X_fine)   # 3D field
    P = np.sum(Ajx * phizrx, axis=1)   # 2D field

    H0 = hankel1(0, cfg.omega / cfg.c_w)
    TL = 20 * np.log10(np.abs(4 * P / H0)) - 2.5

    # --- 2D TL field (x, z) ---
    interp_z = interp1d(X, Z, axis=0, kind="linear", fill_value="extrapolate")
    Z_fine = interp_z(X_fine)
    phi_2d = interp_phi(X_fine)
    P_2d = np.sum(Ajx[:, np.newaxis, :] * phi_2d, axis=2)

    P_2d_amp = np.abs(4 * P_2d / H0)
    P_2d_amp = np.where(P_2d_amp < 1e-10, 1e-10, P_2d_amp)
    TL_2d = 20 * np.log10(P_2d_amp) - 2.5

    return {
        "X": X, "Z": Z,
        "X_fin": X_fine, "D_fin": D_fine, "dx": dx,
        "TL": TL, "TL_2d": TL_2d, "Z_fine": Z_fine,
        "phis_array": phis_array, "kjx_array": kjx_array,
        "N_tot": N_tot,
        "iz": iz, "izs": izs,
    }


if __name__ == "__main__":
    cfg = SimulationConfig()
    results = compute_propagation(cfg)
    print(f"Number of modes: {cfg.nmod}")
    print(f"TL computed on {len(results['X_fin'])} points of the fine grid.")