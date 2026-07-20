"""
Propagation.py
==============
Calcul de la propagation acoustique couplée multi-modes (wedge).
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


def _coupled_modes_rhs(Ac, kj, dkj, C, TT, theta):
    """Second membre dA/dx pour les amplitudes modales couplées, à x fixe."""
    delta_theta = -theta[:, None] + theta[None, :]
    V_term = (C - C.T) * np.exp(1j * delta_theta)
    T_term = TT * np.exp(1j * delta_theta)

    return (
        - 0.5 * (dkj / kj) * Ac
        - 0.5 * (V_term @ (kj * Ac)) / kj
        - (T_term @ Ac) / kj
    )


def compute_propagation(cfg: SimulationConfig):
    """Calcule le champ de propagation modale couplée sur la grille fine X_fine."""
    # --- Extraction des paramètres de configuration ---
    c_w, c_s = cfg.c_w, cfg.c_s
    p_w, p_s = cfg.p_w, cfg.p_s
    omega, eta = cfg.omega, cfg.eta
    beta_eau = cfg.beta_eau
    beta_s1, beta_s2 = cfg.beta_sediment_1, cfg.beta_sediment_2
    z_trans = cfg.z_transition
    nmod = cfg.nmod

    # --- Profil de milieu (grille grossière) ---
    D, X, Z, DZ_W, N_w, N_tot = build_medium_mesh(cfg, cfg.R_factor[-1])
    Z = np.array(Z)

    # --- Grilles d'interpolation fines ---
    X_fine = np.linspace(cfg.X_0, cfg.X_fin, cfg.N_Grilles_fines)
    D_fine = np.linspace(cfg.D_0, cfg.D_fin, cfg.N_Grilles_fines)
    dx = X_fine[1] - X_fine[0]

    # --- Indices récepteur / source ---
    iz = np.argmin(np.abs(Z[0] - cfg.zr))
    izs = np.argmin(np.abs(Z[0] - cfg.zs))

    # --- Modes propres et nombres d'onde (partie réelle) ---
    all_roots, all_phis = Modes_compute(cfg)

    # --- Couplage dû à l'atténuation ---
    T = attenuation_terms(
        all_phis, all_roots, DZ_W, N_w, p_w, p_s, c_w, c_s,
        omega, eta, beta_eau, beta_s1, beta_s2, z_trans,
    )

    # --- Partie imaginaire des nombres d'onde ---
    all_roots = compute_imaginary_wavenumbers(
        all_roots, all_phis, DZ_W, N_w, omega, eta, p_w, p_s, c_w, c_s,
        beta_eau=beta_eau,
        beta_sediment_1=beta_s1,
        beta_sediment_2=beta_s2,
        z_transition=z_trans,
    )
    all_roots_re = [np.real(k) for k in all_roots]

    # --- Couplage bathymétrique (pente du fond) ---
    all_dmodes = compute_mode_derivatives_z(all_phis, DZ_W, N_w)
    all_C1ij = compute_couplingterms_2(
        all_phis, all_dmodes, all_roots_re, DZ_W, N_w, p_w, p_s, c_w, c_s, omega
    )

    kjx_array = np.array(all_roots)
    phis_array = np.array(all_phis)
    c1jl_array = np.array(all_C1ij)
    Tjl_array = np.array(T)
    dhdx = np.gradient(D_fine, dx)

    # --- Interpolations sur la grille fine ---
    interp_kj_re = interp1d(X, np.real(kjx_array), axis=0, kind="linear", fill_value="extrapolate")
    interp_phi = interp1d(X, phis_array, axis=0, kind="linear", fill_value="extrapolate")
    interp_c1jl = interp1d(X, c1jl_array, axis=0, kind="linear", fill_value="extrapolate")
    interp_tjl = interp1d(X, Tjl_array, axis=0, kind="linear", fill_value="extrapolate")

    kj_fine_re = interp_kj_re(X_fine)
    dkjx = np.zeros_like(kj_fine_re)
    dkjx[1:-1, :] = (kj_fine_re[2:, :] - kj_fine_re[:-2, :]) / (2 * dx)
    interp_dkj = interp1d(X_fine, dkjx, axis=0, kind="linear", fill_value="extrapolate")

    thetajx = cumulative_trapezoid(kj_fine_re, X_fine, axis=0, initial=0)

    # --- Condition initiale (source) ---
    phi0 = phis_array[0, izs, :]
    k0 = kjx_array[0, :]
    Ac = 0.5 * np.conj(phi0) / k0

    nx = len(X_fine)
    Ajx = np.zeros((nx, nmod), dtype=complex)
    Ajx[0, :] = Ac

    # --- Intégration RK4 des amplitudes modales ---
    for ii in range(1, nx):
        x0, x1 = X_fine[ii - 1], X_fine[ii]
        xm = x0 + 0.5 * dx
        theta0, theta1 = thetajx[ii - 1], thetajx[ii]
        theta_mid = 0.5 * (theta0 + theta1)

        kj_m, dkj_m = interp_kj_re(xm), interp_dkj(xm)
        C_m = interp_c1jl(xm) * dhdx[ii]
        TT_m = interp_tjl(xm)

        K1 = _coupled_modes_rhs(
            Ac, interp_kj_re(x0), interp_dkj(x0),
            interp_c1jl(x0) * dhdx[ii - 1], interp_tjl(x0), theta0,
        )
        K2 = _coupled_modes_rhs(Ac + 0.5 * dx * K1, kj_m, dkj_m, C_m, TT_m, theta_mid)
        K3 = _coupled_modes_rhs(Ac + 0.5 * dx * K2, kj_m, dkj_m, C_m, TT_m, theta_mid)
        K4 = _coupled_modes_rhs(
            Ac + dx * K3, interp_kj_re(x1), interp_dkj(x1),
            interp_c1jl(x1) * dhdx[ii], interp_tjl(x1), theta1,
        )

        Ac += (dx / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)
        Ajx[ii, :] = Ac

    Ajx *= np.exp(1j * thetajx)

    # --- Calcul du TL 1D (à la profondeur recepteur zr) ---
    phizrx = interp_phi(X_fine)[:, iz, :]
    P = np.sum(Ajx * phizrx, axis=1) / np.sqrt(X_fine)

    H0 = hankel1(0, omega / c_w)
    TL = 20 * np.log10(np.abs(4 * P / H0)) - 2.5

    # --- Calcul du champ TL 2D (x, z) ---
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
    print(f"Nombre de modes : {cfg.nmod}")
    print(f"TL calculé sur {len(results['X_fin'])} points de la grille fine.")