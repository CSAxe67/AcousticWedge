"""
Propagation_Vcoupling.py
========================
Variante de `Propagation.py` utilisant les coefficients de couplage `V`
(calcules via `ComputeAllCjl` a partir des modes perturbes, methode des
differences finies sur beta) au lieu des coefficients `C1ij` (derivee
analytique des modes en z).

Le second membre du systeme couple utilise ici la formulation :
    dA/dx = -0.5*(dk/k)*A + 0.5*(coupling @ (k*A))/k
    avec coupling = (V - V.T + T)

alors que la version C1ij utilise :
    dA/dx = -0.5*(dk/k)*A - 0.5*(V_term @ (k*A))/k - (T_term @ A)/k

Aucune figure n'est generee ici — voir `Plot_wedge.py` pour la visualisation.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.special import hankel1

from MediumParam import (
    MediumParam,
    c_w,
    beta, R_factor, eta,
    beta_eau, beta_sediment_1, beta_sediment_2,
    omega, nmod, z_transition,
    zr, zs, D_0, X_0, X_fin, D_fin, N_Grilles_fines,
)
from Modescompute import Modes_compute, compute_imaginary_wavenumbers
from attenuation import attenuation_terms
from coupling import compute_couplingterms_1


# ============================================================
# SECOND MEMBRE DU SYSTEME COUPLE (formulation V)
# ============================================================
def _coupled_modes_rhs(Ac, kj, dkj, V, TT, theta):
    """Second membre dA/dx, formulation avec coefficients V (ComputeAllCjl)."""
    delta_theta = -theta[:, None] + theta[None, :]
    coupling = (V - V.T + TT) * np.exp(1j * delta_theta)

    return (
        - 0.5 * (dkj / kj) * Ac
        + 0.5 * (coupling @ (kj * Ac)) / kj
    )


# ============================================================
# CALCUL PRINCIPAL
# ============================================================
def compute_propagation_Vcoupling():
    """
    Calcule le champ de propagation modale couplee sur la grille fine X_fin,
    en utilisant les coefficients de couplage V (ComputeAllCjl).

    Returns
    -------
    dict : meme structure que `Propagation.compute_propagation`
    """
    # --- Profil de milieu (grille grossiere) ---
    D, X, Z, DZ_W, N_w, N_tot, c_w_loc, c_s_loc, p_w, p_s = MediumParam(beta, R_factor[-1])
    Z = np.array(Z)

    # --- Grilles d'interpolation fines ---
    X_fine = np.linspace(X_0, X_fin, N_Grilles_fines)
    D_fine = np.linspace(D_0, D_fin, N_Grilles_fines)
    dx = X_fine[1] - X_fine[0]

    # --- Indices recepteur / source ---
    iz = np.argmin(np.abs(Z[0] - zr))
    izs = np.argmin(np.abs(Z[0] - zs))

    # --- Modes propres et nombres d'onde (partie reelle) ---
    all_roots, all_phis = Modes_compute(beta, nmod)

    # --- Modes et vp perturbes (pour la derivee de couplage V) ---
    all_roots_der, all_phis_der = Modes_compute(beta, nmod, compute_der=True)

    # --- Couplage du a la pente du fond (methode differences finies) ---
    V = compute_couplingterms_1(all_phis, all_phis_der, D, DZ_W, N_w, beta, X, p_w, p_s)

    # --- Couplage du a l'attenuation ---
    T = attenuation_terms(
        all_phis, all_roots, DZ_W, N_w, p_w, p_s, c_w_loc, c_s_loc,
        omega, eta, beta_eau, beta_sediment_1, beta_sediment_2, z_transition,
    )

    # --- Ajout de la partie imaginaire (attenuation) aux nombres d'onde ---
    all_roots = compute_imaginary_wavenumbers(
        all_roots, all_phis, DZ_W, N_w, omega, eta, p_w, p_s, c_w_loc, c_s_loc,
        beta_eau=beta_eau,
        beta_sediment_1=beta_sediment_1,
        beta_sediment_2=beta_sediment_2,
        z_transition=z_transition,
    )

    kjx_array = np.array(all_roots)     # (N_pos, nmod)
    phis_array = np.array(all_phis)     # (N_pos, N_tot, nmod)
    cjl_array = np.array(V)             # (N_pos, nmod, nmod)
    Tjl_array = np.array(T)             # (N_pos, nmod, nmod)

    # --- Interpolations sur la grille fine ---
    interp_kj_re = interp1d(X, np.real(kjx_array), axis=0, kind="linear", fill_value="extrapolate")
    interp_phi = interp1d(X, phis_array, axis=0, kind="linear", fill_value="extrapolate")
    interp_cjl = interp1d(X, cjl_array, axis=0, kind="linear", fill_value="extrapolate")
    interp_tjl = interp1d(X, Tjl_array, axis=0, kind="linear", fill_value="extrapolate")

    kj_fine_re = interp_kj_re(X_fine)
    dkjx = np.zeros_like(kj_fine_re)
    dkjx[1:-1, :] = (kj_fine_re[2:, :] - kj_fine_re[:-2, :]) / (2 * dx)
    interp_dkj = interp1d(X_fine, dkjx, axis=0, kind="linear", fill_value="extrapolate")

    thetajx = cumulative_trapezoid(kj_fine_re, X_fine, axis=0, initial=0)

    # --- Conditions initiales (source) ---
    phi0 = phis_array[0, izs, :]
    k0 = kjx_array[0, :]
    Ac = 0.5 * np.conj(phi0) / k0

    nx = len(X_fine)
    Ajx = np.zeros((nx, nmod), dtype=complex)
    Ajx[0, :] = Ac

    # --- Integration RK4 des amplitudes modales couplees ---
    for ii in range(1, nx):
        x0, x1 = X_fine[ii - 1], X_fine[ii]
        xm = x0 + 0.5 * dx
        theta0, theta1 = thetajx[ii - 1], thetajx[ii]
        theta_mid = 0.5 * (theta0 + theta1)

        kj_m, dkj_m = interp_kj_re(xm), interp_dkj(xm)
        V_m = interp_cjl(xm)
        TT_m = interp_tjl(xm)

        K1 = _coupled_modes_rhs(
            Ac, interp_kj_re(x0), interp_dkj(x0), interp_cjl(x0), interp_tjl(x0), theta0,
        )
        K2 = _coupled_modes_rhs(Ac + 0.5 * dx * K1, kj_m, dkj_m, V_m, TT_m, theta_mid)
        K3 = _coupled_modes_rhs(Ac + 0.5 * dx * K2, kj_m, dkj_m, V_m, TT_m, theta_mid)
        K4 = _coupled_modes_rhs(
            Ac + dx * K3, interp_kj_re(x1), interp_dkj(x1), interp_cjl(x1), interp_tjl(x1), theta1,
        )

        Ac = Ac + (dx / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)
        Ajx[ii, :] = Ac

    Ajx *= np.exp(1j * thetajx)

    # --- Champ de pression / TL en zr (1D) ---
    phizrx = interp_phi(X_fine)[:, iz, :]
    P = np.sum(Ajx * phizrx, axis=1) / np.sqrt(X_fine)

    H0 = hankel1(0, omega / c_w)
    TL = 20 * np.log10(np.abs(4 * P / H0)) - 2.5

    # --- Champ TL 2D (x, z) ---
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
    results = compute_propagation_Vcoupling()
    print(f"Nombre de modes : {nmod}")
    print(f"TL calcule sur {len(results['X_fin'])} points de la grille fine.")