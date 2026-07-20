"""
Computeinterrcoeff.py
======================
Coefficients d'integration piecewise (trapezes ponderes par la densite)
et coefficients de couplage inter-modes :
    - ComputeAttCouplingTerms : couplage du a l'attenuation (T_jl)
    - ComputeAllCjl           : couplage bathymetrique par derivee finie sur beta (V_jl)
    - compute_all_C1ij        : couplage bathymetrique analytique a l'interface (C1ij)
"""

import numpy as np

# ============================================================
# COUPLAGE DU A L'ATTENUATION (T_jl)
# ============================================================
def attenuation_terms(all_phis, all_roots_re, DZ_W, N_w, p_w, p_s, c_w, c_s,
                             omega, eta, beta_eau, beta_sediment_1, beta_sediment_2,
                             z_transition):
    """
    Matrice de couplage T_jl = omega^2 * eta * <phi_j | f_z | phi_l>, ou
    f_z(z) porte le profil d'attenuation (eau/sediment, avec gradient
    optionnel au-dela de z_transition).

    Vectorise sur (j, l) : le calcul se ramene a un produit matriciel
    phi^T @ diag(f_z) @ phi avec une correction d'extremite (equivalent a
    N appels a `CoefIntegrationPiecewise_0(0, ...)`, mais sans boucle Python).
    """
    all_Coupling = []
    for i in range(len(all_phis)):
        phi = all_phis[i]      # (N_tot, nmod)
        dz = DZ_W[i]
        nw = N_w[i]
        N_tot = phi.shape[0]

        f_z = np.zeros(N_tot)
        f_z[:nw] = beta_eau / (p_w * c_w**2)
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

        # Equivalent vectorise de CoefIntegrationPiecewise_0(0, f_z_row, phi_j*phi_l, dz)
        # pour toutes les paires (j, l) simultanement : la correction d'interface
        # a i_int=0 annule entierement la contribution du point z=0.
        integral_jl = dz * (phi.T @ (f_z[:, np.newaxis] * phi)
                            - np.outer(f_z[0] * phi[0, :], phi[0, :]))

        Coupling = (omega**2 * eta) * integral_jl
        all_Coupling.append(Coupling)

    return all_Coupling


