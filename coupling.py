"""
Computeinterrcoeff.py
======================
    - ComputeAllCjl           : couplage bathymetrique par derivee finie sur beta (V_jl)
    - compute_all_C1ij        : couplage bathymetrique analytique a l'interface (C1ij)
"""

import numpy as np
from integration import compute_piecwise_integral
from integration import compute_piecewise_scalarpdct


# ============================================================
# COUPLAGE BATHYMETRIQUE — DERIVEE FINIE SUR BETA (V_jl)
# ============================================================
def compute_coupling_terms_1(all_phis, all_phis_der, D, DZ_W, N_w, beta, X, rho_w, rho_s):
    """
    Coefficients de couplage V_jl = <dphi_j/dh | phi_l>_rho, ou dphi/dh est
    estimee par difference finie entre les modes a h et h + beta*dz.
    """
    all_Cjl = []

    for i in range(len(all_phis)):
        phi0 = all_phis[i]       # modes a h
        phi1 = all_phis_der[i]   # modes a h + dh
        dz = DZ_W[i]
        nw = N_w[i]
        dh = beta * dz

        dphi_dh = (phi1 - phi0) / dh

        M1 = dphi_dh.T   # (nmod, N_tot)
        M2 = phi0        # (N_tot, nmod)

        Cjl = compute_piecewise_scalarpdct(nw, M1, M2, dz, rho_w, rho_s)
        all_Cjl.append(Cjl)

    return all_Cjl


# ============================================================
# COUPLAGE BATHYMETRIQUE — FORMULE ANALYTIQUE A L'INTERFACE (C1ij)
# ============================================================
def compute_couplingterms_x(modes, dmodes, wnum, nw, p_w, p_s, c_w, c_s, omega):
    """
    Matrice C1ij (couplage bathymetrique, cf. reference [2] Eq. 18-20) pour
    UNE position x donnee.

    Vectorise sur (j, l) par broadcasting numpy (evite la double boucle
    Python en O(nmod^2), couteuse pour nmod ~ 100-150).

    Parametres
    ----------
    modes  : (N_tot, nmod) — modes normalises a cette position (phi_j)
    dmodes : (N_tot, nmod) — derivees dphi_j/dz a cette position
    wnum   : (nmod,)       — nombres d'onde k_j a cette position
    nw     : int           — indice de l'interface eau/sediment
    p_w, p_s, c_w, c_s, omega : scalaires

    Retourne
    --------
    C1ij : (nmod, nmod)
    """
    ih = nw
    gammaw = 1.0 / p_w
    gammab = 1.0 / p_s

    phi_ih = modes[ih, :]     # (nmod,)
    dphi_ih = dmodes[ih, :]   # (nmod,)
    kj2 = wnum**2

    # Terme qui ne depend que de j (indice "ligne")
    term_j = gammaw * (omega / c_w) ** 2 - gammab * (omega / c_s) ** 2 + (gammab - gammaw) * kj2

    # b1[j, l] = phi_ih[j]*phi_ih[l]*term_j[j] - (p_s - p_w)*gammab^2*dphi_ih[j]*dphi_ih[l]
    b1 = np.outer(phi_ih * term_j, phi_ih) - (p_s - p_w) * gammab**2 * np.outer(dphi_ih, dphi_ih)

    # denom[j, l] = kl^2 - kj^2
    denom = kj2[np.newaxis, :] - kj2[:, np.newaxis]
    np.fill_diagonal(denom, 1.0)  # evite la division par 0 (diagonale traitee a part)

    C1ij = -b1 / denom
    np.fill_diagonal(C1ij, phi_ih**2 * (gammab - gammaw) / 2.0)  # Eq. (18), cas j == l

    return C1ij


def compute_couplingterms_2(all_modes, all_dmodes, all_roots, DZ_W, N_w,
                      p_w, p_s, c_w, c_s, omega):
    """Calcule C1ij pour toutes les positions x (voir compute_C1ij)."""
    all_C1ij = []
    for i in range(len(all_modes)):
        C1 = compute_couplingterms_x(
            modes=all_modes[i],
            dmodes=all_dmodes[i],
            wnum=all_roots[i],
            nw=N_w[i],
            p_w=p_w, p_s=p_s, c_w=c_w, c_s=c_s, omega=omega,
        )
        all_C1ij.append(-C1)

    return all_C1ij