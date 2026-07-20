"""
Modescompute.py
================
Calcul des modes propres et des nombres d'onde (partie reelle) par
methode aux differences finies, avec extrapolation de Richardson sur
trois pas de discretisation verticale.

Fournit egalement :
    - l'ajout de la partie imaginaire (attenuation) aux nombres d'onde
    - la derivee verticale des modes (utilisee pour le couplage C1ij)
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
from multiprocessing import Pool, cpu_count
from functools import partial
from config import SimulationConfig
from geometry import build_medium_mesh
from integration import compute_piecwise_integral
from integration import compute_piecewise_scalarpdct


# ============================================================
# RESOLUTION DU PROBLEME AUX VALEURS PROPRES (une position x)
# ============================================================
def _process_one(i, DZ_W, N_w, N_tot, c_w, c_s, p_w, p_s, nmod, omega):
    """Modes propres symetrises (eigh_tridiagonal) a la position d'indice i."""
    dz = DZ_W[i]
    nw = N_w[i]

    diag = np.zeros(N_tot)
    upper = np.zeros(N_tot - 1)

    diag[1:nw] = -2.0 / dz**2 + (omega / c_w) ** 2
    diag[nw + 1:N_tot - 1] = -2.0 / dz**2 + (omega / c_s) ** 2
    diag[0] = diag[-1] = 0.0

    upper[1:nw - 1] = 1.0 / dz**2
    upper[nw + 1:N_tot - 2] = 1.0 / dz**2
    upper[0] = upper[-1] = 0.0

    # Interface eau/sediment
    dp, dm = p_w, p_s
    cp, cm = c_w, c_s
    diag[nw] = (omega**2 * (dm / ((dp + dm) * cp**2) + dp / ((dp + dm) * cm**2)) - 2.0 / dz**2)
    upper[nw - 1] = (2.0 / dz**2) * (dp / (dp + dm))
    upper[nw] = (2.0 / dz**2) * (dm / (dp + dm))

    n_sel = int(nmod)

    eigenvalues, eigenvectors = eigh_tridiagonal(diag, upper)

    lam = eigenvalues[::-1][:n_sel]
    vecs = eigenvectors[:, ::-1][:, :n_sel]
    k_values = np.real(np.sqrt(lam.astype(complex)))
    modes = vecs

    # Correction de signe (convention : positif a l'indice 10)
    signs = np.where(modes[10, :] < 0, -1, 1)
    modes *= signs[np.newaxis, :]

    # Normalisation via le produit scalaire pondere par la densite
    I_mat = compute_piecewise_scalarpdct(nw, modes.T, modes, dz, p_w, p_s)
    normes = np.sqrt(np.diag(I_mat))
    modes = modes / normes

    return k_values, modes


# ============================================================
# EXTRAPOLATION DE RICHARDSON SUR 3 PAS DE DISCRETISATION
# ============================================================
def _run_richardson(cfg: SimulationConfig, process_fn, compute_der=False):
    """
    Logique commune aux variantes symetrisee (`Modes_compute`) et non
    symetrisee (`Modescompute_Nonsym.Modes_compute_nonsym`) : construction
    du maillage aux 3 resolutions de `cfg.R_factor`, resolution parallele
    du probleme aux valeurs propres via `process_fn`, puis extrapolation
    de Richardson (ordre 2 en dz) sur les nombres d'onde.

    Parametres
    ----------
    cfg : SimulationConfig
        Configuration globale (cfg.Deriv_step, cfg.R_factor, cfg.nmod, cfg.omega).
    process_fn : callable
        Fonction resolvant le probleme aux valeurs propres a une position
        (signature identique a `_process_one` / `_process_one_nonsym`).
    compute_der : bool
        Si True, calcule les modes/valeurs propres perturbes (pour
        `ComputeAllCjl`), a `Hmax` reduit de `beta*dz`.

    Retourne
    --------
    extrapolated_roots : list of np.ndarray (nmod,)
        Nombres d'onde reels extrapoles, un tableau par position x.
    all_modes : list of np.ndarray (N_tot, nmod)
        Modes au maillage le plus fin (R_factor[2]), un tableau par position x.
    """
    R_factor = np.asarray(cfg.R_factor, dtype=float)
    if R_factor.shape != (3,):
        raise ValueError("cfg.R_factor doit contenir exactement 3 valeurs (grossier -> fin).")

    nmod = cfg.nmod
    omega = cfg.omega

    mesh0 = build_medium_mesh(cfg, R_factor[0], compute_der)
    mesh1 = build_medium_mesh(cfg, R_factor[1], compute_der)
    mesh2 = build_medium_mesh(cfg, R_factor[2], compute_der)

    n_pos = len(mesh0.Z)
    print(f"Calcul parallele sur {n_pos} positions...")

    worker = partial(
    process_fn,
    DZ_W=mesh0.dz,
    N_w=mesh0.N_w,
    N_tot=mesh0.N_tot,
    c_w=cfg.c_w,
    c_s=cfg.c_s,
    p_w=cfg.p_w,
    p_s=cfg.p_s,
    nmod=cfg.nmod,
    omega=cfg.omega,
)
    worker_1 = partial(
    process_fn,
    DZ_W=mesh1.dz,
    N_w=mesh1.N_w,
    N_tot=mesh1.N_tot,
    c_w=cfg.c_w,
    c_s=cfg.c_s,
    p_w=cfg.p_w,
    p_s=cfg.p_s,
    nmod=cfg.nmod,
    omega=cfg.omega,
)

    worker_2 = partial(
        process_fn,
        DZ_W=mesh2.dz,
        N_w=mesh2.N_w,
        N_tot=mesh2.N_tot,
        c_w=cfg.c_w,
        c_s=cfg.c_s,
        p_w=cfg.p_w,
        p_s=cfg.p_s,
        nmod=cfg.nmod,
        omega=cfg.omega,
    )

    # Un seul pool de process reutilise pour les 3 resolutions
    # (evite de creer/detruire 3 pools de process successifs)
    with Pool(cpu_count()) as pool:
        results = pool.map(worker, range(n_pos))
        results_1 = pool.map(worker_1, range(len(mesh1.Z)))
        results_2 = pool.map(worker_2, range(len(mesh2.Z)))

    all_roots = [r[0] for r in results]
    all_roots_1 = [r[0] for r in results_1]
    all_roots_2 = [r[0] for r in results_2]
    all_modes = [r[1] for r in results_2]

    # --- Extrapolation de Richardson (ordre 2 en dz, base sur R_factor) ---
    r = 1.0 / R_factor  # pas relatifs, ex. [1, 0.667, 0.5]
    M = np.column_stack([np.ones(3), r**2, r**4])
    M_inv = np.linalg.inv(M)

    extrapolated_roots = []
    for i in range(n_pos):
        K_matrix = np.vstack([all_roots[i], all_roots_1[i], all_roots_2[i]])
        coeff = M_inv @ K_matrix
        extrapolated_roots.append(coeff[0, :])  # terme d'ordre 0 = valeur extrapolee

    return extrapolated_roots, all_modes


def Modes_compute(cfg: SimulationConfig, compute_der=False):
    """Variante symetrisee (`eigh_tridiagonal`, cf. `_process_one`). Voir `_run_richardson`."""
    return _run_richardson(cfg, _process_one, compute_der)


# ============================================================
# PARTIE IMAGINAIRE DES NOMBRES D'ONDE (ATTENUATION)
# ============================================================
def compute_imaginary_wavenumbers(all_roots, all_phis, DZ_W, N_w,
                                   omega, eta, p_w, p_s, c_w, c_s,
                                   beta_eau=0.0,
                                   beta_sediment_1=0.5,
                                   beta_sediment_2=0.5,
                                   z_transition=None):
    """Ajoute la partie imaginaire (attenuation) aux nombres d'onde reels."""
    all_roots_complex = []
    for i in range(len(all_roots)):
        kj_re = all_roots[i]
        phi = all_phis[i]
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

        M1 = f_z.reshape(1, -1)
        M2 = phi**2
        integral_val = compute_piecwise_integral(nw, M1, M2, dz).flatten()
        kj_im = (omega**2 * eta * integral_val) / kj_re

        all_roots_complex.append(kj_re.astype(complex) + 1j * kj_im)

    return all_roots_complex


# ============================================================
# DERIVEE VERTICALE DES MODES (differences finies centrees)
# ============================================================
def compute_mode_derivatives_z(all_modes, DZ_W, N_w):
    """Derivee dphi/dz par differences finies centrees, avec traitement
    special de l'interface eau/sediment (moyenne des derivees a gauche/droite)."""
    all_dmodes = []

    for i, modes in enumerate(all_modes):
        dz = DZ_W[i]
        nw = N_w[i]
        N = modes.shape[0]
        dm = modes.copy()

        # Dirichlet : derivee nulle en surface et au fond
        dm[0, :] = 0.0
        dm[-1, :] = 0.0

        dm[1:nw, :] = (modes[2:nw + 1, :] - modes[0:nw - 1, :]) / (2.0 * dz)
        dm[nw + 1:N - 1, :] = (modes[nw + 2:N, :] - modes[nw:N - 2, :]) / (2.0 * dz)

        # Interface : moyenne des derivees a gauche et a droite
        d_left = (modes[nw, :] - modes[nw - 1, :]) / dz
        d_right = (modes[nw + 1, :] - modes[nw, :]) / dz
        dm[nw, :] = 0.5 * (d_left + d_right)

        all_dmodes.append(dm)

    return all_dmodes