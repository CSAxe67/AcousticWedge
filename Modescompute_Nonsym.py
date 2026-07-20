"""
Modescompute_Nonsym.py
=======================
Variante non symetrisee (matrice pleine + `scipy.linalg.eig`) du calcul
des modes propres, fidele a l'implementation MATLAB d'origine. Plus
couteuse que `Modescompute._process_one` (methode symetrisee, utilisee
en production) ; conservee ici pour comparaison/validation croisee
uniquement.

Reutilise la logique commune (maillage 3 resolutions + extrapolation de
Richardson) de `Modescompute._run_richardson` pour eviter toute
duplication de code.
"""

import numpy as np
from scipy.linalg import eig

from config import SimulationConfig
from Modescompute import _run_richardson


def _normalize_modes(modes, nw, dz, p_w, p_s):
    """Normalisation trapezoidale ponderee par la densite (utilisee par _process_one_nonsym)."""
    N = modes.shape[0]
    w = np.empty(N)
    w[:nw] = dz / p_w
    w[nw:] = dz / p_s
    w[0] *= 0.5
    w[-1] *= 0.5
    norms = np.sqrt(np.einsum("ij,i->j", modes**2, w))
    norms[norms == 0] = 1.0
    return modes / norms[np.newaxis, :]


def _process_one_nonsym(i, DZ_W, N_w, N_tot, c_w, c_s, p_w, p_s, nmod, omega):
    """
    Variante non symetrisee (matrice pleine + `scipy.linalg.eig`), fidele a
    l'implementation MATLAB d'origine. Plus couteuse que `_process_one` ;
    conservee pour comparaison/validation croisee.
    """
    dz = DZ_W[i]
    nw = N_w[i]
    N = N_tot

    rho = np.empty(N)
    c = np.empty(N)
    rho[:nw] = p_w
    rho[nw:] = p_s
    c[:nw] = c_w
    c[nw:] = c_s

    main = -2.0 / dz**2 + (omega / c) ** 2
    lower = np.full(N, 1.0 / dz**2)
    upper = np.full(N, 1.0 / dz**2)

    # Dirichlet surface / fond (comme MATLAB : main[0] et main[-1] non modifies)
    lower[0] = lower[1] = upper[0] = 0.0
    lower[-1] = upper[-2] = upper[-1] = 0.0

    # Interface eau/sediment
    main[nw] = (omega**2 * (
        p_s / ((p_w + p_s) * c_w**2) + p_w / ((p_w + p_s) * c_s**2)
    ) - 2.0 / dz**2)
    upper[nw] = (2.0 / dz**2) * p_w / (p_w + p_s)
    lower[nw] = (2.0 / dz**2) * p_s / (p_w + p_s)

    A = np.diag(main) + np.diag(upper[:-1], 1) + np.diag(lower[1:], -1)
    lam, vecs = eig(A)

    idx = np.argsort(lam.real)[::-1]
    lam = lam[idx].real[:nmod]
    vecs = vecs[:, idx].real[:, :nmod]

    k_values = np.sqrt(np.maximum(lam, 0.0))

    signs = np.sign(vecs[10, :])
    signs[signs == 0] = 1
    vecs *= signs[np.newaxis, :]

    modes = _normalize_modes(vecs, nw, dz, p_w, p_s)
    return k_values, modes


def Modes_compute_nonsym(cfg: SimulationConfig, compute_der=False):
    """Variante non symetrisee de `Modescompute.Modes_compute`. Voir `_run_richardson`."""
    return _run_richardson(cfg, _process_one_nonsym, compute_der)