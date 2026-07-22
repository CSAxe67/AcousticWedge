"""
Modescompute.py
================
Eigenmode and wavenumber (real part) computation via a finite-difference
method, with Richardson extrapolation over three vertical discretization
steps.

Also provides:
    - the addition of the imaginary part (attenuation) to the wavenumbers
    - the vertical derivative of the modes (used for C1ij coupling)
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
from multiprocessing import Pool, cpu_count
from functools import partial

from config import SimulationConfig
from geometry import build_medium_mesh
from integration import compute_piecwise_integral, compute_piecewise_scalarpdct


# ============================================================
# EIGENVALUE PROBLEM RESOLUTION (single position x)
# ============================================================
def _process_one(i, DZ_W, N_w, N_tot, c_w, c_s, p_w, p_s, nmod, omega):
    """Symmetrized eigenmodes (eigh_tridiagonal) at position index i."""
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

    # Water/sediment interface
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

    # Sign convention: positive at index 10
    signs = np.where(modes[10, :] < 0, -1, 1)
    modes *= signs[np.newaxis, :]

    # Normalization via the density-weighted inner product
    I_mat = compute_piecewise_scalarpdct(nw, modes.T, modes, dz, p_w, p_s)
    normes = np.sqrt(np.diag(I_mat))
    modes = modes / normes

    return k_values, modes


# ============================================================
# RICHARDSON EXTRAPOLATION OVER 3 DISCRETIZATION STEPS
# ============================================================
def _run_richardson(cfg: SimulationConfig, process_fn, compute_der=False):
    """
    Logic shared by the symmetrized variant (`Modes_compute`) and the
    non-symmetrized one (`Modescompute_Nonsym.Modes_compute_nonsym`):
    building the mesh at the 3 resolutions in `cfg.R_factor`, solving the
    eigenvalue problem in parallel via `process_fn`, then applying
    Richardson extrapolation (2nd order in dz) to the wavenumbers.

    Parameters
    ----------
    cfg : SimulationConfig
        Global configuration (cfg.Deriv_step, cfg.R_factor, cfg.nmod, cfg.omega).
    process_fn : callable
        Function solving the eigenvalue problem at a single position
        (same signature as `_process_one` / `_process_one_nonsym`).
    compute_der : bool
        If True, computes the perturbed modes/eigenvalues (for
        `compute_coupling_terms_1`), at `Hmax` reduced by `beta*dz`.

    Returns
    -------
    extrapolated_roots : list of np.ndarray (nmod,)
        Extrapolated real wavenumbers, one array per position x.
    all_modes : list of np.ndarray (N_tot, nmod)
        Modes at the finest mesh (R_factor[2]), one array per position x.
    """
    R_factor = np.asarray(cfg.R_factor, dtype=float)
    if R_factor.shape != (3,):
        raise ValueError("cfg.R_factor must contain exactly 3 values (coarse -> fine).")

    nmod = cfg.nmod
    omega = cfg.omega

    mesh0 = build_medium_mesh(cfg, R_factor[0], compute_der)
    mesh1 = build_medium_mesh(cfg, R_factor[1], compute_der)
    mesh2 = build_medium_mesh(cfg, R_factor[2], compute_der)

    n_pos = len(mesh0.Z)
    print(f"Parallel computation over {n_pos} positions...")

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

    # A single process pool reused across the 3 resolutions
    # (avoids creating/destroying 3 successive process pools)
    with Pool(cpu_count()) as pool:
        results = pool.map(worker, range(n_pos))
        results_1 = pool.map(worker_1, range(len(mesh1.Z)))
        results_2 = pool.map(worker_2, range(len(mesh2.Z)))

    all_roots = [r[0] for r in results]
    all_roots_1 = [r[0] for r in results_1]
    all_roots_2 = [r[0] for r in results_2]
    all_modes = [r[1] for r in results_2]

    # --- Richardson extrapolation (2nd order in dz, based on R_factor) ---
    r = 1.0 / R_factor  # relative steps, e.g. [1, 0.667, 0.5]
    M = np.column_stack([np.ones(3), r**2, r**4])
    M_inv = np.linalg.inv(M)

    extrapolated_roots = []
    for i in range(n_pos):
        K_matrix = np.vstack([all_roots[i], all_roots_1[i], all_roots_2[i]])
        coeff = M_inv @ K_matrix
        extrapolated_roots.append(coeff[0, :])  # 0th-order term = extrapolated value

    return extrapolated_roots, all_modes


def Modes_compute(cfg: SimulationConfig, compute_der=False):
    """Symmetrized variant (`eigh_tridiagonal`, see `_process_one`). See `_run_richardson`."""
    return _run_richardson(cfg, _process_one, compute_der)


# ============================================================
# IMAGINARY PART OF THE WAVENUMBERS (ATTENUATION)
# ============================================================
def compute_imaginary_wavenumbers(cfg: SimulationConfig, all_roots, all_phis, DZ_W, N_w):
    """
    Adds the imaginary part (attenuation) to the real wavenumbers.

    Parameters
    ----------
    cfg : SimulationConfig
        Global configuration. Uses cfg.p_w, cfg.p_s, cfg.c_w, cfg.c_s,
        cfg.omega, cfg.eta, cfg.beta_water, cfg.beta_sediment_1,
        cfg.beta_sediment_2, cfg.z_transition.
    all_roots : list of np.ndarray, each (nmod,)
        Real wavenumbers at each position x.
    all_phis : list of np.ndarray, each (N_tot, nmod)
        Normalized modes at each position x.
    DZ_W : list of float
        Vertical mesh step at each position x.
    N_w : list of int
        Water/sediment interface index at each position x.

    Returns
    -------
    list of np.ndarray of complex, each (nmod,)
        Complex wavenumbers (real part unchanged, imaginary part = attenuation).
    """
    if not (len(all_roots) == len(all_phis) == len(DZ_W) == len(N_w)):
        raise ValueError(
            "all_roots, all_phis, DZ_W and N_w must have the same length "
            f"(got {len(all_roots)}, {len(all_phis)}, {len(DZ_W)}, {len(N_w)})."
        )

    p_w, p_s = cfg.p_w, cfg.p_s
    c_w, c_s = cfg.c_w, cfg.c_s
    omega, eta = cfg.omega, cfg.eta
    beta_water = cfg.beta_water
    beta_sediment_1, beta_sediment_2 = cfg.beta_sediment_1, cfg.beta_sediment_2
    z_transition = cfg.z_transition

    all_roots_complex = []
    for i in range(len(all_roots)):
        kj_re = all_roots[i]
        phi = all_phis[i]
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

        M1 = f_z.reshape(1, -1)
        M2 = phi**2
        integral_val = compute_piecwise_integral(nw, M1, M2, dz).flatten()
        kj_im = (omega**2 * eta * integral_val) / kj_re

        all_roots_complex.append(kj_re.astype(complex) + 1j * kj_im)

    return all_roots_complex


# ============================================================
# VERTICAL DERIVATIVE OF THE MODES (centered finite differences)
# ============================================================
def compute_mode_derivatives_z(all_modes, DZ_W, N_w):
    """
    Derivative dphi/dz via centered finite differences, with special
    handling of the water/sediment interface (average of the left/right
    derivatives).

    Parameters
    ----------
    all_modes : list of np.ndarray, each (N_tot, nmod)
        Normalized modes at each position x.
    DZ_W : list of float
        Vertical mesh step at each position x.
    N_w : list of int
        Water/sediment interface index at each position x.

    Returns
    -------
    list of np.ndarray, each (N_tot, nmod)
        Vertical derivative dphi/dz at each position x.
    """
    if not (len(all_modes) == len(DZ_W) == len(N_w)):
        raise ValueError(
            "all_modes, DZ_W and N_w must have the same length "
            f"(got {len(all_modes)}, {len(DZ_W)}, {len(N_w)})."
        )

    all_dmodes = []

    for i, modes in enumerate(all_modes):
        dz = DZ_W[i]
        nw = N_w[i]
        N = modes.shape[0]
        dm = modes.copy()

        # Dirichlet: zero derivative at the surface and at the bottom
        dm[0, :] = 0.0
        dm[-1, :] = 0.0

        dm[1:nw, :] = (modes[2:nw + 1, :] - modes[0:nw - 1, :]) / (2.0 * dz)
        dm[nw + 1:N - 1, :] = (modes[nw + 2:N, :] - modes[nw:N - 2, :]) / (2.0 * dz)

        # Interface: average of the left and right derivatives
        d_left = (modes[nw, :] - modes[nw - 1, :]) / dz
        d_right = (modes[nw + 1, :] - modes[nw, :]) / dz
        dm[nw, :] = 0.5 * (d_left + d_right)

        all_dmodes.append(dm)

    return all_dmodes