"""
two_layer_modes.py
===================

Normal-mode toolbox for the two-layer (Pekeris-type) waveguide

        phi_zz + (omega^2/c^2) phi = k^2 phi ,   0 <= z <= H
        phi(0) = 0,   phi(H) = 0
        phi and (1/rho) phi_z continuous at z = h

with water layer (c1, rho1) for z < h and bottom/sediment layer (c2, rho2)
for z > h.  Implements both mode classes:

    * water-borne modes   : oscillatory in the water column, evanescent
                             (sinh/cosh) in the bottom layer
    * bottom-borne modes  : oscillatory (sin/cos) in *both* layers, as
                             defined in the accompanying derivation

and everything needed to build the mode-coupling coefficients

    V_lj = ( d(phi_l)/dh , phi_j ) = int_0^H (1/rho) (d phi_l/dh) phi_j dz

that appear in adiabatic / coupled-mode range-dependent propagation models
when the water depth h = h(r) varies with range.

All closed-form expressions (dispersion relations, normalization
integrals, d k_j/dh via the implicit function theorem, and the four
class-pair coupling-coefficient integrals) were derived analytically and
cross-checked against an independent finite-difference / quadrature
pipeline; see `coupling_coefficient_numeric` below for that same
independent check, callable at any parameter set.

Author: derivation & implementation for Daniel (ACM Group, Univ. Wuppertal)
"""

from __future__ import annotations

import warnings
from typing import List, Literal, Optional, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

from pekeris import Waveguide, Mode, Branch


def get_modes(
    wg: Waveguide,
    h: float,
    branch: Branch,
    points_per_half_period: int = 12,
    min_points_per_interval: int = 24,
    margin: float = 1e-4,
) -> List[Mode]:
    """Find all wavenumbers k_j(h) for `branch` and package them as Modes."""
    ks = wg.find_wavenumbers(
        h, branch, points_per_half_period, min_points_per_interval, margin=margin
    )
    return [Mode(wg, branch, h, k, index=i) for i, k in enumerate(ks)]


# =====================================================================
# 3. Elementary antiderivatives used by the coupling-coefficient formulas
# =====================================================================


def _F(c, x):
    """int_0^x z cos(cz) dz-type helper: [sin(cx) - cx cos(cx)] / c^2."""
    return np.where(
        np.abs(c) < 1e-12,
        0.0,
        (np.sin(c * x) - c * x * np.cos(c * x)) / np.where(c == 0, 1.0, c**2),
    )


def _Fs(c, x):
    """Hyperbolic analogue: [cx cosh(cx) - sinh(cx)] / c^2."""
    return np.where(
        np.abs(c) < 1e-12,
        0.0,
        (c * x * np.cosh(c * x) - np.sinh(c * x)) / np.where(c == 0, 1.0, c**2),
    )


def _sinhc(c, D):
    """sinh(cD)/c, with the c -> 0 limit D handled."""
    return D if abs(c) < 1e-12 else np.sinh(c * D) / c


def _sinc_(c, D):
    """sin(cD)/c, with the c -> 0 limit D handled."""
    return D if abs(c) < 1e-12 else np.sin(c * D) / c


def J_ss(a, b, D):
    """int_0^D sinh(au) sinh(bu) du."""
    return 0.5 * (_sinhc(a + b, D) - _sinhc(a - b, D))


def K_cs(a, b, D):
    """int_0^D u cosh(au) sinh(bu) du."""
    return 0.5 * (_Fs(a + b, D) + _Fs(b - a, D))


def L_ss(a, b, D):
    """int_0^D sin(au) sin(bu) du."""
    return 0.5 * (_sinc_(a - b, D) - _sinc_(a + b, D))


def M_cs(a, b, D):
    """int_0^D u cos(au) sin(bu) du."""
    return 0.5 * (_F(a + b, D) + _F(b - a, D))


def S_sc(a, b, D):
    """int_0^D sinh(au) sin(bu) du."""
    return (a * np.cosh(a * D) * np.sin(b * D) - b * np.sinh(a * D) * np.cos(b * D)) / (
        a**2 + b**2
    )


def T_fun(a, b, D):
    """int_0^D u cosh(au) sin(bu) du."""
    num = (
        (b**2 - a**2) * np.cosh(a * D) * np.sin(b * D)
        + 2 * a * b * np.sinh(a * D) * np.cos(b * D)
        + a * D * (a**2 + b**2) * np.sinh(a * D) * np.sin(b * D)
        - b * D * (a**2 + b**2) * np.cosh(a * D) * np.cos(b * D)
    )
    return num / (a**2 + b**2) ** 2


# =====================================================================
# 5. Attenuative coupling matrix
#      T_jl = omega^2 eta int_0^H phi_j phi_l f_z(z) dz,
#      f_z = 0 (z<h), f1 (h<=z<h_t), f1 + z*f2 (z>=h_t)
# =====================================================================
#
# f_z vanishes identically in the water column, so only the bottom-layer
# branch of each mode ever contributes; splitting at h_t,
#
#   T_jl = (omega^2 eta / (N_j N_l)) * [ f1*I0(j,l) + f2*I1(j,l) ],
#   I0(j,l) = int_h^H   phi_j phi_l      dz   (raw, un-normalized modes)
#   I1(j,l) = int_{h_t}^H  z  phi_j phi_l  dz
#
# Unlike V_lj there is no orthogonality short-cut (f_z is not constant
# over the bottom layer), so this is evaluated directly for every (j,l)
# pair, diagonal included. Three new u-weighted antiderivatives are
# needed for I1, built from the elementary helpers Gc, Gs below.


def _Gc(c, D):
    """int_0^D u cosh(cu) du = (D/c) sinh(cD) - (cosh(cD)-1)/c^2."""
    if abs(c) < 1e-12:
        return D**2 / 2
    return (D / c) * np.sinh(c * D) - (np.cosh(c * D) - 1) / c**2


def _Gs(c, D):
    """int_0^D u cos(cu) du = (D/c) sin(cD) + (cos(cD)-1)/c^2."""
    if abs(c) < 1e-12:
        return D**2 / 2
    return (D / c) * np.sin(c * D) + (np.cos(c * D) - 1) / c**2


def K_ss(a, b, D):
    """int_0^D u sinh(au) sinh(bu) du."""
    return 0.5 * (_Gc(a + b, D) - _Gc(a - b, D))


def M_ss(a, b, D):
    """int_0^D u sin(au) sin(bu) du."""
    return 0.5 * (_Gs(a - b, D) - _Gs(a + b, D))


def R_fun(a, b, D):
    """int_0^D u sinh(au) cos(bu) du."""
    num = (
        (b**2 - a**2) * np.sinh(a * D) * np.cos(b * D)
        - 2 * a * b * np.cosh(a * D) * np.sin(b * D)
        + a * D * (a**2 + b**2) * np.cosh(a * D) * np.cos(b * D)
        + b * D * (a**2 + b**2) * np.sinh(a * D) * np.sin(b * D)
    )
    return num / (a**2 + b**2) ** 2


def W_fun(a, b, D):
    """int_0^D u sinh(au) sin(bu) du."""
    num = (
        a * D * (a**2 + b**2) * np.cosh(a * D) * np.sin(b * D)
        - b * D * (a**2 + b**2) * np.sinh(a * D) * np.cos(b * D)
        + (b**2 - a**2) * np.sinh(a * D) * np.sin(b * D)
        + 2 * a * b * np.cosh(a * D) * np.cos(b * D)
        - 2 * a * b
    )
    return num / (a**2 + b**2) ** 2


# =====================================================================
# 4. Coupling coefficients  V_lj = ( d phi_l/dh , phi_j )
# =====================================================================


def _U1(l: Mode, j: Mode) -> float:
    """Layer-1 (water column) contribution, common to all class pairs."""
    h = l.h
    return (
        -(l.k * l.kp)
        / (2 * l.wg.rho1 * l.kappa1)
        * (_F(l.kappa1 + j.kappa1, h) + _F(j.kappa1 - l.kappa1, h))
    )


def coupling_coefficient(l: Mode, j: Mode) -> float:
    """
    Closed-form coupling coefficient V_lj = (d phi_l/dh, phi_j), for any
    combination of branches ('water'/'bottom') for l and j, including the
    same-mode (diagonal, l == j) term.
    """
    if l.wg is not j.wg or l.h != j.h:
        raise ValueError("modes l and j must share the same Waveguide and depth h")

    if l.branch == j.branch and np.isclose(l.k, j.k, rtol=1e-10):
        # Same-mode (diagonal) term: the overlap-integral formula used below
        # for l != j relies on orthogonality to kill the -phi_l/N_l^2 * dN_l/dh
        # piece of d(phihat_l)/dh; for l == j that piece survives instead of
        # the overlap integral. Differentiating N_l^2 = int (1/rho) phi_l^2 dz
        # (Leibniz rule, moving interface at z=h) gives the closed form
        #   V_ll = [sin^2(kappa1_l h) / (2 N_l^2)] * (1/rho2 - 1/rho1),
        # valid for both branches (verified against finite differences).
        phi_h = np.sin(l.kappa1 * l.h)
        return (phi_h**2 / (2 * l.N**2)) * (1 / l.wg.rho2 - 1 / l.wg.rho1)

    rho2 = l.wg.rho2
    D = l.D
    u1 = _U1(l, j)

    if l.branch == "water" and j.branch == "water":
        u2 = (j.A / rho2) * (
            l.dA * J_ss(l.g2, j.g2, D) + l.A * l.g2p * K_cs(l.g2, j.g2, D)
        )
    elif l.branch == "bottom" and j.branch == "bottom":
        u2 = (j.C / rho2) * (
            l.dC * L_ss(l.x2, j.x2, D) + l.C * l.x2p * M_cs(l.x2, j.x2, D)
        )
    elif l.branch == "water" and j.branch == "bottom":
        u2 = -(j.C / rho2) * (
            l.dA * S_sc(l.g2, j.x2, D) + l.A * l.g2p * T_fun(l.g2, j.x2, D)
        )
    elif l.branch == "bottom" and j.branch == "water":
        u2 = -(j.A / rho2) * (
            l.dC * S_sc(j.g2, l.x2, D) + l.C * l.x2p * R_fun(j.g2, l.x2, D)
        )
    else:
        raise ValueError("branch must be 'water' or 'bottom'")

    return (u1 + u2) / (l.N * j.N)


def coupling_matrix(modes: List[Mode]) -> np.ndarray:
    """
    Build the (len(modes_l) x len(modes_j)) matrix of coupling coefficients
    V[p, q] = coupling_coefficient(modes_l[p], modes_j[q]), diagonal
    (same-mode) entries included.
    """
    V = np.empty((len(modes), len(modes)))
    for p, l in enumerate(modes):
        for q, j in enumerate(modes):
            V[p, q] = coupling_coefficient(l, j)
    return V


# ---------------------------------------------------------------------
# Independent numerical (finite-difference + quadrature) cross-check.
# Slow, but useful to validate the closed-form results at any parameter
# point, exactly as was done when deriving the formulas above.
# ---------------------------------------------------------------------


# def coupling_coefficient_numeric(
#     wg: Waveguide, mode_l: Mode, mode_j: Mode, eps: float = 1e-8
# ) -> float:
#     if not np.isclose(mode_l.h, mode_j.h, eps):
#         raise ValueError(
#             "coupling is supposed to occur between modes at the same location"
#         )

#     def integrand(z: float):
#         rho = wg.rho1 * (z < mode_l.h) + wg.rho2 * (z >= mode_l.h)
#         dphi = mode_l.dphi(z)
#         phi = mode_j.phi(z)
#         return(phi * dphi / rho).item()

#     I1, _ = quad(integrand, 0, mode_l.h, limit=300)
#     I2, _ = quad(integrand, mode_l.h, wg.H, limit=300)

#     return I1 + I2


def coupling_coefficient_numeric(
    wg: Waveguide,
    branch_l: Branch,
    idx_l: int,
    branch_j: Branch,
    idx_j: int,
    h: float,
    eps: float = 1e-6,
) -> float:
    """Brute-force finite-difference / quadrature evaluation of V_lj."""

    def phihat(branch, idx, hh, z):
        k = wg.find_wavenumbers(hh, branch)[idx]
        N = np.sqrt(wg.norm2(k, hh, branch))
        return wg.phi_raw(z, k, hh, branch) / N

    def dphihat_dh(branch, idx, hh, z):
        return (phihat(branch, idx, hh + eps, z) - phihat(branch, idx, hh - eps, z)) / (
            2 * eps
        )

    def integrand(z):
        rho = wg.rho1 if z < h else wg.rho2
        dphi = dphihat_dh(branch_l, idx_l, h, z)[0]
        phij = phihat(branch_j, idx_j, h, z)[0]
        return dphi * phij / rho

    I1, _ = quad(integrand, 0, h, limit=300)
    I2, _ = quad(integrand, h, wg.H, limit=300)
    return I1 + I2


def coupling_coefficient_T(
    j: Mode, l: Mode, h_t: float, f1: float, f2: float, eta: float
) -> float:
    """
    Closed-form attenuative coupling coefficient

        T_jl = omega^2 eta (phi_j, phi_l f_z)   [plain product, no 1/rho weight]

    for any combination of branches, any (j, l) including j == l. `h_t`
    is the depth (h <= h_t <= H) where the attenuation profile switches
    from the constant f1 to f1 + z*f2.
    """
    if j.wg is not l.wg or j.h != l.h:
        raise ValueError("modes j and l must share the same Waveguide and depth h")
    if not (j.h <= h_t <= j.wg.H):
        raise ValueError("h_t must lie in [h, H]")

    wg = j.wg
    D = j.D  # H - h
    Dt = wg.H - h_t
    rho2 = wg.rho2  # unused directly (T has no 1/rho weight) but kept for clarity

    if j.branch == "water" and l.branch == "water":
        pref = j.A * l.A
        I0 = pref * J_ss(j.g2, l.g2, D)
        I1 = pref * (wg.H * J_ss(j.g2, l.g2, Dt) - K_ss(j.g2, l.g2, Dt))
    elif j.branch == "bottom" and l.branch == "bottom":
        pref = j.C * l.C
        I0 = pref * L_ss(j.x2, l.x2, D)
        I1 = pref * (wg.H * L_ss(j.x2, l.x2, Dt) - M_ss(j.x2, l.x2, Dt))
    elif j.branch == "water" and l.branch == "bottom":
        pref = -j.A * l.C
        I0 = pref * S_sc(j.g2, l.x2, D)
        I1 = pref * (wg.H * S_sc(j.g2, l.x2, Dt) - W_fun(j.g2, l.x2, Dt))
    elif j.branch == "bottom" and l.branch == "water":
        # T_jl = T_lj (plain product, no derivative asymmetry): reuse the
        # water-bottom formula with the roles of j and l swapped.
        pref = -l.A * j.C
        I0 = pref * S_sc(l.g2, j.x2, D)
        I1 = pref * (wg.H * S_sc(l.g2, j.x2, Dt) - W_fun(l.g2, j.x2, Dt))
    else:
        raise ValueError("branch must be 'water' or 'bottom'")

    return wg.omega**2 * eta * (f1 * I0 + f2 * I1) / (j.N * l.N)


def coupling_matrix_T(
    modes: List[Mode], h_t: float, f1: float, f2: float, eta: float
) -> np.ndarray:
    """
    Build the (len(modes_j) x len(modes_l)) matrix of attenuative coupling
    coefficients T[p, q] = coupling_coefficient_T(modes_j[p], modes_l[q],
    h_t, f1, f2, eta), diagonal entries included (T has no orthogonality
    short-cut, so unlike coupling_matrix there is nothing special about j==l).
    """
    T = np.empty((len(modes), len(modes)))
    for p, j in enumerate(modes):
        for q, l in enumerate(modes):
            T[p, q] = coupling_coefficient_T(j, l, h_t, f1, f2, eta)
    return T


# =====================================================================
# 5. Plotting routines
# =====================================================================


def plot_dispersion(
    wg: Waveguide,
    h: float,
    branch: Branch,
    ax=None,
    n: int = 2000,
    margin: float = 1e-4,
):
    """Plot G_branch(k, h) vs k and mark its roots (the k_j at this h)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if branch == "water":
        kmin, kmax, G = (
            wg.omega / wg.c2 * (1 + margin),
            wg.omega / wg.c1 * (1 - margin),
            wg.Gw,
        )
    else:
        kmin, kmax, G = (
            wg.omega / wg.c2 * margin,
            wg.omega / wg.c2 * (1 - margin),
            wg.Gb,
        )

    ks = np.linspace(kmin, kmax, n)
    vals = np.array([G(k, h) for k in ks])

    # G has interior poles (coth/cot blow-ups) wherever the trial k makes
    # sinh(gamma2*D) or sin(kappa2*D) vanish; a fixed y-range would either
    # swamp the genuine zero-crossings (if sized for the poles) or clip
    # them into flat plateaus (if sized too small). Instead pick the range
    # from the bulk of the (finite) values and let the pole spikes run off
    # the visible axis, which reads as a normal-looking dispersion curve.
    finite = vals[np.isfinite(vals)]
    scale = 3 * np.nanpercentile(np.abs(finite), 90) if finite.size else 1.0
    scale = max(scale, 1e-12)

    ax.plot(ks, vals, lw=1.2)
    ax.axhline(0, color="k", lw=0.7)

    roots = wg.find_wavenumbers(h, branch, n=n, margin=margin)
    ax.plot(
        roots, np.zeros_like(roots), "o", color="crimson", zorder=5, label="$k_j$ roots"
    )

    ax.set_xlabel("horizontal wavenumber $k$")
    ax.set_ylabel(f"$G_{{{branch}}}(k, h={h:g})$")
    ax.set_title(f"Dispersion relation ({branch}-borne modes), $h={h:g}$")
    ax.set_ylim(-scale, scale)
    ax.legend()
    return ax


def plot_modes(
    wg: Waveguide,
    modes: List[Mode],
    ax=None,
    n_z: int = 800,
    normalize: bool = True,
    offset: float = 0.0,
):
    """
    Plot one or several mode shapes phi_j(z) with depth z on the vertical
    axis increasing downward (oceanographic convention). `offset` shifts
    successive curves horizontally by that amount, useful when overlaying
    many modes on one axis.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 6))

    z = np.linspace(0, wg.H, n_z)
    for i, m in enumerate(modes):
        phi = m.phi(z) if normalize else wg.phi_raw(z, m.k, m.h, m.branch)
        style = "-" if m.branch == "water" else "--"
        color = "tab:blue" if m.branch == "water" else "tab:orange"
        ax.plot(
            phi + i * offset,
            z,
            style,
            color=color,
            label=f"{m.branch} #{m.index}  $k={m.k:.4f}$",
        )

    ax.axhline(
        modes[0].h, color="gray", lw=1, ls=":", label=f"interface $h={modes[0].h:g}$"
    )
    ax.axhline(wg.H, color="k", lw=1)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\hat\phi_j(z)$" + (" (offset)" if offset else ""))
    ax.set_ylabel("depth $z$")
    ax.set_title("Normal modes")
    ax.legend(fontsize=8, loc="lower right")
    return ax


def plot_coupling_matrix(
    V: np.ndarray, labels_l=None, labels_j=None, ax=None, cmap: str = "RdBu_r"
):
    """Heat-map visualization of a coupling-coefficient matrix V_lj."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    vmax = np.nanmax(np.abs(V)) if np.isfinite(V).any() else 1.0
    im = ax.imshow(V, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="$V_{lj}$")

    if labels_l is not None:
        ax.set_yticks(range(len(labels_l)))
        ax.set_yticklabels(labels_l, fontsize=8)
    if labels_j is not None:
        ax.set_xticks(range(len(labels_j)))
        ax.set_xticklabels(labels_j, fontsize=8, rotation=90)

    ax.set_xlabel("mode $j$")
    ax.set_ylabel("mode $l$")
    ax.set_title("Coupling coefficients $V_{lj}$")
    return ax


# =====================================================================
# 6. Demonstration
# =====================================================================


def main():
    import matplotlib.pyplot as plt

    # --- example two-layer waveguide -----------------------------------
    # non-round numbers deliberately: exact round depths can put kappa2*(H-h)
    # or gamma2*(H-h) at an exact multiple of pi for some mode, which makes
    # the raw (un-normalized) amplitude C or A blow up even though the
    # normalized mode and V_lj stay perfectly finite -- see the
    # `_ILL_CONDITIONED_TOL` warning in Mode.compute().
    wg = Waveguide(
        c1=1500.0, c2=1800.0, rho1=1.0, rho2=1.5, H=400, omega=2 * np.pi * 50.0
    )  # 80 Hz source
    h = 100.0

    water_modes = get_modes(wg, h, "water")
    bottom_modes = get_modes(wg, h, "bottom")
    print(
        f"Found {len(water_modes)} water-borne modes, "
        f"{len(bottom_modes)} bottom-borne modes at h = {h} m"
    )
    print("water k_j:", [f"{m.k:.5f}" for m in water_modes])
    print("bottom k_j:", [f"{m.k:.5f}" for m in bottom_modes])

    # --- coupling coefficients ------------------------------------------
    all_modes = water_modes + bottom_modes
    V = coupling_matrix(all_modes)
    labels = [f"{m.branch[0]}{m.index}" for m in all_modes]

    if len(water_modes) >= 2:
        l, j = water_modes[0], water_modes[1]
        v_analytic = coupling_coefficient(l, j)
        v_numeric = coupling_coefficient_numeric(wg, "water", 0, "water", 1, h)
        print(
            f"\nV_ww[0,1]: closed-form = {v_analytic:.6e}, "
            f"finite-difference check = {v_numeric:.6e}"
        )

    if len(water_modes) >= 2:
        l, j = water_modes[0], water_modes[0]
        v_analytic = coupling_coefficient(l, j)
        v_numeric = coupling_coefficient_numeric(wg, "water", 0, "water", 0, h)
        print(
            f"\nV_ww[0,0]: closed-form = {v_analytic:.6e}, "
            f"finite-difference check = {v_numeric:.6e}"
        )

    # --- plots ------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.5))
    plot_dispersion(wg, h, "water", ax=axes[0])
    plot_dispersion(wg, h, "bottom", ax=axes[1])
    plot_modes(wg, (water_modes[:2] + bottom_modes[:3]), ax=axes[2])
    plot_coupling_matrix(V, labels_l=labels, labels_j=labels, ax=axes[3])
    fig.tight_layout()
    fig.savefig("two_layer_modes_demo.png", dpi=150)
    print("\nSaved demo figure to two_layer_modes_demo.png")


def test():
    import matplotlib.pyplot as plt

    wg = Waveguide(c1=1500, c2=1800, rho1=1.0, rho2=1.5, H=400, omega=2 * np.pi * 50)
    water_modes = get_modes(wg, 100, "water")


if __name__ == "__main__":
    test()
