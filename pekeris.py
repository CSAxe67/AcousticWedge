from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

from scipy.optimize import brentq

import numpy as np

# Interface-amplitude denominators [sinh(gamma2*D) for water-borne modes,
# sin(kappa2*D) for bottom-borne modes] occasionally pass very close to zero
# for special combinations of h, H and k_j (e.g. round-number depths that
# make kappa2*D an exact multiple of pi). Right at such a point C (or A) and
# N blow up together while phi/N and the coupling coefficients stay
# perfectly finite, but forming C, N separately in floating point loses
# precision catastrophically. This threshold flags the condition so it is
# reported rather than silently returned as a wrong number.
_ILL_CONDITIONED_TOL = 1e-6

Branch = Literal["water", "bottom"]

# =====================================================================
# 0. Helper method for root finding
# =====================================================================


def _scan_interval_for_roots(
    G, h, a, b, phase_fn, points_per_half_period, min_pts, validate_tol
):
    """
    Find all roots of the pole-free, smooth function k -> G(k, h) on the
    *open* interval (a, b), sampling with a point count driven by the actual
    phase excursion of `phase_fn` across (a, b) rather than a fixed budget --
    see the docstring of `Waveguide.find_wavenumbers` for why this matters.
    Used for every sub-interval between consecutive hard breakpoints (band
    edges and, for the bottom branch, the analytic cot poles).
    """
    span = b - a
    eps = max(span * 1e-9, 1e-14)
    a_in, b_in = a + eps, b - eps
    if b_in <= a_in:
        a_in, b_in = a, b  # degenerate (extremely narrow) interval: use as-is

    dphase = abs(phase_fn(b_in) - phase_fn(a_in))
    npts = max(min_pts, int(np.ceil(points_per_half_period * dphase / np.pi)) + 2)

    ks = np.linspace(a_in, b_in, npts)
    vals = G(ks, h)

    roots = []
    for i in range(npts - 1):
        v0, v1 = vals[i], vals[i + 1]
        if not (np.isfinite(v0) and np.isfinite(v1)):
            continue
        if v0 == 0.0:
            roots.append(ks[i])
            continue
        if v0 * v1 < 0.0:
            r = brentq(G, ks[i], ks[i + 1], args=(h,))
            if abs(G(r, h)) < validate_tol:
                roots.append(r)
            # else: sign flip came from a pole, not a genuine zero -- discard.
    return roots


# =====================================================================
# 1. Waveguide: geometry / dispersion relations / vertical wavenumbers
# =====================================================================


@dataclass
class Waveguide:
    """Two-layer waveguide parameters (SI or any consistent unit system)."""

    c1: float  # sound speed, water layer
    c2: float  # sound speed, bottom layer  (c2 > c1 assumed)
    rho1: float  # density, water layer
    rho2: float  # density, bottom layer
    H: float  # total depth (rigid/pressure-release floor at z = H)
    omega: float  # angular frequency

    # ---- vertical wavenumbers -----------------------------------------
    def kappa1(self, k):
        """Vertical wavenumber in the water layer, sqrt(omega^2/c1^2 - k^2)."""
        return np.sqrt(self.omega**2 / self.c1**2 - k**2)

    def gamma2(self, k):
        """Evanescent decay rate in the bottom layer (water-borne modes)."""
        return np.sqrt(k**2 - self.omega**2 / self.c2**2)

    def kappa2(self, k):
        """Vertical wavenumber in the bottom layer (bottom-borne modes)."""
        return np.sqrt(self.omega**2 / self.c2**2 - k**2)

    # ---- dispersion relations G(k, h) = 0 ------------------------------
    def Gw(self, k, h):
        """Dispersion function for water-borne modes."""
        D = self.H - h
        kappa, g = self.kappa1(k), self.gamma2(k)
        return (kappa / self.rho1) * np.cos(kappa * h) + (g / self.rho2) / np.tanh(
            g * D
        ) * np.sin(kappa * h)

    def Gb(self, k, h):
        """Dispersion function for bottom-borne modes."""
        D = self.H - h
        kappa, x2 = self.kappa1(k), self.kappa2(k)
        return (kappa / self.rho1) * np.cos(kappa * h) + (x2 / self.rho2) / np.tan(
            x2 * D
        ) * np.sin(kappa * h)

    def G(self, k, h, branch: Branch):
        return self.Gw(k, h) if branch == "water" else self.Gb(k, h)

    # ---- partial derivatives of G, needed for dk/dh via IFT ------------
    def dGw_dh(self, k, h):
        D = self.H - h
        kappa, g = self.kappa1(k), self.gamma2(k)
        return (
            -(kappa**2 / self.rho1) * np.sin(kappa * h)
            + (g**2 / self.rho2) / np.sinh(g * D) ** 2 * np.sin(kappa * h)
            + (g * kappa / self.rho2) / np.tanh(g * D) * np.cos(kappa * h)
        )

    def dGw_dk(self, k, h):
        D = self.H - h
        kappa, g = self.kappa1(k), self.gamma2(k)
        return (
            -(k / (self.rho1 * kappa)) * np.cos(kappa * h)
            + (k * h / self.rho1) * np.sin(kappa * h)
            + (k / (self.rho2 * g)) / np.tanh(g * D) * np.sin(kappa * h)
            - (k * D / self.rho2) / np.sinh(g * D) ** 2 * np.sin(kappa * h)
            - (k * h * g / (self.rho2 * kappa)) / np.tanh(g * D) * np.cos(kappa * h)
        )

    def dGb_dh(self, k, h):
        D = self.H - h
        kappa, x2 = self.kappa1(k), self.kappa2(k)
        return (
            -(kappa**2 / self.rho1) * np.sin(kappa * h)
            + (x2**2 / self.rho2) / np.sin(x2 * D) ** 2 * np.sin(kappa * h)
            + (x2 * kappa / self.rho2) / np.tan(x2 * D) * np.cos(kappa * h)
        )

    def dGb_dk(self, k, h):
        D = self.H - h
        kappa, x2 = self.kappa1(k), self.kappa2(k)
        return (
            -(k / (self.rho1 * kappa)) * np.cos(kappa * h)
            + (k * h / self.rho1) * np.sin(kappa * h)
            - (k / (self.rho2 * x2)) / np.tan(x2 * D) * np.sin(kappa * h)
            + (k * D / self.rho2) / np.sin(x2 * D) ** 2 * np.sin(kappa * h)
            - (k * h * x2 / (self.rho2 * kappa)) / np.tan(x2 * D) * np.cos(kappa * h)
        )

    def dk_dh(self, k, h, branch: Branch):
        """d k_j / dh via the implicit function theorem applied to G(k,h)=0."""
        if branch == "water":
            return -self.dGw_dh(k, h) / self.dGw_dk(k, h)
        else:
            return -self.dGb_dh(k, h) / self.dGb_dk(k, h)

    def bottom_cot_poles(self, h: float, margin: float = 1e-4) -> np.ndarray:
        """Compute the exact locations of the poles of the Dispersion relation for bottom bourne modes

        Args:
            h (float): interface position
            margin (float, optional): _description_. Defaults to 1e-4.

        Raises:
            ValueError: _description_

        Returns:
            np.ndarray: a collection of poles
        """
        D = self.H - h
        kmax = self.omega / self.c2 * (1 - margin)
        n_max = int(np.floor(self.omega / self.c2 * D / np.pi - 1.0e-9))
        poles = []
        for n in range(1, n_max + 1):
            ksq = (self.omega / self.c2) ** 2 - (n * np.pi / D) ** 2
            if ksq <= 0:
                break
            k = np.sqrt(ksq)
            if 0 < k < kmax:
                poles.append(k)
        return np.array(sorted(poles))

    # ---- root finding: k_j(h) for a given branch -----------------------
    def find_wavenumbers(
        self,
        h: float,
        branch: Branch,
        points_per_half_period: int = 12,
        min_points_per_interval: int = 24,
        margin: float = 1e-4,
        validate_tol: float = 1e-6,
    ) -> np.ndarray:
        if branch == "water":
            kmin = self.omega / self.c2 * (1 + margin)
            kmax = self.omega / self.c1 * (1 - margin)
            G = self.Gw
            breakpoints = np.array([kmin, kmax])
        elif branch == "bottom":
            kmin = self.omega / self.c2 * margin
            kmax = self.omega / self.c2 * (1 - margin)
            G = self.Gb
            poles = self.bottom_cot_poles(h, margin=margin)
            breakpoints = np.concatenate(([kmin], poles, [kmax]))
        else:
            raise ValueError("branch must be 'water' or 'bottom'")

        phase = (
            lambda k: self.kappa1(k) * h
        )  # shared oscillatory driver (cos/sin(kappa h))

        roots = []
        for a, b in zip(breakpoints[:-1], breakpoints[1:]):
            if b <= a:
                continue
            roots.extend(
                _scan_interval_for_roots(
                    G,
                    h,
                    a,
                    b,
                    phase,
                    points_per_half_period,
                    min_points_per_interval,
                    validate_tol,
                )
            )
        return np.array(sorted(roots))

    # ---- normalization integrals N^2 = (phi_j, phi_j) -------------------
    def norm2(self, k, h, branch: Branch):
        D = self.H - h
        kappa = self.kappa1(k)
        term1 = (1 / (2 * self.rho1)) * (
            h - np.sin(kappa * h) * np.cos(kappa * h) / kappa
        )
        if branch == "water":
            g = self.gamma2(k)
            term2 = (np.sin(kappa * h) ** 2 / (2 * self.rho2)) * (
                (1 / np.tanh(g * D)) / g - D / np.sinh(g * D) ** 2
            )
        else:
            x2 = self.kappa2(k)
            term2 = (np.sin(kappa * h) ** 2 / (2 * self.rho2)) * (
                D / np.sin(x2 * D) ** 2 - (1 / np.tan(x2 * D)) / x2
            )
        return term1 + term2

    # ---- raw (un-normalized) mode shapes ---------------------------------
    def phi_water_raw(self, z, k, h):
        """Un-normalized water-borne mode shape, evaluated at array z."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        D = self.H - h
        kappa, g = self.kappa1(k), self.gamma2(k)
        A = np.sin(kappa * h) / np.sinh(g * D)
        out = np.where(z < h, np.sin(kappa * z), A * np.sinh(g * (self.H - z)))
        return out

    def phi_bottom_raw(self, z, k, h):
        """Un-normalized bottom-borne mode shape, evaluated at array z."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        D = self.H - h
        kappa, x2 = self.kappa1(k), self.kappa2(k)
        C = -np.sin(kappa * h) / np.sin(x2 * D)
        out = np.where(z < h, np.sin(kappa * z), C * np.sin(x2 * (z - self.H)))
        return out

    def phi_raw(self, z, k, h, branch: Branch):
        return (
            self.phi_water_raw(z, k, h)
            if branch == "water"
            else self.phi_bottom_raw(z, k, h)
        )

    def dphi_bottom_raw(self, z, k, h):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        D = self.H - h
        kappa1, kappa2 = self.kappa1(k), self.kappa2(k)
        C = np.sin(kappa1 * h) / np.sin(kappa2 * D)
        out = np.where(
            z < h,
            kappa1 * np.cos(kappa1 * z),
            C * kappa2 * np.cos(kappa2 * (z - self.H)),
        )
        return out

    def dphi_water_raw(self, z, k, h):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        D = self.H - h
        kappa1, gamma2 = self.kappa1(k), self.gamma2(k)
        A = np.sin(kappa1 * h) / np.sinh(gamma2 * D)
        out = np.where(
            z < h,
            kappa1 * np.cos(kappa1 * z),
            A * gamma2 * np.cosh(gamma2 * (z - self.H)),
        )
        return out

    def dphi_raw(self, z, k, h, branch: Branch):
        return (
            self.dphi_water_raw(z, k, h)
            if branch == "water"
            else self.dphi_bottom_raw(z, k, h)
        )

    def rho_of(self, z, h):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        return np.where(z < h, self.rho1, self.rho2)


# =====================================================================
# 2. Mode objects: a wavenumber together with everything the coupling-
#    coefficient formulas need (normalization, dA/dh or dC/dh, dk/dh...)
# =====================================================================


@dataclass
class Mode:
    """A single normalized eigenmode of the Waveguide at a given h."""

    wg: Waveguide
    branch: Branch
    h: float
    k: float
    index: int = -1  # book-keeping only (mode order at this h)

    # filled in by `compute()`
    N: float = field(init=False, default=0.0)
    kp: float = field(init=False, default=0.0)  # dk/dh
    kappa1: float = field(init=False, default=0.0)
    kappa1p: float = field(init=False, default=0.0)
    D: float = field(init=False, default=0.0)
    # water-borne extras
    g2: float = field(init=False, default=0.0)
    g2p: float = field(init=False, default=0.0)
    A: float = field(init=False, default=0.0)
    dA: float = field(init=False, default=0.0)
    # bottom-borne extras
    x2: float = field(init=False, default=0.0)
    x2p: float = field(init=False, default=0.0)
    C: float = field(init=False, default=0.0)
    dC: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.compute()

    def compute(self):
        wg, k, h = self.wg, self.k, self.h
        self.D = wg.H - h
        self.kappa1 = wg.kappa1(k)
        self.kp = wg.dk_dh(k, h, self.branch)
        self.kappa1p = -(k / self.kappa1) * self.kp
        self.N = np.sqrt(wg.norm2(k, h, self.branch))

        if self.branch == "water":
            g, D = wg.gamma2(k), self.D
            self.g2 = g
            self.g2p = (k / g) * self.kp
            denom = np.sinh(g * D)
            if abs(denom) < _ILL_CONDITIONED_TOL:
                warnings.warn(
                    f"water-borne mode k={k:.6g} at h={h:.6g}: sinh(gamma2*D)="
                    f"{denom:.3e} is nearly singular; A, N and dA/dh may lose "
                    "precision (the normalized mode itself remains finite). "
                    "Perturb h slightly if this matters.",
                    RuntimeWarning,
                )
            self.A = np.sin(self.kappa1 * h) / denom
            self.dA = (
                np.cos(self.kappa1 * h)
                * (self.kappa1 + h * self.kappa1p)
                * np.sinh(g * D)
                - np.sin(self.kappa1 * h) * np.cosh(g * D) * (self.g2p * D - g)
            ) / denom**2
        else:
            x2, D = wg.kappa2(k), self.D
            self.x2 = x2
            self.x2p = -(k / x2) * self.kp
            denom = np.sin(x2 * D)
            if abs(denom) < _ILL_CONDITIONED_TOL:
                warnings.warn(
                    f"bottom-borne mode k={k:.6g} at h={h:.6g}: sin(kappa2*D)="
                    f"{denom:.3e} is nearly singular; C, N and dC/dh may lose "
                    "precision (the normalized mode itself remains finite). "
                    "Perturb h slightly if this matters.",
                    RuntimeWarning,
                )
            self.C = -np.sin(self.kappa1 * h) / denom
            self.dC = (
                -(
                    np.cos(self.kappa1 * h)
                    * (self.kappa1 + h * self.kappa1p)
                    * np.sin(x2 * D)
                    - np.sin(self.kappa1 * h) * np.cos(x2 * D) * (self.x2p * D - x2)
                )
                / denom**2
            )

    # ---- convenience: evaluate the *normalized* mode shape --------------
    def phi(self, z):
        return self.wg.phi_raw(z, self.k, self.h, self.branch) / self.N

    def dphi(self, z):
        return self.wg.dphi_raw(z, self.k, self.h, self.branch) / self.N
