import numpy as np
from dataclasses import dataclass
from config import SimulationConfig


@dataclass
class Mesh:
    """Description of the domain mesh."""

    D: np.ndarray
    X: np.ndarray
    Z: list[np.ndarray]
    dz: list[float]
    N_w: list[int]
    N_tot: int


def build_medium_mesh(cfg: SimulationConfig, r_factor_val: float, compute_derivative: bool = False):
    """Generates the spatial mesh and the depth profiles."""
    N_tot_1 = int(r_factor_val * cfg.N_tot)
    dz = cfg.Hmax / (N_tot_1 - 1)

    D = np.linspace(cfg.D_0, cfg.D_fin, cfg.N_totX)
    X = np.linspace(cfg.X_0, cfg.X_fin, cfg.N_totX)

    Z = []
    DZ_W = []

    if not compute_derivative:
        N_w = [max(round(D[i] / dz), 2) for i in range(cfg.N_totX)]
        for i in range(cfg.N_totX):
            z = [dz * k for k in range(N_tot_1)]
            Z.append(z)
            DZ_W.append(dz)
        return Mesh(
            D=D,
            X=X,
            Z=Z,
            dz=DZ_W,
            N_w=N_w,
            N_tot=N_tot_1,
        )
    else:
        dh = cfg.Deriv_step * dz
        D_der = []
        for i in range(cfg.N_totX):
            z = [dz * k for k in range(N_tot_1)]
            D_der.append(D[i] - dh)
            Z.append(z)
            DZ_W.append(dz)
        N_w = [max(round(D_der[i] / dz), 2) for i in range(cfg.N_totX)]
        return Mesh(
            D=D_der,
            X=X,
            Z=Z,
            dz=DZ_W,
            N_w=N_w,
            N_tot=N_tot_1,
        )


def compute_nombres_modes(cfg: SimulationConfig) -> int:
    """Estimates the number of modes based on the configuration."""
    return int((cfg.Hmax / np.pi) * np.sqrt((cfg.omega / cfg.c_s) ** 2 - np.cos(cfg.alpha) * (cfg.omega / cfg.c_w) ** 2)) + 3