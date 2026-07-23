import numpy as np
from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    """Global parameters for the acoustic propagation simulation."""

    # --- Wedge geometry ---
    X_0: float = 1.0
    X_fin: float = 4000.0
    D_0: float = 200
    D_fin: float = 0.0
    N_tot: int = 1500
    N_totX: int = 50
    Hmax: float = 1500
    N_Grilles_fines: int = 4000

    # --- Medium (sound speeds, densities) ---
    c_w: float = 1500.0
    c_s: float = 1700.0
    p_w: float = 1.0
    p_s: float = 1.0

    # --- Physical and numerical parameters ---
    freq: float = 25.0
    alpha: float = np.pi / 2.2
    Deriv_step: int = 5
    # field(default_factory=...) is used for mutable defaults (lists)
    R_factor: list[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])

    beta_water: float = 0.0
    beta_sediment_1: float = 0.5
    beta_sediment_2: float = 0.5
    z_transition: float = 1000.0
    nmod: int = 42

    # --- Source / receiver position ---
    zr: float = 30.0
    zs: float = 100.0

    # --- Dynamically computed properties ---
    @property
    def omega(self) -> float:
        """Angular frequency, computed automatically from the frequency."""
        return 2 * np.pi * self.freq

    @property
    def eta(self) -> float:
        """Attenuation factor."""
        return 1.0 / (40.0 * np.pi * np.log10(np.exp(1)))

    @property
    def angle_deg(self) -> float:
        """
        Wedge angle, in degrees, computed from the bathymetric slope
        between (X_0, D_0) and (X_fin, D_fin).

        Not to be confused with `alpha` (critical angle threshold used
        to estimate the number of modes).
        """
        return np.degrees(np.arctan((self.D_0 - self.D_fin) / (self.X_fin - self.X_0)))
    @property
    def dh_dx(self) -> float:
        """
        Bathymetric slope (dh/dx) of the wedge.
        
        Corresponds to tan(angle) in radians: (D_0 - D_fin) / (X_fin - X_0).
        """
        return (self.D_0 - self.D_fin) / (self.X_fin - self.X_0)