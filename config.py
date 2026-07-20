import numpy as np
from dataclasses import dataclass, field

@dataclass
class SimulationConfig:
    """Paramètres globaux pour la simulation de propagation acoustique."""
    
    # --- Géométrie du Wedge ---
    X_0: float = 1.0
    X_fin: float = 4000.0
    D_0: float = 1455.88
    D_fin: float = 0.0
    N_tot: int = 2500
    N_totX: int = 3000
    Hmax: float = 5000.0
    N_Grilles_fines: int = 4000

    # --- Milieu (vitesses, densités) ---
    c_w: float = 1500.0
    c_s: float = 1700.0
    p_w: float = 1.0
    p_s: float = 1.0

    # --- Physique et Numérique ---
    freq: float = 25.0
    alpha: float = np.pi / 2.2
    Deriv_step: int = 4
    # Utilisation de field(default_factory=...) pour les listes modifiables
    R_factor: list[float] = field(default_factory=lambda: [1.0, 1.5, 2.0]) 
    
    beta_eau: float = 0.0
    beta_sediment_1: float = 0.5
    beta_sediment_2: float = 0.5
    z_transition: float = 1000.0
    nmod: int = 142

    # --- Position Source / Récepteur ---
    zr: float = 30.0
    zs: float = 1000.0

    # --- Propriétés calculées dynamiquement ---
    @property
    def omega(self) -> float:
        """Pulsation calculée automatiquement à partir de la fréquence."""
        return 2 * np.pi * self.freq

    @property
    def eta(self) -> float:
        """Facteur d'atténuation."""
        return 1.0 / (40.0 * np.pi * np.log10(np.exp(1)))


