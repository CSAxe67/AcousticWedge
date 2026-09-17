# AcousticWedge

Coupled multi-mode acoustic propagation solver for a range-dependent
wedge-shaped waveguide (shallow water over a sloping sediment bottom).

The model computes normal modes at each range, couples them through
bathymetric slope and attenuation terms, integrates the coupled modal
amplitudes with an RK4 scheme, and reconstructs the Transmission Loss
(TL) field.

## Physical model

- **Environment**: water column over a fluid sediment half-space,
  separated by a sloping interface (the "wedge"). Sound speed and
  density are constant within each layer (`c_w`, `c_s`, `p_w`, `p_s`).
- **Modes**: computed at each range by finite differences on the
  vertical wave equation, with Richardson extrapolation (2nd order in
  `dz`) over three mesh resolutions for improved accuracy.
- **Mode coupling**:
  - *Attenuation-induced* (`T_jl`): volume attenuation in water and
    sediment, optionally with a depth-dependent gradient in the
    sediment beyond `z_transition`.
  - *Bathymetric* (bottom-slope-induced): two independent formulations
    are implemented —
    - `C1ij`: analytical derivative of the modes at the interface
      (used by default, see `Propagation.py`), derived in [1], Section 4.2.
    - `V_jl`: finite-difference derivative of the modes with respect
      to depth, obtained by recomputing the modes at a perturbed
      bathymetry (`Propagation_Vcoupling.py`, currently on hold).
- **Propagation**: the coupled modal amplitudes are integrated along
  range with an RK4 scheme, then combined with the local modes to
  reconstruct the pressure field and the TL (dB re 1 m). The coupled
  mode propagation equation follows the formulation of [2].

## Project structure

| File | Role |
|---|---|
| `config.py` | `SimulationConfig` dataclass: all physical and numerical parameters (geometry, sound speeds, densities, attenuation, source/receiver positions). Single source of truth for the whole pipeline. |
| `geometry.py` | Builds the spatial mesh (`Mesh` dataclass) and the depth profile at a given vertical resolution; estimates the number of modes needed (`compute_nombres_modes`). |
| `integration.py` | Piecewise trapezoidal integration coefficients (density-weighted), used for mode normalization and coupling integrals. |
| `Modescompute.py` | Eigenmode / wavenumber computation (finite differences + Richardson extrapolation), imaginary part (attenuation) of the wavenumbers, vertical derivative of the modes. |
| `Modescompute_Nonsym.py` | Non-symmetrized eigenvalue solver (`scipy.linalg.eig` on the full matrix), kept only for cross-validation against the symmetrized solver used in production. |
| `attenuation.py` | Attenuation-induced coupling matrix `T_jl`. |
| `coupling.py` | Bathymetric coupling matrices: `V_jl` (finite difference on beta) and `C1ij` (analytical, at the interface). |
| `Propagation.py` | Main solver: assembles modes + coupling terms, runs the RK4 range integration, returns the TL field (1D and 2D). Uses the `C1ij` coupling formulation. |
| `Propagation_Vcoupling.py` | Alternative solver using the `V_jl` coupling formulation. **Not yet migrated to the current `config`/`geometry` architecture — do not use as-is.** |
| `plot_wedge.py` | Plots TL(x) against a reference curve (point-by-point error) and the 2D TL field. |

## Installation

```bash
pip install numpy scipy matplotlib
```

Tested with Python >= 3.10 (uses `list[float]` style type hints).

## Usage

```python
from config import SimulationConfig
from Propagation import compute_propagation

cfg = SimulationConfig()          # default parameters (see config.py)
results = compute_propagation(cfg)

# results["TL"]     : 1D Transmission Loss at receiver depth zr (dB)
# results["TL_2d"]  : 2D TL field (range x depth)
# results["X_fin"]  : fine range grid
# results["Z_fine"] : fine depth grid (2D field)
```

To reproduce validation figures against a reference run:

```bash
python plot_wedge.py
```

This expects a reference TL(x) file (two columns: range in km, TL in
dB) in the working directory, set via `REF_FILE` at the top of
`plot_wedge.py`. Which reference file is relevant depends on your
`config.SimulationConfig` (wedge angle, frequency, source/receiver
depths, etc.) — e.g. `TL_SourceImg_20_deg.txt` for a ~20 deg wedge,
or `wedgea.TL` for the ~2.9 deg case shown below. Adjust `REF_FILE`
(and, if needed, the sign of the loaded `TL_ref` column) to match
the reference you're comparing against.

## Configuration

All parameters live in `config.SimulationConfig`. Key groups:

- **Geometry**: `X_0, X_fin, D_0, D_fin, Hmax, N_tot, N_totX` -- wedge
  extent, water depth at the source/far end, and mesh resolution.
- **Medium**: `c_w, c_s, p_w, p_s` -- sound speed and density in water
  and sediment.
- **Attenuation**: `beta_water, beta_sediment_1, beta_sediment_2,
  z_transition` -- attenuation coefficients (dB/wavelength) and the
  depth at which the sediment attenuation transitions from
  `beta_sediment_1` to `beta_sediment_2`.
- **Numerics**: `nmod` (number of modes retained), `R_factor` (3 mesh
  refinement factors used for Richardson extrapolation), `Deriv_step`
  (finite-difference step for the bathymetric derivative).
- **Source/receiver**: `zr, zs` -- receiver and source depths.

`omega` and `eta` are derived automatically from `freq` and are
exposed as read-only properties.

## Validated cases (working example)

The figures below were obtained with a wedge angle of ~2.9 deg,
validated against the `wedgea.TL` reference. To reproduce them,
apply the following changes on top of the default setup:

**`config.py`** — use this parameter set:

```python
X_0: float = 1.0
X_fin: float = 4000
D_0: float = 200
D_fin: float = 0
N_tot: int = 1500
N_totX: int = 50
Hmax: float = 1500
N_Grilles_fines: int = 4000

c_w: float = 1500.0
c_s: float = 1700.0
p_w: float = 1.0
p_s: float = 1.0

freq: float = 25.0
alpha: float = np.pi / 2.1  # aberture angle
Deriv_step: int = 4
R_factor: list[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])

beta_water: float = 0.0
beta_sediment_1: float = 0.5
beta_sediment_2: float = 0.5
z_transition: float = 4000.0

zr: float = 30.0
zs: float = 100
```

**`plot_wedge.py`** — point to the `wedgea.TL` reference file and
flip the sign of the reference TL column:

```python
REF_FILE = "wedgea.TL"
...
TL_ref = -ref_data[:, 1]
```

**`Propagation.py`** — use the 2D-field normalization instead of the
default 3D one:

```python
P = np.sum(Ajx * phizrx, axis=1)   # 2D field
# instead of:
# P = np.sum(Ajx * phizrx, axis=1) / np.sqrt(X_fine)   # 3D field
```

### Results

**1D TL vs reference (`wedgea`) + point-by-point error:**

![TL 1D vs wedgea](figures/TL_f25_0Hz_nmod44_deriv4_Ntot1500_NtotX50_Hmax1500_betaval0_5_ztrans4000_0_3deg.png)

**2D TL field:**

![TL 2D field](figures/TL_2D_field_f25_0Hz_nmod44_ztrans4000_0_betaval0_5_3deg.png)

With this configuration, the mean absolute error against the
`wedgea` reference is ~0.41 dB over most of the range, with larger
deviations (~1.5-1.8 dB) beyond 3300 m.

## References

[1] M. Trofimov, S. Kozitskiy, A. Zakharenko, and P. Petrov,
"Formal Derivations of Mode Coupling Equations in Underwater Acoustics:
How the Method of Multiple Scales Results in an Expansion over Eigenfunctions
and the Vectorized WKBJ Solution for the Amplitudes,"
*Journal of Marine Science and Engineering*, vol. 11, no. 4, p. 797, 2023.
doi: [10.3390/jmse11040797](https://doi.org/10.3390/jmse11040797).
— the `C1ij` bathymetric coupling coefficients used in `coupling.py` /
`Propagation.py` are derived in Section 4.2.

[2] P. S. Petrov, M. S. Kazak, and T. N. Petrova,
"A Generalization of WKBJ Method for Solving a System Describing
Propagation of Coupled Modes in Underwater Acoustics,"
*Physica D: Nonlinear Phenomena*, 2022.
doi: [10.1016/j.physd.2022.133744](https://doi.org/10.1016/j.physd.2022.133744).
— the coupled-mode propagation equation implemented in `Propagation.py`
follows this formulation.


