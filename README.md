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
      (used by default, see `Propagation.py`).
    - `V_jl`: finite-difference derivative of the modes with respect
      to depth, obtained by recomputing the modes at a perturbed
      bathymetry (`Propagation_Vcoupling.py`, currently on hold).
- **Propagation**: the coupled modal amplitudes are integrated along
  range with an RK4 scheme, then combined with the local modes to
  reconstruct the pressure field and the TL (dB re 1 m).

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

To reproduce the validation figures against a reference run:

```bash
python plot_wedge.py
```

This expects a reference file `TL_SourceImg_20_deg.txt` (two columns:
range in km, TL in dB) in the working directory.

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


