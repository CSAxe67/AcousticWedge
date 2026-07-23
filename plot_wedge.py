"""
plot_wedge.py
=============
Visualization of the propagation results computed by `Propagation.py`:
    - TL(x) curve at the receiver depth, compared against a reference
    - point-by-point absolute error
    - 2D TL field (x, z)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from config import SimulationConfig
from Propagation import compute_propagation
from Propagation_Vcoupling import compute_propagation_Vcoupling

REF_FILE = "wedgea.TL"


def load_reference(path):
    """Loads a reference TL(x) curve from a two-column text file (range in km, TL in dB)."""
    ref_data = np.loadtxt(path)
    x_ref = ref_data[:, 0] * 1000
    TL_ref = -ref_data[:, 1]
    return x_ref, TL_ref


def plot_tl_1d(cfg: SimulationConfig, results, x_ref, TL_ref):
    """TL(x) curve + point-by-point absolute error against the reference."""
    X_fin, TL = results["X_fin"], results["TL"]
    deriv_step = cfg.Deriv_step  # finite-difference step factor (bathymetric derivative)

    TL_interp = interp1d(X_fin, TL, kind="linear", fill_value="extrapolate")(x_ref)
    err_abs = np.abs(TL_interp - TL_ref)
    sum_err = np.sum(err_abs)
    mae = np.mean(err_abs)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(X_fin, TL, label=f"Python RK4 (SAE={sum_err:.1f} dB)", linewidth=1.5, color="steelblue")
    ax.plot(x_ref, TL_ref, label="wedgea (reference)", linewidth=1.5, linestyle="--", color="tomato")
    ax.set_xlim([cfg.X_0, cfg.X_fin])
    ax.set_ylim([-90, 0])
    ax.set_ylabel("TL (dB re 1 m)")
    ax.set_title(f"TL — f={cfg.freq} Hz, wedge angle={cfg.angle_deg:.1f} deg")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    param_text = (
        f"nmod  = {cfg.nmod}\n"
        f"deriv_step = {deriv_step}\n"
        f"N_tot = {cfg.N_tot}\n"
        f"N_totX = {cfg.N_totX}\n"
        f"Hmax  = {cfg.Hmax}"
    )
    ax.text(0.98, 0.97, param_text, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    ax2 = axes[1]
    ax2.plot(x_ref, err_abs, label=f"Python vs wedgea (MAE={mae:.4f} dB)",
             linewidth=1.2, color="steelblue")
    ax2.set_xlim([cfg.X_0, cfg.X_fin])
    ax2.set_xlabel("Range (m)")
    ax2.set_ylabel("|error| (dB)")
    ax2.set_title("Point-by-point absolute error vs wedgea")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fname = (
        f"TL_f{cfg.freq}Hz_nmod{cfg.nmod}_deriv{deriv_step}_Ntot{cfg.N_tot}_NtotX{cfg.N_totX}"
        f"_Hmax{cfg.Hmax}_betaval{cfg.beta_sediment_1}_ztrans{cfg.z_transition}_{cfg.angle_deg:.0f}deg.png"
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved: {fname}")


def plot_tl_2d(cfg: SimulationConfig, results):
    """2D TL field (x, z)."""
    X_fin, TL_2d, Z_fine = results["X_fin"], results["TL_2d"], results["Z_fine"]
    X_grid = np.repeat(X_fin[:, np.newaxis], Z_fine.shape[1], axis=1)

    plt.figure(figsize=(14, 6))
    mesh = plt.pcolormesh(X_grid, Z_fine, TL_2d, cmap="jet", vmin=-60, vmax=0, shading="auto")

    plt.title(f"Transmission Loss (TL) 2D field — f = {cfg.freq} Hz, wedge angle = {cfg.angle_deg:.1f} deg",
              fontsize=14, fontweight="bold")
    plt.xlabel("Distance / Range (m)", fontsize=12)
    plt.ylabel("Depth (m)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.xlim([cfg.X_0, cfg.X_fin])
    plt.ylim([cfg.Hmax, 0])

    cbar = plt.colorbar(mesh, pad=0.01, aspect=25)
    cbar.set_label("TL (dB re 1 m)", fontsize=12)

    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()

    fname = f"TL_2D_field_f{cfg.freq}Hz_nmod{cfg.nmod}_ztrans{cfg.z_transition}_betaval{cfg.beta_sediment_1}_{cfg.angle_deg:.0f}deg.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"2D TL field saved as: {fname}")


if __name__ == "__main__":
    cfg = SimulationConfig()
    results = compute_propagation(cfg)
    # results = compute_propagation_Vcoupling(cfg)
    x_ref, TL_ref = load_reference(REF_FILE)

    plot_tl_1d(cfg, results, x_ref, TL_ref)
    plot_tl_2d(cfg, results)