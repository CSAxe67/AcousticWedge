"""
Plot_wedge.py
=============
Visualisation des resultats de propagation calcules par `Propagation.py` :
    - courbe TL(x) a la profondeur recepteur, comparee a une reference
    - erreur absolue point a point
    - champ TL 2D (x, z)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from config import SimulationConfig
from Propagation import compute_propagation

cfg = SimulationConfig()
Hmax, N_totX, N_tot = cfg.Hmax, cfg.N_totX, cfg.N_tot
freq, nmod, beta = cfg.freq, cfg.nmod, cfg.Deriv_step
z_transition, beta_sediment_1 = cfg.z_transition, cfg.beta_sediment_1

REF_FILE = "TL_SourceImg_20_deg.txt"


def load_reference(path):
    ref_data = np.loadtxt(path)
    x_ref = ref_data[:, 0] * 1000
    TL_ref = ref_data[:, 1]
    return x_ref, TL_ref


def plot_tl_1d(results, x_ref, TL_ref):
    """Courbe TL(x) + erreur absolue vs reference."""
    X_fin, TL = results["X_fin"], results["TL"]

    TL_interp = interp1d(X_fin, TL, kind="linear", fill_value="extrapolate")(x_ref)
    err_abs = np.abs(TL_interp - TL_ref)
    sum_err = np.sum(err_abs)
    mae = np.mean(err_abs)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(X_fin, TL, label=f"Python RK4 (SAE={sum_err:.1f} dB)", linewidth=1.5, color="steelblue")
    ax.plot(x_ref, TL_ref, label="wedgea (reference)", linewidth=1.5, linestyle="--", color="tomato")
    ax.set_xlim([0, 4000])
    ax.set_ylim([-90, 0])
    ax.set_ylabel("TL (dB re 1 m)")
    ax.set_title(f"TL — f={freq} Hz")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    param_text = (
        f"nmod  = {nmod}\n"
        f"beta  = {beta}\n"
        f"N_tot = {N_tot}\n"
        f"N_totX = {N_totX}\n"
        f"Hmax  = {Hmax}"
    )
    ax.text(0.98, 0.97, param_text, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    ax2 = axes[1]
    ax2.plot(x_ref, err_abs, label=f"Python vs wedgea (MAE={mae:.4f} dB)",
             linewidth=1.2, color="steelblue")
    ax2.set_xlim([0, 4000])
    ax2.set_xlabel("Range (m)")
    ax2.set_ylabel("|erreur| (dB)")
    ax2.set_title("Erreur absolue point a point vs wedgea")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fname = (
        f"TL_f{freq}Hz_nmod{nmod}_beta{beta}_Ntot{N_tot}_NtotX{N_totX}"
        f"_Hmax{Hmax}_betaval{beta_sediment_1}_ztrans{z_transition}_20deg.png"
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure sauvegardee : {fname}")


def plot_tl_2d(results):
    """Champ TL 2D (x, z)."""
    X_fin, TL_2d, Z_fine = results["X_fin"], results["TL_2d"], results["Z_fine"]
    X_grid = np.repeat(X_fin[:, np.newaxis], Z_fine.shape[1], axis=1)

    plt.figure(figsize=(14, 6))
    mesh = plt.pcolormesh(X_grid, Z_fine, TL_2d, cmap="jet", vmin=-60, vmax=0, shading="auto")

    plt.title(f"Champ de Transmission Loss (TL) 2D — f = {freq} Hz", fontsize=14, fontweight="bold")
    plt.xlabel("Distance / Range (m)", fontsize=12)
    plt.ylabel("Profondeur (m)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.xlim([0, 4000])
    plt.ylim([Hmax, 0])

    cbar = plt.colorbar(mesh, pad=0.01, aspect=25)
    cbar.set_label("TL (dB re 1 m)", fontsize=12)

    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()

    fname = f"TL_2D_champ_f{freq}Hz_nmod{nmod}_ztrans{z_transition}_betaval{beta_sediment_1}_20deg.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Champ TL 2D sauvegarde sous : {fname}")


if __name__ == "__main__":
    results = compute_propagation()
    x_ref, TL_ref = load_reference(REF_FILE)

    plot_tl_1d(results, x_ref, TL_ref)
    plot_tl_2d(results)