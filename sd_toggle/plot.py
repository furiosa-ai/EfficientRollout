"""Paper-quality boundary visualization for SD toggle decision.

All plot functions accept L_accept as primary parameter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .config import SDToggleConfig, load_config
from .roofline import predict_speedup, find_boundary_B


def _setup_style():
    """Set matplotlib publication style."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_boundary(
    config: SDToggleConfig,
    gamma: int,
    L_accept: float,
    ax=None,
    S_range: Optional[list[int]] = None,
    B_range: tuple[float, float] = (1.0, 80.0),
    color: str = "blue",
    label: Optional[str] = None,
    **kwargs,
):
    """Plot single boundary curve (B* vs S where speedup=1).

    Args:
        config: Calibrated configuration
        gamma: Number of draft tokens
        L_accept: Expected accepted length
        ax: Matplotlib axes (created if None)
        S_range: Sequence lengths to evaluate
        B_range: Search range for B*
        color: Line color
        label: Legend label
    """
    import matplotlib.pyplot as plt
    _setup_style()

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    if S_range is None:
        S_range = [256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192]

    S_vals = []
    B_vals = []
    for S in S_range:
        B_star = find_boundary_B(S, gamma, L_accept, config, B_range)
        if B_star is not None:
            S_vals.append(S)
            B_vals.append(B_star)

    if S_vals:
        lbl = label or f"B* (γ={gamma}, L={L_accept:.1f})"
        ax.plot(S_vals, B_vals, "o-", color=color, label=lbl, **kwargs)
        # Fill: SD beneficial below the curve
        ax.fill_between(S_vals, B_vals, 0, alpha=0.1, color=color)

    ax.set_xlabel("Sequence Length (S)")
    ax.set_ylabel("Batch Size (B)")
    ax.set_title(f"SD Viability Boundary — {config.model.name}")
    ax.legend()
    return ax


def plot_boundary_with_empirical(
    config: SDToggleConfig,
    gamma: int,
    L_accept: float,
    csv_path: str | Path,
    ax=None,
    pow2_filter: bool = True,
    S_range: Optional[list[int]] = None,
    B_range: tuple[float, float] = (1.0, 80.0),
):
    """Plot boundary curve with empirical data points overlaid.

    Green points = SD beneficial empirically, Red = SD harmful.
    """
    import matplotlib.pyplot as plt
    from .fit import load_csv
    _setup_style()

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Plot boundary curve
    plot_boundary(config, gamma, L_accept, ax=ax, S_range=S_range,
                  B_range=B_range, color="navy")

    # Load empirical data
    rows = load_csv(csv_path)
    lookup = {}
    for r in rows:
        key = (r["batch"], r["seq"], r["gamma"], r["mode"])
        lookup[key] = r["median_ms"]

    pow2_set = {1, 2, 4, 8, 16, 32, 64, 128}

    bs_pairs = sorted(set(
        (r["batch"], r["seq"])
        for r in rows
        if r["gamma"] == gamma and r["mode"] == "drafter_decode"
    ))

    green_B, green_S = [], []
    red_B, red_S = [], []

    for B, S in bs_pairs:
        if pow2_filter and B not in pow2_set:
            continue

        T_T_key = (B, S, 0, "target_decode")
        T_D_key = (B, S, gamma, "drafter_decode")
        T_V_key = (B, S, gamma, "target_verify")

        if T_T_key not in lookup or T_D_key not in lookup or T_V_key not in lookup:
            continue

        T_T = lookup[T_T_key]
        T_D = lookup[T_D_key]
        T_V = lookup[T_V_key]

        if T_T <= 0:
            continue

        r_emp = T_D / T_T
        v_emp = T_V / T_T
        denom = gamma * r_emp + v_emp
        if denom <= 0:
            continue
        empirical_speedup = L_accept / denom

        if empirical_speedup > 1.0:
            green_B.append(B)
            green_S.append(S)
        else:
            red_B.append(B)
            red_S.append(S)

    ax.scatter(green_S, green_B, c="green", marker="o", s=30,
               alpha=0.7, label=f"SD beneficial ({len(green_B)})", zorder=5)
    ax.scatter(red_S, red_B, c="red", marker="x", s=30,
               alpha=0.7, label=f"SD harmful ({len(red_B)})", zorder=5)
    ax.legend()
    ax.set_title(f"{config.model.name} γ={gamma} L={L_accept:.1f}"
                 + (" (pow2)" if pow2_filter else ""))
    return ax


def plot_boundary_multi(
    configs: list[SDToggleConfig],
    gammas: list[int],
    L_accepts: list[float],
    csv_paths: Optional[list[str | Path]] = None,
    pow2_filter: bool = True,
    output_path: Optional[str | Path] = None,
    S_range: Optional[list[int]] = None,
):
    """Multi-panel boundary figure: models × L_accept values.

    Creates a grid: rows = models, columns = L_accept values.
    Each panel shows boundary curve + empirical points for a given gamma.

    Args:
        configs: List of SDToggleConfig (one per model)
        gammas: Gamma values (separate figure per gamma)
        L_accepts: L_accept values for columns
        csv_paths: Optional CSV paths for empirical overlay (one per model)
        pow2_filter: Only show power-of-2 batch sizes
        output_path: Save path (without extension — saves both .png and .pdf)
        S_range: Sequence lengths for boundary computation
    """
    import matplotlib.pyplot as plt
    _setup_style()

    n_models = len(configs)
    n_L = len(L_accepts)

    for gamma in gammas:
        fig, axes = plt.subplots(n_models, n_L, figsize=(4.5 * n_L, 4 * n_models),
                                  squeeze=False)

        for i, config in enumerate(configs):
            for j, L in enumerate(L_accepts):
                ax = axes[i, j]

                if csv_paths and i < len(csv_paths) and csv_paths[i]:
                    plot_boundary_with_empirical(
                        config, gamma, L, csv_paths[i],
                        ax=ax, pow2_filter=pow2_filter, S_range=S_range,
                    )
                else:
                    plot_boundary(config, gamma, L, ax=ax, S_range=S_range)

                if i == 0:
                    ax.set_title(f"L_accept={L:.1f}")
                if j == 0:
                    ax.set_ylabel(f"{config.model.name}\nBatch Size")

        fig.suptitle(f"SD Viability Boundary — γ={gamma}"
                     + (" (pow2 batches)" if pow2_filter else ""),
                     fontsize=14, y=1.02)
        plt.tight_layout()

        if output_path:
            base = Path(output_path)
            base.parent.mkdir(parents=True, exist_ok=True)
            stem = f"{base.stem}_gamma{gamma}"
            fig.savefig(base.parent / f"{stem}.png", dpi=300)
            fig.savefig(base.parent / f"{stem}.pdf")
            print(f"Saved: {base.parent / stem}.png/.pdf")

        plt.close(fig)


def plot_Laccept_sweep(
    config: SDToggleConfig,
    gamma: int,
    L_accepts: list[float],
    ax=None,
    S_range: Optional[list[int]] = None,
    output_path: Optional[str | Path] = None,
):
    """Plot boundary curves for multiple L_accept values on same axes.

    Shows how the viability region shifts with acceptance quality.
    """
    import matplotlib.pyplot as plt
    _setup_style()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.figure

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(L_accepts)))

    for L, color in zip(L_accepts, colors):
        plot_boundary(config, gamma, L, ax=ax, S_range=S_range,
                      color=color, label=f"L={L:.1f}")

    ax.set_title(f"{config.model.name} γ={gamma} — L_accept sweep")
    ax.legend()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        print(f"Saved: {path}")

    return ax
