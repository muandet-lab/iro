"""Reusable CVaR plotting helpers inspired by DGIL non-linear simulation notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:  # Optional dependency; plotting falls back to raw lines if unavailable.
    from scipy.interpolate import make_interp_spline
except Exception:  # pragma: no cover - optional import
    make_interp_spline = None


def _smooth_xy(x: np.ndarray, y: np.ndarray, *, points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    if make_interp_spline is None:
        return x, y
    if x.size < 4:
        return x, y
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_max <= x_min:
        return x, y
    x_new = np.linspace(x_min, x_max, points)
    spline = make_interp_spline(x, y, k=3)
    y_new = spline(x_new)
    return x_new, y_new


def plot_cvar_alpha_curves(
    summary_df: pd.DataFrame,
    *,
    output_png: Path | None = None,
    output_pdf: Path | None = None,
    split: str = "val",
    smooth: bool = True,
    algorithm_order: tuple[str, ...] = ("erm", "groupdro", "iro"),
    title: str | None = None,
) -> plt.Figure:
    """Plot CVaR-vs-alpha curves with ±1 std ribbons.

    Expected columns in ``summary_df``:
    - ``algorithm``
    - ``alpha_op``
    - ``cvar_mean``
    - ``cvar_std``
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "erm": "#1f77b4",
        "groupdro": "#222222",
        "iro": "#d62728",
    }
    linestyles = {
        "erm": ":",
        "groupdro": "--",
        "iro": "-",
    }
    labels = {
        "erm": "ERM",
        "groupdro": "GroupDRO",
        "iro": "IRO",
    }

    for alg in algorithm_order:
        g = summary_df[summary_df["algorithm"] == alg].sort_values("alpha_op")
        if g.empty:
            continue
        x = g["alpha_op"].to_numpy(dtype=float)
        y = g["cvar_mean"].to_numpy(dtype=float)
        y_std = g["cvar_std"].to_numpy(dtype=float)
        color = colors.get(alg, None)
        linestyle = linestyles.get(alg, "-")
        label = labels.get(alg, alg)

        if smooth:
            x_s, y_s = _smooth_xy(x, y)
            x_up, y_up = _smooth_xy(x, y + y_std)
            x_dn, y_dn = _smooth_xy(x, y - y_std)
            ax.plot(x_s, y_s, color=color, linestyle=linestyle, linewidth=3.0, label=label)
            ax.fill_between(x_up, y_dn, y_up, color=color, alpha=0.18)
        else:
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=3.0, label=label)
            ax.fill_between(x, y - y_std, y + y_std, color=color, alpha=0.18)

    ax.set_xlabel(r"$\lambda_{op}$", fontsize=13)
    ax.set_ylabel(r"CVaR$_{\lambda_{op}}$(group risk)", fontsize=13)
    ax.set_title(title or f"iWildCam CVaR Curve ({split} split)", fontsize=15)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=180, bbox_inches="tight")
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")
    return fig


def plot_alpha_risk_distribution(
    alpha_to_risks: dict[float, np.ndarray],
    *,
    output_png: Path | None = None,
    title: str = "Distribution of group risks by preference alpha",
) -> plt.Figure:
    """Plot DGIL-style risk distributions across alpha values (KDE + colorbar)."""

    alphas = sorted(alpha_to_risks.keys())
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(float(min(alphas)), float(max(alphas)))

    for alpha in alphas:
        risks = np.asarray(alpha_to_risks[alpha], dtype=float)
        if risks.size == 0:
            continue
        color = cmap(norm(float(alpha)))
        if risks.size > 2 and float(np.std(risks)) > 1e-9:
            sns.kdeplot(risks, ax=ax, color=color, linewidth=2.0)
        else:
            ax.axvline(float(np.mean(risks)), color=color, linewidth=2.0)

    ax.set_title(title)
    ax.set_xlabel("group risk")
    ax.set_ylabel("density")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\alpha$", labelpad=5, size=12)
    fig.tight_layout()

    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=180, bbox_inches="tight")
    return fig


__all__ = ["plot_cvar_alpha_curves", "plot_alpha_risk_distribution"]

