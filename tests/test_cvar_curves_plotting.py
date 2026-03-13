from __future__ import annotations

from pathlib import Path

import pandas as pd

from iro.visualization.cvar_curves import plot_alpha_risk_distribution, plot_cvar_alpha_curves


def test_plot_cvar_alpha_curves_writes_outputs(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {"algorithm": "erm", "alpha_op": 0.0, "cvar_mean": 1.0, "cvar_std": 0.1},
            {"algorithm": "erm", "alpha_op": 1.0, "cvar_mean": 2.0, "cvar_std": 0.2},
            {"algorithm": "groupdro", "alpha_op": 0.0, "cvar_mean": 1.1, "cvar_std": 0.1},
            {"algorithm": "groupdro", "alpha_op": 1.0, "cvar_mean": 2.1, "cvar_std": 0.2},
            {"algorithm": "iro", "alpha_op": 0.0, "cvar_mean": 0.9, "cvar_std": 0.1},
            {"algorithm": "iro", "alpha_op": 1.0, "cvar_mean": 1.9, "cvar_std": 0.2},
        ]
    )
    out_png = tmp_path / "curve.png"
    out_pdf = tmp_path / "curve.pdf"
    fig = plot_cvar_alpha_curves(summary, output_png=out_png, output_pdf=out_pdf, smooth=False)
    assert out_png.exists()
    assert out_pdf.exists()
    assert fig is not None


def test_plot_alpha_risk_distribution_writes_png(tmp_path: Path) -> None:
    alpha_to_risks = {
        0.0: [0.9, 1.0, 1.1, 1.2],
        0.5: [1.0, 1.1, 1.3, 1.4],
        1.0: [1.2, 1.4, 1.5, 1.8],
    }
    out_png = tmp_path / "risk_dist.png"
    fig = plot_alpha_risk_distribution(alpha_to_risks, output_png=out_png)
    assert out_png.exists()
    assert fig is not None

