"""
Generate an EKF example trajectory with 2-sigma uncertainty bounds.

Run from this directory with:
    PYTHONPATH=../src python3 example_ekf_2sigma_trajectory.py

Output:
    ekf_2sigma_trajectory.png
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from test_ekf import (
    DT,
    _compute_metrics,
    _make_circle,
    _run_ekf,
)


STATE_LABELS = ("px [m]", "py [m]", "theta [rad]", "v [m/s]", "omega [rad/s]")


def _confidence_ellipse_points(mean_xy: np.ndarray, cov_xy: np.ndarray, n: int = 80) -> np.ndarray:
    """Return points for the 2-sigma covariance ellipse in the x-y plane."""
    vals, vecs = np.linalg.eigh(cov_xy)
    vals = np.maximum(vals, 0.0)
    radii = 2.0 * np.sqrt(vals)

    angles = np.linspace(0.0, 2.0 * math.pi, n)
    unit = np.vstack((np.cos(angles), np.sin(angles)))
    ellipse = vecs @ np.diag(radii) @ unit
    return ellipse.T + mean_xy


def main() -> None:
    rng = np.random.default_rng(123)
    init_state, controls = _make_circle(T=320, v0=2.0, delta=0.18)
    result = _run_ekf(controls, init_state, rng, gps_rate=4)
    metrics = _compute_metrics(result)

    t = DT * np.arange(1, len(result.true_states) + 1)
    errors = result.est_means - result.true_states
    errors[:, 2] = np.arctan2(np.sin(errors[:, 2]), np.cos(errors[:, 2]))
    sigma2 = 2.0 * np.sqrt(np.maximum(np.diagonal(result.est_covs, axis1=1, axis2=2), 0.0))

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    ax_xy = fig.add_subplot(gs[:, 0])
    ax_xy.plot(result.true_states[:, 0], result.true_states[:, 1], "k-", lw=2.0, label="truth")
    ax_xy.plot(result.est_means[:, 0], result.est_means[:, 1], color="tab:blue", lw=1.7, label="EKF mean")

    ellipse_idxs = np.linspace(0, len(result.est_means) - 1, 18, dtype=int)
    for idx in ellipse_idxs:
        ellipse = _confidence_ellipse_points(result.est_means[idx, :2], result.est_covs[idx, :2, :2])
        ax_xy.plot(ellipse[:, 0], ellipse[:, 1], color="tab:orange", alpha=0.35, lw=1.0)

    ax_xy.set_title("Constant-turn trajectory")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.legend(loc="best")

    state_axes = [fig.add_subplot(gs[i // 2, 1 + i % 2]) for i in range(5)]
    for i, ax in enumerate(state_axes):
        ax.fill_between(t, -sigma2[:, i], sigma2[:, i], color="tab:orange", alpha=0.22, label="+/- 2 sigma")
        ax.plot(t, errors[:, i], color="tab:blue", lw=1.1, label="estimate error")
        ax.axhline(0.0, color="0.2", lw=0.8, alpha=0.45)
        ax.set_title(STATE_LABELS[i])
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.25)

    state_axes[0].legend(loc="upper right")
    fig.suptitle(
        f"EKF 2-sigma bounds: coverage={metrics.frac_2sig:.3f}, "
        f"pos RMSE={metrics.rmse_pos:.3f} m, mean NEES={metrics.mean_nees:.2f}"
    )

    output = "ekf_2sigma_trajectory.png"
    fig.savefig(output, dpi=180)
    print(f"Saved {output}")
    print(f"2-sigma coverage: {metrics.frac_2sig:.3f}")


if __name__ == "__main__":
    main()
