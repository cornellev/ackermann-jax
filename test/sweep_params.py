"""
sweep_contact_params.py — Batched parameter sweep for Ackermann contact/wheel tuning.

Strategy
--------
Because AckermannCarModel holds Python-level scalars inside its parameter
dataclasses, we cannot vmap *through* the model constructor.  Instead we:

  1. Build a single base model with default params.
  2. Write a `simulate_one` function that accepts (k_n, c_n, I_w_fac, tau_spin)
     as plain JAX scalars, patches the relevant fields on-the-fly inside the
     scan body using those scalars, and returns scalar score metrics.
  3. vmap `simulate_one` over a flat grid of parameter combinations.
  4. Reshape results back to the grid and plot a heatmap per metric.

Patching strategy: rather than reconstructing the full model inside the scan
(which would re-trigger Python-level dataclass construction), we pass the
swept parameters directly into a custom `xdot_patched` closure that overrides
k_n, c_n, I_w, and b_w at the point of use.  Everything else (geometry,
tire model, chassis inertia) stays fixed from the base model.

Metrics scored (all dimensionless or normalised):
  S1  bounce      — max(p_z) - p_z[0]           : car left the ground?     (lower=better)
  S2  settle_vz   — RMS of v_z over last 70% of settle phase                (lower=better)
  S3  pitch_osc   — RMS of w_B[1] (pitch rate) during weave phase           (lower=better)
  S4  roll_osc    — RMS of w_B[0] (roll rate) during weave phase            (lower=better)
  S5  omega_rough — RMS of Δω_W per step during weave (wheel jitter)        (lower=better)
  S6  z_drift     — |mean(p_z) - p_z[0]| during weave (rode off ground?)   (lower=better)
  composite       — weighted sum of normalised metrics                       (lower=better)
"""

from __future__ import annotations

import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

jax.config.update("jax_enable_x64", False)

from ackermann_jax import ( #noqa
    AckermannCarInput,
    AckermannCarModel,
    AckermannCarState,
    default_params,
    default_state,
)

# ── Sweep grid definition ─────────────────────────────────────────────────────
# Edit these to change the search space.

# K_N_VALUES    = [500., 1000., 1500., 2000., 3000., 4000.]   # N/m
K_N_VALUES = np.arange(100, 4500, 50, dtype=np.float32)   # N/m
# C_N_VALUES    = [30.,  60.,   100.,  150.,  200.,  300.]     # N·s/m
C_N_VALUES = np.arange(10, 500, 5, dtype=np.float32)      # N·s/m
# IW_FAC_VALUES = [1.,   2.,    5.,    10.]                    # inertia scale
IW_FAC_VALUES = [10.]
# TAU_SPIN_VALUES = [0.05, 0.1, 0.2, 0.3]                     # wheel settle time [s]
TAU_SPIN_VALUES = [0.05]

# Simulation settings
DT        = 0.01      # fine dt — sweep finds params that also work at coarse dt
T_SETTLE  = 1.5       # [s]
T_WEAVE   = 3.0       # [s]  shorter for sweep speed
V_CMD     = 1.0
Z0        = 0.08      # initial height [m]

# Scoring weights  [bounce, settle_vz, pitch_osc, roll_osc, omega_rough, z_drift]
WEIGHTS = np.array([2.0, 0.5, 3.0, 3.0, 2.5, 2.0])

# ── Base model (geometry/tires/chassis fixed) ─────────────────────────────────
_BASE_PARAMS = default_params()
_BASE_MODEL  = AckermannCarModel(_BASE_PARAMS)

# ── Patched simulation ────────────────────────────────────────────────────────

def _normal_forces_patched(p_i_W, v_i_W, k_n, c_n):
    """Drop-in replacement for model._normal_forces using swept scalars."""
    z0  = _BASE_PARAMS.contact.z0
    z   = p_i_W[:, 2]
    vz  = v_i_W[:, 2]
    d   = z0 - z
    ddot = -vz
    Fz  = k_n * jax.nn.relu(d) + c_n * jax.nn.relu(ddot) * (d > 0)
    return jnp.maximum(0.0, Fz)


def simulate_one(k_n, c_n, I_w_fac, tau_spin):
    """
    Run settle + weave for one parameter combination.
    All arguments are JAX scalars (float32) so this function is vmappable.

    Returns a (6,) float32 array of raw metric values.
    """
    p   = _BASE_PARAMS
    geom = p.geom

    m_wheel = 0.05
    I_w = I_w_fac * 0.5 * m_wheel * geom.wheel_radius ** 2
    b_w = I_w / tau_spin

    N_settle = int(T_SETTLE / DT)
    N_weave  = int(T_WEAVE  / DT)
    N_total  = N_settle + N_weave

    x0 = default_state(z0=Z0)

    # ── Custom xdot that uses swept contact / wheel params ────────────────────
    def xdot_patched(x: AckermannCarState, u: AckermannCarInput) -> AckermannCarState:
        p_W   = x.p_W
        R_WB  = x.R_WB
        v_W   = x.v_W
        w_B   = x.w_B
        omega_w = x.omega_W

        delta_i = geom.ackermann_front_angles(u.delta)
        r_B     = geom.wheel_contact_points_body()

        c = jnp.cos(delta_i)
        s = jnp.sin(delta_i)
        t_B = jnp.stack([c, s, jnp.zeros_like(c)], axis=-1)
        n_B = jnp.stack([-s, c, jnp.zeros_like(c)], axis=-1)

        R   = R_WB.as_matrix()
        t_W = (R @ t_B.T).T
        n_W = (R @ n_B.T).T
        z_W = jnp.array([0., 0., 1.], dtype=jnp.float32)

        p_i_W      = p_W[None, :] + (R @ r_B.T).T
        w_cross_r  = jnp.cross(w_B[None, :], r_B)
        v_i_W      = v_W[None, :] + (R @ w_cross_r.T).T

        # ← patched contact
        Fz = _normal_forces_patched(p_i_W, v_i_W, k_n, c_n)

        v_t = jnp.sum(t_W * v_i_W, axis=-1)
        v_n = jnp.sum(n_W * v_i_W, axis=-1)
        kappa, alpha = _BASE_MODEL._slip(omega_w, v_t, v_n)
        Fx, Fy = _BASE_MODEL._tire_forces(kappa, alpha, Fz)

        f_i_W = Fx[:, None] * t_W + Fy[:, None] * n_W + Fz[:, None] * z_W[None, :]
        F_W   = jnp.sum(f_i_W, axis=0)

        r_i_W = p_i_W - p_W[None, :]
        tau_W = jnp.sum(jnp.cross(r_i_W, f_i_W), axis=0)
        tau_B = R.T @ tau_W

        tau_cmd  = p.motor.mask() * u.tau_w
        # ← patched wheel inertia / damping
        domega_w = (tau_cmd - geom.wheel_radius * Fx - b_w * omega_w) / I_w

        g_W  = jnp.array([0., 0., -p.chassis.g], dtype=jnp.float32)
        dv_W = F_W / p.chassis.mass + g_W

        I_b  = p.chassis.I_body
        dW_B = jnp.linalg.solve(I_b, tau_B - jnp.cross(w_B, I_b @ w_B))
        dp_W = v_W

        import jaxlie as _jaxlie
        return AckermannCarState(
            p_W=dp_W,
            R_WB=_jaxlie.SO3.identity(),
            v_W=dv_W,
            w_B=dW_B,
            omega_W=domega_w,
        )

    def step_patched(x, u):
        xdot = xdot_patched(x, u)
        # semi-implicit Euler
        v_W_next     = x.v_W     + DT * xdot.v_W
        w_B_next     = x.w_B     + DT * xdot.w_B
        omega_w_next = x.omega_W + DT * xdot.omega_W
        p_W_next     = x.p_W     + DT * v_W_next

        import jaxlie as _jaxlie
        R_next = x.R_WB @ _jaxlie.SO3.exp(w_B_next * DT)
        return AckermannCarState(
            p_W=p_W_next, R_WB=R_next,
            v_W=v_W_next, w_B=w_B_next, omega_W=omega_w_next,
        )

    # ── Scan body ─────────────────────────────────────────────────────────────
    def scan_body(carry, k):
        x, integ_v, integ_s = carry
        k = k.astype(jnp.int32)

        t_weave = jnp.maximum(0.0, k * DT - T_SETTLE)
        v_ref   = jnp.where(k < N_settle, 0.0, V_CMD)
        psi_cmd = jnp.where(
            k < N_settle, 0.0,
            0.30 * jnp.sin(2.0 * jnp.pi * 0.4 * t_weave)
        )

        tau_w, integ_v = _BASE_MODEL.map_velocity_to_wheel_torques(
            x=x, v_cmd=v_ref, integral_state=integ_v,
            dt=DT, Kp=40., Ki=2., tau_max=0.35, integ_max=0.5,
            use_traction_limit=True,
        )
        delta, integ_s = _BASE_MODEL.map_heading_to_steering(
            x=x, psi_cmd=psi_cmd, integral_state=integ_s,
            dt=DT, Kp=3., Ki=2., Kd=0.4, delta_max=0.35, integ_max=0.5,
        )

        u      = AckermannCarInput(delta=delta, tau_w=tau_w)
        x_next = step_patched(x, u)

        log = jnp.array([
            x_next.p_W[2],          # 0  p_z
            x_next.v_W[2],          # 1  v_z
            x_next.w_B[0],          # 2  roll rate
            x_next.w_B[1],          # 3  pitch rate
            x_next.omega_W[0],      # 4  wheel 0 speed
            x_next.omega_W[1],      # 5  wheel 1 speed
            x_next.omega_W[2],      # 6  wheel 2 speed
            x_next.omega_W[3],      # 7  wheel 3 speed
        ], dtype=jnp.float32)

        return (x_next, integ_v, integ_s), log

    ks = jnp.arange(N_total, dtype=jnp.float32)
    (_, _, _), logs = jax.lax.scan(
        scan_body,
        (x0, jnp.float32(0.0), jnp.float32(0.0)),
        ks,
    )
    # logs: (N_total, 8)

    p_z     = logs[:, 0]
    v_z     = logs[:, 1]
    roll_r  = logs[:, 2]
    pitch_r = logs[:, 3]
    omega   = logs[:, 4:8]   # (N_total, 4)

    settle_slice = slice(0, N_settle)
    weave_slice  = slice(N_settle, N_total)

    # S1: bounce — how far above start did p_z get?
    s1_bounce = jnp.maximum(0.0, jnp.max(p_z) - Z0)

    # S2: settle quality — RMS of v_z in second half of settle phase
    settle_2nd = slice(N_settle // 2, N_settle)
    s2_settle  = jnp.sqrt(jnp.mean(v_z[settle_2nd] ** 2))

    # S3: pitch oscillation during weave
    s3_pitch = jnp.sqrt(jnp.mean(pitch_r[weave_slice] ** 2))

    # S4: roll oscillation during weave
    s4_roll  = jnp.sqrt(jnp.mean(roll_r[weave_slice] ** 2))

    # S5: wheel speed roughness (mean finite-difference of omega)
    domega   = jnp.diff(omega[weave_slice], axis=0) / DT
    s5_rough = jnp.sqrt(jnp.mean(domega ** 2))

    # S6: z drift during weave (car leaving ground)
    s6_zdrift = jnp.abs(jnp.mean(p_z[weave_slice]) - Z0)

    return jnp.array([s1_bounce, s2_settle, s3_pitch, s4_roll, s5_rough, s6_zdrift],
                     dtype=jnp.float32)


# ── Build flat parameter grid ─────────────────────────────────────────────────

def build_grid():
    combos = list(itertools.product(K_N_VALUES, C_N_VALUES, IW_FAC_VALUES, TAU_SPIN_VALUES))
    arr    = np.array(combos, dtype=np.float32)   # (N_combos, 4)
    return arr, combos

# ── Run sweep ─────────────────────────────────────────────────────────────────

def run_sweep():
    grid, combos = build_grid()
    N = len(combos)
    print(f"Sweeping {N} parameter combinations "
          f"({len(K_N_VALUES)} k_n × {len(C_N_VALUES)} c_n × "
          f"{len(IW_FAC_VALUES)} I_w_fac × {len(TAU_SPIN_VALUES)} tau_spin)")

    k_n_arr    = jnp.array(grid[:, 0])
    c_n_arr    = jnp.array(grid[:, 1])
    iw_arr     = jnp.array(grid[:, 2])
    tau_arr    = jnp.array(grid[:, 3])

    sim_batched = jax.jit(jax.vmap(simulate_one))

    print("JIT compiling…", flush=True)
    t0      = time.perf_counter()
    metrics = sim_batched(k_n_arr, c_n_arr, iw_arr, tau_arr)
    metrics.block_until_ready()
    t_compile = time.perf_counter() - t0
    print(f"  compile + run: {t_compile:.1f} s")

    print("Second run (execution only)…", flush=True)
    t0      = time.perf_counter()
    metrics = sim_batched(k_n_arr, c_n_arr, iw_arr, tau_arr)
    metrics.block_until_ready()
    t_run   = time.perf_counter() - t0
    print(f"  run only: {t_run*1e3:.1f} ms  ({t_run/N*1e3:.3f} ms/combo)")

    metrics_np = np.array(metrics)   # (N, 6)
    return metrics_np, grid, combos

# ── Scoring and ranking ───────────────────────────────────────────────────────

METRIC_NAMES = ["bounce [m]", "settle_vz [m/s]", "pitch_rms [rad/s]",
                "roll_rms [rad/s]", "omega_rough [rad/s²]", "z_drift [m]"]

def compute_composite(metrics_np):
    # Normalise each metric to [0,1] across the sweep
    mn  = metrics_np.min(axis=0, keepdims=True)
    mx  = metrics_np.max(axis=0, keepdims=True)
    rng = np.maximum(mx - mn, 1e-8)
    norm = (metrics_np - mn) / rng                    # (N, 6)
    composite = norm @ WEIGHTS / WEIGHTS.sum()        # (N,)
    return composite, norm

def print_top_k(metrics_np, grid, combos, k=10):
    composite, _ = compute_composite(metrics_np)
    order = np.argsort(composite)

    print(f"\n{'─'*90}")
    print(f"{'Rank':>4}  {'k_n':>6}  {'c_n':>5}  {'I_fac':>5}  {'τ_spin':>6}  "
          + "  ".join(f"{n[:10]:>10}" for n in METRIC_NAMES) + "  composite")
    print(f"{'─'*90}")
    for rank, idx in enumerate(order[:k]):
        kn, cn, iw, ts = grid[idx]
        vals = "  ".join(f"{metrics_np[idx, j]:10.4f}" for j in range(6))
        print(f"{rank+1:>4}  {kn:>6.0f}  {cn:>5.0f}  {iw:>5.0f}  {ts:>6.3f}  "
              f"{vals}  {composite[idx]:.4f}")

    print(f"\n{'─'*90}")
    print("Top-1 parameter set:")
    best = order[0]
    kn, cn, iw, ts = grid[best]
    print(f"  k_n     = {kn:.0f}  N/m")
    print(f"  c_n     = {cn:.0f}  N·s/m")
    print(f"  I_w_fac = {iw:.0f}  (× physical inertia)")
    print(f"  tau_spin= {ts:.3f} s")
    m_wheel = 0.05
    r_w     = 0.03
    I_w = iw * 0.5 * m_wheel * r_w**2
    b_w = I_w / ts
    omn = np.sqrt(4 * kn / 1.5)
    zeta = cn / (2 * np.sqrt(kn * 1.5 / 4))
    print(f"\n  Derived:")
    print(f"  I_w  = {I_w:.5f} kg·m²")
    print(f"  b_w  = {b_w:.5f} N·m·s")
    print(f"  ωn   = {omn:.2f} rad/s  (fn = {omn/(2*np.pi):.2f} Hz)")
    print(f"  ζ    = {zeta:.3f}")
    return order[0], grid[order[0]]

# ── Heatmap plots ─────────────────────────────────────────────────────────────

def plot_heatmaps(metrics_np, grid, combos):
    """
    For each (I_w_fac, tau_spin) pair, plot a k_n × c_n heatmap of each metric
    and the composite score.
    """
    composite, _ = compute_composite(metrics_np)
    all_metrics  = np.concatenate([metrics_np, composite[:, None]], axis=1)
    all_names    = METRIC_NAMES + ["composite (lower=better)"]

    kn_vals  = np.array(K_N_VALUES)
    cn_vals  = np.array(C_N_VALUES)
    iw_vals  = np.array(IW_FAC_VALUES)
    tau_vals = np.array(TAU_SPIN_VALUES)

    Nkn, Ncn, Niw, Ntau = len(kn_vals), len(cn_vals), len(iw_vals), len(tau_vals)
    n_metrics = all_metrics.shape[1]

    for iw_idx, iw in enumerate(iw_vals):
        for tau_idx, tau in enumerate(tau_vals):
            fig, axes = plt.subplots(2, 4, figsize=(18, 8))
            axes = axes.flatten()
            fig.suptitle(f"I_w_fac={iw:.0f}  tau_spin={tau:.3f} s", fontsize=13)

            # Select rows matching this (iw, tau)
            mask = (grid[:, 2] == iw) & (grid[:, 3] == tau)
            sub_grid    = grid[mask]                    # (Nkn*Ncn, 4)
            sub_metrics = all_metrics[mask]             # (Nkn*Ncn, n_metrics+1)

            for m_idx in range(n_metrics):
                ax = axes[m_idx]
                mat = np.full((Nkn, Ncn), np.nan)

                for row_idx in range(sub_grid.shape[0]):
                    kn  = sub_grid[row_idx, 0]
                    cn  = sub_grid[row_idx, 1]
                    ki  = np.where(kn_vals == kn)[0][0]
                    ci  = np.where(cn_vals == cn)[0][0]
                    mat[ki, ci] = sub_metrics[row_idx, m_idx]

                # Diverging colormap for composite, sequential for raw metrics
                cmap  = "RdYlGn_r"
                im    = ax.imshow(mat, aspect="auto", cmap=cmap,
                                  origin="lower",
                                  norm=mcolors.PowerNorm(gamma=0.5,
                                                         vmin=np.nanmin(mat),
                                                         vmax=np.nanmax(mat)))
                plt.colorbar(im, ax=ax, shrink=0.8)

                ax.set_xticks(range(Ncn))
                ax.set_xticklabels([f"{v:.0f}" for v in cn_vals], fontsize=7)
                ax.set_yticks(range(Nkn))
                ax.set_yticklabels([f"{v:.0f}" for v in kn_vals], fontsize=7)
                ax.set_xlabel("c_n [N·s/m]", fontsize=8)
                ax.set_ylabel("k_n [N/m]",   fontsize=8)
                ax.set_title(all_names[m_idx], fontsize=9)

                # Star the best cell
                best_flat = np.nanargmin(mat)
                bi, bj    = np.unravel_index(best_flat, mat.shape)
                ax.plot(bj, bi, "w*", markersize=12)

            for ax in axes[n_metrics:]:
                ax.set_visible(False)

            fig.tight_layout()

    plt.show()

# ── Trajectory plot for best params ──────────────────────────────────────────

def plot_best_trajectory(best_idx, best_params):
    kn, cn, iw, ts = best_params
    print(f"\nPlotting trajectory for best params: "
          f"k_n={kn:.0f}  c_n={cn:.0f}  I_fac={iw:.0f}  tau={ts:.3f}")

    N_settle = int(T_SETTLE / DT)
    N_weave  = int(T_WEAVE  / DT)
    N_total  = N_settle + N_weave
    t_arr    = np.arange(N_total) * DT

    # Re-run scalar simulation for full logging
    logs_list = {"p_z": [], "v_z": [], "roll": [], "pitch": [], "yaw": [],
                 "omega": [], "Fz_mean": []}

    x0 = default_state(z0=Z0)

    from ackermann_jax.parameters import ContactParams, WheelParams
    import copy
    p_new  = default_params()
    m_w    = 0.05
    I_w    = float(iw) * 0.5 * m_w * 0.03**2
    b_w_   = I_w / float(ts)

    # Rebuild model with best params
    from ackermann_jax.parameters import ContactParams, WheelParams
    import dataclasses
    new_contact = dataclasses.replace(p_new.contact, k_n=float(kn), c_n=float(cn))
    new_wheels  = dataclasses.replace(p_new.wheels,  I_w=I_w, b_w=b_w_)
    new_p       = dataclasses.replace(p_new, contact=new_contact, wheels=new_wheels)
    best_model  = AckermannCarModel(new_p)

    x = x0
    iv = is_ = 0.0
    for k in range(N_total):
        t_w   = max(0.0, k * DT - T_SETTLE)
        v_ref = 0.0 if k < N_settle else V_CMD
        psi   = 0.0 if k < N_settle else 0.30 * float(jnp.sin(2*jnp.pi*0.4*t_w))

        tau_w, iv = best_model.map_velocity_to_wheel_torques(
            x=x, v_cmd=v_ref, integral_state=jnp.float32(iv),
            dt=DT, Kp=40., Ki=2., tau_max=0.35, integ_max=0.5,
            use_traction_limit=True)
        delta, is_ = best_model.map_heading_to_steering(
            x=x, psi_cmd=psi, integral_state=jnp.float32(is_),
            dt=DT, Kp=3., Ki=2., Kd=0.4, delta_max=0.35, integ_max=0.5)

        u  = AckermannCarInput(delta=delta, tau_w=tau_w)
        x  = best_model.step(x=x, u=u, dt=DT)
        iv, is_ = float(iv), float(is_)

        R   = np.array(x.R_WB.as_matrix())
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        rll = float(np.arctan2(R[2, 1], R[2, 2]))
        pch = float(np.arcsin(-R[2, 0]))

        logs_list["p_z"].append(float(x.p_W[2]))
        logs_list["v_z"].append(float(x.v_W[2]))
        logs_list["roll"].append(np.rad2deg(rll))
        logs_list["pitch"].append(np.rad2deg(pch))
        logs_list["yaw"].append(np.rad2deg(yaw))
        logs_list["omega"].append(np.array(x.omega_W))

    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(14, 10))
    fig.suptitle(f"Best params: k_n={kn:.0f}  c_n={cn:.0f}  "
                 f"I_fac={iw:.0f}  tau_spin={ts:.3f} s", fontsize=12)

    axes[0,0].plot(t_arr, logs_list["p_z"])
    axes[0,0].axhline(Z0, color="k", linestyle="--", linewidth=0.7, label=f"z0={Z0}")
    axes[0,0].set_ylabel("p_z [m]"); axes[0,0].legend(fontsize=8)
    axes[0,0].axvline(T_SETTLE, color="r", linewidth=0.5)

    axes[0,1].plot(t_arr, logs_list["v_z"])
    axes[0,1].axhline(0, color="k", linewidth=0.4)
    axes[0,1].set_ylabel("v_z [m/s]")
    axes[0,1].axvline(T_SETTLE, color="r", linewidth=0.5)

    axes[1,0].plot(t_arr, logs_list["roll"],  label="roll")
    axes[1,0].plot(t_arr, logs_list["pitch"], label="pitch")
    axes[1,0].set_ylabel("[deg]"); axes[1,0].legend(fontsize=8)
    axes[1,0].axvline(T_SETTLE, color="r", linewidth=0.5)

    axes[1,1].plot(t_arr, logs_list["yaw"])
    axes[1,1].set_ylabel("yaw [deg]")
    axes[1,1].axvline(T_SETTLE, color="r", linewidth=0.5)

    omega_arr = np.array(logs_list["omega"])
    for i, lbl in enumerate(["FL","FR","RL","RR"]):
        axes[2,0].plot(t_arr, omega_arr[:,i], label=lbl, linewidth=0.7)
    axes[2,0].set_ylabel("ω_wheel [rad/s]"); axes[2,0].legend(fontsize=7)
    axes[2,0].axvline(T_SETTLE, color="r", linewidth=0.5)
    axes[2,0].set_xlabel("t [s]")

    # Damping ratio and ωn annotation
    omn  = np.sqrt(4 * kn / 1.5)
    zeta = cn / (2 * np.sqrt(kn * 1.5 / 4))
    axes[2,1].axis("off")
    axes[2,1].text(0.1, 0.6,
        f"Contact spring:\n"
        f"  k_n  = {kn:.0f} N/m\n"
        f"  c_n  = {cn:.0f} N·s/m\n"
        f"  ωn   = {omn:.2f} rad/s  (fn={omn/(2*np.pi):.2f} Hz)\n"
        f"  ζ    = {zeta:.3f}\n\n"
        f"Wheel inertia:\n"
        f"  I_fac   = {iw:.0f}\n"
        f"  tau_spin= {ts:.3f} s\n"
        f"  I_w  = {iw*0.5*0.05*0.0009:.5f} kg·m²\n"
        f"  b_w  = {iw*0.5*0.05*0.0009/ts:.5f} N·m·s",
        transform=axes[2,1].transAxes, fontsize=10, family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    metrics_np, grid, combos = run_sweep()

    best_idx, best_params = print_top_k(metrics_np, grid, combos, k=10)

    # plot_heatmaps(metrics_np, grid, combos)
    plot_best_trajectory(best_idx, best_params)
