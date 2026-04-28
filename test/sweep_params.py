"""
sweep_contact_params.py — Batched parameter sweep for Ackermann contact/wheel tuning.

Strategy
--------
Because AckermannCarModel holds Python-level scalars inside its parameter
dataclasses, we cannot vmap *through* the model constructor.  Instead we:

  1. Build a single base model with default params.
  2. Write a `simulate_one` function that accepts
     (k_n, c_n, I_w_fac, tau_spin, c_kappa, c_alpha)
     as plain JAX scalars, patches the relevant fields on-the-fly inside the
     scan body, and returns scalar score metrics.
  3. vmap `simulate_one` over a flat grid of parameter combinations.
  4. Reshape results back to the grid and plot heatmaps.

Swept parameters (6 total):
  k_n      — contact spring stiffness      [N/m]
  c_n      — contact spring damping        [N·s/m]
  I_w_fac  — wheel inertia scale factor    [–]
  tau_spin — wheel speed settling time     [s]
  c_kappa  — longitudinal tire stiffness   [N/–]
  c_alpha  — lateral tire stiffness        [N/rad]

Metrics scored (all lower=better):
  S1  bounce       — max(p_z) - p_z[0]              : car left ground?
  S2  settle_vz    — RMS v_z over last 50% of settle
  S3  pitch_osc    — RMS w_B[1] during weave
  S4  roll_osc     — RMS w_B[0] during weave
  S5  omega_rough  — RMS Δω_W/dt during weave (jitter)
  S6  z_drift      — |mean(p_z) - p_z[0]| during weave
  S7  slip_cycling — wheel direction-reversal rate during weave
  composite        — weighted sum of normalised metrics
"""

from __future__ import annotations

import itertools
import time
import addcopyfighandler #noqa

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
# WARNING: total combos = product of all lengths. Keep grids coarse first.

# K_N_VALUES      = [500.,  1000., 1500., 2000., 3000.]   # N/m
K_N_VALUES = [1500.]
# C_N_VALUES      = [30.,   60.,   100.,  150.,  250.]     # N·s/m
C_N_VALUES = [50.]
IW_FAC_VALUES   = [1.,    2.,    5.,    10.]             # inertia scale factor
TAU_SPIN_VALUES = [0.05,  0.1,   0.2,   0.3]            # wheel settle time [s]
C_KAPPA_VALUES  = [5.,    10.,   20.,   30.]             # longitudinal tire stiffness
C_ALPHA_VALUES  = [5.,    10.,   20.,   25.]             # lateral tire stiffness

# Column indices in the flat grid array — update if you reorder above
_COL_KN, _COL_CN, _COL_IW, _COL_TAU, _COL_CK, _COL_CA = 0, 1, 2, 3, 4, 5

# Simulation settings
DT        = 0.01      # fine dt — sweep finds params that also work at coarse dt
T_SETTLE  = 1.5       # [s]
T_WEAVE   = 3.0       # [s]  shorter for sweep speed
V_CMD     = 1.0
Z0        = 0.08      # initial height [m]

# Scoring weights — [bounce, settle_vz, pitch_osc, roll_osc, omega_rough, z_drift, slip_cycling]
WEIGHTS = np.array([3.0, 1.5, 4.0, 4.0, 6.0, 2.0, 1.0])

METRIC_NAMES = [
    "bounce [m]",
    "settle_vz [m/s]",
    "pitch_rms [rad/s]",
    "roll_rms [rad/s]",
    "omega_rough [rad/s²]",
    "z_drift [m]",
    "slip_cycling [–]",
]

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


def simulate_one(k_n, c_n, I_w_fac, tau_spin, c_kappa, c_alpha):
    """
    Run settle + weave for one parameter combination.
    All arguments are JAX scalars (float32) — fully vmappable.

    Returns a (7,) float32 array of raw metric values.
    """
    p    = _BASE_PARAMS
    geom = p.geom
    tp   = p.tires

    m_wheel = 0.05
    I_w = I_w_fac * 0.5 * m_wheel * geom.wheel_radius ** 2
    b_w = I_w / tau_spin

    N_settle = int(T_SETTLE / DT)
    N_weave  = int(T_WEAVE  / DT)
    N_total  = N_settle + N_weave

    x0 = default_state(z0=Z0)

    # ── Custom xdot that uses swept contact / wheel params ────────────────────
    def xdot_patched(x: AckermannCarState, u: AckermannCarInput):
        """Returns (xdot, v_t, kappa, Fx, tau_cmd) — intermediates needed for
        the exact semi-implicit wheel integration."""
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

        # ← patched slip: uses base eps_v but swept c_kappa / c_alpha
        kappa_max = 2.0
        rw   = geom.wheel_radius
        eps_v = tp.eps_v
        denom = jnp.sqrt(v_t * v_t + eps_v * eps_v)
        kappa = kappa_max * jnp.tanh((rw * omega_w - v_t) / denom / kappa_max)
        alpha = 0.5 * jnp.tanh(jnp.arctan2(v_n, jnp.abs(v_t) + eps_v) / 0.5)

        # ← patched tire forces with swept c_kappa / c_alpha
        Fx_star = c_kappa * kappa
        Fy_star = -c_alpha * alpha
        Fmax = tp.mu * Fz
        mag   = jnp.sqrt(Fx_star**2 + Fy_star**2 + tp.eps_force)
        scale = jnp.minimum(1.0, Fmax / jnp.sqrt(mag**2 + Fmax**2))
        Fx = scale * Fx_star
        Fy = scale * Fy_star

        Fx *= jnp.array([0.0, 0.0, 1.0, 1.0])

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
        xdot_state = AckermannCarState(
            p_W=dp_W,
            R_WB=_jaxlie.SO3.identity(),
            v_W=dv_W,
            w_B=dW_B,
            omega_W=domega_w,
        )
        return xdot_state, v_t, kappa, Fx, tau_cmd

    def step_patched(x, u):
        xdot, v_t, kappa, Fx, tau_cmd = xdot_patched(x, u)
        Fx *= jnp.array([0.0, 0.0, 1.0, 1.0]) # only consider rear wheels

        # Semi-implicit Euler for translational / rotational body states
        v_W_next = x.v_W + DT * xdot.v_W
        w_B_next = x.w_B + DT * xdot.w_B
        p_W_next = x.p_W + DT * v_W_next

        import jaxlie as _jaxlie
        R_next = x.R_WB @ _jaxlie.SO3.exp(w_B_next * DT)

        # Exact analytical integration for wheel speeds (unconditionally stable).
        # Linearises Fx(omega) around the current op-point:
        #   Fx(omega) ≈ Fx_k + dFx/domega * (omega - omega_k)
        # giving a linear ODE whose solution is:
        #   omega_next = omega_inf + (omega_k - omega_inf) * exp(-lambda * dt)
        rw    = geom.wheel_radius
        eps_v = tp.eps_v
        denom_vt       = jnp.sqrt(v_t * v_t + eps_v * eps_v)
        kappa_raw      = (rw * x.omega_W - v_t) / denom_vt
        sech2          = 1.0 - jnp.tanh(kappa_raw / 2.0) ** 2
        dkappa_domega  = (rw / denom_vt) * sech2
        dFx_domega     = c_kappa * dkappa_domega
        lam            = (b_w + rw * dFx_domega) / I_w
        omega_inf      = (tau_cmd - rw * (Fx - dFx_domega * x.omega_W)) / (b_w + rw * dFx_domega)
        omega_w_next   = omega_inf + (x.omega_W - omega_inf) * jnp.exp(-lam * DT)

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
            use_traction_limit=False, # use actual parameters and not what is in the base model
        )
        Fz_approx = p.chassis.mass * p.chassis.g / 4.0
        tau_fric = c_kappa * 1.0 * p.tires.mu * Fz_approx * geom.wheel_radius
        tau_w = jnp.clip(tau_w, -tau_fric, tau_fric)

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

    # S7: wheel direction reversals during weave — slip limit-cycling indicator
    # count steps where any wheel flips sign, normalised to [0, 1]
    omega_weave = omega[weave_slice]                          # (N_weave, 4)
    sign_flips  = jnp.diff(jnp.sign(omega_weave), axis=0)    # nonzero = flip
    s7_slip     = jnp.sum(sign_flips != 0).astype(jnp.float32) / float(N_weave * 4)

    return jnp.array(
        [s1_bounce, s2_settle, s3_pitch, s4_roll, s5_rough, s6_zdrift, s7_slip],
        dtype=jnp.float32,
    )


# ── Build flat parameter grid ─────────────────────────────────────────────────

def build_grid():
    combos = list(itertools.product(
        K_N_VALUES, C_N_VALUES, IW_FAC_VALUES, TAU_SPIN_VALUES,
        C_KAPPA_VALUES, C_ALPHA_VALUES,
    ))
    arr = np.array(combos, dtype=np.float32)   # (N_combos, 6)
    return arr, combos

# ── Run sweep ─────────────────────────────────────────────────────────────────

def run_sweep():
    grid, combos = build_grid()
    N = len(combos)
    print(f"Sweeping {N} parameter combinations:")
    print(f"  {len(K_N_VALUES)} k_n  × {len(C_N_VALUES)} c_n  × "
          f"{len(IW_FAC_VALUES)} I_fac × {len(TAU_SPIN_VALUES)} tau_spin × "
          f"{len(C_KAPPA_VALUES)} c_kappa × {len(C_ALPHA_VALUES)} c_alpha")

    sim_batched = jax.jit(jax.vmap(simulate_one))

    print("JIT compiling…", flush=True)
    t0      = time.perf_counter()
    metrics = sim_batched(
        jnp.array(grid[:, _COL_KN]),
        jnp.array(grid[:, _COL_CN]),
        jnp.array(grid[:, _COL_IW]),
        jnp.array(grid[:, _COL_TAU]),
        jnp.array(grid[:, _COL_CK]),
        jnp.array(grid[:, _COL_CA]),
    )
    metrics.block_until_ready()
    t_compile = time.perf_counter() - t0
    print(f"  compile + first run: {t_compile:.1f} s")

    print("Second run (execution only)…", flush=True)
    t0      = time.perf_counter()
    metrics = sim_batched(
        jnp.array(grid[:, _COL_KN]),
        jnp.array(grid[:, _COL_CN]),
        jnp.array(grid[:, _COL_IW]),
        jnp.array(grid[:, _COL_TAU]),
        jnp.array(grid[:, _COL_CK]),
        jnp.array(grid[:, _COL_CA]),
    )
    metrics.block_until_ready()
    t_run = time.perf_counter() - t0
    print(f"  run only: {t_run*1e3:.1f} ms  ({t_run/N*1e3:.3f} ms/combo)")

    return np.array(metrics), grid, combos

# ── Scoring and ranking ───────────────────────────────────────────────────────

def compute_composite(metrics_np):
    mn  = metrics_np.min(axis=0, keepdims=True)
    mx  = metrics_np.max(axis=0, keepdims=True)
    rng = np.maximum(mx - mn, 1e-8)
    norm      = (metrics_np - mn) / rng
    composite = norm @ WEIGHTS / WEIGHTS.sum()
    return composite, norm

def print_top_k(metrics_np, grid, combos, k=10):
    composite, _ = compute_composite(metrics_np)
    order = np.argsort(composite)

    col_w = 10
    hdr   = (f"{'Rank':>4}  {'k_n':>6}  {'c_n':>5}  {'Ifac':>4}  "
             f"{'τspin':>5}  {'Ckap':>5}  {'Calp':>5}  "
             + "  ".join(f"{n[:col_w]:>{col_w}}" for n in METRIC_NAMES)
             + "  composite")
    print(f"\n{'─'*len(hdr)}")
    print(hdr)
    print(f"{'─'*len(hdr)}")

    for rank, idx in enumerate(order[:k]):
        kn, cn, iw, ts, ck, ca = grid[idx]
        vals = "  ".join(f"{metrics_np[idx, j]:{col_w}.4f}" for j in range(len(METRIC_NAMES)))
        print(f"{rank+1:>4}  {kn:>6.0f}  {cn:>5.0f}  {iw:>4.0f}  "
              f"{ts:>5.3f}  {ck:>5.0f}  {ca:>5.0f}  {vals}  {composite[idx]:.4f}")

    print(f"\n{'─'*len(hdr)}")
    best = order[0]
    kn, cn, iw, ts, ck, ca = grid[best]
    print("Top-1 parameter set:")
    print(f"  k_n     = {kn:.0f}  N/m")
    print(f"  c_n     = {cn:.0f}  N·s/m")
    print(f"  I_w_fac = {iw:.0f}")
    print(f"  tau_spin= {ts:.3f} s")
    print(f"  c_kappa = {ck:.0f}")
    print(f"  c_alpha = {ca:.0f}")
    m_w = 0.05; r_w = 0.03
    I_w  = iw * 0.5 * m_w * r_w**2
    b_w_ = I_w / ts
    omn  = np.sqrt(4 * kn / 1.5)
    zeta = cn / (2 * np.sqrt(kn * 1.5 / 4))
    print(f"\n  Derived:")
    print(f"  I_w  = {I_w:.5f} kg·m²  |  b_w = {b_w_:.5f} N·m·s")
    print(f"  ωn   = {omn:.2f} rad/s  (fn = {omn/(2*np.pi):.2f} Hz)  |  ζ = {zeta:.3f}")
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
# ── Heatmap plots ─────────────────────────────────────────────────────────────

def _safe_imshow(ax, mat, cmap="RdYlGn_r"):
    """imshow with PowerNorm, gracefully handles constant/all-nan slices."""
    finite_vals = mat[np.isfinite(mat)]
    if finite_vals.size == 0:
        ax.text(0.5, 0.5, "no finite data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.axis("off")
        return None
    vmin = float(np.nanmin(mat))
    vmax = float(np.nanmax(mat))
    if np.isclose(vmin, vmax):
        norm = mcolors.NoNorm()
    else:
        safe_vmin = max(vmin, 0.0)
        safe_vmax = max(vmax, safe_vmin + 1e-8)
        norm = mcolors.PowerNorm(gamma=0.5, vmin=safe_vmin, vmax=safe_vmax)
    return ax.imshow(mat, aspect="auto", cmap=cmap, origin="lower", norm=norm)


def plot_heatmaps(metrics_np, grid, combos):
    """
    Outer loop: (c_kappa, c_alpha) pairs  →  one figure per pair.
    Inner heatmap: k_n (y-axis) × c_n (x-axis).
    Each cell shows the MIN composite/metric over (I_w_fac, tau_spin),
    so the best wheel params are automatically selected per contact-param cell.
    White star marks the best cell per subplot.
    """
    composite, _ = compute_composite(metrics_np)
    all_metrics  = np.concatenate([metrics_np, composite[:, None]], axis=1)
    all_names    = METRIC_NAMES + ["composite (lower=better)"]
    n_metrics    = all_metrics.shape[1]

    kn_vals = np.array(sorted(set(grid[:, _COL_KN].tolist())))
    cn_vals = np.array(sorted(set(grid[:, _COL_CN].tolist())))
    ck_vals = np.array(sorted(set(grid[:, _COL_CK].tolist())))
    ca_vals = np.array(sorted(set(grid[:, _COL_CA].tolist())))
    Nkn, Ncn = len(kn_vals), len(cn_vals)

    # Index lookup maps for speed
    kn_idx = {float(v): i for i, v in enumerate(kn_vals)}
    cn_idx = {float(v): i for i, v in enumerate(cn_vals)}

    for ck in ck_vals:
        for ca in ca_vals:
            fig, axes = plt.subplots(2, 4, figsize=(20, 9))
            axes = axes.flatten()
            fig.suptitle(
                f"c_kappa={ck:.0f}  c_alpha={ca:.0f}  "
                f"(min over I_w_fac & tau_spin per cell)",
                fontsize=12,
            )

            mask    = (grid[:, _COL_CK] == ck) & (grid[:, _COL_CA] == ca)
            sub_g   = grid[mask]
            sub_met = all_metrics[mask]

            for m_idx in range(n_metrics):
                ax  = axes[m_idx]
                mat = np.full((Nkn, Ncn), np.inf)

                for row_idx in range(sub_g.shape[0]):
                    ki  = kn_idx[float(sub_g[row_idx, _COL_KN])]
                    ci  = cn_idx[float(sub_g[row_idx, _COL_CN])]
                    val = float(sub_met[row_idx, m_idx])
                    if val < mat[ki, ci]:
                        mat[ki, ci] = val
                mat[mat == np.inf] = np.nan

                im = _safe_imshow(ax, mat)
                if im is not None:
                    plt.colorbar(im, ax=ax, shrink=0.8)
                    if np.any(np.isfinite(mat)):
                        bi, bj = np.unravel_index(np.nanargmin(mat), mat.shape)
                        ax.plot(bj, bi, "w*", markersize=12)

                step_x = max(1, Ncn // 7)
                step_y = max(1, Nkn // 7)
                ax.set_xticks(range(0, Ncn, step_x))
                ax.set_xticklabels(
                    [f"{cn_vals[i]:.0f}" for i in range(0, Ncn, step_x)],
                    fontsize=6, rotation=45,
                )
                ax.set_yticks(range(0, Nkn, step_y))
                ax.set_yticklabels(
                    [f"{kn_vals[i]:.0f}" for i in range(0, Nkn, step_y)],
                    fontsize=6,
                )
                ax.set_xlabel("c_n [N·s/m]", fontsize=8)
                ax.set_ylabel("k_n [N/m]",   fontsize=8)
                ax.set_title(all_names[m_idx], fontsize=9)

            for ax in axes[n_metrics:]:
                ax.set_visible(False)
            fig.tight_layout()

    plt.show()


# ── Trajectory plot for best params ──────────────────────────────────────────

def plot_best_trajectory(best_idx, best_params):
    kn, cn, iw, ts, ck, ca = best_params
    print(f"\nPlotting trajectory for best params: "
          f"k_n={kn:.0f}  c_n={cn:.0f}  I_fac={iw:.0f}  "
          f"tau={ts:.3f}  c_kappa={ck:.0f}  c_alpha={ca:.0f}")

    N_settle = int(T_SETTLE / DT)
    N_weave  = int(T_WEAVE  / DT)
    N_total  = N_settle + N_weave
    t_arr    = np.arange(N_total) * DT

    logs_list = {"p_z": [], "v_z": [], "roll": [], "pitch": [],
                 "yaw": [], "omega": []}

    x0    = default_state(z0=Z0)
    p_new = default_params()

    import dataclasses
    m_w = 0.05; r_w = 0.03
    I_w  = float(iw) * 0.5 * m_w * r_w**2
    b_w_ = I_w / float(ts)

    new_contact = dataclasses.replace(p_new.contact, k_n=float(kn), c_n=float(cn))
    new_wheels  = dataclasses.replace(p_new.wheels,  I_w=I_w, b_w=b_w_)
    new_tires   = dataclasses.replace(p_new.tires,   C_kappa=float(ck), C_alpha=float(ca))
    new_p       = dataclasses.replace(p_new, contact=new_contact,
                                      wheels=new_wheels, tires=new_tires)
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
    fig.suptitle(
        f"Best params: k_n={kn:.0f}  c_n={cn:.0f}  I_fac={iw:.0f}  "
        f"tau_spin={ts:.3f} s  c_kappa={ck:.0f}  c_alpha={ca:.0f}",
        fontsize=11,
    )

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

    omn  = np.sqrt(4 * kn / 1.5)
    zeta = cn / (2 * np.sqrt(kn * 1.5 / 4))
    I_w_val = iw * 0.5 * 0.05 * 0.03**2
    axes[2,1].axis("off")
    axes[2,1].text(0.05, 0.97,
        f"Contact spring:\n"
        f"  k_n  = {kn:.0f} N/m\n"
        f"  c_n  = {cn:.0f} N·s/m\n"
        f"  ωn   = {omn:.2f} rad/s  (fn={omn/(2*np.pi):.2f} Hz)\n"
        f"  ζ    = {zeta:.3f}\n\n"
        f"Wheel inertia:\n"
        f"  I_fac    = {iw:.0f}\n"
        f"  tau_spin = {ts:.3f} s\n"
        f"  I_w      = {I_w_val:.5f} kg·m²\n"
        f"  b_w      = {I_w_val/ts:.5f} N·m·s\n\n"
        f"Tire model:\n"
        f"  C_kappa  = {ck:.0f}\n"
        f"  C_alpha  = {ca:.0f}",
        transform=axes[2,1].transAxes, fontsize=9, family="monospace",
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
