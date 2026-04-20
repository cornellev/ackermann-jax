"""
Trace ekf_step using JAX's built-in Perfetto profiler.

Usage:
    python test/trace_ekf_step.py

Outputs a trace directory that can be loaded in https://ui.perfetto.dev.

The script:
  1. JIT-compiles ekf_step with x0/u0 and blocks until ready, printing
     compile+run time so you can see the compilation overhead.
  2. Builds a realistic EKF state from the car dynamics simulation.
  3. Warms up with N_WARMUP additional calls (pure execution, no compile).
  4. Runs ekf_step inside jax.profiler.trace() for N_TRACE steps.
"""

import jax
import jax.numpy as jnp
import jaxlie

from ackermann_jax import (
    AckermannCarModel,
    AckermannCarInput,
    AckermannCarState,
    default_params,
    default_state,
)
from ackermann_jax.ekf import EKFState, ekf_step, ERROR_DIM

# ── Config ────────────────────────────────────────────────────────────

TRACE_DIR = "/Users/lucaslibshutz/Documents/Repos/autonomy/ackermann-jax/ekf_step_trace"
N_WARMUP = 5    # JIT warm-up calls (not traced)
N_TRACE = 20    # calls captured in the trace

DT = 0.01
Q_SCALE = 1e-6
R_GPS_SCALE = 2.25
R_GYRO_SCALE = 1e-4
R_GRAVITY_SCALE = 1e-2
R_WHEELS_SCALE = 1e-4

# ── Measurement functions (same as test_ekf.py) ───────────────────────

def h_gps(x: AckermannCarState):
    return x.p_W

def h_gyro(x: AckermannCarState):
    return x.w_B

def h_wheels(x: AckermannCarState):
    return x.omega_W

def make_h_gravity(g: float):
    g_up_W = jnp.array([0.0, 0.0, g], dtype=jnp.float32)
    def h_gravity(x: AckermannCarState):
        return x.R_WB.as_matrix().T @ g_up_W
    return h_gravity

# ── Build a realistic initial EKF state ──────────────────────────────

def build_ekf_state(model: AckermannCarModel) -> tuple:
    """
    Run the dynamics for a short settle phase to get a realistic x_nom,
    then wrap it in an EKFState.  Also returns a representative (u, measurements).
    """
    x = default_state(z0=0.10)
    P0 = 1e-4 * jnp.eye(ERROR_DIM)

    dt = DT
    integ_v = jnp.array(0.0, dtype=jnp.float32)
    integ_s = jnp.array(0.0, dtype=jnp.float32)

    # Settle: run ~0.5 s of dynamics so the car is on the ground
    N_settle = 50
    for _ in range(N_settle):
        tau_w, integ_v = model.map_velocity_to_wheel_torques(
            x=x, v_cmd=1.0, integral_state=integ_v,
            dt=dt, Kp=40.0, Ki=2.0, tau_max=0.35,
        )
        delta, integ_s = model.map_heading_to_steering(
            x=x, psi_cmd=0.0, integral_state=integ_s, dt=dt,
        )
        u = AckermannCarInput(delta=delta, tau_w=tau_w)
        x = model.step(x=x, u=u, dt=dt)

    ekf = EKFState(x_nom=x, P=P0)

    # Build one step's worth of inputs / measurements
    tau_w, _ = model.map_velocity_to_wheel_torques(
        x=x, v_cmd=1.0, integral_state=integ_v,
        dt=dt, Kp=40.0, Ki=2.0, tau_max=0.35,
    )
    delta, _ = model.map_heading_to_steering(
        x=x, psi_cmd=0.0, integral_state=integ_s, dt=dt,
    )
    u = AckermannCarInput(delta=delta, tau_w=tau_w)

    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    g = model.params.chassis.g

    meas = {
        "gps":     x.p_W    + jnp.sqrt(R_GPS_SCALE)     * jax.random.normal(k1, x.p_W.shape),
        "gyro":    x.w_B    + jnp.sqrt(R_GYRO_SCALE)    * jax.random.normal(k2, x.w_B.shape),
        "gravity": x.R_WB.as_matrix().T @ jnp.array([0., 0., g], dtype=jnp.float32)
                            + jnp.sqrt(R_GRAVITY_SCALE)  * jax.random.normal(k3, (3,)),
        "wheels":  x.omega_W + jnp.sqrt(R_WHEELS_SCALE) * jax.random.normal(k4, x.omega_W.shape),
    }

    return ekf, u, meas

# ── Traced function ───────────────────────────────────────────────────

def run_ekf_step_sequence(ekf, model, u, meas, Q, R_gps, R_gyro, R_gravity, R_wheels, dt):
    """One full predict + 4 sequential updates (matches test_ekf.py pattern)."""
    h_gravity = make_h_gravity(model.params.chassis.g)
    ekf = ekf_step(model, ekf, u, meas["gps"], h_gps, Q, R_gps, dt)
    ekf = ekf_step(model, ekf, u, meas["gyro"], h_gyro, Q, R_gyro, dt)
    ekf = ekf_step(model, ekf, u, meas["gravity"], h_gravity, Q, R_gravity, dt)
    ekf = ekf_step(model, ekf, u, meas["wheels"], h_wheels, Q, R_wheels, dt)
    return ekf

# ── Main ──────────────────────────────────────────────────────────────

def main():
    import time

    params = default_params()
    model = AckermannCarModel(params)

    Q        = Q_SCALE        * jnp.eye(ERROR_DIM)
    R_gps    = R_GPS_SCALE    * jnp.eye(3)
    R_gyro   = R_GYRO_SCALE   * jnp.eye(3)
    R_grav   = R_GRAVITY_SCALE * jnp.eye(3)
    R_wheels = R_WHEELS_SCALE  * jnp.eye(4)

    # ── Step 1: compile with x0/u0 ────────────────────────────────────
    x0 = default_state(z0=0.10)
    u0 = AckermannCarInput(
        delta=jnp.array(0.0, dtype=jnp.float32),
        tau_w=jnp.zeros(4, dtype=jnp.float32),
    )
    ekf0 = EKFState(x_nom=x0, P=1e-4 * jnp.eye(ERROR_DIM))
    g = params.chassis.g
    meas0 = {
        "gps":     x0.p_W,
        "gyro":    x0.w_B,
        "gravity": x0.R_WB.as_matrix().T @ jnp.array([0., 0., g], dtype=jnp.float32),
        "wheels":  x0.omega_W,
    }
    kwargs0 = dict(
        model=model, u=u0, meas=meas0,
        Q=Q, R_gps=R_gps, R_gyro=R_gyro, R_gravity=R_grav, R_wheels=R_wheels,
        dt=DT,
    )

    print("Compiling ekf_step (x0/u0)...")
    t0 = time.perf_counter()
    ekf_compiled = run_ekf_step_sequence(ekf=ekf0, **kwargs0)
    jax.block_until_ready(ekf_compiled)
    t_compile = time.perf_counter() - t0
    print(f"  compile + run: {t_compile * 1e3:.1f} ms")

    # One more call with x0 to isolate pure execution cost post-compile
    t0 = time.perf_counter()
    ekf_compiled = run_ekf_step_sequence(ekf=ekf_compiled, **kwargs0)
    jax.block_until_ready(ekf_compiled)
    t_run0 = time.perf_counter() - t0
    print(f"  run only:      {t_run0 * 1e3:.1f} ms")

    # ── Step 2: build settled EKF state ──────────────────────────────
    ekf, u, meas = build_ekf_state(model)
    step_kwargs = dict(
        model=model, u=u, meas=meas,
        Q=Q, R_gps=R_gps, R_gyro=R_gyro, R_gravity=R_grav, R_wheels=R_wheels,
        dt=DT,
    )

    # ── Step 3: warm-up calls with settled state ──────────────────────
    print(f"\nWarming up with settled state ({N_WARMUP} calls)...")
    ekf_cur = ekf
    for _ in range(N_WARMUP):
        ekf_cur = run_ekf_step_sequence(ekf=ekf_cur, **step_kwargs)
        jax.block_until_ready(ekf_cur)

    # ── Step 4: trace ─────────────────────────────────────────────────
    print(f"Tracing {N_TRACE} calls → {TRACE_DIR}")
    ekf_cur = ekf
    with jax.profiler.trace(TRACE_DIR, create_perfetto_link=True):
        for i in range(N_TRACE):
            with jax.profiler.TraceAnnotation(f"ekf_step_{i}"):
                ekf_cur = run_ekf_step_sequence(ekf=ekf_cur, **step_kwargs)
            jax.block_until_ready(ekf_cur)

    print(f"\nDone.  Open https://ui.perfetto.dev and load the trace from:\n  {TRACE_DIR}")


if __name__ == "__main__":
    main()
