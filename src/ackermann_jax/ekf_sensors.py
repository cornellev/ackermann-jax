"""
Real-time EKF prediction and update loop.
 
Reads sensor data from the sensor shared-memory block, runs one EKF
predict-update cycle per iteration, and publishes the estimated vehicle
state to the Kalman shared-memory block.
 
Sensor to EKF mapping
---------------------
.. code-block:: text
 
    Sensor field          SHM index   EKF usage
    --------------------  ---------   ----------------------------
    steering turn angle   d[6]        u.delta  [rad]
    throttle              d[18]       u.tau_w  (scaled by _TAU_MAX, RWD mask)
    GPS latitude          d[14]       z_gps x  [m, local ENU]
    GPS longitude         d[15]       z_gps y  [m, local ENU]
    RPM front-left        d[8]        omega_W[0] [rad/s]
    RPM front-right       d[9]        omega_W[1] [rad/s]
    RPM rear-left         d[11]       omega_W[2] [rad/s]
    RPM rear-right        d[12]       omega_W[3] [rad/s]
 
"""

import math
import time
import jax.numpy as jnp
import jaxlie
from ackermann_jax.read_sensor_shm import SensorShmReader
from ackermann_jax.write_kalman_shm import KalmanShmWriter
from ackermann_jax.ekf import EKFState, ekf_predict, ekf_update, ERROR_DIM
from ackermann_jax.car import (
    AckermannCarModel,
    AckermannCarInput,
    AckermannCarState,
    default_params,
    default_state,
)

# indexes for easy access
_IDX_GLOBAL_TS = 0
_IDX_TURN_ANGLE = 6  # steering turn_angle
_IDX_RPM_FL = 8  # front-left RPM
_IDX_RPM_FR = 9  # front-right RPM
_IDX_RPM_RL = 11  # rear-left RPM
_IDX_RPM_RR = 12  # rear-right RPM
_IDX_GPS_LAT = 14
_IDX_GPS_LON = 15
_IDX_THROTTLE = 18  # motor throttle

# useful constants
_RPM_TO_RADS = 2.0 * math.pi / 60.0
_R_EARTH = 6_371_000.0 
_R_GPS_VAL = 1e-4  # GPS position noise variance
_R_WHEELS_VAL = 1e-4  # wheel encoder noise variance
_Q_SCALE = 1e-6  # process-noise scale
_P0_SCALE = 1e-4  # initial covariance scale
_TAU_MAX = 0.35  # peak wheel torque
_DT_DEFAULT = 0.01  # nominal step, also used when timestamps are stale
_DT_MAX = 1.0  # cap: larger gaps are treated as a re-init guard


def h_gps_2d(x):
    return x.p_W[:2]


def h_wheels(x):
    return x.omega_W


def _latlon_to_local_xy(lat, lon, lat0, lon0):
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    x = dlon * math.cos(math.radians(lat0)) * _R_EARTH
    y = dlat * _R_EARTH
    return x, y


def main() -> None:
    """
    Run the real-time EKF loop. Each iteration reads the sensor SHM,
    runs predict, updates state parameters, and publishes the result
    to the Kalman SHM. The loop runs until interrupted, at which point 
    the sensor reader and Kalman writer are closed cleanly.
    """
    params = default_params()
    model = AckermannCarModel(params)

    reader = SensorShmReader()
    writer = KalmanShmWriter()

    Q = _Q_SCALE * jnp.eye(ERROR_DIM)
    P0 = _P0_SCALE * jnp.eye(ERROR_DIM)
    R_gps_mat = _R_GPS_VAL * jnp.eye(2)
    R_wheels_mat = _R_WHEELS_VAL * jnp.eye(4)

    _motor_mask = jnp.array([0.0, 0.0, 1.0, 1.0], dtype=jnp.float32)

    ekf = None
    lat0 = None
    lon0 = None
    prev_ts_s = None

    try:
        while True:
            loop_start = time.monotonic()
            snap = reader.read_snapshot()
            if snap is None:
                time.sleep(_DT_DEFAULT)
                continue
            _, d = snap

            ts_us = int(d[_IDX_GLOBAL_TS])
            ts_s = ts_us * 1e-6
            lat = float(d[_IDX_GPS_LAT])
            lon = float(d[_IDX_GPS_LON])
            delta = float(d[_IDX_TURN_ANGLE])
            throttle = float(d[_IDX_THROTTLE])
            omega_meas = jnp.array(
                [
                    float(d[_IDX_RPM_FL]) * _RPM_TO_RADS,
                    float(d[_IDX_RPM_FR]) * _RPM_TO_RADS,
                    float(d[_IDX_RPM_RL]) * _RPM_TO_RADS,
                    float(d[_IDX_RPM_RR]) * _RPM_TO_RADS,
                ],
                dtype=jnp.float32,
            )
            if lat0 is None and lat != 0.0 and lon != 0.0:
                lat0, lon0 = lat, lon
                print(f"[ekf_pred] GPS origin set: ({lat0:.7f}, {lon0:.7f})")
            if ekf is None:
                if lat0 is None:
                    time.sleep(_DT_DEFAULT)
                    continue
                gx, gy = _latlon_to_local_xy(lat, lon, lat0, lon0)
                p0 = jnp.array(
                    [gx, gy, float(params.geom.wheel_radius)],
                    dtype=jnp.float32,
                )
                x0 = AckermannCarState(
                    p_W=p0,
                    R_WB=jaxlie.SO3.identity(),
                    v_W=jnp.zeros(3, dtype=jnp.float32),
                    w_B=jnp.zeros(3, dtype=jnp.float32),
                    omega_W=omega_meas,
                )
                ekf = EKFState(x_nom=x0, P=P0)
                prev_ts_s = ts_s
                time.sleep(_DT_DEFAULT)
                continue

            dt = ts_s - prev_ts_s if prev_ts_s is not None else _DT_DEFAULT
            if dt <= 0.0 or dt > _DT_MAX:
                dt = _DT_DEFAULT
            prev_ts_s = ts_s

            tau_w = _motor_mask * (throttle * _TAU_MAX)
            u = AckermannCarInput(
                delta=jnp.array(delta, dtype=jnp.float32),
                tau_w=tau_w.astype(jnp.float32),
            )

            ekf = ekf_predict(model, ekf, u, Q, dt)

            if lat != 0.0 and lon != 0.0 and lat0 is not None:
                gx, gy = _latlon_to_local_xy(lat, lon, lat0, lon0)
                z_gps = jnp.array([gx, gy], dtype=jnp.float32)
                ekf = ekf_update(ekf, z_gps, h_gps_2d, R_gps_mat)

            ekf = ekf_update(ekf, omega_meas, h_wheels, R_wheels_mat)

            writer.write_state(ekf, ts_us)

            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, _DT_DEFAULT - elapsed)
            time.sleep(sleep_time)

    finally:
        reader.close()
        writer.close()


if __name__ == "__main__":
    main()
