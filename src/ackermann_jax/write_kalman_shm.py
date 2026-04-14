"""
Write the EKF state estimate to a POSIX shared-memory block for consumption
by :class:`~ackermann_jax.read_kalman_shm.KalmanShmReader` or any other
process that maps the same block.
 
Shared-memory layout
--------------------
.. code-block:: text
 
    Offset   Size   Type        Field
    ------   ----   ---------   -------------------------------------------
    0        4      uint32      seq-lock counter (odd = write in progress)
    4        8      uint64      hardware timestamp [µs]
    12       12     float32×3   p_W      world-frame position [m]
    24       16     float32×4   R_WB     rotation quaternion (w, x, y, z)
    40       12     float32×3   v_W      world-frame velocity [m/s]
    52       12     float32×3   w_B      body-frame angular velocity [rad/s]
    64       16     float32×4   omega_W  wheel angular velocities [rad/s]
"""
import struct
from multiprocessing import shared_memory, resource_tracker

from ackermann_jax.ekf import EKFState

SHM_NAME = "kalman_shm"
KALMAN_FMT = "<Q17f"
KALMAN_SIZE = struct.calcsize(KALMAN_FMT)
SEQ_FMT = "<I"
SEQ_SIZE = struct.calcsize(SEQ_FMT)
BLOCK_SIZE = SEQ_SIZE + KALMAN_SIZE


class KalmanShmWriter:
    '''
    Create and maintain the Kalman output shared-memory block.
 
    Creates the POSIX shared-memory block on construction and exposes :meth:`write_state`
    to publish a new :class:`~ackermann_jax.ekf.EKFState` atomically.
 
 
    Example::
 
        writer = KalmanShmWriter()
        # inside EKF loop:
        writer.write_state(ekf_state, timestamp_us)
        # on shutdown:
        writer.close()
    '''
    def __init__(self, name: str = SHM_NAME):
        '''
        Create the shared-memory block.
        '''
        try:
            old = shared_memory.SharedMemory(name=name, create=False)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass

        self.shm = shared_memory.SharedMemory(name=name, size=BLOCK_SIZE, create=True)
        resource_tracker.unregister(self.shm._name, "shared_memory")
        self.buf = self.shm.buf
        self._seq: int = 0
        self.buf[:BLOCK_SIZE] = bytes(BLOCK_SIZE)

    def write_state(self, ekf: EKFState, timestamp: int) -> None:
        '''
        Atomically publish a new EKF state estimate.
        
        Args:
            ekf: Current :class:`~ackermann_jax.ekf.EKFState`.  Only the
                nominal trajectory ``ekf.x_nom`` is published; the covariance
                ``ekf.P`` is not written to shared memory.
            timestamp: Hardware timestamp in **microseconds** (``uint64``),
                typically sourced from the sensor shared-memory block.
        '''
        x = ekf.x_nom
        p = x.p_W
        wxyz = x.R_WB.wxyz
        v = x.v_W
        w = x.w_B
        om = x.omega_W

        self._seq = (self._seq + 1) | 1
        struct.pack_into(SEQ_FMT, self.buf, 0, self._seq)

        payload = struct.pack(
            KALMAN_FMT,
            int(timestamp),
            float(p[0]),
            float(p[1]),
            float(p[2]),
            float(wxyz[0]),
            float(wxyz[1]),
            float(wxyz[2]),
            float(wxyz[3]),
            float(v[0]),
            float(v[1]),
            float(v[2]),
            float(w[0]),
            float(w[1]),
            float(w[2]),
            float(om[0]),
            float(om[1]),
            float(om[2]),
            float(om[3]),
        )
        self.buf[SEQ_SIZE : SEQ_SIZE + KALMAN_SIZE] = payload

        self._seq += 1
        struct.pack_into(SEQ_FMT, self.buf, 0, self._seq)

    def close(self) -> None:
        '''
        Detach from the shared-memory block without unlinking it.
        '''
        if self.shm is not None:
            self.shm.close()

    def unlink(self) -> None:
        '''
        Destroy the shared-memory block.
        '''
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass
