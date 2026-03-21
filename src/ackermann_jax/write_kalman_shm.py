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
    def __init__(self, name: str = SHM_NAME):
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
        if self.shm is not None:
            self.shm.close()

    def unlink(self) -> None:
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass
