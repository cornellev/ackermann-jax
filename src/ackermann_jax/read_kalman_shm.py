"""
Read the EKF state estimate from a POSIX shared-memory block written by
:class:`~ackermann_jax.write_kalman_shm.KalmanShmWriter`.

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

import time
import struct
from multiprocessing import shared_memory, resource_tracker

SHM_NAME = "kalman_shm"
KALMAN_FMT = "<Q17f"
KALMAN_SIZE = struct.calcsize(KALMAN_FMT)
SEQ_FMT = "<I"
SEQ_SIZE = struct.calcsize(SEQ_FMT)
BLOCK_SIZE = SEQ_SIZE + KALMAN_SIZE


def _read_seq(buf) -> int:
    return struct.unpack_from(SEQ_FMT, buf, 0)[0]


class KalmanShmReader:
    """
    Attach to the Kalman output shared-memory block and read EKF states.
    Can be used by any process that wants to consume the estimated
    car state published by the EKF process.  Returns ``None`` if the block does not exist yet.

    Attributes:
        available (bool): ``True`` if the shared-memory block was found and
            successfully attached; ``False`` otherwise.

    Example::

        reader = KalmanShmReader()
        if reader.available:
            snap = reader.read_snapshot_dict()
            print(snap["p_W"])
        reader.close()
    """

    def __init__(self, name: str = SHM_NAME):
        """
        Attach to an existing shared-memory block.
        """
        self.available = False
        self._shm = None
        self._buf = None

        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            resource_tracker.unregister(shm._name, "shared_memory")
        except FileNotFoundError:
            print("Kalman SHM not found.")
            return

        if shm.size < BLOCK_SIZE:
            shm.close()
            raise RuntimeError(f"SHM too small: {shm.size} < {BLOCK_SIZE}")

        self._shm = shm
        self._buf = shm.buf
        self.available = True

    def close(self) -> None:
        """
        Detach from the shared-memory block.
        """
        if self._shm is not None:
            self._shm.close()
            self._shm = None
            self._buf = None
            self.available = False

    def read_snapshot(self):
        """
        Read one consistent snapshot.
 
        Returns:
            ``(seq, data)`` on success, where
 
            - **seq** (*int*) — even seq-lock counter at read time.
            - **data** (*tuple*) — 18-element tuple:
 
              .. code-block:: text
 
                  [0]      global_ts   uint64     hardware timestamp [µs]
                  [1:4]    p_W         float32×3  world-frame position [m]
                  [4:8]    R_WB_wxyz   float32×4  quaternion (w, x, y, z)
                  [8:11]   v_W         float32×3  world-frame velocity [m/s]
                  [11:14]  w_B         float32×3  body angular velocity [rad/s]
                  [14:18]  omega_W     float32×4  wheel speeds [rad/s]
 
            ``None`` if :attr:`available` is ``False``.
        """
        if not self.available:
            return None

        buf = self._buf
        while True:
            seq1 = _read_seq(buf)
            if seq1 & 1:
                continue

            data = struct.unpack_from(KALMAN_FMT, buf, SEQ_SIZE)

            seq2 = _read_seq(buf)
            if seq1 == seq2 and not (seq2 & 1):
                return seq2, data

    def read_snapshot_dict(self):
        """
        Read a snapshot and return it as a structured dictionary.
 
        Returns:
            A dict with keys:
 
            - ``"seq"`` (*int*) — seq-lock counter.
            - ``"global_ts"`` (*int*) — hardware timestamp [µs].
            - ``"p_W"`` (*list[float]*) — ``[x, y, z]`` world position [m].
            - ``"R_WB_wxyz"`` (*list[float]*) — ``[w, x, y, z]`` quaternion.
            - ``"v_W"`` (*list[float]*) — ``[vx, vy, vz]`` world velocity [m/s].
            - ``"w_B"`` (*list[float]*) — ``[wx, wy, wz]`` body angular velocity [rad/s].
            - ``"omega_W"`` (*list[float]*) — ``[FL, FR, RL, RR]`` wheel speeds [rad/s].
 
            ``None`` if :attr:`available` is ``False``.
        """
        snap = self.read_snapshot()
        if snap is None:
            return None

        seq, d = snap
        return {
            "seq": seq,
            "global_ts": d[0],
            "p_W": list(d[1:4]),
            "R_WB_wxyz": list(d[4:8]),
            "v_W": list(d[8:11]),
            "w_B": list(d[11:14]),
            "omega_W": list(d[14:18]),
        }


def main():
    """
    Pprint EKF state snapshots at 10 Hz until interrupted.
    """
    RATE = 10
    PERIOD = 1 / RATE
    reader = KalmanShmReader()
    if not reader.available:
        return 1
    try:
        while True:
            snap = reader.read_snapshot_dict()
            if snap is not None:
                print(snap)
            time.sleep(PERIOD)
    finally:
        reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
