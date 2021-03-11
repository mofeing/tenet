from pycompss.api.task import task
from pycompss.api.parameter import INOUT, IN, OUT
import numpy as np


@task(shape=IN, returns=np.ndarray)
def init_qubit_zero(shape) -> np.ndarray:
    print("[TENET][DEBUG] Call to init_qubit_zero: shape=" + str(shape))

    arr = np.zeros(shape, dtype=np.csingle)
    arr.flat[0] = 1
    return arr


@task(shape=IN)
def init_qubit_one(shape) -> np.ndarray:
    print("[TENET][DEBUG] Call to init_qubit_one: shape=" + str(shape))

    arr = np.zeros(shape, dtype=np.csingle)
    arr.flat[1] = 1
    return arr


@task(psi=INOUT, op=IN)
def apply_op1(psi, op):
    print("[TENET][DEBUG] Call to apply_op1\n\tpsi.shape=" + str(psi.shape))

    orig_shape = psi.shape
    psi = psi.reshape((2, -1))
    psi = np.dot(op, psi)
    psi.reshape(orig_shape)


@task(a=INOUT, b=INOUT, op=IN)
def apply_op2(a: np.ndarray, idx_a: int, b: np.ndarray, idx_b: int, op: np.ndarray):
    print("[TENET][DEBUG] Call to apply_op2\n\ta.shape=" + str(a.shape) + "\n\tb.shape=" +
          str(b.shape) + "\n\tida=" + str(idx_a) + "\n\tidb=" + str(idx_b))

    # Contract tensors
    op = op.reshape((2, 2, 2, 2))
    c = np.tensordot(a, b, axes=(idx_a, idx_b))
    c = np.tensordot(c, op, axes=[(0, 2), (0, 1)])

    # SVD
    c = c.transpose(0, 2, 1, 3)
    chi = c.shape[0]
    c = c.reshape(chi*2, -1)
    (u, s, v) = np.linalg.svd(c, compute_uv=True)

    b = v
    a = u * s
