from pycompss.api.task import task
from pycompss.api.parameter import INOUT, IN, OUT
import numpy as np


@task(shape=IN, returns=np.ndarray)
def init_qubit_zero(shape) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.csingle)
    arr.flat[0] = 1
    return arr


@task(shape=IN)
def init_qubit_one(shape) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.csingle)
    arr.flat[1] = 1
    return arr


@task(psi=INOUT, op=IN)
def apply_op1(psi, op):
    orig_shape = psi.shape
    psi = psi.reshape((2, -1))
    psi = np.dot(op, psi)
    psi.reshape(orig_shape)


@task(a=INOUT, b=INOUT, op=IN)
def apply_op2(a: np.ndarray, idx_a: int, b: np.ndarray, idx_b: int, op: np.ndarray):
    # Contract tensors
    op = op.reshape((2, 2, 2, 2))
    c = np.tensordot(a, b, axes=(idx_a, idx_b))
    c = np.tensordot(c, op, axes=[(1, 2), (0, 1)])

    # SVD
    c = c.transpose(0, 2, 1, 3)
    chi = c.shape[0]
    c = c.reshape(chi*2, -1)
    (u, s, v) = np.linalg.svd(c, compute_uv=True)

    b = v
    a = u * s