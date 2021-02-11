from abc import abstractmethod
from .circuit import Circuit
from pycompss.api.task import task
from pycompss.api.parameter import INOUT, IN, OUT
from .operator import Operator, S
import numpy as np
from numpy.linalg import svd
from functools import singledispatchmethod
from typing import Tuple


class Network:
    def __init__(self, n: int):
        assert n > 0
        self._n = 0
        self._tensor = []

    def len(self) -> int:
        """
        Returns the number of qubits of the qubit.
        """
        return self._n

    @property
    def n(self) -> int:
        return self._n

    @abstractmethod
    def distance(self, a: int, b: int) -> int:
        """
        Computes the topological distance between any two qubits `a` and `b`.
        """
        raise NotImplementedError

    @abstractmethod
    def swap_path(self, a: int, b: int):
        """
        Computes the shortest ordering of SWAP operations such applying will make `a` and `b` qubits neighbours topologically.

        `b` is the qubit target of the SWAPs.
        """
        raise NotImplementedError

    @abstractmethod
    def amplitude(self, bitstring: str) -> complex:
        """
        Computes the amplitude of the given bitstring.
        """
        raise NotImplementedError

    @singledispatchmethod
    def apply(self, target, op):
        raise NotImplementedError

    @apply.register
    def _(self, target: int, op: Operator):
        """
        Apply a single-qubit operator.
        """
        assert 0 <= target < self.n

        __apply_op1(self._tensor[target], op)

    @apply.register
    def _(self, target: Tuple[int, int], op: Operator):
        """
        Apply a double-qubit operator.
        """
        a = target[0]
        b = target[1]
        assert 0 <= a < self.n
        assert 0 <= b < self.n

        # TODO permute op?
        __apply_op2(self._tensor[a], , self._tensor[b], , op)

    def run(self, circuit: Circuit):
        """
        Applies the circuit by making calls to `apply`.
        """
        assert self.len() == circuit.len()

        for (target, op) in circuit:
            self.apply(target, op)


@task(shape=IN, returns=np.ndarray)
def __init_qubit_zero(shape) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.csingle)
    arr[0, :] = 1
    return arr


@task(psi=INOUT, op=IN)
def __apply_op1(psi: np.ndarray, op: np.ndarray):
    orig_shape = psi.shape
    psi.reshape((2, -1))
    psi = op * psi
    psi.reshape(orig_shape)


@task(a=INOUT, b=INOUT, op=IN)
def __apply_op2(a: np.ndarray, idx_a: int, b: np.ndarray, idx_b: int, op: np.ndarray):
    # TODO contract tensors
    c = np.einsum()

    # TODO contract operator
    c = np.einsum()

    # TODO svd
    (u, s, v) = svd(c, compute_uv=True)

    a = u * s
    b = v
