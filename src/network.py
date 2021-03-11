from abc import abstractmethod
from circuit import Circuit
from gate import Gate, S, Swap
import numpy as np
from numpy.linalg import svd
from typing import Tuple, List
from kernel import apply_op1, apply_op2


class Network(object):
    def __init__(self, n: int):
        assert n > 0
        self._n = n
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
    def path(self, a: int, b: int) -> List[int]:
        """
        Computes the shortest path between qubits `a` and `b`.
        """
        raise NotImplementedError

    @abstractmethod
    def common_idx(self, a: int, b: int) -> Tuple[int, int]:
        """
        Returns the common edge between `a` and `b` in index number.
        """
        raise NotImplementedError

    @abstractmethod
    def amplitude(self, bitstring: str) -> complex:
        """
        Computes the amplitude of the given bitstring.
        """
        raise NotImplementedError

    def apply(self, target, op):
        if isinstance(target, int):
            self.__apply_int(target, op)
        elif isinstance(target, Tuple):
            self.__apply_tuple(target, op)
        else:
            raise NotImplementedError

    def __apply_int(self, target: int, op: Gate):
        """
        Apply a single-qubit operator.
        """
        assert 0 <= target < self.n
        apply_op1(self._tensor[target], op.mat())

    def __apply_tuple(self, target: Tuple[int, int], op: Gate):
        """
        Apply a double-qubit operator.
        """
        a = target[0]
        b = target[1]
        assert 0 <= a < self.n
        assert 0 <= b < self.n

        swaps = []
        if self.distance(a, b) > 1:
            swaps = self.path(a, b)

        # Swap qubits 'till a and b are contiguous
        for c in swaps:
            (idx_a, idx_c) = self.common_idx(a, c)
            apply_op2(
                self._tensor[a], idx_a, self._tensor[c], idx_c, Swap().mat())
            a = c

        # Call kernel
        (idx_a, idx_b) = self.common_idx(a, b)
        apply_op2(
            self._tensor[a], idx_a, self._tensor[b], idx_b, op.mat())

        # Reverse back Swaps
        for c in reversed(swaps):
            (idx_a, idx_c) = self.common_idx(a, c)
            apply_op2(
                self._tensor[a], idx_a, self._tensor[b], idx_b, Swap().mat())
            a = c

    def run(self, circuit: Circuit):
        """
        Applies the circuit by making calls to `apply`.
        """
        assert self.len() == circuit.len()

        for (target, op) in circuit:
            self.apply(target, op)
