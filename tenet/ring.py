import numpy as np
from typing import Tuple
from .network import Network, __init_qubit_zero


class Ring(Network):
    """
    A Matrix Product State where the first and last qubits are connected forming a ring.
    """

    def __init__(self, n: int, chi: int):
        assert chi > 2
        Network.__init__(self, n)
        self._tensors = [__init_qubit_zero((2, chi, chi)) for _ in range(n)]

    def distance(self, a: int, b: int) -> int:
        return min(abs(a-b), self.n - abs(a-b))

    def path(self, a: int, b: int):
        assert self.distance(a, b) > 1

        n = self.len()
        clockwise = self.distance(a, (b - 1) % n) > self.distance(a, (b+1) % n)

        order = []
        head = a
        while head != b:
            head = (head + 1 if clockwise else head - 1) % n
            order.append(head)

        order.append(b)
        return order
