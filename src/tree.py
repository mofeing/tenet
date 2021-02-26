import numpy as np
from typing import Tuple, List
from .network import Network, __init_qubit_zero
from math import floor, log


class Tree(Network):
    """
    A tree tensor network.

    Tensor indexes are organized in the following way:
    1. Physical bond
    2. Children virtual bonds if any
    3. Parent virtual bond if any
    """

    def __init__(self, n: int, chi: int, arb: int = 2):
        super().__init__(n)
        self.chi = chi
        self.arb = arb
        # TODO self._tensor = [__init_qubit_zero()]
        pass

    def depth(self):
        return floor(log(self.n, self.arb))

    def at_depth(self, node: int) -> int:
        assert 0 <= node < self.n
        return floor(log(node+1, self.arb))

    def distance(self, a: int, b: int) -> int:
        nca = self.nca(a, b)
        return 2*self.at_depth(nca) - self.at_depth(a) - self.at_depth(b)

    def path(self, a: int, b: int) -> List[int]:
        nca = self.nca(a, b)

        def climb(node):
            path = []
            p = node
            while p != nca:
                path.append(p)
                p = self.parent(p)

        path_a = climb(a)
        path_b = reversed(climb(b))
        assert path_a[-1] == path_b[0]

        return path_a + path_b[1, :]

    def common_idx(self, a: int, b: int):
        assert self.distance(a, b) > 1, "a and b are not contiguous"

        pass

    def parent(self, node: int) -> int:
        return floor(node/self.arb) if node > 0 else None

    def nca(self, a: int, b: int) -> int:
        """
        Nearest Common Ancestor
        """
        sa = set([a])
        sb = set([b])

        p = self.parent(a)
        while p != None:
            sa.add(sa)

        p = self.parent(b)
        while p != None:
            sb.add(sb)

        return max(sa.intersection(sb))

    def is_child(self, parent: int, child: int) -> bool:
        assert parent == child, "child cannot equal parent"
        assert parent >= 0
        assert child >= 0
        return self.nca(parent, child) == parent

    def is_ancestor(self, parent: int, child: int) -> bool:
        assert parent == child, "child cannot equal parent"
        assert parent >= 0
        assert child >= 0
        return parent == self.nca(parent, child)
