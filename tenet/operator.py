from abc import abstractclassmethod
from numpy import ndarray, array, csingle
from math import cos, sin, pi, sqrt
from cmath import rect


class Operator:
    @abstractclassmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractclassmethod
    def mat(self) -> ndarray:
        """
        Returns the matrix representation of the operator.
        """
        raise NotImplementedError


class I(Operator):
    def name(self) -> str:
        return "I"

    def mat(self) -> ndarray:
        return array([1, 0], [0, 1], dtype=csingle)


class X(Operator):
    def name(self) -> str:
        return "X"

    def mat(self) -> ndarray:
        return array([[0, 1], [1, 0]], dtype=csingle)


class Y(Operator):
    def name(self) -> str:
        return "Y"

    def mat(self) -> ndarray:
        return array([[0, -1j], [1j, 0]], dtype=csingle)


class Z(Operator):
    def name(self) -> str:
        return "Z"

    def mat(self) -> ndarray:
        return array([[1, 0], [0, -1]], dtype=csingle)


class S(Operator):
    def name(self) -> str:
        return "S"

    def mat(self) -> ndarray:
        return array([[1, 0], [0, 1j]], dtype=csingle)


class Sd(Operator):
    def name(self) -> str:
        return "S dagger"

    def mat(self) -> ndarray:
        return array([[1, 0], [0, -1j]], dtype=csingle)


class T(Operator):
    def name(self) -> str:
        return "T"

    def mat(self) -> ndarray:
        return array([[1, 0], [0, rect(1, pi/4)]], dtype=csingle)


class Td(Operator):
    def name(self) -> str:
        return "T dagger"

    def mat(self) -> ndarray:
        return array([1, 0], [0, rect(1, -pi/4)], dtype=csingle)


class Rx(Operator):
    def name(self) -> str:
        return "Rx"

    def __init__(self, theta: float):
        self.theta = theta

    def mat(self) -> ndarray:
        return array([
            [cos(self.theta/2), -1j * sin(self.theta/2)],
            [-1j * sin(self.theta/2), cos(self.theta/2)]
        ], dtype=csingle)


class Ry(Operator):
    def name(self) -> str:
        return "Ry"

    def __init__(self, theta: float):
        self.theta = theta

    def mat(self) -> ndarray:
        return array([
            [cos(self.theta/2), -sin(self.theta/2)],
            [sin(self.theta/2), cos(self.theta/2)]
        ], dtype=csingle)


class Rz(Operator):
    def name(self) -> str:
        return "Rz"

    def __init__(self, theta: float):
        self.theta = theta

    def mat(self) -> ndarray:
        return array([
            [rect(1, - self.theta/2), 0],
            [0, rect(1, self.theta/2)]
        ], dtype=csingle)


class H(Operator):
    def name(self) -> str:
        return "H"

    def mat(self) -> ndarray:
        return 1/sqrt(2) * array([[1, 1], [1, -1]])


class U3(Operator):
    def name(self) -> str:
        return "U3"

    def __init__(self, theta: float, phi: float, lambd: float):
        assert 0 <= theta <= pi
        assert 0 <= phi <= 2*pi
        assert 0 <= lambd <= 2*pi
        self.theta = theta
        self.phi = phi
        self.lambd = lambd

    def mat(self):
        return array([
            [cos(self.theta/2), - rect(1, self.lambd) * sin(self.theta/2)],
            [
                rect(1, self.phi) * sin(self.theta/2),
                rect(1, self.lambd + self.phi) * cos(self.theta/2)
            ]
        ], dtype=csingle)


class Controlled(Operator):
    def name(self) -> str:
        return "C-"  # TODO

    def __init__(self, op: Operator):
        assert op.size == (2, 2)
        self.op = op

    def mat(self) -> ndarray:
        return array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, self.op[1, 1], self.op[1, 2]],
            [0, 0, self.op[2, 1], self.op[2, 2]]
        ], dtype=csingle)


def CX(Operator):
    def name(self) -> str:
        return "CX"

    def __init__(self):
        return Controlled(X())


def CY(Operator):
    def name(self) -> str:
        return "CY"

    def __init__(self):
        return Controlled(Y())


def CZ(Operator):
    def name(self) -> str:
        return "CZ"

    def __init__(self):
        return Controlled(Z())


class Swap(Operator):
    def name(self) -> str:
        return "SWAP"

    def mat(self) -> ndarray:
        return array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=csingle)
