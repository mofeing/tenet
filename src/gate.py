from abc import abstractclassmethod
from numpy import ndarray, array, csingle
from math import cos, sin, pi, sqrt
from cmath import rect


class Gate:
    @abstractclassmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractclassmethod
    def mat(self) -> ndarray:
        """
        Returns the matrix representation of the Gate.
        """
        raise NotImplementedError


class I(Gate):
    def name(self) -> str:
        return "I"

    def mat(self) -> ndarray:
        return array([[1, 0], [0, 1]], dtype=csingle)


class X(Gate):
    def name(self) -> str:
        return "X"

    def mat(self) -> ndarray:
        return array([[0, 1], [1, 0]], csingle)


class Y(Gate):
    def name(self) -> str:
        return "Y"

    def mat(self) -> ndarray:
        return array([[0, -1j], [1j, 0]], csingle)


class Z(Gate):
    def name(self) -> str:
        return "Z"

    def mat(self) -> ndarray:
        return array([[1, 0], [0, -1]], csingle)


class S(Gate):
    def name(self) -> str:
        return "S"

    def mat(self) -> ndarray:
        return array([[1, 0], [0, 1j]], csingle)


class Sd(Gate):
    def name(self) -> str:
        return "S dagger"

    def mat(self) -> ndarray:
        return array([[1, 0], [0, -1j]], csingle)


class T(Gate):
    def name(self) -> str:
        return "T"

    def mat(self) -> ndarray:
        return array([[1, 0], [0, rect(1, pi/4)]], csingle)


class Td(Gate):
    def name(self) -> str:
        return "T dagger"

    def mat(self) -> ndarray:
        return array([1, 0], [0, rect(1, -pi/4)], csingle)


class Rx(Gate):
    def name(self) -> str:
        return "Rx"

    def __init__(self, theta: float):
        self.theta = theta

    def mat(self) -> ndarray:
        return array([
            [cos(self.theta/2), -1j * sin(self.theta/2)],
            [-1j * sin(self.theta/2), cos(self.theta/2)]
        ], csingle)


class Ry(Gate):
    def name(self) -> str:
        return "Ry"

    def __init__(self, theta: float):
        self.theta = theta

    def mat(self) -> ndarray:
        return array([
            [cos(self.theta/2), -sin(self.theta/2)],
            [sin(self.theta/2), cos(self.theta/2)]
        ], csingle)


class Rz(Gate):
    def name(self) -> str:
        return "Rz"

    def __init__(self, theta: float):
        self.theta = theta

    def mat(self) -> ndarray:
        return array([
            [rect(1, - self.theta/2), 0],
            [0, rect(1, self.theta/2)]
        ], csingle)


class H(Gate):
    def name(self) -> str:
        return "H"

    def mat(self) -> ndarray:
        return 1/sqrt(2) * array([[1, 1], [1, -1]])


class U3(Gate):
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
        ], csingle)


class Controlled(Gate):
    def name(self) -> str:
        return "C-"  # TODO

    def __init__(self, op: Gate, name=None):
        assert op.mat().shape == (2, 2)
        self.op = op

    def mat(self) -> ndarray:
        return array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, self.op[0, 0], self.op[0, 1]],
            [0, 0, self.op[1, 0], self.op[1, 1]]
        ], csingle)


class CX(Gate):
    def name(self) -> str:
        return "CX"

    def mat(self):
        op = X().mat()
        return array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, op[0, 0], op[0, 1]],
            [0, 0, op[1, 0], op[1, 1]]
        ], csingle)


class CY(Gate):
    def name(self) -> str:
        return "CY"

    def mat(self):
        op = Y().mat()
        return array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, op[0, 0], op[0, 1]],
            [0, 0, op[1, 0], op[1, 1]]
        ], csingle)


class CZ(Gate):
    def name(self) -> str:
        return "CZ"

    def mat(self):
        op = Z().mat()
        return array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, op[0, 0], op[0, 1]],
            [0, 0, op[1, 0], op[1, 1]]
        ], csingle)


class Swap(Gate):
    def name(self) -> str:
        return "SWAP"

    def mat(self) -> ndarray:
        return array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], csingle)
