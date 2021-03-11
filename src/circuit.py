from typing import Union, Tuple
import networkx as nx
from uuid import uuid4
from gate import Gate, I


class Circuit(object):
    """
    A graph representing a quantum circuit.
    """

    def __init__(self, n: int):
        assert n > 0
        self.__n = n
        aux = [{'id': uuid4(), 'target': i, 'operator': I()}
               for i in range(n)]
        self.__tail = [i['id'] for i in aux]
        self.__head = self.__tail
        self.__graph = nx.DiGraph()
        for obj in aux:
            self.__graph.add_node(obj['id'], attr_dict={
                                  'target': obj['target'], 'operator': obj['operator']})

    @property
    def graph(self) -> nx.DiGraph:
        return self.__graph

    def len(self) -> int:
        return self.__n

    @property
    def n(self) -> int:
        return self.__n

    def add_gate(self, target, op):
        if isinstance(target, int):
            self.__add_gate_int(target, op)
        elif isinstance(target, Tuple):
            self.__add_gate_tuple(target, op)
        else:
            raise NotImplementedError

    def __add_gate_int(self, target: int, op: Gate):
        assert op.mat().shape == (2, 2)
        assert 0 <= target < self.n

        uuid_from = self.__head[target]
        uuid_to = uuid4()
        self.__graph.add_node(uuid_to, attr_dict={
                              'target': target, 'operator': op})
        self.__graph.add_edge(uuid_from, uuid_to)
        self.__head[target] = uuid_to

    def __add_gate_tuple(self, target: Tuple[int, int], op: Gate):
        assert op.mat().shape == (4, 4)
        assert 0 <= target[0] < self.n
        assert 0 <= target[1] < self.n

        a = self.__head[target[0]]
        b = self.__head[target[1]]

        uuid_to = uuid4()
        self.__graph.add_node(uuid_to, attr_dict={
                              'target': target, 'operator': op})
        self.__graph.add_edge(a, uuid_to)
        self.__graph.add_edge(b, uuid_to)

        self.__head[target[0]] = uuid_to
        self.__head[target[1]] = uuid_to

    def depth(self) -> int:
        return nx.algorithms.dag.dag_longest_path_length(self.__graph)

    def __iter__(self):
        self.__it = nx.topological_sort(self.__graph)
        return self

    def __next__(self):
        node = next(self.__it)
        attr = self.__graph.nodes[node]['attr_dict']
        return (attr['target'], attr['operator'])
