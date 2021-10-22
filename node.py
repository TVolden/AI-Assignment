from math import e, tanh
from typing import Sequence, Tuple, Union

from numpy.lib.function_base import gradient

class Node:
    def __init__(self, value: float, parents = None):
        self.value = value
        self.grad = 0.0
        if parents is None:
            parents = []
        self.parents = parents

    def nullify_gradient(self):
        self.grad = 0

    def __repr__(self):
        return "Node (value=%.4f, gradient=%.4f)" % (self.value, self.grad)

    def backprop(self, df_dnode):
        self.grad += df_dnode
        for parent, gradient in self.parents:
            parent.backprop(gradient * df_dnode)

    def backward(self):
        """
        Computes the gradient of every (upstream) node in the computational graph w.r.t. node.
        """
        self.backprop(1.0)  # The gradient of a node w.r.t. itself is 1 by definition.

    def __add__(self, other):
        return Node(self.value + other.value, [(self, 1.0), (other, 1.0)])
    
    def __mul__(self, other):
        return Node(self.value * other.value, [(self, other.value), (other, self.value)])
    
    def __pow__(self, power: Union[float, int]):
        assert type(power) in {float, int}, "power must be float or int"
        return Node(self.value ** power, [(self, power * self.value ** (power - 1))])
    
    def __neg__(self):
        return Node(-1.0) * self

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def tanh(self):
        return Node(tanh(self.value), [(self, 1 - tanh(self.value) ** 2)])
    
    def relu(self):
        return Node(self.value if self.value > 0.0 else 0.0, [(self, 1.0 if self.value > 0.0 else 0.0)])

    def exp(self):
        return Node(e ** self.value, [(self, e ** self.value)])

def square(node: Node) -> Node:
    return Node(node.value**2, [(node, 2*node.value)])

def plus(a: Node, b: Node) -> Node:
    return Node(a.value + b.value, [(a, a.value), (b, b.value)])

def multiply(a: Node, b: Node) -> Node:
    return Node(a.value*b.value, [(a,a.value), (b, b.value)])