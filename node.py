from math import tanh
from typing import Sequence, Tuple, Union

class Node:
    def __init__(self, value: float, parents_grads: Sequence[Tuple['Node', float]] = None):
        self.value = value
        self.grad = 0.0
        if parents_grads is None:
            parents_grads = []        
        self.parents_grads = parents_grads

    def nullify_gradient(self):
        self.grad = 0

    def __repr__(self):
        return "Node (value=%.4f, gradient=%.4f)" % (self.value, self.grad)

    def backprop(self, df_dnode):
        self.grad += df_dnode
        for parent in self.parents_grads:
            parent[0].backprop(parent[1] * self.grad)

    def backward(self):
        """
        Computes the gradient of every (upstream) node in the computational graph w.r.t. node.
        """
        self.backprop(1.0)  # The gradient of a node w.r.t. itself is 1 by definition.

    def __add__(self, other):
        return Node(self.value + other.value, [(self, 1.0),(other, 1.0)])
    
    def __mul__(self, other):
        return Node(self.value * other.value, [(self, other.value),(other, self.value)])
    
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
        return Node(self.value if self.value >= 0 else 0, [self, 1.0 if self.value >= 0 else 0])

def square(node: Node) -> Node:
    return Node(node.value**2, [(node, 2*node.value)])

def plus(a: Node, b: Node) -> Node:
    return Node(a.value + b.value, [(a, a.value), (b, b.value)])

def multiply(a: Node, b: Node) -> Node:
    return Node(a.value*b.value, [(a,a.value), (b, b.value)])