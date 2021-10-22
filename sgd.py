
# Stochastic Gradient Descent
from typing import Sequence
from node import Node

class SGD:
  def __init__(self, parameters: Sequence[Node], learning_rate: float):
    self.parameters = parameters
    self.learning_rate = learning_rate

  def zero_grad(self):
    """ Set the gradient to zero for all parameters """
    for parameter in self.parameters:
        parameter.nullify_gradient()

  def step(self):
    """Performs a single step of SGD for each parameter: p = p - learning_rate * grad_p """
    for parameter in self.parameters:
        parameter.value -= self.learning_rate * parameter.grad