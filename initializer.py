from typing import Sequence
from node import Node
import random

class Initializer:

  def init_weights(self, n_in, n_out) -> Sequence[Sequence[Node]]:    
    raise NotImplementedError

  def init_weights_flat(self, n_in):
    raise NotImplementedError

  def init_bias(self, n_out) -> Sequence[Node]:
    raise NotImplementedError


class NormalInitializer(Initializer):

  def __init__(self, mean=0.000001, std=0.000001):
    self.mean = mean
    self.std = std

  def init_weights(self, n_in, n_out):
    return [[Node(random.gauss(self.mean, self.std)) for _ in range(n_out)] for _ in range(n_in)]

  def init_weights_flat(self, n_in):
    return [Node(random.gauss(self.mean, self.std)) for _ in range(n_in)]

  def init_bias(self, n_out):
    return [Node(0.0) for _ in range(n_out)]