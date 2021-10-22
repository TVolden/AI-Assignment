from initializer import *
from nn_layer import NNLayer
from node import Node

class DenseLayer(NNLayer):
    def __init__(self, n_in: int, n_out: int, act_fn, initializer: Initializer = NormalInitializer()):
        """
          n_in: the number of inputs to the layer
          n_out: the number of output neurons in the layer
          act_fn: the non-linear activation function for each neuron
          initializer: The initializer to use to initialize the weights and biases
        """
        self.weights = initializer.init_weights(n_in, n_out)
        self.bias = initializer.init_bias(n_out)
        self.n_out = n_out
        self.n_in = n_in
        self.act_fn = act_fn
    
    def __repr__(self):    
        return 'Weights: ' + repr(self.weights) + ' Biases: ' + repr(self.bias)

    def parameters(self) -> Sequence[Node]:
      """Returns all the vars of the layer (weights + biases) as a single flat list"""
      output = []
      for i in range(self.n_in):
        for j in range(self.n_out):
          output.append(self.weights[i][j])
      output.extend(self.bias)
      return output
      
    def forward(self, inputs: Sequence[Node]) -> Sequence[Node]:
        """ 
        inputs: A n_in length vector of Var's corresponding to the previous layer outputs or the data if it's the first layer.

        Computes the forward pass of the dense layer: For each output neuron, j, it computes: act_fn(weights[i][j]*inputs[i] + bias[j])
        Returns a vector of Vars that is n_out long.
        """
        assert len(self.weights) == len(inputs), "weights and inputs must match in first dimension"        
        output = []
        for j in range(self.n_out):
          val = Node(0)
          for i in range(len(inputs)):
            val += self.weights[i][j] * inputs[i]
          output.append(self.act_fn(val + self.bias[j]))
        return output