import time
from typing import Sequence
from nn_layer import NNLayer
from node import Node

class MultiLayeredNetwork:
    def __init__(self, layers:Sequence[NNLayer]) -> None:
        self.layers = layers
    
    def parameters(self) -> Sequence[Node]:
        """
        Returns all the parameters of the layers as a flat list
        """
        output = []
        for layer in self.layers:
            output.extend(layer.parameters()) # Get a flat list of parameters from each layer
        return output

    def forward(self, input: Sequence[Node]) -> Sequence[Node]:
        """
        Computes the forward pass of the multi layered network.
        """
        x = input
        for layer in self.layers:
            x = layer.forward(x) # The output is passed on as input for the next layer
                    
        # Return the last output
        return x