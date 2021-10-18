from typing import Sequence
from initializer import *
from node import Node
from math import prod

"""
A basic convolutional neural network, implemented by Thomas Volden.

Sources of inspiration:
- https://www.analyticsvidhya.com/blog/2021/08/beginners-guide-to-convolutional-neural-network-with-implementation-in-python/
- https://en.wikipedia.org/wiki/Convolutional_neural_network
"""
class Conv:
    def __init__(self, n_filters:int, filter_size:int, act_fn, input_shape:tuple[int, int, int], initializer: Initializer = NormalInitializer()):
        """
        n_filters: Number of output filters
        filter_size: The shared height and width of the filter
        act_fn: The activation function
        input_shape: The height, width and channels of the input
        """
        self.filters = n_filters
        self.filter_size = filter_size
        self.act_fn = act_fn
        self.input_shape = input_shape
        self.weights = initializer.init_weights(filter_size^2 * n_filters)
        self.bias = initializer.init_bias(n_filters)

    def forward(self, inputs: Sequence[Node]) -> Sequence[Node]:
        """ 
        inputs: input values from the two dimentional value map and number of channels

        Check to see if the lenght of inputs are equal to the product of the defined shape (x*y*c)
        
        The strider is automatically set to 1,1 since I don't need to change it.
        """
        assert prod(self.input_shape) == len(inputs), "The input must meet the defined shape"
        output = []
        steps_x = self.input_shape[0] - self.filter_size + 1 # Calculate max number of moves to the left
        steps_y = self.input_shape[1] - self.filter_size + 1 # Calculate max number of moves to the bottom
        
        for f in range(self.filters): # For each filter bank
            for _ in range(self.input_shape[2]): # For each channel
                for y in range(steps_y): # For each move to the bottom
                    for x in range(steps_x): # For each move to the left
                        val = Node(0)
                        # Append j,i value from filter bank to x,y value
                        for i in range(self.filter_size):
                            for j in range(self.filter_size):
                                filter = f * self.filter_size^2 # Skip whole filters banks
                                filter_row = i * self.filter_size # Skip a filder width for each row
                                row = (i+y)*self.input_shape[0] # Skip input width plus stride y offset for each row
                                col = x + j # Skip stride x offset for each column
                                val += self.weights[filter + filter_row + j] * inputs[row + col]
                        
                        # Apply the filter bank bias and parse it through a non-lineary activation function
                        output.append(self.act_fn(val + self.bias[f]))
        return output