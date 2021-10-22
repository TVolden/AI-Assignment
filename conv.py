from math import prod
import time
from typing import Sequence
from initializer import *
from nn_layer import NNLayer
from node import Node

"""
A basic convolutional neural network, implemented by Thomas Volden.

Sources of inspiration:
- https://www.analyticsvidhya.com/blog/2021/08/beginners-guide-to-convolutional-neural-network-with-implementation-in-python/
- https://en.wikipedia.org/wiki/Convolutional_neural_network
"""
class Conv(NNLayer):
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
        self.weights = initializer.init_weights_flat(pow(filter_size,2) * input_shape[2] * n_filters)
        self.bias = initializer.init_bias(n_filters)

    def parameters(self) -> Sequence[Node]:
      """Returns all the vars of the layer (weights + biases) as a single flat list"""
      output = [w for w in self.weights]
      output.extend(self.bias)
      return output

    def forward(self, inputs: Sequence[Node]) -> Sequence[Node]:
        """ 
        inputs: input values from the three dimentional value map of width, height and channel.

        Check to see if the lenght of inputs are equal to the product of the defined shape (x*y*z)
        
        The strider is automatically set to 1,1 since I don't need to change it.
        """
        assert prod(self.input_shape) == len(inputs), "The input must meet the defined shape"
        output = []
        steps_x = self.input_shape[0] - self.filter_size + 1 # Calculate max number of moves to the right
        steps_y = self.input_shape[1] - self.filter_size + 1 # Calculate max number of moves to the bottom
        sqr_filter = pow(self.filter_size,2)

        for f in range(self.filters): # For each filter bank
            # We going to stride through the whole input (width, height) and all channelse to produce a 2D feature map.
            filter = f * (sqr_filter + self.input_shape[2]) # Skip whole filter banks

            for y in range(steps_y): # For each move to the bottom
                for x in range(steps_x): # For each move to the right
                    val = Node(0)
                    
                    #start_time = time.time() 
                    # Append j,i and k values from filter bank to x,y and all channels
                    for i in range(self.filter_size):
                        row = (i + y) * self.input_shape[0] # Skip input width plus stride y offset for each row
                        for j in range(self.filter_size):
                            filter_row = j * self.filter_size # Skip a filter width for each row
                            col = x + j # Skip stride x offset for each column
                            # For each width and height in the input, we go through all channels, apply weights and append the result
                            for k in range(self.input_shape[2]):
                                filter_chan = k * sqr_filter # Skip a height and width layer for each channel
                                chan = k * self.input_shape[0] * self.input_shape[1] # Skip input width and height for each channel
                                val += self.weights[filter + filter_chan + filter_row + j] * inputs[chan + row + col] # Apply weight
                    
                    #if (time.time() - start_time) > 0.00001:
                    #    print(f"x:{x}, y:{y}, f:{f}")
                    #    print("## TIME %s sec" % ((time.time() - start_time)))

                    # Apply the filter bank bias and parse it through a non-lineary activation function
                    output.append(self.act_fn(val + self.bias[f]))

            

        # The output size should be equal to (x - filter_size + 1) * (y - filter_size + 1) * filters
        assert prod([s - self.filter_size + 1 for s in self.input_shape[:2]]) * self.filters == len(output), "The output must meet the reduced shapes"

        return output 