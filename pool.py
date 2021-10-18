from typing import Sequence
from node import Node
from math import prod

"""
A basic pooling layer, implemented by Thomas Volden.

Sources of inspiration:
- https://www.analyticsvidhya.com/blog/2021/08/beginners-guide-to-convolutional-neural-network-with-implementation-in-python/
- https://en.wikipedia.org/wiki/Convolutional_neural_network
"""
class PoolingStrategy:
    def compute(values:Sequence[Node]) -> Node:
        raise NotImplementedError

class MaxPooling(PoolingStrategy):
    def compute(values: Sequence[Node]) -> Node:
        return max(values) # Return the maximum number of the pool

class Pool:
    def __init__(self, stride:int, shape:int, input_shape:tuple[int, int, int], strategy:PoolingStrategy = MaxPooling()):
        """
        stride: The number of positions to move for each iteration (both left and down)
        shape: The size of the pool (both width and height)
        input_shape: The height, width and number of channels of the input
        stategy: The stategy used for pooling. Standard set to max pooling
        """
        self.stride = stride
        self.shape = shape
        self.input_shape = input_shape
        self.strategy = strategy

    def pool(self, inputs: Sequence[Node]) -> Sequence[Node]:
        """ 
        inputs: input values from the two dimentional value map and number of channels

        Check to see if the lenght of inputs are equal to the product of the defined shape (x*y*c)
        """
        assert prod(self.input_shape) == len(inputs), "The input must meet the defined shape"
        output = []
        steps_x = int((self.input_shape[0] - self.filter_size) / self.stride) + 1 # Calculate max number of moves to the left giving the strider value
        steps_y = int((self.input_shape[1] - self.filter_size) / self.stride) + 1 # Calculate max number of moves to the bottom giving the strider value
        
        for c in range(self.input_shape[2]): # For each channel
            for y in range(steps_y): # For each move to the bottom
                for x in range(steps_x): # For each move to the left
                    # Append j,i value from filter bank to x,y value
                    values = []
                    for i in range(self.shape):
                        for j in range(self.shape):
                            channel = c * self.input_shape[0] * self.input_shape[1] # Skip whole areas (height*width)
                            stride_y = y * self.stride # Calculate the stride offset for the y direction
                            stride_x = x * self.stride # Calculate the stride offset for the x direction
                            row = (i + stride_y) * self.input_shape[0] # Jump over a width for each row plus y stride offset
                            col = stride_x + j # The column plus x stride offset
                            values.append(inputs[channel + row + col])
                    # Parse the value through our pooling strategy to get one value
                    output.append(self.strategy.compute(values))
        return output