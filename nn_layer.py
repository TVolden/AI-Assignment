from typing import Sequence
from node import Node

class NNLayer:
    def parameters(self) -> Sequence[Node]:
        return []
    
    def forward(self, inputs: Sequence[Node]) -> Sequence[Node]:
        pass