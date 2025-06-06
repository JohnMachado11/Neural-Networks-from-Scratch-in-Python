import numpy as np


# Sigmoid activation
class Activation_Sigmoid:
    # Forward pass
    def forward(self, inputs):
        # Save input and calculate / save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    # Backward pass
    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output