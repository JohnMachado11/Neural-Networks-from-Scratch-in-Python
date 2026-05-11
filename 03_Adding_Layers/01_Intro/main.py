import numpy as np

"""
Input Layer   ---   Hidden Layer 1   ---   Hidden Layer 2

 4 Inputs     ->     3 Neurons        ->     3 Neurons
"""

# Batch of 3 samples, each with 4 input features
inputs = [[1.0, 2.0, 3.0, 2.5], # Sample 1: 4 features fed to all 3 neurons in Hidden Layer 1
          [2.0, 5.0, -1.0, 2.0], # Sample 2: 4 features fed to all 3 neurons in Hidden Layer 1
          [-1.5, 2.7, 3.3, -0.8]] # Sample 3: 4 features fed to all 3 neurons in Hidden Layer 1

weights = [[0.2, 0.8, -0.5, 1.0], # Neuron 1 in Hidden Layer 1 weights 
           [0.5, -0.91, 0.26, -0.5], # Neuron 2 in Hidden Layer 1 weights
           [-0.26, -0.27, 0.17, 0.87]] # Neuron 3 in Hidden Layer 1 weights

biases = [2, 3, 0.5] # Neurons 1 -> 3 in Hidden Layer 1 biases

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
print(layer1_outputs) 

"""
Hidden Layer 1 Outputs (one row per sample, one column per neuron):

[[ 4.8    1.21   2.385] # Sample 1: data coming from the 3 neurons in Hidden Layer 1, fed to each of the 3 neurons in Hidden Layer 2
[ 8.9   -1.81   0.2  ] # Sample 2: data coming from the 3 neurons in Hidden Layer 1, fed to each of the 3 neurons in Hidden Layer 2
[ 1.41   1.051  0.026]] # Sample 3: data coming from the 3 neurons in Hidden Layer 1, fed to each of the 3 neurons in Hidden Layer 2
"""

weights2 = [[0.1, -0.14, 0.5], # Neuron 1 in Hidden Layer 2 weights 
            [-0.5, 0.12, -0.33], # Neuron 2 in Hidden Layer 2 weights
            [-0.44, 0.73, -0.13]] # Neuron 3 in Hidden Layer 2 weights

biases2 = [-1, 2, -0.5] # Neurons 1 -> 3 in Hidden Layer 2 biases

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)

""" 
Hidden Layer 2 Outputs:

[[ 0.5031  -1.04185 -2.03875]
[ 0.2434  -2.7332  -5.7633 ]
[-0.99314  1.41254 -0.35655]]
"""