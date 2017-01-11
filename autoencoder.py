import numpy as np
import mnist as mn

class Layer:
    def __init__(self, n_in, width, include_bias=True):
        self.width = width + 1 if include_bias else width
        self.values = np.ones(self.width)
        self.weights = np.random.rand(width, n_in) / 10.0
        self.has_bias = include_bias

    def evaluate(self, s):
        res = np.maximum(np.matmul(self.weights, s), 0.0)
        if self.has_bias:
            return np.append(res, [1.0]) # optimize?
        else:
            return res
        
class Net:
    def __init__(self, input_n):
        self.input_n = input_n
        self.layers = []
    
    def add_layer(self, width, include_bias=True):
        if len(self.layers) == 0:
            layer_inputs = self.input_n + 1 # Add 1 for bias
        else:
            layer_inputs = self.layers[-1].width
        layer = Layer(layer_inputs, width, include_bias)
        self.layers.append(layer)
    
    def evaluate(self, s):
        """Takes an numpy array of floats of length input_n produces one of length output_n"""
        prev_out = np.append(s, [1.0]) # add bias in input layer
        for layer in self.layers:
            prev_out = layer.evaluate(prev_out)
        
        return prev_out
            
    def train(samples):
        for s in samples:
            pass

trn_samp = list(mn.read(dataset='training', path='./mnist'))

print("Loading...")
print(len(trn_samp[0][1][0]))
mn.show(trn_samp[1][1])

in_size = 28 * 28

net = Net(in_size)
net.add_layer(25)
net.add_layer(in_size, include_bias=False) # output layer

output = net.evaluate(mn.to_vec(trn_samp[1][1]))
mn.show(mn.from_vec(output))