import numpy as np
# import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))


class Layer:
    def __init__(self, nb_node, dim, mode):
        # Initialize weights to random values
        # nbNode: number of nodes in the layer
        # dim: dimension of the input vector
        if mode == 'random':
            self.Weights = np.random.uniform(-1, 1, (nb_node, dim+1))
        elif mode == 'ones':
            self.Weights = np.ones((nb_node, dim+1))
        self._bias = 1
        self._nb_node = nb_node
    
    def output(self, x):
        input_vector = np.concatenate(([self._bias], x))
        linear_combination = self.Weights.dot(input_vector)
        return sigmoid(linear_combination)


class NN:
    def __init__(self, network_size, input_size, mode='random'):
        # networkSize: an array of length equal to the number of layers.
        # Each coordinate is the number of nodes in the layer
        # inputSize: the dimension of the input vector
        self.Layers = []
        input_dim = input_size
        for nb_node in network_size:
            layer = Layer(nb_node, input_dim, mode)
            self.Layers.append(layer)
            input_dim = nb_node
            
    def output(self, x):
        for layer in self.Layers:
            x = layer.Output(x)
        return x


# *****************************************
# Main called function
# *****************************************
if __name__ == '__main__':
    network = NN(network_size=np.array([3, 2]), input_size=4, mode='ones')
    print network.output(np.array([1, 2, 3, 4]))