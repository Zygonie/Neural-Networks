import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
class Layer:
    def __init__(self, nbNode, dim, mode):
        # Initialize weights to random values
        # nbNode: number of nodes in the layer
        # dim: dimension of the input vector
        if mode=='random':
            self.Weights = np.random.uniform(-1, 1, (nbNode, dim+1))
        elif mode=='ones':
            self.Weights = np.ones((nbNode, dim+1))
        self.Biais = 1
        self.NbNode = nbNode
    
    def Output(self, x):  
        inputVector = np.concatenate(([self.Biais],x))
        linearCombination = self.Weights.dot(inputVector)
        return sigmoid(linearCombination)
    
class NN:
    def __init__(self, networkSize, inputSize, mode='random'):
        # networkSize: an array of length equal to the number of layers. Each coordinate is the number of nodes in the layer
        # inputSize: the dimension of the input vector
        self.Layers = []
        inputDim = inputSize
        for nbNode in networkSize:
            layer = Layer(nbNode, inputDim, mode)
            self.Layers.append(layer)
            inputDim = nbNode
            
    def Output(self, input):
        for layer in self.Layers:
            input = layer.Output(input)
        return input

#*****************************************
# Main called function
#*****************************************
if __name__ == '__main__':
    network=NN(networkSize=np.array([3,2]), inputSize=4, mode='ones')
    print network.Output(np.array([1,2,3,4]))