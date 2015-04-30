import numpy as np
import time
import matplotlib.pyplot as plt


def sigmoid(x_input):
    return 1 / (1 + np.exp(-x_input))


def circle(x_in):
    return np.sqrt(x_in[0, :]**2 + x_in[1, :]**2)


class Layer:
    def __init__(self, nb_input, nb_output, mode):
        # Initialize weights to random values
        # nbNode: number of nodes in the layer
        # dim: dimension of the input vector
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.bias = []
        if mode == 'random':
            self.weights = 20 * np.random.uniform(-1, 1, (nb_output, nb_input+1))
        elif mode == 'ones':
            self.weights = np.ones((nb_output, nb_input+1))
    
    def output(self, x_input, one_dimension=False):
        x_input = np.array(x_input)
        if x_input.ndim == 1 and not one_dimension:
            x_input = x_input.reshape(-1, 1)
        self.bias = np.ones((1, x_input.shape[1]))
        input_vector = np.concatenate((self.bias, x_input))
        a = self.weights.dot(input_vector)
        z = sigmoid(a)
        return z


class MLP:
    def __init__(self, topology, mode='random'):
        # topology: an array of length equal to the number of layers.
        # Each coordinate is the number of nodes in the layer
        self.layers = []
        self.layer_outputs = []
        self.error = []
        self.learning_rate = None
        self.one_dimension = topology[0] == 1
        for input_dim, output_dim in zip(topology[:-1], topology[1:]):
            self.layers.append(Layer(input_dim, output_dim, mode))

    def train(self, x_train, y_train, learning_rate=0.2):
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        if x_train.ndim == 1 and not self.one_dimension:
            x_train = x_train.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(1, -1)
        self.learning_rate = learning_rate
        self.forward_propagation(x_train)
        self.back_propagation(y_train)
        self.update_weights()

    def forward_propagation(self, x_fw):
        x_fw = np.array(x_fw)
        self.layer_outputs = [x_fw]
        self.layers[0].z = x_fw
        for layer in self.layers:
            self.layer_outputs.append(layer.output(self.layer_outputs[-1], self.one_dimension))
        return self.layer_outputs[-1]

    def back_propagation(self, y_bw):
        self.error = [np.concatenate((np.zeros((1, y_bw.shape[1])), self.layer_outputs[-1] - y_bw))]
        for layer, layer_output in zip(reversed(self.layers), reversed(self.layer_outputs[:-1])):
            layer_output = np.concatenate((layer.bias, layer_output))
            next_error = self.error[-1][1:]
            temp = layer.weights.T.dot(next_error) * layer_output * (1-layer_output)
            self.error.append(temp)
        self.error.reverse()

    def update_weights(self):
        for layer, error, layer_output in zip(self.layers, self.error[1:], self.layer_outputs[:-1]):
            error = error[1:]
            layer_output = np.concatenate((layer.bias, layer_output))
            layer.weights -= self.learning_rate * error.dot(layer_output.T) \
                             + 0.01 * layer.weights/error.shape[1]

    def output(self, x_input):
        a = np.array(x_input)
        if a.ndim == 1 and not self.one_dimension:
            a = a.reshape(-1, 1)
        for layer in self.layers:
            a = layer.output(a)
        return a


def test_function(x_in):
    # return sigmoid(x_in)
    # return x_in**3
    return np.sin(x_in)


# *****************************************
# Main called function
# *****************************************
if __name__ == '__main__':
    # Simulation d'une fonction x^2
    np.random.seed(100)
    network = MLP(topology=[1, 128, 128, 128, 128, 1])
    x = np.arange(-10, 10.1, 0.25)
    x = x.reshape(1, x.shape[0])
    y = test_function(x)
    y_min = np.min(y)
    y_max = np.max(y)
    y_to_fit = 0.9 * (y - y_min) / (y_max - y_min)
    plt.plot(x.flatten(), y.flatten(), '.r')
    plt.ion()
    plt.ylim((y_min-0.1, y_max+0.1))
    plt.show(False)  # pas bloquant
    for n in xrange(int(1e5)):
        network.train(x, y_to_fit, learning_rate=0.0005)
        if n > 1e4:
            fit = [(network.output(i)[0, 0] * (y_max - y_min) + y_min) / 0.9 for i in x.T]
            plt.clf()
            plt.title('GO!')
            plt.plot(x.flatten(), y.flatten(), '.r')
            plt.plot(x.flatten(), fit)
            plt.draw()
            time.sleep(0.001)
    print 'Fonction square'
    for i in range(5):
        res = fit[i]
        print '{}\t{:.4f}\texpected {:.4f}'.format(x[0, i], res, y[i])
    # Simulation d'une fonction circle
    network = MLP(topology=[2, 3, 1])
    np.random.seed(100)
    x = np.random.uniform(-1, 1, (2, 10))
    y = circle(x)
    y_min = np.min(y)
    y_max = np.max(y)
    y = 0.9 * (y - y_min) / (y_max - y_min)
    for n in xrange(int(1e3)):
        network.train(x, y, learning_rate=0.9)
    print 'Fonction circle'
    for i in x.T:
        res = (network.output(i)[0, 0] * (y_max - y_min) + y_min) / 0.9
        print '{}\t{:.4f}\texpected {:.4f}'.format(i, res, circle(i.reshape(-1, 1))[0])
    # Simulation d'une fonction XOR
    network = MLP(topology=[2, 2, 4, 1])
    X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    y = np.array([0, 1, 1, 0])
    for n in xrange(int(1e3)):
        network.train(X, y, learning_rate=0.9)
    print 'Fonction XOR'
    print 'Input\tOutput\tQuantized'
    for i in [[0, 0], [1, 0], [0, 1], [1, 1]]:
        print '{}\t{:.4f}\t{}'.format(i, network.output(i)[0, 0], 1*(network.output(i)[0] > .5))
    # Simulation d'une fonction OR
    network = MLP(topology=[2, 2, 4, 1])
    X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    y = np.array([0, 1, 1, 1])
    for n in xrange(int(1e3)):
        network.train(X, y, learning_rate=0.9)
    print 'Fonction OR'
    print 'Input\tOutput\tQuantized'
    for i in [[0, 0], [1, 0], [0, 1], [1, 1]]:
        print '{}\t{:.4f}\t{}'.format(i, network.output(i)[0, 0], 1*(network.output(i)[0] > .5))