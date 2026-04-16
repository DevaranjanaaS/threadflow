import numpy as np

class SGD:
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.lr = lr

    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for key in layer.params:
                    layer.params[key] -= self.lr * layer.grads[key]

class MomentumSGD:
    def __init__(self, layers, lr=0.01, momentum=0.9):
        self.layers = layers
        self.lr = lr
        self.momentum = momentum
        self.velocities = []
        for layer in layers:
            v_layer = {}
            if hasattr(layer, 'params'):
                for key in layer.params:
                    v_layer[key] = np.zeros_like(layer.params[key])
            self.velocities.append(v_layer)

    def step(self):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for key in layer.params:
                    # v = momentum * v - lr * grad
                    self.velocities[i][key] = self.momentum * self.velocities[i][key] - self.lr * layer.grads[key]
                    # w = w + v
                    layer.params[key] += self.velocities[i][key]
