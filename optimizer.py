import numpy as np
from mpi4py import MPI

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
                    self.velocities[i][key] = self.momentum * self.velocities[i][key] - self.lr * layer.grads[key]
                    layer.params[key] += self.velocities[i][key]

class QuantumOptimizer:
    """
    Standout Feature: Distributed signSGD with Majority Voting.
    Reduces communication payload and provides robustness via voting logic.
    """
    def __init__(self, layers, lr=0.001):
        self.layers = layers
        self.lr = lr

    def step(self, comm, size):
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for key in layer.params:
                    # 1. Compute local gradient sign
                    local_sign = np.sign(layer.grads[key]).astype(np.int32)
                    
                    # 2. Majority Vote: Sum signs across all ranks
                    global_votes = np.zeros_like(local_sign, dtype=np.int32)
                    comm.Allreduce(local_sign, global_votes, op=MPI.SUM)
                    
                    # 3. Update: If most ranks say "Go right", we go right.
                    # We move by a fixed step size (LR) in the direction of the majority.
                    layer.params[key] -= self.lr * np.sign(global_votes)
