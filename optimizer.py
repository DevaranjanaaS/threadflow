import numpy as np
from mpi4py import MPI

class SGD:
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.lr = lr

    def zero_grad(self):
        for layer in self.layers:
            layer.grads = {}

    def step(self):
        """Standard Step used for Sequential or Accumulation modes."""
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for key in layer.params:
                    if key in layer.grads:
                        layer.params[key] -= self.lr * layer.grads[key]

class OverlapOptimizer:
    """
    ELITE FEATURE: Non-blocking Gradient Synchronization.
    Starts Allreduce in the background as soon as each layer's 
    backward pass is complete.
    """
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.lr = lr
        self.handles = []
        self.sync_grads = {}

    def start_sync(self, layer_idx, layer, comm):
        if not hasattr(layer, 'params'): return
        
        if layer_idx not in self.sync_grads:
            self.sync_grads[layer_idx] = {}

        for key in layer.params:
            if key not in self.sync_grads[layer_idx]:
                self.sync_grads[layer_idx][key] = np.zeros_like(layer.grads[key])
            
            # Use Non-blocking Iallreduce
            handle = comm.Iallreduce(layer.grads[key], self.sync_grads[layer_idx][key], op=MPI.SUM)
            self.handles.append(handle)

    def wait_and_step(self, size):
        MPI.Request.Waitall(self.handles)
        self.handles = []

        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for key in layer.params:
                    # Update with Averaged Gradients
                    layer.params[key] -= self.lr * (self.sync_grads[i][key] / size)
