import numpy as np

class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # He Initialization: mean=0, std=sqrt(2/in_dim)
        self.params['W'] = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.params['b'] = np.zeros((1, out_dim))
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.params['W'] + self.params['b']

    def backward(self, grad_output):
        # grad_output shape: (batch_size, out_dim)
        # self.x shape: (batch_size, in_dim)
        # Accumulate gradients (needed for micro-batching)
        grad_W = self.x.T @ grad_output
        grad_b = np.sum(grad_output, axis=0, keepdims=True)
        
        if 'W' not in self.grads:
            self.grads['W'] = grad_W
            self.grads['b'] = grad_b
        else:
            self.grads['W'] += grad_W
            self.grads['b'] += grad_b
            
        return grad_output @ self.params['W'].T

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask

class CrossEntropyLoss:
    def __call__(self, logits, y):
        batch_size = logits.shape[0]
        # Numerical stability: subtract max to prevent overflow in exp
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted_logits)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        
        # Loss: -log(p_y) averaged over batch
        # We use a small epsilon to prevent log(0)
        correct_logprobs = -np.log(probs[np.arange(batch_size), y] + 1e-15)
        loss = np.mean(correct_logprobs)
        
        # Gradient of CE with Softmax: (probs - target) / batch_size
        d_logits = probs.copy()
        d_logits[np.arange(batch_size), y] -= 1
        d_logits /= batch_size
        
        return loss, d_logits
