class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def params(self):
        # Useful for distributed syncing later
        all_params = []
        for layer in self.layers:
            if hasattr(layer, 'params'):
                all_params.append(layer.params)
        return all_params
