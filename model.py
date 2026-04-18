class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output, on_layer_backward=None):
        for i, layer in enumerate(reversed(self.layers)):
            grad_output = layer.backward(grad_output)
            if on_layer_backward:
                # We need the original index for the optimizer
                original_idx = len(self.layers) - 1 - i
                on_layer_backward(original_idx, layer)
        return grad_output

    def params(self):
        # Useful for distributed syncing later
        all_params = []
        for layer in self.layers:
            if hasattr(layer, 'params'):
                all_params.append(layer.params)
        return all_params
