import numpy as np
from engine import Linear, ReLU, CrossEntropyLoss

def numerical_gradient(f, x, epsilon=1e-7):
    """Computes numerical gradient using finite difference."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_val = x[ix]
        
        # f(x + eps)
        x[ix] = old_val + epsilon
        pos = f(x)
        
        # f(x - eps)
        x[ix] = old_val - epsilon
        neg = f(x)
        
        # (f(x+eps) - f(x-eps)) / 2*eps
        grad[ix] = (pos - neg) / (2 * epsilon)
        
        # Restore
        x[ix] = old_val
        it.iternext()
    return grad

def test_gradients():
    print("Starting Phase 2 Gradient Checking...")
    np.random.seed(42)
    
    # Setup small network and dummy data
    batch_size = 3
    in_dim = 10
    hid_dim = 8
    num_classes = 5
    
    x = np.random.randn(batch_size, in_dim)
    y = np.random.randint(0, num_classes, size=batch_size)
    
    l1 = Linear(in_dim, hid_dim)
    r1 = ReLU()
    l2 = Linear(hid_dim, num_classes)
    loss_fn = CrossEntropyLoss()
    
    # 1. Forward Pass
    h1 = l1.forward(x)
    a1 = r1.forward(h1)
    logits = l2.forward(a1)
    loss, d_logits = loss_fn(logits, y)
    
    # 2. Backward Pass
    d_a1 = l2.backward(d_logits)
    d_h1 = r1.backward(d_a1)
    l1.backward(d_h1)
    
    # 3. Numerical Check for L1 Weights
    def f_W1(W):
        original_W = l1.params['W'].copy()
        l1.params['W'] = W
        # Re-run forward pass
        _h1 = l1.forward(x)
        _a1 = r1.forward(_h1)
        _logits = l2.forward(_a1)
        _loss, _ = loss_fn(_logits, y)
        l1.params['W'] = original_W
        return _loss

    grad_W1_num = numerical_gradient(f_W1, l1.params['W'])
    
    # Relative error: ||grad - num_grad|| / (||grad|| + ||num_grad||)
    diff = np.linalg.norm(l1.grads['W'] - grad_W1_num) / (
        np.linalg.norm(l1.grads['W']) + np.linalg.norm(grad_W1_num)
    )
    
    print(f"L1 Weights Gradient Relative Diff: {diff:.2e}")
    
    # Check L1 bias too
    def f_b1(b):
        original_b = l1.params['b'].copy()
        l1.params['b'] = b
        _h1 = l1.forward(x)
        _a1 = r1.forward(_h1)
        _logits = l2.forward(_a1)
        _loss, _ = loss_fn(_logits, y)
        l1.params['b'] = original_b
        return _loss

    grad_b1_num = numerical_gradient(f_b1, l1.params['b'])
    diff_b = np.linalg.norm(l1.grads['b'] - grad_b1_num) / (
        np.linalg.norm(l1.grads['b']) + np.linalg.norm(grad_b1_num)
    )
    print(f"L1 Bias Gradient Relative Diff:    {diff_b:.2e}")

    assert diff < 1e-7, "L1 Weights gradient check failed!"
    assert diff_b < 1e-7, "L1 Bias gradient check failed!"
    
    print("✅ Phase 2 Validation Successful: Analytical gradients are correct.")

if __name__ == "__main__":
    try:
        test_gradients()
    except Exception as e:
        print(f"❌ Phase 2 Validation Failed: {e}")
        exit(1)
