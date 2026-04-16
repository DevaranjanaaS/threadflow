import numpy as np
import time
from data_loader import MNISTLoader, get_batches
from engine import Linear, ReLU, CrossEntropyLoss
from model import Model
from optimizer import SGD

def train_seq():
    print("Starting Phase 3: Sequential Training...")
    
    # 1. Load Data
    loader = MNISTLoader()
    (x_train, y_train), (x_test, y_test) = loader.load()
    
    # 2. Define Model
    # Input: 784 -> Hidden: 128 -> Output: 10
    model = Model([
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    ])
    
    optimizer = SGD(model.layers, lr=0.1)
    loss_fn = CrossEntropyLoss()
    
    epochs = 5
    batch_size = 64
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        iterations = 0
        
        for batch_x, batch_y in get_batches(x_train, y_train, batch_size):
            # Forward
            logits = model.forward(batch_x)
            loss, d_logits = loss_fn(logits, batch_y)
            
            # Backward
            model.backward(d_logits)
            
            # Optimize
            optimizer.step()
            
            total_loss += loss
            iterations += 1
            
        # Evaluation
        test_logits = model.forward(x_test)
        preds = np.argmax(test_logits, axis=1)
        accuracy = np.mean(preds == y_test)
        
        elapsed = time.time() - start_time
        avg_loss = total_loss / iterations
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {accuracy:.4f} | Time: {elapsed:.2f}s")

    print("\n✅ Phase 3 Validation: Sequential training complete.")
    if accuracy > 0.92:
        print(f"Final Test Accuracy {accuracy:.4f} target > 0.92 met.")
    else:
        print(f"Final Test Accuracy {accuracy:.4f} target > 0.92 NOT met. Suggest tuning.")

if __name__ == "__main__":
    np.random.seed(42)
    train_seq()
