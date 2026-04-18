import numpy as np
import time
import os
import json
from mpi4py import MPI
from engine import Linear, ReLU, CrossEntropyLoss
from model import Model
from optimizer import OverlapOptimizer

def generate_synthetic_data(num_samples=10000, num_features=20, num_classes=3):
    """Generates synthetic classification data (Blobs)."""
    np.random.seed(42)
    X = np.random.randn(num_samples, num_features).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    
    # Add some structure so it's learnable: add class-specific bias to features
    for i in range(num_classes):
        X[y == i] += i * 2.0 
        
    return X, y

def train_custom():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. PARAMETERS
    FEATURES = 20
    CLASSES = 3
    LR = 0.01
    BATCH = 64
    EPOCHS = 10

    # 2. DATA GENERATION
    if rank == 0:
        print(f"🧬 Generating Synthetic Numerical Dataset...")
        print(f"📊 Features: {FEATURES} | Classes: {CLASSES} | Samples: 10,000")
    
    X_full, y_full = generate_synthetic_data(num_samples=10000, num_features=FEATURES, num_classes=CLASSES)
    
    # Sharding
    shard = len(X_full) // size
    X_train = X_full[rank*shard : (rank+1)*shard]
    y_train = y_full[rank*shard : (rank+1)*shard]
    
    # Simple Test Set on Rank 0
    X_test, y_test = generate_synthetic_data(num_samples=1000, num_features=FEATURES, num_classes=CLASSES)

    # 3. ELITE MODEL SETUP
    model = Model([
        Linear(FEATURES, 64),
        ReLU(),
        Linear(64, 32),
        ReLU(),
        Linear(32, CLASSES)
    ])

    # Sync initial weights
    for layer in model.layers:
        if hasattr(layer, 'params'):
            for key in layer.params:
                comm.Bcast(layer.params[key], root=0)

    optimizer = OverlapOptimizer(model.layers, lr=LR)
    loss_fn = CrossEntropyLoss()

    if rank == 0:
        print(f"🚀 Training Elite Engine on Custom Numeric Data ({size} Ranks)...")

    # Clear stats for live dashboard
    if rank == 0:
        with open("live_stats.json", "w") as f: pass

    for epoch in range(EPOCHS):
        indices = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), BATCH):
            bx = X_train[indices[i:i+BATCH]]
            by = y_train[indices[i:i+BATCH]]
            if len(bx) < BATCH: continue

            logits = model.forward(bx)
            loss, d_logits = loss_fn(logits, by)
            model.backward(d_logits, on_layer_backward=lambda idx, layer: optimizer.start_sync(idx, layer, comm))
            optimizer.wait_and_step(size)

        # Evaluate progress
        if rank == 0:
            test_logits = model.forward(X_test)
            acc = np.mean(np.argmax(test_logits, axis=1) == y_test)
            print(f"Epoch {epoch+1}/{EPOCHS} | Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")
            
            # Stream to telemetry
            stats = {"epoch": epoch + 1, "accuracy": float(acc), "loss": float(loss), "timestamp": time.time()}
            with open("live_stats.json", "a") as f:
                f.write(json.dumps(stats) + "\n")

if __name__ == "__main__":
    train_custom()
