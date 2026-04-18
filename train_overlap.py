import numpy as np
import time
import os
import json
from mpi4py import MPI
from data_loader import MNISTLoader
from engine import Linear, ReLU, CrossEntropyLoss
from model import Model
from optimizer import OverlapOptimizer

def train_overlap(lr=0.01, epochs=5, batch_size=128):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ONLY Rank 0 resets the live stats file to avoid race conditions
    if rank == 0:
        with open("live_stats.json", "w") as f:
            pass
    
    loader = MNISTLoader()
    (x_train_full, y_train_full), (x_test, y_test) = loader.load()
    
    # Split data
    shard_size = len(x_train_full) // size
    x_train = x_train_full[rank*shard_size : (rank+1)*shard_size]
    y_train = y_train_full[rank*shard_size : (rank+1)*shard_size]
    
    # Define a larger model to see the benefits of overlap
    model = Model([
        Linear(784, 1024), 
        ReLU(), 
        Linear(1024, 512),
        ReLU(),
        Linear(512, 10)
    ])
    
    # Sync initial params
    for layer in model.layers:
        if hasattr(layer, 'params'):
            for key in layer.params:
                comm.Bcast(layer.params[key], root=0)
    
    optimizer = OverlapOptimizer(model.layers, lr=lr)
    loss_fn = CrossEntropyLoss()
    
    if rank == 0:
        print(f"⚡ Overlap Engine: {size} Ranks | Method: Non-blocking Iallreduce")

    results = []

    for epoch in range(epochs):
        t0 = time.time()
        indices = np.random.permutation(len(x_train))
        for i in range(0, len(x_train), batch_size):
            bx, by = x_train[indices[i:i+batch_size]], y_train[indices[i:i+batch_size]]
            if len(bx) < batch_size: continue
            
            # Forward
            logits = model.forward(bx)
            loss, d_logits = loss_fn(logits, by)
            
            # Backward with communication overlapping
            # start_sync is called for each layer right after its gradients are computed
            model.backward(d_logits, on_layer_backward=lambda idx, layer: optimizer.start_sync(idx, layer, comm))
            
            # Wait for all communications to finish and update weights
            optimizer.wait_and_step(size)
            
        dur = time.time() - t0
        
        if rank == 0:
            test_logits = model.forward(x_test)
            acc = np.mean(np.argmax(test_logits, axis=1) == y_test)
            print(f"Epoch {epoch+1} | Acc: {acc:.4f} | Time: {dur:.2f}s")
            
            # LIVE LOGGING FOR DASHBOARD
            stats = {
                "epoch": epoch + 1,
                "accuracy": float(acc),
                "loss": float(loss),
                "timestamp": time.time()
            }
            with open("live_stats.json", "a") as f:
                f.write(json.dumps(stats) + "\n")
            
            results.append({"epoch": epoch+1, "acc": float(acc), "time": dur})

    if rank == 0:
        with open(f"overlap_results_{size}.json", "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    np.random.seed(42 + MPI.COMM_WORLD.Get_rank())
    train_overlap()
