import numpy as np
import time
from mpi4py import MPI
from data_loader import MNISTLoader
from engine import Linear, ReLU, CrossEntropyLoss
from model import Model
from optimizer import QuantumOptimizer

def train_quantum(lr=0.001, epochs=3, batch_size=128):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    loader = MNISTLoader()
    (x_train_full, y_train_full), (x_test, y_test) = loader.load()
    
    shard_size = len(x_train_full) // size
    x_train = x_train_full[rank*shard_size : (rank+1)*shard_size]
    y_train = y_train_full[rank*shard_size : (rank+1)*shard_size]
    
    model = Model([
        Linear(784, 512), 
        ReLU(), 
        Linear(512, 10)
    ])
    
    for layer in model.layers:
        if hasattr(layer, 'params'):
            for key in layer.params:
                comm.Bcast(layer.params[key], root=0)
    
    optimizer = QuantumOptimizer(model.layers, lr=lr)
    loss_fn = CrossEntropyLoss()
    
    if rank == 0:
        print(f"🚀 QuantumSync Training: {size} Ranks | Method: signSGD Majority Voting")

    for epoch in range(epochs):
        t0 = time.time()
        indices = np.random.permutation(len(x_train))
        for i in range(0, len(x_train), batch_size):
            bx, by = x_train[indices[i:i+batch_size]], y_train[indices[i:i+batch_size]]
            if len(bx) < batch_size: continue
            
            # Forward + Backward
            logits = model.forward(bx)
            loss, d_logits = loss_fn(logits, by)
            model.backward(d_logits)
            
            # QUANTUM SYNC STEP (Includes Allreduce)
            optimizer.step(comm, size)
            
        dur = time.time() - t0
        if rank == 0:
            test_logits = model.forward(x_test)
            acc = np.mean(np.argmax(test_logits, axis=1) == y_test)
            print(f"Epoch {epoch+1} | Accuracy: {acc:.4f} | Time: {dur:.2f}s")

if __name__ == "__main__":
    np.random.seed(42 + MPI.COMM_WORLD.Get_rank())
    train_quantum()
