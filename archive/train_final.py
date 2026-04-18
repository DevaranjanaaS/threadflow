import numpy as np
import time
import sys
from mpi4py import MPI
from data_loader import MNISTLoader, get_batches
from engine import Linear, ReLU, CrossEntropyLoss
from model import Model
from optimizer import MomentumSGD

def train_parallel(lr=0.01, momentum=0.9, epochs=5, batch_size=64):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    loader = MNISTLoader()
    (x_train_full, y_train_full), (x_test, y_test) = loader.load()
    
    num_samples = x_train_full.shape[0]
    samples_per_rank = num_samples // size
    start_idx = rank * samples_per_rank
    end_idx = (rank + 1) * samples_per_rank if rank != size - 1 else num_samples
    
    x_train = x_train_full[start_idx:end_idx]
    y_train = y_train_full[start_idx:end_idx]
    
    model = Model([
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    ])
    
    # Sync weights
    for layer in model.layers:
        if hasattr(layer, 'params'):
            for key in layer.params:
                comm.Bcast(layer.params[key], root=0)
    
    optimizer = MomentumSGD(model.layers, lr=lr, momentum=momentum)
    loss_fn = CrossEntropyLoss()
    
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        indices = np.random.permutation(len(x_train))
        x_train_shuff = x_train[indices]
        y_train_shuff = y_train[indices]
        
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train_shuff[i:i+batch_size]
            batch_y = y_train_shuff[i:i+batch_size]
            
            logits = model.forward(batch_x)
            loss, d_logits = loss_fn(logits, batch_y)
            model.backward(d_logits)
            
            # Sync gradients
            for layer in model.layers:
                if hasattr(layer, 'params'):
                    for key in layer.params:
                        comm.Allreduce(MPI.IN_PLACE, layer.grads[key], op=MPI.SUM)
                        layer.grads[key] /= size
            
            optimizer.step()
        
        duration = time.time() - epoch_start
        epoch_times.append(duration)
        
        if rank == 0:
            test_logits = model.forward(x_test)
            acc = np.mean(np.argmax(test_logits, axis=1) == y_test)
            print(f"Rank 0 | Epoch {epoch+1} | Acc: {acc:.4f} | Time: {duration:.2f}s")
            
    if rank == 0:
        avg_epoch_time = np.mean(epoch_times)
        print(f"\nTraining finished for {size} processes.")
        print(f"Average epoch time: {avg_epoch_time:.2f}s")
        print(f"Final Accuracy: {acc:.4f}")

if __name__ == "__main__":
    np.random.seed(42 + MPI.COMM_WORLD.Get_rank())
    # Finer tuning for 95%+ accuracy
    train_parallel(lr=0.01, momentum=0.9, epochs=5, batch_size=32)
