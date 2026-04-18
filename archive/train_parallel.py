import numpy as np
import time
from mpi4py import MPI
from data_loader import MNISTLoader, get_batches
from engine import Linear, ReLU, CrossEntropyLoss
from model import Model
from optimizer import SGD

def train_parallel():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Starting Phase 4: MPI Parallel Training with {size} processes...")

    # 1. Load Data
    loader = MNISTLoader()
    (x_train_full, y_train_full), (x_test, y_test) = loader.load()
    
    # 2. Shard Data
    # Each rank gets a slice of the training data
    num_samples = x_train_full.shape[0]
    samples_per_rank = num_samples // size
    
    start_idx = rank * samples_per_rank
    end_idx = (rank + 1) * samples_per_rank if rank != size - 1 else num_samples
    
    x_train = x_train_full[start_idx:end_idx]
    y_train = y_train_full[start_idx:end_idx]
    
    # 3. Define Model
    model = Model([
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    ])
    
    # 4. Synchronize initial weights (Rank 0 -> All)
    # This is critical so everyone starts at the same point in the loss landscape
    for layer in model.layers:
        if hasattr(layer, 'params'):
            for key in layer.params:
                comm.Bcast(layer.params[key], root=0)
    
    optimizer = SGD(model.layers, lr=0.1)
    loss_fn = CrossEntropyLoss()
    
    epochs = 5
    batch_size = 64 # Local batch size
    
    for epoch in range(epochs):
        epoch_start = time.time()
        local_loss = 0
        iterations = 0
        
        # shuffle local data
        indices = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train_shuffled[i:i + batch_size]
            batch_y = y_train_shuffled[i:i + batch_size]
            
            # Forward
            logits = model.forward(batch_x)
            loss, d_logits = loss_fn(logits, batch_y)
            
            # Backward
            model.backward(d_logits)
            
            # 5. ALLREDUCE GRADIENTS
            # We average the gradients across all processes
            for layer in model.layers:
                if hasattr(layer, 'params'):
                    for key in layer.params:
                        # Sum gradients from all ranks
                        comm.Allreduce(MPI.IN_PLACE, layer.grads[key], op=MPI.SUM)
                        # Average them
                        layer.grads[key] /= size
            
            # Optimize
            optimizer.step()
            
            local_loss += loss
            iterations += 1
            
        # Synchronize loss for reporting
        avg_loss = comm.reduce(local_loss / iterations, op=MPI.SUM, root=0)
        if rank == 0:
            avg_loss /= size
            # Evaluation on test set (usually done on one rank or in parallel)
            test_logits = model.forward(x_test)
            preds = np.argmax(test_logits, axis=1)
            accuracy = np.mean(preds == y_test)
            
            duration = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Test Acc: {accuracy:.4f} | Time: {duration:.2f}s")

    if rank == 0:
        print("\n✅ Phase 4 Validation: Parallel training complete.")
        print(f"Final Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # Ensure consistent initialization across ranks for testing reproducibility
    np.random.seed(42 + MPI.COMM_WORLD.Get_rank())
    train_parallel()
