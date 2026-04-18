import numpy as np
import time
from mpi4py import MPI
from data_loader import MNISTLoader
from engine import Linear, ReLU, CrossEntropyLoss
from model import Model
from optimizer import SGD

def train_pipeline(num_micro_batches=8, batch_size=128, epochs=3):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 1. DEFINE A DEEP MODEL AND SPLIT IT
    # Let's create a 12-layer deep network
    # We will distribute them across the ranks
    full_layers = [
        Linear(784, 512), ReLU(),
        Linear(512, 256), ReLU(),
        Linear(256, 128), ReLU(),
        Linear(128, 64), ReLU(),
        Linear(64, 10)
    ]
    
    # Split layers among ranks
    layers_per_rank = np.array_split(full_layers, size)
    my_layers = layers_per_rank[rank].tolist()
    
    model = Model(my_layers)
    optimizer = SGD(model.layers, lr=0.001)
    loss_fn = CrossEntropyLoss() if rank == size - 1 else None
    
    loader = MNISTLoader()
    (x_train, y_train), (x_test, y_test) = loader.load()
    
    micro_batch_size = batch_size // num_micro_batches
    
    if rank == 0:
        print(f"🚀 Pipelined Engine active with {size} ranks")
        print(f"📦 Total Layers: {len(full_layers)} | Micro-batches: {num_micro_batches}")

    for epoch in range(epochs):
        t0 = time.time()
        for i in range(0, len(x_train), batch_size):
            optimizer.zero_grad()
            bx = x_train[i:i+batch_size]
            by = y_train[i:i+batch_size]
            if len(bx) < batch_size: continue

            # --- PIPELINE START ---
            
            # Storage for micro-batch data (needed for backward pass)
            activations = [] # Store inputs to my segment
            outputs = []     # Store outputs from my segment
            
            # 1. FORWARD PASS PIPELINE
            for m in range(num_micro_batches):
                m_start, m_end = m * micro_batch_size, (m+1) * micro_batch_size
                
                # Get Input
                if rank == 0:
                    input_data = bx[m_start:m_end]
                else:
                    # Receive from previous rank
                    input_data = comm.recv(source=rank-1, tag=m)
                
                activations.append(input_data)
                
                # Compute
                out = model.forward(input_data)
                outputs.append(out)
                
                # Pass Forward
                if rank < size - 1:
                    comm.send(out, dest=rank+1, tag=m)
                else:
                    # Last rank calculates loss (optional logging here)
                    pass

            # 2. BACKWARD PASS PIPELINE
            # Gradients of outputs from my segment
            grad_outputs = [None] * num_micro_batches
            
            for m in range(num_micro_batches-1, -1, -1):
                if rank == size - 1:
                    # Last rank starts the backward flow
                    m_start, m_end = m * micro_batch_size, (m+1) * micro_batch_size
                    _, grad_in = loss_fn(outputs[m], by[m_start:m_end])
                    grad_outputs[m] = grad_in
                else:
                    # Receive gradient from next rank
                    grad_outputs[m] = comm.recv(source=rank+1, tag=m+100)
                
                # Local backward
                # We need to set the model's 'cached' activations to the ones 
                # corresponding to this specific micro-batch
                # Model.backward handles the reverse iteration internally
                
                # Manually run backward on my layers for this micro-batch
                current_grad = grad_outputs[m]
                for layer in reversed(model.layers):
                    # We need to ensure the layer has the correct 'x' (input)
                    # This is tricky because model.forward overwrote it.
                    # Simple fix: We'll store inputs in the forward pass.
                    pass 
                
                # Re-run forward quickly to set the 'x' or modify engine to accept x in backward
                # Let's choose the cleaner way: engine.py layers already store self.x
                # But they only store the LAST one. 
                # For Pipeline, we need to temporarily restore the 'x' for this micro-batch.
                
                # --- TRICKY PART: RESTORING STATE ---
                # A better way is to modify engine.py, but for this script 
                # we will just re-apply the correct inputs before backward.
                temp_x = activations[m]
                for layer in model.layers:
                    layer.x = temp_x
                    temp_x = layer.forward(temp_x)
                
                # Now model.backward correctly uses the stored 'x' for this micro-batch
                grad_to_pass = model.backward(grad_outputs[m])
                
                # Pass Gradient Backward
                if rank > 0:
                    comm.send(grad_to_pass, dest=rank-1, tag=m+100)
            
            # 3. OPTIMIZE
            # After all micro-batches are processed, update weights
            optimizer.step()

        # End of epoch
        if rank == 0:
            # Simple test on Rank 0 is not enough, but for demo:
            print(f"Epoch {epoch+1} completed in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    train_pipeline()
