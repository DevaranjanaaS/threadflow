import numpy as np
import time
import json
import os
from mpi4py import MPI
from data_loader import MNISTLoader
from engine import Linear, ReLU, CrossEntropyLoss
from model import Model
from optimizer import MomentumSGD

def train_mega_workload(batch_size=512, accum_steps=20):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    loader = MNISTLoader()
    (x_train_full, y_train_full), (x_test, y_test) = loader.load()
    
    # Strong Scaling: We divide the work
    shard_size = len(x_train_full) // size
    x_train = x_train_full[rank*shard_size : (rank+1)*shard_size]
    y_train = y_train_full[rank*shard_size : (rank+1)*shard_size]
    
    model = Model([
        Linear(784, 4096), 
        ReLU(), 
        Linear(4096, 4096), 
        ReLU(), 
        Linear(4096, 10)
    ])
    
    for layer in model.layers:
        if hasattr(layer, 'params'):
            for key in layer.params:
                comm.Bcast(layer.params[key], root=0)
    
    optimizer = MomentumSGD(model.layers, lr=0.01, momentum=0.9)
    loss_fn = CrossEntropyLoss()
    
    max_iters = 50 

    if rank == 0:
        print(f"🔥 Starting Mega Workload Benchmark ({size} Ranks) 🔥")

    t_start = time.time()
    # Ensure every rank does exactly its share
    num_iters = min(len(x_train) // batch_size, max_iters)
    
    for i in range(num_iters):
        start_idx = i * batch_size
        bx, by = x_train[start_idx:start_idx+batch_size], y_train[start_idx:start_idx+batch_size]
        
        logits = model.forward(bx)
        loss, d_logits = loss_fn(logits, by)
        model.backward(d_logits)
        
        if (i + 1) % accum_steps == 0:
            for layer in model.layers:
                if hasattr(layer, 'params'):
                    for key in layer.params:
                        comm.Allreduce(MPI.IN_PLACE, layer.grads[key], op=MPI.SUM)
                        layer.grads[key] /= size
            optimizer.step()
            
    total_time = time.time() - t_start
    
    if rank == 0:
        print(f"✅ Rank {size} Total Time: {total_time:.2f}s")
        with open(f"mega_result_{size}.json", "w") as f:
            json.dump({"time": float(total_time)}, f)

if __name__ == "__main__":
    train_mega_workload()
