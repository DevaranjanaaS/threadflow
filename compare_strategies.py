import numpy as np
import time
import json
import os
from mpi4py import MPI
from data_loader import MNISTLoader
from engine import Linear, ReLU, CrossEntropyLoss
from model import Model
from optimizer import MomentumSGD, QuantumOptimizer

def train_engine(mode="regular", accum_steps=1):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    loader = MNISTLoader()
    (x_train_full, y_train_full), (x_test, y_test) = loader.load()
    
    shard_size = len(x_train_full) // size
    x_train = x_train_full[rank*shard_size : (rank+1)*shard_size]
    y_train = y_train_full[rank*shard_size : (rank+1)*shard_size]
    
    model = Model([Linear(784, 512), ReLU(), Linear(512, 10)])
    
    for layer in model.layers:
        if hasattr(layer, 'params'):
            for key in layer.params:
                comm.Bcast(layer.params[key], root=0)
    
    if mode == "quantum":
        optimizer = QuantumOptimizer(model.layers, lr=0.001)
    else:
        optimizer = MomentumSGD(model.layers, lr=0.01, momentum=0.9)
        
    loss_fn = CrossEntropyLoss()
    
    metrics = {"compute_times": [], "comm_times": [], "epoch_times": []}
    max_iters = 100 
    batch_size = 128

    for epoch in range(1): # 1 epoch is enough for timing profile
        t_epoch_start = time.time()
        t_compute = 0
        t_comm = 0
        
        indices = np.random.permutation(len(x_train))
        for i in range(0, min(len(x_train), max_iters * batch_size), batch_size):
            bx, by = x_train[indices[i:i+batch_size]], y_train[indices[i:i+batch_size]]
            if len(bx) < batch_size: continue
            
            # COMPUTE
            t0 = time.time()
            logits = model.forward(bx)
            loss, d_logits = loss_fn(logits, by)
            model.backward(d_logits)
            t_compute += (time.time() - t0)
            
            # COMM
            if (i // batch_size + 1) % accum_steps == 0:
                t1 = time.time()
                if mode == "quantum":
                    optimizer.step(comm, size)
                else:
                    for layer in model.layers:
                        if hasattr(layer, 'params'):
                            for key in layer.params:
                                comm.Allreduce(MPI.IN_PLACE, layer.grads[key], op=MPI.SUM)
                                layer.grads[key] /= size
                    optimizer.step()
                t_comm += (time.time() - t1)
            
        epoch_dur = time.time() - t_epoch_start
        if rank == 0:
            metrics["epoch_times"].append(float(epoch_dur))
            metrics["compute_times"].append(float(t_compute))
            metrics["comm_times"].append(float(t_comm))
            print(f"Mode: {mode} | Total: {epoch_dur:.2f}s (Comp: {t_compute:.2f}s, Comm: {t_comm:.2f}s)")

    if rank == 0: return metrics
    return None

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "regular"
    accum = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    res = train_engine(mode, accum)
    if MPI.COMM_WORLD.Get_rank() == 0:
        with open(f"compare_{mode}_{accum}.json", "w") as f:
            json.dump(res, f)
