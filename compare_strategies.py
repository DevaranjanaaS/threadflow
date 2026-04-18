import numpy as np
import time
import json
import os
import sys
from mpi4py import MPI
from data_loader import DatasetLoader
from engine import Linear, ReLU, CrossEntropyLoss
from model import Model
from optimizer import SGD, OverlapOptimizer

def run_duel_benchmark(mode="overlap", accum_steps=1, dataset='mnist'):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    loader = DatasetLoader(dataset=dataset)
    (x_train_full, y_train_full), (x_test, y_test) = loader.load()
    
    # Standard Sharding
    shard_size = len(x_train_full) // size
    x_train = x_train_full[rank*shard_size : (rank+1)*shard_size]
    y_train = y_train_full[rank*shard_size : (rank+1)*shard_size]
    
    # A standard 2-layer MLP for consistent benchmarking
    model = Model([Linear(784, 512), ReLU(), Linear(512, 10)])
    
    # Sync initial weights for fair start
    for layer in model.layers:
        if hasattr(layer, 'params'):
            for key in layer.params:
                comm.Bcast(layer.params[key], root=0)
    
    if mode == "overlap":
        optimizer = OverlapOptimizer(model.layers, lr=0.01)
    else:
        optimizer = SGD(model.layers, lr=0.01)
        
    loss_fn = CrossEntropyLoss()
    
    metrics = {"compute_times": [], "comm_times": [], "epoch_times": []}
    max_iters = 100 
    batch_size = 128

    for epoch in range(1):
        t_epoch_start = time.time()
        t_compute = 0
        t_comm = 0
        
        indices = np.random.permutation(len(x_train))
        for i in range(0, min(len(x_train), max_iters * batch_size), batch_size):
            bx, by = x_train[indices[i:i+batch_size]], y_train[indices[i:i+batch_size]]
            if len(bx) < batch_size: continue
            
            # --- PHASE 1: COMPUTE ---
            t0 = time.time()
            logits = model.forward(bx)
            loss, d_logits = loss_fn(logits, by)
            
            if mode == "overlap":
                # Start communication during backward pass
                model.backward(d_logits, on_layer_backward=lambda idx, layer: optimizer.start_sync(idx, layer, comm))
            else:
                # Standard blocking backward
                model.backward(d_logits)
            t_compute += (time.time() - t0)
            
            # --- PHASE 2: SYNC & STEP ---
            # For "accumulation", this condition fires every N steps
            # For "overlap", this fires every step (but it's a 'wait' rather than a full sync)
            if (i // batch_size + 1) % accum_steps == 0:
                t1 = time.time()
                if mode == "overlap":
                    optimizer.wait_and_step(size)
                else:
                    # Blocking sync
                    for layer in model.layers:
                        if hasattr(layer, 'params'):
                            for key in layer.params:
                                comm.Allreduce(MPI.IN_PLACE, layer.grads[key], op=MPI.SUM)
                                layer.grads[key] /= (size * accum_steps)
                    optimizer.step()
                    optimizer.zero_grad() # Prepare for next accumulation block
                t_comm += (time.time() - t1)
            
        epoch_dur = time.time() - t_epoch_start
        if rank == 0:
            metrics["epoch_times"].append(float(epoch_dur))
            metrics["compute_times"].append(float(t_compute))
            metrics["comm_times"].append(float(t_comm))
            print(f"[{dataset}] {mode.upper()} | Time: {epoch_dur:.2f}s (Comp: {t_compute:.2f}s, Comm: {t_comm:.2f}s)")

    return metrics if rank == 0 else None

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "overlap"
    accum = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    dataset = sys.argv[3] if len(sys.argv) > 3 else "mnist"
    
    res = run_duel_benchmark(mode, accum, dataset)
    if MPI.COMM_WORLD.Get_rank() == 0:
        with open(f"duel_{mode}_{accum}_{dataset}.json", "w") as f:
            json.dump(res, f)
