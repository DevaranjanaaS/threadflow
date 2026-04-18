import numpy as np
import os
import tarfile
import urllib.request
import time
import json
from mpi4py import MPI
from engine import Linear, ReLU, CrossEntropyLoss
from model import Model
from optimizer import OverlapOptimizer

def download_cifar100(data_dir='./data/cifar100'):
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    tar_path = os.path.join(data_dir, "cifar-100-binary.tar.gz")
    if not os.path.exists(tar_path):
        print("📡 Downloading CIFAR-100 (Heavyweight Dataset ~160MB)...")
        urllib.request.urlretrieve(url, tar_path)
        
    # Extract
    if not os.path.exists(os.path.join(data_dir, "cifar-100-binary")):
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
            
    return os.path.join(data_dir, "cifar-100-binary")

def load_cifar100_binary(path, type='train'):
    file_path = os.path.join(path, f"{type}.bin")
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    # CIFAR-100 binary format:
    # <1 byte coarse label><1 byte fine label><3072 bytes image>
    # We want the 'fine label' (index 1)
    record_size = 3074
    num_samples = len(data) // record_size
    reshaped = data.reshape(num_samples, record_size)
    
    labels = reshaped[:, 1].astype(np.int64)
    images = reshaped[:, 2:].astype(np.float32) / 255.0
    return images, labels

def train_cifar():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. SETUP DATA
    if rank == 0:
        cifar_path = download_cifar100()
    comm.Barrier()
    cifar_path = './data/cifar100/cifar-100-binary'
    
    x_full, y_full = load_cifar100_binary(cifar_path, 'train')
    x_test, y_test = load_cifar100_binary(cifar_path, 'test')
    
    # Sharding
    shard = len(x_full) // size
    x_train = x_full[rank*shard : (rank+1)*shard]
    y_train = y_full[rank*shard : (rank+1)*shard]

    # 2. DEFINE DEEP CIFAR MODEL
    # Input: 32*32*3 = 3072 | Output: 100 classes
    model = Model([
        Linear(3072, 1024),
        ReLU(),
        Linear(1024, 512),
        ReLU(),
        Linear(512, 100)
    ])

    # Sync initial weights
    for layer in model.layers:
        if hasattr(layer, 'params'):
            for key in layer.params:
                comm.Bcast(layer.params[key], root=0)

    optimizer = OverlapOptimizer(model.layers, lr=0.005) # Lower LR for harder data
    loss_fn = CrossEntropyLoss()

    if rank == 0:
        print(f"🌆 Training Elite Engine on CIFAR-100 ({size} Ranks)...")
        print(f"📊 100 Categories | 3072 RGB Features | 50,000 Samples")

    # Clear stats for telemetry
    if rank == 0:
        with open("live_stats.json", "w") as f: pass

    for epoch in range(10):
        t0 = time.time()
        indices = np.random.permutation(len(x_train))
        for i in range(0, len(x_train), 128):
            bx = x_train[indices[i : i+128]]
            by = y_train[indices[i : i+128]]
            if len(bx) < 128: continue

            logits = model.forward(bx)
            loss, d_logits = loss_fn(logits, by)
            model.backward(d_logits, on_layer_backward=lambda idx, layer: optimizer.start_sync(idx, layer, comm))
            optimizer.wait_and_step(size)

        dur = time.time() - t0
        if rank == 0:
            test_logits = model.forward(x_test[:1000]) # Test subset for speed
            acc = np.mean(np.argmax(test_logits, axis=1) == y_test[:1000])
            print(f"Epoch {epoch+1}/10 | Accuracy: {acc*100:.2f}% | Loss: {loss:.4f} | Time: {dur:.2f}s")
            
            # Stream to live dashboard
            stats = {"epoch": epoch+1, "accuracy": float(acc), "loss": float(loss), "timestamp": time.time()}
            with open("live_stats.json", "a") as f:
                f.write(json.dumps(stats) + "\n")

if __name__ == "__main__":
    train_cifar()
