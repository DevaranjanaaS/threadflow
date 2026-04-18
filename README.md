# 🧵 ThreadFlow Elite: Distributed Neural Engine

ThreadFlow Elite is a high-performance, distributed neural network training framework built from scratch using Python, NumPy, and MPI. It is designed to demonstrate advanced parallel computing strategies, specifically focusing on the trade-off between **communication latency hiding** and **frequency reduction**.

## 🚀 Key Features

### 1. The Parallel Duel (Optimization Strategies)
ThreadFlow pits two world-class distributed strategies against each other:
*   **Overlap SGD (Latency Hiding)**: Uses non-blocking `MPI_Iallreduce` to mask network synchronization by performing communication in the background while the CPU computes the next layer's gradients.
*   **Accumulation SGD (Frequency Reduction)**: Minimizes expensive MPI handshakes by buffering gradients locally and only synchronizing once every N steps (Accumulation).

### 2. Live Training Dashboard
A premium, dark-themed visualizer built with `matplotlib` that streams live Loss and Accuracy curves from the distributed ranks using a real-time telemetry bridge.

### 3. Dataset Intelligence
Fully integrated support for both **MNIST** (Digits) and **Fashion-MNIST** (Clothing) with automated sharding and performance benchmarking across both datasets.

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

*Note: Requires an MPI implementation (e.g., OpenMPI or MPICH) installed on your system.*

## 📈 Usage

### Run the Live Demo
**Terminal 1 (The Engine):**
```bash
mpirun --oversubscribe -n 4 .venv/bin/python3 train_overlap.py
```
**Terminal 2 (The Eyes):**
```bash
.venv/bin/python3 live_dashboard.py
```

### Run the Analytics Suite
To generate the "Parallel Duel" report comparing datasets and strategies:
```bash
python3 run_comparisons.py
python3 generate_dashboard_advanced.py
# Open dashboard_advanced.html in your browser
```

## 📁 System Architecture
*   `engine.py`: Core neural network components (Linear, ReLU, CrossEntropy).
*   `optimizer.py`: Elite parallel optimizers (Overlap & Accumulation).
*   `data_loader.py`: Universal dataset loader with stable AWS mirrors.
*   `archive/`: Contains experimental features (QuantumSync, Pipeline Parallelism, etc.).
