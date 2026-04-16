# ThreadFlow Neural Engine

A high-performance, transparent neural network training engine built from scratch using Python, NumPy, and mpi4py.

## Features
- **Scratch Implementation**: Core neural network components (layers, optimizers, loss functions) implemented using only NumPy.
- **Parallel Training**: Data parallelism implementation using MPI (`mpi4py`) for scaling training across multiple CPU cores.
- **High Performance**: Optimized for speed and transparency, achieving high accuracy on the MNIST dataset.
- **Benchmarking Tools**: Scripts to run benchmarks and analyze scaling performance.
- **Visualization Dashboard**: Automatically generated HTML dashboard to visualize training results and scaling efficiency.

## Project Structure
- `engine.py`: Core neural network engine logic.
- `model.py`: Model architecture definitions.
- `optimizer.py`: Implementation of optimization algorithms (e.g., SGD).
- `data_loader.py`: Parallel data loading and preprocessing.
- `train_parallel.py`: Main script for parallel training using MPI.
- `train_professional.py`: Advanced training script with logging and result tracking.
- `run_benchmarks.py`: Utility to run benchmarks across different process counts.
- `generate_dashboard.py`: Tool to generate a visual performance dashboard.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install numpy mpi4py`.
3. Ensure you have an MPI implementation installed on your system (e.g., OpenMPI or MPICH).

## Usage
To run the parallel training:
```bash
mpirun -np 4 python train_parallel.py
```

To run benchmarks:
```bash
python run_benchmarks.py
```
