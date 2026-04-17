import subprocess
import os
import json
import sys

def run_benchmarks():
    counts = [1, 2, 4, 8]
    print(f"🚀 Starting ThreadFlow Scaling Benchmark: [1, 2, 4, 8] Ranks")

    if os.path.exists("./.venv/bin/python3"):
        py_exec = "./.venv/bin/python3"
    elif os.path.exists("./venv/bin/python3"):
        py_exec = "./venv/bin/python3"
    else:
        py_exec = sys.executable
    
    # CRITICAL: Prevent NumPy from using all cores per MPI process
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    for c in counts:
        print(f"\nRunning with {c} processes...")
        cmd = ["mpirun", "--oversubscribe", "-n", str(c), py_exec, "train_professional.py"]
        try:
            subprocess.run(cmd, env=env, check=True)
        except Exception as e:
            print(f"Failed to run {c} ranks: {e}")

    summary = {}
    for c in counts:
        fname = f"results_{c}.json"
        if os.path.exists(fname):
            with open(fname, 'r') as f:
                summary[c] = json.load(f)
    
    with open("scaling_summary.json", "w") as f:
        json.dump(summary, f)
    
    print("\n✅ Benchmark complete. Run 'python3 generate_dashboard.py' to see results.")

if __name__ == "__main__":
    run_benchmarks()
