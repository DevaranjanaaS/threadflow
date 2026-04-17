import subprocess
import os
import json

def run_comprehensive_benchmarks():
    ranks = [1, 2, 4]
    strategies = [
        ("regular", 1),   # Classic
        ("regular", 8),   # Accumulation
        ("quantum", 1)    # QuantumSync
    ]
    
    master_results = {}
    
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    print("🚀 Starting Comprehensive Distributed Strategy Comparison...")
    
    for mode, accum in strategies:
        strat_key = f"{mode}_acc{accum}"
        master_results[strat_key] = {}
        
        for r in ranks:
            print(f"\nBenchmarking strategy: {strat_key} | Rank: {r}")
            cmd = ["mpirun", "--oversubscribe", "-n", str(r), "./venv/bin/python3", "compare_strategies.py", mode, str(accum)]
            try:
                subprocess.run(cmd, env=env, check=True)
                with open(f"compare_{mode}_{accum}.json", "r") as f:
                    master_results[strat_key][r] = json.load(f)
            except Exception as e:
                print(f"Failed {strat_key} at rank {r}: {e}")

    with open("comprehensive_results.json", "w") as f:
        json.dump(master_results, f)
    print("\n✅ Comprehensive results saved to comprehensive_results.json")

if __name__ == "__main__":
    run_comprehensive_benchmarks()
