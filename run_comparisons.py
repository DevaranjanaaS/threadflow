import subprocess
import os
import json

def run_duel_benchmarks():
    ranks = [1, 2, 4, 8]
    strategies = [
        ("overlap", 1),   # Pure Overlap
        ("regular", 8)    # Pure Accumulation (Blocking every 8 steps)
    ]
    datasets = ["mnist", "fashion"]
    
    master_results = {}
    
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    print("⚔️  Starting The Parallel Duel: Overlap vs Accumulation...")
    
    if os.path.exists("./.venv/bin/python3"):
        py_exec = "./.venv/bin/python3"
    elif os.path.exists("./venv/bin/python3"):
        py_exec = "./venv/bin/python3"
    else:
        import sys
        py_exec = sys.executable

    for ds in datasets:
        master_results[ds] = {}
        for mode, accum in strategies:
            strat_key = f"{mode}_acc{accum}"
            master_results[ds][strat_key] = {}
            
            for r in ranks:
                print(f"[{ds.upper()}] Strategy: {mode.upper()} | Rank: {r}")
                cmd = ["mpirun", "--oversubscribe", "-n", str(r), py_exec, "compare_strategies.py", mode, str(accum), ds]
                try:
                    subprocess.run(cmd, env=env, check=True)
                    with open(f"duel_{mode}_{accum}_{ds}.json", "r") as f:
                        master_results[ds][strat_key][r] = json.load(f)
                except Exception as e:
                    print(f"Failed {strat_key} at rank {r}: {e}")

    with open("comprehensive_results.json", "w") as f:
        json.dump(master_results, f)
    print("\n✅ Duel results saved to comprehensive_results.json")

if __name__ == "__main__":
    run_duel_benchmarks()
