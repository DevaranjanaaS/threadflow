import json
import os

def generate_duel_dashboard():
    if not os.path.exists("comprehensive_results.json"):
        print("comprehensive_results.json not found!")
        return

    with open("comprehensive_results.json", "r") as f:
        data = json.load(f)

    datasets = ["mnist", "fashion"]
    strats = ["overlap_acc1", "regular_acc8"]
    labels = {"overlap_acc1": "Overlap (Hiding Latency)", "regular_acc8": "Accumulation (Reducing Frequency)"}
    ranks = [1, 2, 4, 8]
    
    analysis = {}
    for ds in datasets:
        analysis[ds] = {s: {"scaling": [], "efficiency": [], "throughput": [], "comm": []} for s in strats}
        for s in strats:
            for r in ranks:
                r_str = str(r)
                if ds in data and s in data[ds] and r_str in data[ds][s]:
                    metrics = data[ds][s][r_str]
                    avg_total = sum(metrics["epoch_times"]) / len(metrics["epoch_times"])
                    avg_comp = sum(metrics["compute_times"]) / len(metrics["compute_times"])
                    avg_comm = sum(metrics["comm_times"]) / len(metrics["comm_times"])
                    
                    analysis[ds][s]["scaling"].append(avg_total)
                    analysis[ds][s]["efficiency"].append((avg_comp / avg_total) * 100)
                    analysis[ds][s]["throughput"].append(12800 / avg_total)
                    analysis[ds][s]["comm"].append(avg_comm)
                else:
                    for k in analysis[ds][s]: analysis[ds][s][k].append(0)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The Parallel Duel: Overlap vs Accumulation</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #030712; --card: #111827; --primary: #60a5fa; --secondary: #a78bfa; --accent: #f43f5e; --text: #f9fafb;
        }}
        body {{ font-family: 'Outfit', sans-serif; background-color: var(--bg); color: var(--text); margin: 0; padding: 40px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{ text-align: center; margin-bottom: 60px; }}
        h1 {{ font-size: 3.5rem; font-weight: 800; background: linear-gradient(to right, #60a5fa, #f43f5e); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 30px; }}
        .card {{ background: var(--card); border: 1px solid #374151; padding: 30px; border-radius: 24px; }}
        .full-width {{ grid-column: span 2; }}
        h2 {{ color: var(--primary); margin-top: 0; }}
        .versus {{ display: flex; align-items: center; justify-content: center; gap: 40px; margin-bottom: 40px; }}
        .strat-box {{ padding: 20px 40px; border-radius: 16px; border: 1px solid #374151; text-align: center; }}
        .vs-label {{ font-size: 2rem; font-weight: 800; color: var(--accent); }}
        .tagline {{ color: #9ca3af; font-size: 0.9rem; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>THE PARALLEL DUEL</h1>
            <p>Latency Hiding (Overlap) vs. Frequency Reduction (Accumulation)</p>
        </header>

        <div class="versus">
            <div class="strat-box" style="border-color: #60a5fa">
                <div style="color: #60a5fa; font-weight: 800; font-size: 1.2rem;">OVERLAP SGD</div>
                <div class="tagline">"Masking Communication with Work"</div>
            </div>
            <div class="vs-label">VS</div>
            <div class="strat-box" style="border-color: #f43f5e">
                <div style="color: #f43f5e; font-weight: 800; font-size: 1.2rem;">ACCUMULATION SGD</div>
                <div class="tagline">"Reducing Network Frequency"</div>
            </div>
        </div>

        <div class="grid">
            <div class="card full-width">
                <h2>Cross-Dataset Throughput (Samples / Sec)</h2>
                <p>Higher is better. How each strategy performs on MNIST vs Fashion-MNIST.</p>
                <canvas id="throughputChart" height="100"></canvas>
            </div>
            
            <div class="card">
                <h2>Communication Cost Analysis (MNIST)</h2>
                <p>Measures time spent waiting for network synchronization.</p>
                <canvas id="commChartMnist"></canvas>
            </div>

            <div class="card">
                <h2>Communication Cost Analysis (Fashion)</h2>
                <p>Observing if data complexity increases sync overhead.</p>
                <canvas id="commChartFashion"></canvas>
            </div>
            
            <div class="card full-width">
                <h2>Parallel Efficiency Spectrum</h2>
                <p>Measures how close to "Linear Scaling" each method stays.</p>
                <canvas id="efficiencyChart" height="100"></canvas>
            </div>
        </div>
    </div>

    <script>
        const processors = [1, 2, 4, 8];
        const colors = {{ overlap: "#60a5fa", accumulation: "#f43f5e" }};
        const data = {json.dumps(analysis)};

        // Throughput Grouped Bar
        new Chart(document.getElementById('throughputChart'), {{
            type: 'bar',
            data: {{
                labels: processors,
                datasets: [
                    {{ label: "Overlap (MNIST)", data: data.mnist.overlap_acc1.throughput, backgroundColor: colors.overlap, stack: 'mnist' }},
                    {{ label: "Accumulation (MNIST)", data: data.mnist.regular_acc8.throughput, backgroundColor: colors.accumulation, stack: 'mnist' }},
                    {{ label: "Overlap (Fashion)", data: data.fashion.overlap_acc1.throughput, backgroundColor: colors.overlap + '88', stack: 'fashion' }},
                    {{ label: "Accumulation (Fashion)", data: data.fashion.regular_acc8.throughput, backgroundColor: colors.accumulation + '88', stack: 'fashion' }}
                ]
            }},
            options: {{ scales: {{ y: {{ grid: {{ color: '#374151' }} }} }} }}
        }});

        // Comm Chart MNIST
        new Chart(document.getElementById('commChartMnist'), {{
            type: 'line',
            data: {{
                labels: processors,
                datasets: [
                    {{ label: "Overlap Sync", data: data.mnist.overlap_acc1.comm, borderColor: colors.overlap, fill: false }},
                    {{ label: "Accumulation Sync", data: data.mnist.regular_acc8.comm, borderColor: colors.accumulation, fill: false }}
                ]
            }},
            options: {{ scales: {{ y: {{ grid: {{ color: '#374151' }} }} }} }}
        }});

        // Comm Chart Fashion
        new Chart(document.getElementById('commChartFashion'), {{
            type: 'line',
            data: {{
                labels: processors,
                datasets: [
                    {{ label: "Overlap Sync", data: data.fashion.overlap_acc1.comm, borderColor: colors.overlap, fill: false }},
                    {{ label: "Accumulation Sync", data: data.fashion.regular_acc8.comm, borderColor: colors.accumulation, fill: false }}
                ]
            }},
            options: {{ scales: {{ y: {{ grid: {{ color: '#374151' }} }} }} }}
        }});

        // Efficiency Chart
        new Chart(document.getElementById('efficiencyChart'), {{
            type: 'bar',
            data: {{
                labels: ["MNIST: Overlap", "MNIST: Accum", "Fashion: Overlap", "Fashion: Accum"],
                datasets: [
                    {{ label: "Rank 2", data: [data.mnist.overlap_acc1.efficiency[1], data.mnist.regular_acc8.efficiency[1], data.fashion.overlap_acc1.efficiency[1], data.fashion.regular_acc8.efficiency[1]], backgroundColor: "#3b82f6" }},
                    {{ label: "Rank 4", data: [data.mnist.overlap_acc1.efficiency[2], data.mnist.regular_acc8.efficiency[2], data.fashion.overlap_acc1.efficiency[2], data.fashion.regular_acc8.efficiency[2]], backgroundColor: "#8b5cf6" }},
                    {{ label: "Rank 8", data: [data.mnist.overlap_acc1.efficiency[3], data.mnist.regular_acc8.efficiency[3], data.fashion.overlap_acc1.efficiency[3], data.fashion.regular_acc8.efficiency[3]], backgroundColor: "#ec4899" }}
                ]
            }},
            options: {{ scales: {{ y: {{ max: 100, grid: {{ color: '#374151' }} }} }} }}
        }});
    </script>
</body>
</html>
    """
    with open("dashboard_advanced.html", "w") as f:
        f.write(html)
    print("✅ Parallel Duel Dashboard generated: dashboard_advanced.html")

if __name__ == "__main__":
    generate_duel_dashboard()
