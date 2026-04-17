import json
import os

def generate_improved_dashboard():
    if not os.path.exists("comprehensive_results.json"):
        print("comprehensive_results.json not found!")
        return

    with open("comprehensive_results.json", "r") as f:
        data = json.load(f)

    # Strategies we care about
    strats = ["regular_acc1", "regular_acc8", "quantum_acc1"]
    strat_labels = {
        "regular_acc1": "Classic",
        "regular_acc8": "Accumulation",
        "quantum_acc1": "QuantumSync"
    }
    
    ranks = [1, 2, 4]
    
    # Scaling Data (Total Time)
    scaling_data = {}
    throughput_data = {}
    efficiency_data = {}
    
    batch_size = 128
    num_iters = 100
    images_per_epoch = batch_size * num_iters # approx for timing

    for s in strats:
        scaling_data[s] = []
        throughput_data[s] = []
        efficiency_data[s] = []
        
        for r in ranks:
            r_str = str(r)
            if r_str in data[s]:
                metrics = data[s][r_str]
                avg_total = sum(metrics["epoch_times"]) / len(metrics["epoch_times"])
                avg_comp = sum(metrics["compute_times"]) / len(metrics["compute_times"])
                avg_comm = sum(metrics["comm_times"]) / len(metrics["comm_times"])
                
                scaling_data[s].append(avg_total)
                throughput_data[s].append(images_per_epoch / avg_total)
                efficiency_data[s].append((avg_comp / avg_total) * 100)
            else:
                scaling_data[s].append(0)
                throughput_data[s].append(0)
                efficiency_data[s].append(0)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ThreadFlow Advanced Analytics</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #030712;
            --card: #111827;
            --primary: #60a5fa;
            --secondary: #a78bfa;
            --accent: #f43f5e;
            --text: #f9fafb;
        }}
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 40px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{ text-align: center; margin-bottom: 60px; }}
        h1 {{ 
            font-size: 4rem; font-weight: 800; margin: 0;
            background: linear-gradient(to right, var(--primary), var(--secondary));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 30px; }}
        .card {{
            background: var(--card); border: 1px solid #374151;
            padding: 30px; border-radius: 24px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }}
        .full-width {{ grid-column: span 2; }}
        h2 {{ margin-top: 0; color: var(--primary); font-size: 1.5rem; border-left: 4px solid var(--secondary); padding-left: 15px; }}
        p {{ color: #9ca3af; margin-bottom: 25px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>ThreadFlow Advanced Analytics</h1>
            <p>Comparative Performance & Scaling Breakdown</p>
        </header>

        <div class="grid">
            <div class="card full-width">
                <h2>Scaling Showdown: Time vs Rank</h2>
                <p>Lower is better. Compares how each strategy's total time changes as ranks increase.</p>
                <canvas id="scalingChart"></canvas>
            </div>
            
            <div class="card">
                <h2>Throughput: Images / Second</h2>
                <p>Higher is better. Measures raw processing speed across ranks.</p>
                <canvas id="throughputChart"></canvas>
            </div>

            <div class="card">
                <h2>Parallel Efficiency (%)</h2>
                <p>Measures "Compute-to-Total" ratio. High scores mean low communication overhead.</p>
                <canvas id="efficiencyChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        const ranks = [1, 2, 4];
        const colors = {{
            "regular_acc1": "#3b82f6",
            "regular_acc8": "#10b981",
            "quantum_acc1": "#f43f5e"
        }};
        const labels = {json.dumps(strat_labels)};

        // Scaling Line Chart
        new Chart(document.getElementById('scalingChart'), {{
            type: 'line',
            data: {{
                labels: ranks,
                datasets: [
                    {{ label: labels['regular_acc1'], data: {json.dumps(scaling_data['regular_acc1'])}, borderColor: colors['regular_acc1'], tension: 0.1, borderWidth: 3 }},
                    {{ label: labels['regular_acc8'], data: {json.dumps(scaling_data['regular_acc8'])}, borderColor: colors['regular_acc8'], tension: 0.1, borderWidth: 3 }},
                    {{ label: labels['quantum_acc1'], data: {json.dumps(scaling_data['quantum_acc1'])}, borderColor: colors['quantum_acc1'], tension: 0.1, borderWidth: 3 }}
                ]
            }},
            options: {{ scales: {{ y: {{ grid: {{ color: '#374151' }}, ticks: {{ color: '#9ca3af' }} }}, x: {{ grid: {{ display: false }}, ticks: {{ color: '#9ca3af' }} }} }}, plugins: {{ legend: {{ labels: {{ color: '#f9fafb', font: {{ size: 14 }} }} }} }} }}
        }});

        // Throughput Bar Chart
        new Chart(document.getElementById('throughputChart'), {{
            type: 'bar',
            data: {{
                labels: ranks,
                datasets: [
                    {{ label: labels['regular_acc1'], data: {json.dumps(throughput_data['regular_acc1'])}, backgroundColor: colors['regular_acc1'] }},
                    {{ label: labels['regular_acc8'], data: {json.dumps(throughput_data['regular_acc8'])}, backgroundColor: colors['regular_acc8'] }},
                    {{ label: labels['quantum_acc1'], data: {json.dumps(throughput_data['quantum_acc1'])}, backgroundColor: colors['quantum_acc1'] }}
                ]
            }},
            options: {{ plugins: {{ legend: {{ labels: {{ color: '#f9fafb' }} }} }}, scales: {{ y: {{ beginAtZero: true, grid: {{ color: '#374151' }} }} }} }}
        }});

        // Efficiency Radar/Polar Chart or Grouped Bar
        new Chart(document.getElementById('efficiencyChart'), {{
            type: 'bar',
            data: {{
                labels: ranks,
                datasets: [
                    {{ label: labels['regular_acc1'], data: {json.dumps(efficiency_data['regular_acc1'])}, backgroundColor: colors['regular_acc1'] }},
                    {{ label: labels['regular_acc8'], data: {json.dumps(efficiency_data['regular_acc8'])}, backgroundColor: colors['regular_acc8'] }},
                    {{ label: labels['quantum_acc1'], data: {json.dumps(efficiency_data['quantum_acc1'])}, backgroundColor: colors['quantum_acc1'] }}
                ]
            }},
            options: {{ scales: {{ y: {{ max: 100, ticks: {{ callback: function(value) {{ return value + "%" }} }}, grid: {{ color: '#374151' }} }} }}, plugins: {{ legend: {{ labels: {{ color: '#f9fafb' }} }} }} }}
        }});
    </script>
</body>
</html>
    """
    with open("dashboard_advanced.html", "w") as f:
        f.write(html)
    print("✅ Advanced Dashboard generated: dashboard_advanced.html")

if __name__ == "__main__":
    generate_improved_dashboard()
