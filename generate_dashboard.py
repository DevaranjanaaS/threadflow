import json
import os

def generate_html():
    if not os.path.exists("scaling_summary.json"):
        print("scaling_summary.json not found!")
        return

    with open("scaling_summary.json", "r") as f:
        data = json.load(f)

    # Prepare labels and datasets for Chart.js
    ranks = sorted([int(r) for r in data.keys()])
    
    # Avg Epoch Time
    avg_times = []
    compute_times = []
    comm_times = []
    for r in ranks:
        avg_times.append(sum(data[str(r)]["epoch_times"]) / len(data[str(r)]["epoch_times"]))
        compute_times.append(sum(data[str(r)]["compute_times"]) / len(data[str(r)]["compute_times"]))
        comm_times.append(sum(data[str(r)]["comm_times"]) / len(data[str(r)]["comm_times"]))

    # Ideal Scaling
    t1 = avg_times[0]
    ideal_times = [t1 / r for r in ranks]
    
    # Speedup
    speedup = [t1 / t for t in avg_times]
    ideal_speedup = [float(r) for r in ranks]

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ThreadFlow Scaling Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0f172a;
            --card: #1e293b;
            --primary: #38bdf8;
            --secondary: #818cf8;
            --text: #f8fafc;
        }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            text_align: center;
            margin-bottom: 50px;
        }}
        h1 {{
            font-weight: 800;
            font-size: 3rem;
            margin-bottom: 10px;
            background: linear-gradient(to right, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}
        .card {{
            background: var(--card);
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
        }}
        .full-width {{
            grid-column: span 2;
        }}
        h2 {{
            margin-top: 0;
            font-size: 1.25rem;
            color: var(--primary);
        }}
        canvas {{
            width: 100% !important;
            height: 350px !important;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>ThreadFlow Analyzer</h1>
            <p>High-Performance MPI Neural Network Profiler</p>
        </header>
        
        <div class="grid">
            <div class="card">
                <h2>Strong Scaling: Time per Epoch</h2>
                <canvas id="scalingChart"></canvas>
            </div>
            <div class="card">
                <h2>Speedup: Ideal vs Actual</h2>
                <canvas id="speedupChart"></canvas>
            </div>
            <div class="card full-width">
                <h2>Bottleneck Analysis: Compute vs Communication</h2>
                <canvas id="bottleneckChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        const ranks = {json.dumps(ranks)};
        
        // Scaling Chart
        new Chart(document.getElementById('scalingChart'), {{
            type: 'line',
            data: {{
                labels: ranks,
                datasets: [
                    {{ label: 'Actual Time (s)', data: {json.dumps(avg_times)}, borderColor: '#38bdf8', tension: 0.1 }},
                    {{ label: 'Ideal Time (s)', data: {json.dumps(ideal_times)}, borderColor: '#64748b', borderDash: [5,5], tension: 0.1 }}
                ]
            }},
            options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color: '#f8fafc' }} }} }} }}
        }});

        // Speedup Chart
        new Chart(document.getElementById('speedupChart'), {{
            type: 'line',
            data: {{
                labels: ranks,
                datasets: [
                    {{ label: 'Actual Speedup', data: {json.dumps(speedup)}, borderColor: '#818cf8', tension: 0.1 }},
                    {{ label: 'Ideal Linear', data: {json.dumps(ideal_speedup)}, borderColor: '#64748b', borderDash: [5,5], tension: 0.1 }}
                ]
            }},
            options: {{ scales: {{ y: {{ beginAtZero: true }} }}, plugins: {{ legend: {{ labels: {{ color: '#f8fafc' }} }} }} }}
        }});

        // Bottleneck Chart
        new Chart(document.getElementById('bottleneckChart'), {{
            type: 'bar',
            data: {{
                labels: ranks,
                datasets: [
                    {{ label: 'Compute Time (s)', data: {json.dumps(compute_times)}, backgroundColor: '#38bdf8' }},
                    {{ label: 'Comm Time (s)', data: {json.dumps(comm_times)}, backgroundColor: '#f43f5e' }}
                ]
            }},
            options: {{ 
                scales: {{ x: {{ stacked: true }}, y: {{ stacked: true }} }},
                plugins: {{ legend: {{ labels: {{ color: '#f8fafc' }} }} }}
            }}
        }});
    </script>
</body>
</html>
    """
    with open("dashboard.html", "w") as f:
        f.write(html)
    print("✅ Dashboard generated: dashboard.html")

if __name__ == "__main__":
    generate_html()
