import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import os
import time

def load_data():
    epochs = []
    accs = []
    losses = []
    
    if not os.path.exists("live_stats.json"):
        return [], [], []
        
    try:
        with open("live_stats.json", "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    epochs.append(data['epoch'])
                    accs.append(data['accuracy'])
                    losses.append(data['loss'])
    except:
        pass # Handle concurrent read/write
        
    return epochs, accs, losses

# 1. SETUP PLOTS
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#030712')
plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.3)

# Initial Styling
for ax in [ax1, ax2]:
    ax.set_facecolor('#111827')
    ax.grid(color='#374151', linestyle='--', alpha=0.5)
    ax.tick_params(colors='#9ca3af')
    for spine in ax.spines.values():
        spine.set_edgecolor('#374151')

ax1.set_title("Loss Trajectory", color='#60a5fa', pad=20, fontdict={'weight': 'bold', 'size': 14})
ax2.set_title("Success Rate (Accuracy)", color='#f43f5e', pad=20, fontdict={'weight': 'bold', 'size': 14})

# Lines
line1, = ax1.plot([], [], color='#60a5fa', linewidth=3, marker='o', markersize=6, markerfacecolor='#030712')
line2, = ax2.plot([], [], color='#f43f5e', linewidth=3, marker='o', markersize=6, markerfacecolor='#030712')

# Header Text
header = fig.text(0.5, 0.95, "ThreadFlow Engine: LIVE TELEMETRY", ha='center', fontsize=20, color='#f9fafb', fontweight='bold')
subheader = fig.text(0.5, 0.9, "Distributed Training Heartbeat", ha='center', fontsize=12, color='#9ca3af')

def update(frame):
    epochs, accs, losses = load_data()
    
    if len(epochs) > 0:
        # Update Lines
        line1.set_data(epochs, losses)
        line2.set_data(epochs, accs)
        
        # Rescale Axes
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        
        # Update Subheader
        latest_acc = f"{accs[-1]*100:.2f}%"
        latest_loss = f"{losses[-1]:.4f}"
        subheader.set_text(f"Current Epoch: {epochs[-1]} | Latest Accuracy: {latest_acc} | Latest Loss: {latest_loss}")

    return line1, line2, subheader

if __name__ == "__main__":
    # Clear old stats when starting dashboard
    # (Actually, let the trainer clear it, but we can do it here for clean testing)
    print("🚀 ThreadFlow Live Dashboard initializing...")
    ani = animation.FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
    plt.show()
