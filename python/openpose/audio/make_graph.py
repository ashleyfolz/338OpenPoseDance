import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================
# FILE PATHS (EDIT THESE)
# ============================
FOOTSTRIKE_CSV = "foot_strikes.csv"
BEATS_CSV      = "emily_mov_1_short.beats.csv"
# ============================

# --- Load footstrike CSV ---
foot = pd.read_csv(FOOTSTRIKE_CSV)

required_fs_cols = {"time_s", "side"}
if not required_fs_cols.issubset(foot.columns):
    raise ValueError(f"Footstrike CSV must contain columns: {required_fs_cols}")

left_times  = foot[foot["side"] == "left"]["time_s"].to_numpy()
right_times = foot[foot["side"] == "right"]["time_s"].to_numpy()

# --- Load beats CSV ---
with open(BEATS_CSV, "r") as f:
    first_line = f.readline().strip()

# extract bpm from first line: "tempo_bpm,103.3594"
try:
    bpm_value = float(first_line.split(",")[1])
except:
    bpm_value = None

# Load the rest normally (skip first row)
beats = pd.read_csv(BEATS_CSV, skiprows=1, names=["beat_time_s"])

beat_times = beats["beat_time_s"].dropna().to_numpy()

# --- Create Scatter Plot ---
plt.figure(figsize=(12,6))

# Plot beats as black circles
plt.scatter(beat_times, np.zeros_like(beat_times), 
            color="black", label=f"Beat Times (BPM {bpm_value:.2f})", s=30)

# Plot left foot strikes as red triangles
plt.scatter(left_times, np.zeros_like(left_times), 
            color="red", label="Left Foot Strike", marker="o", s=60, alpha=0.8)

# Plot right foot strikes as blue triangles
plt.scatter(right_times, np.zeros_like(right_times), 
            color="blue", label="Right Foot Strike", marker="o", s=60, alpha=0.8)

# ----------------------------
# Prettify
# ----------------------------
plt.yticks([])  # remove vertical axis since all events are points on a timeline
plt.xlabel("Time (s)")
plt.title("Beat Times + Footstrike Times (Aligned on Timeline)")
plt.legend()
plt.grid(True, axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
